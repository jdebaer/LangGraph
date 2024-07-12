from dotenv import load_dotenv
load_dotenv('./vars/.env') 

import re
from langchain_community.document_loaders import web_base
from typing import Iterator
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from langchain.graph import add_messages
from langchain import hub
from lanngchain_core.output_parsers import StrOutputParser

NEWLINE_RE = re.compile("\n+")

MAX_RETRIES = 3

SOURCE_URLS = [
    'https://pandas.pydata.org/docs/user_guide/indexing.html',
    'https://pandas.pydata.org/docs/user_guide/groupby.html',
    'https://pandas.pydata.org/docs/user_guide/merging.html'
]

class PandasDocsLoader(web_base.WebBaseLoader):
    def lazy_load(self) -> Iterator[Document]:
        # Lazy load text from the urls in web_path (inherited)
        for web_path in self.web_paths:
            scraping = self._scrape(web_path, bs_kwargs=self.bs_kwargs)
            text = scraping.get_text(**self.bs_get_text_kwargs)
            text = NEWLINE_RE.sub("\n", text)
            metadata = web_base._build_metadata(scraping, web_path)
            yield Document(page_content=text, metadata=metadata)

def chunk_documents(urls: list[str]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            r"In \[[0-9]+\]",
            r"\n+",
            r"\s+"
        ],
        is_separator_regex=True,
        chunk_size=1000
    )

    # Note: PandasDocsLoader(url).load() returns a list with one element of type Document.
    web_page_iterator = [PandasDocsLoader(url).load() for url in urls]

    web_page_list = [web_page for _web_page_list in web_page_iterator for web_page in _web_page_list]
    return text_splitter.split_documents(web_page_list)

def get_retriever() -> BaseRetriever:
    chunks = chunk_documents(SOURCE_URLS)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        collection_name="pandas-rag-chroma",
        embedding=OpenAIEmbeddings(),
    )
    retriever = vectorstore.as_retriever()
    return retriever

unbound_llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
retriever = get_retriever()
tavily_search_tool = TavilySearchResults(max_results=3)

# State

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    chunks: list[Documents]
    candidate_answser: str
    retries: int
    web_fallback: bool

class GraphConfig(TypedDict):
    max_retries: int

# Nodes

VERBOSE = False

# Node who's job is to do the RAG.
def doc_searcher_node(state: State):
    if VERBOSE:
        print("---RETRIEVE---")

    question = convert_to_messages(state["messates"][-1].content
    
    documents = retriever.invoke(question)

    # Return the dict that will update the state.
    # This sort of is the state initialization, done in the first node of our flow.
    return {
        "documents": documents,
        "question": question,
        "web_fallback": True
    }

# Node who's job is to generate the final response.

reponse_generator_prompt = hub.pull("rlm/rag-prompt")

def response_generator_node(state: State):
    if VERBOSE:
        print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    retries = state["retries"] if state.get("retries") is not None else -1

    response_generator_runnable = response_generator_prompt | unbound_llm | StrOutputParser()
    response = rag_runnable.invoke(
                {"context"  : documents,
                 "question" : question
                }
    )
    # Update the state.
    return {
        "retries"          : retries + 1,
        "candidate_answer" : response
    }

# Node who's job it is to rewrite the original question in case of a hallucination.

query_rewriter_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         "You are a question re-writer that converts an input question to "
         "a better version that is optimized for vectorstore retrieval. "
         "Look at the input and try to reason about the underlying "
         "semantic intent / meaning."
        ),
        ("human",
         "Here is the initial question: \n\n {question} \n Formulate an "
         "improved question."
        ),
    ]
)

def query_rewriter_node(state: State):
    if VERBOSE:
        print("---TRANSFORM QUESTION---")
    
    question = state["question"]
    
    query_rewriter_runnable = query_rewriter_prompt | unbound_llm | StrOutputParser()
    rewritten_question = query_rewriter_runnable.invoke({"question",question})
    # Update the state.
    return {"question": rewritten_question}

# Node who's job it is to search the web with Tavily

def web_searcher_node(state: State):
    if VERBOSE:
        print("---WEB SEARCH---")
    
    question = state["question"]
    documents = state["documents"]
    
    search_results = tavily_search_tool.invoke(question)
    
    seach_content = "\n".join[hit["content"] for hit in search_results]
    documents.append(Document(page_content=search_content, metadata={"source": "websearch"}))

    # Update the state. We can ony run web search once in this graph.
    return {"documents": documents, "web_fallback":False}

# Node who's job it is to convert the current candidate answer to an AIMessage.

def response_finalizer_node(state: State):
    if VERBOSE:
        print("---FINALIZE RESPONSE---")

    return {"messages": [AIMessage(content=state["candidate_answser"])]}

# Edges

# These are not tools, but input for .with_structured_output to show the model
# what the structured output should look like.
class GradeHallucinationAnswer(BaseModel):
    """Binary score for hallucination present in generated answer."""

    answer: str = Field(
        description="Response is grounded in the facts, 'yes' or 'no'."
    )

class GradeResponseAnswser(BaseModel):
    """Binary score to assess if response addresses the question."""

    answer: str = Field(
        desscription="Response addresses the question, 'yes' or 'no'."
    )

hall_grader_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader assessing whether an LLM generation is grounded "
            "in / supported by a set of retrieved facts. Give a binary score "
            "'yes' or 'no', where 'yes' means that the answer is grounded in / "
            "supported by the set of facts. IF the generation includes code examples "
            ", make sure those examples are FULLY present in the set of facts, "
            "otherwise always return score 'no'."
        ),
        (
            "human",
            "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"
        ),
    ]
)

answer_grader_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader assessing whether an answer addresses / resolves "
            "a question. Give a binary score 'yes' or 'no', where 'yes' means that "
            "the answer resolves the question."
        ),
        (
            "human",
            "User question: \n\n {question} \n\n LLM generation: {generation}"
        )
    ]
)


def from_response_generator(state:State, config) -> Literal["generate_response", 
                                                            "rewrite_query",
                                                            "search_web",
                                                            "finalize_response"]:
    question = state["question"]
    documents = state["chunks"]
    candidate_answser = state["candidate_answser"]
    web_fallback = state["web_fallback"]
    retries = state["retries"] if state.get["retries"] is not None else -1
    max_retries = config.get("configurable", {}).get("max_retries", MAX_RETRIES)

    if not web_fallback:
        return "finalize_response"

    if VERBOSE:
        print("---HALLUCINATIONS CHECK---")

    hall_grader_runnable = hall_grader_prompt | unbound_llm.with_structure_output(GrateHallucinationAnswer)
    hall_grade: GradeHallucinationAnswer = hall_grader_runnable.invoke(
        {
            "documents"  : documents,
            "generation" : generation
        } 
    )

    if hall_grade.answser == "no":
        IF VERBOSE:
            print("---HALLUCINATION DETECTED, RETRY---")
        return "generate" if retries < max_retries else "search_web"

    if VERBOSE:
        print("---WE ARE GROUNDED IN FACTS--")
        print("---RESPONSE ANSWER CHECK--")

    answer_grader_runnable = answer_grader_prompt | unbound_llm.with_structured_output(GradeResponseAnswser)

    answer_grade: GradeResponseAnswser = answer_grader_runnable.invoke(
        {
            "question" : question,
            "generation" : generation
        }
    )

    if answer_grade.answer == "yes":
        if VERBOSE:
            print("---RESPONSE ANSWERS QUESTION---")
        return "finalize_response"

    else:
        if VERBOSE:
            print("---RESPONSE DOES NOT ANSWER QUESTION---")
        return "rewrite_query" if retries < mas_retries else "search_web"

# Assemble graph.

graph_builder = StateGraph(State, config_schema=GraphConfig)

graph_builder.add_node("doc_searcher", doc_searcher_node)
graph_builder.add_node("response_generator", response_generator_node)
graph_builder.add_node("query_rewriter", query_rewriter_node)
graph_builder.add_node("web_searcher", web_searcher_node)
graph_builder.add_node("response_finalizer", response_finalizer_node)

graph_builder.add_edge(START, "doc_search")
graph_builder.add_edge("doc_search", "response_generator")

graph_builder.add_conditional_edges("response_generator", from_response_generator,
    {
        "generate_response" : "response_generator",
        "rewrite_query"     : "query_rewriter",
        "search_web"        : "web_searcher",
        "finalize_response" : "response_finalizer"
    }
)

graph_builder.add_edge("query_rewriter", "doc_searcher")
graph_builder.add_edge("web_searcher", "response_generator")
graph_builder.add_edge("response_finalizer", END)

graph = graph_builder.compile()

# Testing

VERBOSE = True

question = {"messages": [("human", "How do I calculate sum by groups")]}

for event in graph.stream(question)
    print(event.content)
    print("\n----\n")

















if __name__ == "__main__":
    get_retriever()

