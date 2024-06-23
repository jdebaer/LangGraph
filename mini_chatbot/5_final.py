from dotenv import load_dotenv
load_dotenv('./vars/.env') 

from typing import Annotated, Literal

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langchain_core.pydantic_v1 import BaseModel
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.checkpoint.sqlite import SqliteSaver
memory = SqliteSaver.from_conn_string(":memory:")

class State(TypedDict):
    messages: Annotated[list, add_messages]
    ask_human: bool

class RequestAssistance(BaseModel):
    """Escalate the conversation to an expert. Use this if you are unable to assist directly or if the user requires support beyond your permissions.
    To use this function, relay the user's 'request' so the expert can provide the right guidance.
    """
    request: str

tool = TavilySearchResults(max_results=2)
tools = [tool, RequestAssistance]
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")
llm = llm.bind_tools(tools)

def chatbot(state: State):
    response = llm.invoke(state["messages"])
    ask_human = False
    if response.tool_calls and response.tool_calls[0]["name"] == RequestAssistance.__name__:
        ask_human = True
    return {"messages": [response], "ask_human": ask_human}

def human_node(state: State):
    new_messages = []
    if not isinstance(state["messages"][-1], ToolMessage):	
        new_messages.append(
                           ToolMessage(content="Human didn't interject.", tool_call_id=state["messages"][-1].tool_calls[0]["id"])
                           )
    return {"messages": new_messages, "ask_human": False}

def select_next_node(state: State) -> Literal["human", "tools", "__end__"]:
    if state ["ask_human"]:
        return "human"
    return tools_condition(state)
    
graph_builder = StateGraph(State)

graph_builder.add_node("human", human_node)
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges("chatbot", select_next_node, {"human": "human", "tools": "tools", "__end__": "__end__"})

graph_builder.add_edge("human", "chatbot")
graph_builder.add_edge("tools", "chatbot")

graph_builder.set_entry_point("chatbot")

graph = graph_builder.compile(checkpointer=memory, interrupt_before=["human"])

config = {"configurable": {"thread_id": "1"}}

events = graph.stream(
    {
        "messages": [("user", "What is LangGraph?")]
    },
    config,
    stream_mode="values"
)

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

events = graph.stream(
    {
        "messages": [
            ("user", "Ya that's helpful. Maybe I'll build an autonomous agent with it!")
        ]
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

to_replay = None

for state in graph.get_state_history(config):
    print("Num messages: ", len(state.values["messages"]), "Next: ", state.next)
    print("-" * 80)
    if len(state.values["messages"]) == 2:
        to_replay = state

print(to_replay.next)
print(to_replay.config)

for event in graph.stream(None, to_replay.config, stream_mode="values"):
    if "messages" in event:
        event["messages"][-1].pretty_print()





