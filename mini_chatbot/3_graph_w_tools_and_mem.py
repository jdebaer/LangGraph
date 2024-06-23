from dotenv import load_dotenv
load_dotenv('./vars/.env') 

from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
# New since we're shifting to calling LG's prebuilt libraries.
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.checkpoint.sqlite import SqliteSaver
memory = SqliteSaver.from_conn_string(":memory:")

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")
llm = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}
graph_builder.add_node("chatbot", chatbot)

# We shift to reusing LG's tool node class
tool_node = ToolNode(tools=[tool])
# Slight mod to call tool_node object.
graph_builder.add_node("tools", tool_node)

# New: tools_condition behaves like our self-implemented edge in 2_* but with more enterprise-readiness.
graph_builder.add_conditional_edges("chatbot", tools_condition)	

graph_builder.add_edge("tools", "chatbot")

graph_builder.set_entry_point("chatbot")

graph = graph_builder.compile(checkpointer=memory)

## Standalone way to visualize the graph.
#from PIL import Image
#png = graph.get_graph().draw_mermaid_png()
#with open('./image', "wb") as file:
#    file.write(png)
#    file.close()
#with open('./image', "rb") as file:
#    image = Image.open(file)
#    image.show()

config = {"configurable": {"thread_id": "1"}}

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    # Change made here.
    for event in graph.stream({"messages": ("user", user_input)}, config, stream_mode="values"):
        event["messages"][-1].pretty_print()






