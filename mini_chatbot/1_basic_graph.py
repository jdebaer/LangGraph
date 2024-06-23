from dotenv import load_dotenv
load_dotenv('./vars/.env') 

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

# Every node we define will receive the current State as input and return a value that updates that state.

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # tells LangGraph to append new messages to the existing list, rather than overwriting it.
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

#from langchain_anthropic import ChatAnthropic
#llm = ChatAnthropic(model="claude-3-haiku-20240307")

from langchain_openai import ChatOpenAI
# Ideally put this in an env variable.
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever the node is used.
graph_builder.add_node("chatbot", chatbot)

# Tell graph with what node to start when we call the graph.
graph_builder.set_entry_point("chatbot")

# Tell the graph to exit when this node is reached.
graph_builder.set_finish_point("chatbot")

# This creates a "CompiledGraph" we can use invoke on our state.
graph = graph_builder.compile()

# Standalone way to visualize the graph.
#from PIL import Image
#png = graph.get_graph().draw_mermaid_png()
#with open('./image', "wb") as file:
#    file.write(png)
#    file.close()
#with open('./image', "rb") as file:
#    image = Image.open(file)
#    image.show()

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    for event in graph.stream({"messages": ("user", user_input)}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)






