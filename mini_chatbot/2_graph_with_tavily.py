from dotenv import load_dotenv
load_dotenv('./vars/.env') 

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

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

############ New ############

# Define a simple tool.
from langchain_community.tools.tavily_search import TavilySearchResults
tool = TavilySearchResults(max_results=2)
tools = [tool]

# Tell the model what tools it can call.
llm = llm.bind_tools(tools)

# Model cannot call the tool - we need a graph node to do this. Below we implement a OurToolNode class that checks the most recent message in the state
# and calls tools if the message contains tool_calls. It relies on the LLM's tool_calling` support, which is available in Anthropic, OpenAI, Google Gemini,
# and a number of other LLM providers. In the next step we'll replace this self-built class by LG's ToolNode.

import json
from langchain_core.messages import ToolMessage

class OurToolNode:

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if (messages := inputs.get("messages", [])):				# Return empty dict [] if the key does not exist.
            # Grab the last message.
            message = messages[-1]
        else:
            raise ValueError("No key 'messages' found in input.")

        outputs = []

        # We iterated of the tool calls in the last message.
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])

            outputs.append(
                          ToolMessage(
                                     content		= json.dumps(tool_result),
                                     name		= tool_call["name"],
                                     tool_call_id	= tool_call["id"]
                                     )
                          )
        return {"messages": outputs}

our_tool_node = OurToolNode(tools)

graph_builder.add_node("tools", our_tool_node)

# At this point we have the nodes, but no control flow yet - that is done via the edges in the graph. Conditional edges usually contain "if" statements 
# to route to different nodes depending on the current graph state. These functions receive the current graph state and return a string or list of strings 
# indicating which node(s) to call next.
# Below we implement our own edge funtion that checks for tool_calls in the chatbot's output. We'll replace this by LG's tools_condition in the next step.

from typing import Literal

def our_edge(state:State) -> Literal["tools", "__end__"]:

    # If the last message in the State has tool calls we route to our tool node, otherwise we route to the end since this means we have a final answer.
    if isinstance(state, list):
        ai_message = state[-1]
    elif (messages := state.get("messages", [])):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in state passed to edge: {state}.")

    # In the end this function return the name of the next node in the graph.
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return  "__end__"

graph_builder.add_conditional_edges("chatbot", our_edge, {"tools": "tools", "__end__": "__end__"})	# Call tools if edge str output was "tools" etc.
graph_builder.add_edge("tools", "chatbot")


				













############ New End ############

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever the node is used.
graph_builder.add_node("chatbot", chatbot)

# Tell graph with what node to start when we call the graph.
graph_builder.set_entry_point("chatbot")

# Tell the graph to exit when this node is reached.
#graph_builder.set_finish_point("chatbot")

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
            # New new new
            if isinstance(value["messages"][-1], BaseMessage):
                print("Assistant:", value["messages"][-1].content)






