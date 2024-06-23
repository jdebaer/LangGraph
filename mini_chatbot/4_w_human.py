from dotenv import load_dotenv
load_dotenv('./vars/.env') 

from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.checkpoint.sqlite import SqliteSaver
memory = SqliteSaver.from_conn_string(":memory:")

class State(TypedDict):
    messages: Annotated[list, add_messages]
    ask_human: bool

graph_builder = StateGraph(State)

# The below checks out according to: https://python.langchain.com/v0.1/docs/modules/model_io/chat/function_calling/
# Subclassing Tool is equiv. to using @tool decorator, subclassing pydantic BaseModel is the third way.

from langchain_core.pydantic_v1 import BaseModel

class RequestAssistance(BaseModel):
    """Escalate the conversation to an expert. Use this if you are unable to assist directly or if the user requires support beyond your permissions.
    To use this function, relay the user's 'request' so the expert can provide the right guidance.
    """
    request: str

tool = TavilySearchResults(max_results=2)
# New as well.
tools = [tool, RequestAssistance]
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")
llm = llm.bind_tools(tools)

# We need changes to our chatbot node.

def chatbot(state: State):

    # Basically we're intercepting the response of the model in this node and doing some additional state manipulating based on it.
    # Note that we're changing the state based on our outputs here.

    response = llm.invoke(state["messages"])

    ask_human = False

    if response.tool_calls and response.tool_calls[0]["name"] == RequestAssistance.__name__:
        ask_human = True

    # Didn't test but these must of course match the state we have defined above.
    return {"messages": [response], "ask_human": ask_human}

# We need a new node now, "human_node".

def human_node(state: State):
    new_messages = []

    # Note that we get here because the LLL told us to call a tool. The LLM is now expecting a ToolMessage. If it doesn't get one it will throw an error.
    # So either the human or ourselves in the code here (default) have to append a ToolsMessage.

    if not isinstance(state["messages"][-1], ToolMessage):		# If the human updated the state, it will have done so with an appended ToolMessage.
        # Note that state["messages"][-1] is of type AIMessage if we get in this branch.
        new_messages.append(
                           ToolMessage(content="Human didn't interject.", tool_call_id=state["messages"][-1].tool_calls[0]["id"])
                           )

    # We either append our ToolMessage (default) or []. Note that before we didn't append to the state, only to new_messages.
    return {"messages": new_messages, "ask_human": False}
    
graph_builder.add_node("human", human_node)

# No changes to the below until otherwise noted.

graph_builder.add_node("chatbot", chatbot)

# Tools node does not get RequestAssistance, we're going to handle this in a new "human" node.
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)


# New.
# tools_condition simply checks to see if the chatbot has responded with any tool_calls in its response message. If so, it routes to the "tools" action node. 
# Otherwise, it ends the graph. This does not work for us now: we need to check for the flag in the state first and route to the human node if the flag is set. 
# If the flag is not set, then we let tools_condition run as is.

def select_next_node(state: State):

    if state ["ask_human"]:
        return "human"

    return tools_condition(state)

#graph_builder.add_conditional_edges("chatbot", tools_condition)	
graph_builder.add_conditional_edges("chatbot", select_next_node, {"human": "human", "tools": "tools", "__end__": "__end__"})
graph_builder.add_edge("human", "chatbot")

# New stuff ends here.

graph_builder.add_edge("tools", "chatbot")

graph_builder.set_entry_point("chatbot")

# We interrupt before we enter the "human" action node, so that a human can append a ToolMessage (from the pov of the LLM it's a tool call).
graph = graph_builder.compile(checkpointer=memory, interrupt_before=["human"])






## All the below is interacting with the model and triggering the interrupt. This requires converting this script to ipynb and running it in JN.

user_input = "I need some expert guidance on if cats are better than dogs. Could you request assistance for me?"
config = {"configurable": {"thread_id": "1"}}
# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

#--

snapshot = graph.get_state(config)
snapshot.next
# Shows that 'human' is the next step, so that the interrupt worked.

#--

from langchain_core.messages import AIMessage, ToolMessage

def create_response(response: str, ai_message: AIMessage):
    return ToolMessage(
        content=response,
        tool_call_id=ai_message.tool_calls[0]["id"],
    )

ai_message = snapshot.values["messages"][-1]
human_response = (
    "We, the experts are here to help! Cats are better than dogs."
)
tool_message = create_response(human_response, ai_message)
graph.update_state(config, {"messages": [tool_message]})

#--

graph.get_state(config).values["messages"]

#--

# Starting the graph with None takes it up it where it was interrupted.
events = graph.stream(None, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
