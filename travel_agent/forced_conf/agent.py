from dotenv import load_dotenv
load_dotenv('../vars/.env') 

#from db_setup import *
#from db_setup import db
from tools import *
from utilities import *
from utilities import _print_event 


# State

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # New
    user_info: str


# Chatbot w/ template containing a good system prompt.

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig

class Chatbot:

    # Overriding the constructure only needed because we want to pass on the Runnable.
    # The runnable is a chain of a template + LLM invocation.
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            # New
            # No longer needed as user_info is now part of the state.
            #configurable = config.get("configurable", {})
            #passenger_id = configurable. get("passenger_id", None)
            #state = {**state, "user_info": passenger_id}
            
            result = self.runnable.invoke(state)

            if not result.tool_calls and (not result.content or
				isinstance(result.content, list) and not result.content[0].get("text")):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result} 


llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")

primary_chatbot_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for Swiss Airlines. "
            " Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
            "\nCurrent time: {time}."
        ),
        # placeholder is entirely and in-place replaced by the incoming messages.
        ("placeholder", "{messages}")
    ]
).partial(time=datetime.now())

tools = [
    TavilySearchResults(max_results=1),
    fetch_user_flight_information,
    search_flights,
    lookup_policy,
    update_ticket_to_new_flight,
    cancel_ticket,
    search_car_rentals,
    book_car_rental,
    update_car_rental,
    cancel_car_rental,
    search_hotels,
    book_hotel,
    update_hotel,
    cancel_hotel,
    search_trip_recommendations,
    book_excursion,
    update_excursion,
    cancel_excursion,
]

chatbot_runnable = primary_chatbot_prompt | llm.bind_tools(tools)

# Graph

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import tools_condition

graph_builder = StateGraph(State)

# Nodes
# New
# Create a transparent node that fetches the user info.
def user_info(state: State):
    # We have the state updated with the user info.
    # fetch_user_flight_information has access to passenger_id via context/config.
    return {"user_info": fetch_user_flight_information.invoke({})}

graph_builder.add_node("fetch_user_info", user_info)
graph_builder.add_edge("fetch_user_info", "chatbot")

graph_builder.add_node("chatbot", Chatbot(chatbot_runnable))
graph_builder.add_node("tools", create_tool_node_with_fallback(tools))

# Starting point.
# New
graph_builder.set_entry_point("fetch_user_info")

# Edges.
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

mem = SqliteSaver.from_conn_string(":memory:")

graph = graph_builder.compile(
		checkpointer=mem,
		# New
		interrupt_before=["tools"]
)

# Test code.

import shutil
import uuid

tutorial_questions = [
    "Hi there, what time is my flight?",
#    "Am i allowed to update my flight to something sooner? I want to leave later today.",
#    "Update my flight to sometime next week then",
#    "The next available option is great",
#    "what about lodging and transportation?",
#    "Yeah i think i'd like an affordable hotel for my week-long stay (7 days). And I'll want to rent a car.",
#    "OK could you place a reservation for your recommended hotel? It sounds nice.",
#    "yes go ahead and book anything that's moderate expense and has availability.",
#    "Now for a car, what are my options?",
#    "Awesome let's just get the cheapest option. Go ahead and book for 7 days",
#    "Cool so now what recommendations do you have on excursions?",
#    "Are they available while I'm there?",
#    "interesting - i like the museums, what options are there? ",
#    "OK great pick one and book it for my second day there.",
]

shutil.copy(backup_file, db)

thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        "passenger_id": "3442 587242",
        "thread_id": thread_id
    }
}

_printed = set()

for question in tutorial_questions:
    events = graph.stream(
        {"messages": ("user", question)},
        config=config,
        stream_mode="values"
    )
    
    for event in events:
        _print_event(event, _printed)

    # New ----
    snapshot = graph.get_state(config)  	# Need thread_id.
    
    # No more events are being streamed to us. So either the graph finished, or we were interrupted. If we were interrupted, then shapshot.next will not be empty.
    while snapshot.next:

        user_input = input("Type y to continue or change your request.\n\n")

        if user_input.strip() == "y":
        
            events = graph.stream(
                None,
                config=config,
                stream_mode="values"
            )
    
            for event in events:
                _print_event(event, _printed)
        
        else:
            # The LLM is expecting a ToolMessage, so we need to craft one before we can continue.
            
            events = graph.stream(
	    		{
	    		"messages": [
	     			ToolMessage(tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                                content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input."
                                #content=f"API call denied by user. Tell the user you are ready to assist without asking for specific data."
	    			)
	    		]
	    		},
                config=config,
                stream_mode="values"
            )
    
            for event in events:
                _print_event(event, _printed)
        snapshot = graph.get_state(config)	
