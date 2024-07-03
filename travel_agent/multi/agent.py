from dotenv import load_dotenv
load_dotenv('../vars/.env') 

from tools import *
from utilities import *
from utilities import _print_event 


# State

from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages

#New
# Using Optional here because we want to allow for specific value of None.
def update_dialog_stack(existing_list: list[str], new_element: Optional[str]) -> list[str]:
    if new_element is None:
        return existing_list
    if new_element == "pop":
        # This returns the list minus the last element.
        return existing_list[:-1]
    return existing_list + [new_element]
    


class State(TypedDict):
    messages: Annotated[list[ AnyMessage ], add_messages]
    user_info: str
    #New
    dialog_state: Annotated[list[ Literal["primary_assistant", "flight_assistant", "car_assistent", "hotel_assistant", "excursion_assistant"] ], update_dialog_stack]


# Chatbot w/ template containing a good system prompt.

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
#New
from langchain_core.pydantic_v1 import BaseModel, Field

# All chatbot nodes including the assistants are based on this Chatbot class.
# They do the same thing, only differ in what the runnable does.
# The runnable is the combo of a specific prompt + dedicated llm with specialized toolset for the job.
class Assistant:

    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)

            if not result.tool_calls and (not result.content or
				isinstance(result.content, list) and not result.content[0].get("text")):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result} 

# New new new

# New tool for delegates to return control to supervisor.
class CompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant,
    who can re-route the dialog based on the user's needs."""

    cancel: bool = True
    reason: str

    class Config:
        schema_extra = {
            "example": {
                "cancel": True,
                "reason": "User changed their mind about the current task.",
            },
            "example 2": {
                "cancel": True,
                "reason": "I have fully completed the task.",
            }
            "example 3": {
                "cancel": False,
                "reason": "I need to search the user's emails or calendar for more information.",
            },
        }

# The delegates.
flight_assistant_prompt = ChatPromtTemlate.from_messages(
    # List with a first system message in it and a placeholder for all other messages that will be appended.
    [
        ("system",

        "You are a specialized assistant for handling flight updates. "
        " The primary assistant delegates work to you whenever the user needs help updating their bookings. "
        "Confirm the updated flight details with the customer and inform them of any additional fees. "
        " When searching, be persistent. Expand your query bounds if the first search returns no results. "
        "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
        " Remember that a booking isn't completed until after the relevant tool has successfully been used."
        "\n\nCurrent user flight information:\n<Flights>\n{user_info}\n</Flights>"
        "\nCurrent time: {time}."
        "\n\nIf the user needs help, and none of your tools are appropriate for it, then"
        ' "CompleteOrEscalate" the dialog to the host assistant. Do not waste the user\'s time. Do not make up invalid tools or functions.',
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

flight_assistant_safe_tools = [search_flights]
flight_assistant_sensitive_tools = [update_ticket_to_new_flight, cancel_ticket]
flight_assistant_tools = flight_assistant_safe_tools + flight_assistant_sensitive_tools
flight_assistant_runnable = flight_assistant_prompt | llm.bind_tools(flight_assistant_tools + [CompleteOrEscalate])

hotel_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
        
        "You are a specialized assistant for handling hotel bookings. "
        "The primary assistant delegates work to you whenever the user needs help booking a hotel. "
        "Search for available hotels based on the user's preferences and confirm the booking details with the customer. "
        " When searching, be persistent. Expand your query bounds if the first search returns no results. "
        "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
        " Remember that a booking isn't completed until after the relevant tool has successfully been used."
        "\nCurrent time: {time}."
        '\n\nIf the user needs help, and none of your tools are appropriate for it, then "CompleteOrEscalate" the dialog to the host assistant.'
        " Do not waste the user's time. Do not make up invalid tools or functions."
        "\n\nSome examples for which you should CompleteOrEscalate:\n"
        " - 'what's the weather like this time of year?'\n"
        " - 'nevermind i think I'll book separately'\n"
        " - 'i need to figure out transportation while i'm there'\n"
        " - 'Oh wait i haven't booked my flight yet i'll do that first'\n"
        " - 'Hotel booking confirmed'",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

hotel_assistant_safe_tools = [search_hotels]
hotel_assistant_sensitive_tools = [book_hotel, update_hotel, cancel_hotel]
hotel_assistant_tools = hotel_assistant_safe_tools + hotel_assistant_sensitive_tools
hotel_assistant_runnable = hotel_assistant_prompt | llm.bind_tools(
    hotel_assistant_tools + [CompleteOrEscalate]
)

car_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
        "system",
        "You are a specialized assistant for handling car rental bookings. "
        "The primary assistant delegates work to you whenever the user needs help booking a car rental. "
        "Search for available car rentals based on the user's preferences and confirm the booking details with the customer. "
        " When searching, be persistent. Expand your query bounds if the first search returns no results. "
        "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
        " Remember that a booking isn't completed until after the relevant tool has successfully been used."
        "\nCurrent time: {time}."
        "\n\nIf the user needs help, and none of your tools are appropriate for it, then "
        '"CompleteOrEscalate" the dialog to the host assistant. Do not waste the user\'s time. Do not make up invalid tools or functions.'
        "\n\nSome examples for which you should CompleteOrEscalate:\n"
        " - 'what's the weather like this time of year?'\n"
        " - 'What flights are available?'\n"
        " - 'nevermind i think I'll book separately'\n"
        " - 'Oh wait i haven't booked my flight yet i'll do that first'\n"
        " - 'Car rental booking confirmed'",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

car_assistant_safe_tools = [search_car_rentals]
car_assistant_sensitive_tools = [
    book_car_rental,
    update_car_rental,
    cancel_car_rental,
]
car_assistant_tools = car_assistant_safe_tools + car_assistant_sensitive_tools
car_assistant_runnable = car_assistant_prompt | llm.bind_tools(
    car_assistant_tools + [CompleteOrEscalate]
)

excursion_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
        "system",
        "You are a specialized assistant for handling trip recommendations. "
        "The primary assistant delegates work to you whenever the user needs help booking a recommended trip. "
        "Search for available trip recommendations based on the user's preferences and confirm the booking details with the customer. "
        "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
        " When searching, be persistent. Expand your query bounds if the first search returns no results. "
        " Remember that a booking isn't completed until after the relevant tool has successfully been used."
        "\nCurrent time: {time}."
        '\n\nIf the user needs help, and none of your tools are appropriate for it, then "CompleteOrEscalate" the dialog to the host assistant. Do not waste the user\'s time. Do not make up invalid tools or functions.'
        "\n\nSome examples for which you should CompleteOrEscalate:\n"
        " - 'nevermind i think I'll book separately'\n"
        " - 'i need to figure out transportation while i'm there'\n"
        " - 'Oh wait i haven't booked my flight yet i'll do that first'\n"
        " - 'Excursion booking confirmed!'",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

excursion_assistant_safe_tools = [search_trip_recommendations]
excursion_assistant_sensitive_tools = [book_excursion, update_excursion, cancel_excursion]
excursion_assistant_tools = excursion_assistant_safe_tools + excursion_assistant_sensitive_tools
excursion_assistant_runnable = excursion_assistant_prompt | llm.bind_tools(
    excursion_assistant_tools + [CompleteOrEscalate]
)

# New tools for supervisor to delegate work to the delegates

class ToFlightAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle flight updates and cancellations."""

    request: str = Field(
        description = "Any necessary followup questions the update flight assistant should clarify before proceeding."
    )

class ToHotelAssistant(BaseModel):
    """Transfer work to a specialized assistant to handle hotel bookings."""
    
    location: str = Field(
        description = "The location where the user wants to book a hotel."
    )
    checkin_date: str = Field(description = "The check-in date for the hotel.")
    checkout_date: str = Field(description = "The check-out date for the hotel.")
    request: str = Field(
        description = "Any additional information or request from the user regarding the hotel booking."

    class Config:
        schema_extra = {
            "example": {
                "location": "Zurich",
                "checkin_date": "2023-08-15",
                "checkout_date": "2023-08-20",
                "request": "I prefer a hotel near the city center with a room that has a view."
            }
        }

class ToCarAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle car rental bookings."""

    location: str = Field(
        description = "The location where the user wants to rent a car."
    )
    start_date: str = Field(description = "The start date of the car rental.")
    end_date: str = Field(description = "The end date of the car rental.")
    request: str = Field(
        description = "Any additional information or requests from the user regarding the car rental."
    )

    class Config:
        schema_extra = {
            "example": {
                "location": "Basel",
                "start_date": "2023-07-01",
                "end_date": 2023-07-05",
                "request": "I need a compact car with automatic transmission.",
            }
        }


class ToExcursionAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle trip recommendation and other excursion bookings."""

    location: str = Field(
        description = "The location where the user wants to book a recommended trip."
    )
    request: str = Field(
        desscription = "Any additional information or requests from the user regarding the trip recommendation."
    )
    class Config:
        schema_extra = {
            "example": {
                "location": "Lucerne",
                "request": "The user is interested in outdoor activities and scenic views.",
            }
        }

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
        "system",
        "You are a helpful customer support assistant for Swiss Airlines. "
        "Your primary role is to search for flight information and company policies to answer customer queries. "
        "If a customer requests to update or cancel a flight, book a car rental, book a hotel, or get trip recommendations, "
        "delegate the task to the appropriate specialized assistant by invoking the corresponding tool. You are not able to make these types of changes yourself."
        " Only the specialized assistants are given permission to do this for the user."
        "The user is not aware of the different specialized assistants, so do not mention them; just quietly delegate through function calls. "
        "Provide detailed information to the customer, and always double-check the database before concluding that information is unavailable. "
        " When searching, be persistent. Expand your query bounds if the first search returns no results. "
        " If a search comes up empty, expand your search before giving up."
        "\n\nCurrent user flight information:\n<Flights>\n{user_info}\n</Flights>"
        "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}")
    ]
).partial(time=datetime.now())

primary_assistant_tools = [
    TavilySearchResults(max_results=1),
    search_flights,
    lookup_policy,  
    ToFlightBookingAssistant,
    ToBookCarRental,
    ToHotelBookingAssistant,
    ToBookExcursion,
]

llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")

#sensitive_tool_names = {t.name for t in sensitive_tools}

primary_assistant_runnable = primary_assistant_prompt | llm.bind_tools(primary_assistant_tools)

# We don't have the actual assistant nodes yet (nodes are functions).
# We do this with a factory function.

from typing import Callable
from langchain_core.messages import ToolMessage

def create_assistant_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    
    def entry_node(state: State) -> dict:
        tool_call_id = state["message"][-1].tool_calls[0]["id"]
        return {
            "message": [
                ToolMessage(
                    content = f"The assistant is now the {assistant_name}." 
                    "Reflect on the above conversation between the host assistant and the user."
                    " The user's intent is unsatisfied. Use the provided tools to assist the user."
                    f"Remember, you are {assistant_name},"
                    " and the booking, update, other other action is not complete until"
                    " after you have successfully invoked the appropriate tool."
                    " If the user changes their mind or needs help for other tasks, call the"
                    " CompleteOrEscalate function to let the primary host assistant take control."
                    " Do not mention who you are - just act as the proxy for the assistant.",
                    tool_call_id = tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }
    return entry_node

flight_assistant_entry_node = create_assistant_entry_node("Flight Updates & Booking Assistant", "flight_assistant")
car_assistant_entry_node = create_assistant_entry_node("Car Rental Assistant", "car_assistant")
hotel_assistant_entry_node = create_assistant_entry_node("Hotel Booking Assistant", "hotel_assistant")
excursion_assistant_entry_node = create_assistant_entry_node("Trip Recommendation Assistant", "excursion_assistant")

# Node used for exiting any assistant

def leave_specialized_assistant(state: State) -> dict:
    """Pop the dialog stack and return to the main assistant.

    This lets the full graph explicitly track the dialog flow and delegate control
    to specific sub-graphs.
    """
    
    messages = []
    if state["messages"][-1].tool_calls:
        # Does not support parallel tool calls by the LLM - fix this.

        messages.append(
            ToolMessage(
                content = "Resuming dialog with the host assistant. Please reflect on the past conversation and assist the user as needed.",
                tool_call_id = state["messages"][-1].tool_calls[0]["id"]
            )
        )
        return {
            "dialog_state": "pop",
            "messages": messages
        }

flight_assistant_node = Asssitant(flight_assistant_runnable)
car_assistant_node = Assistant(car_assistant_runnable)
hotel_assistant_node = Assistant(hotel_assistant_runnable)
excursion_assistant_node = Assistant(excursion_assistant_node)

# End new new new

# Graph

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import tools_condition

graph_builder = StateGraph(State)

# Nodes
def user_info(state: State):
    return {"user_info": fetch_user_flight_information.invoke({})}

graph_builder.add_node("fetch_user_info", user_info)
graph_builder.add_edge("fetch_user_info", "primary_assistant")

graph_builder.add_node("primary_assistant", Assistant(primary_assistant_runnable))

# New graph stuff.

graph_builder.add_node("flight_assistant_entry", flight_assistant_entry_node)
graph_builder.add_node("flight_assistant", flight_assistant_node)
graph_builder.add_edge("flight_assistant_entry", "flight_assistant")

graph_builder.add_node("flight_assistant_sensitive_tools", create_tool_node_with_fallback(flight_assistant_sensitive_tools))
graph_builder.add_node("flight_assistant_safe_tools", create_tool_node_with_fallback(flight_assistant_safe_tools))
graph_builder.add_edge("flight_assistant_sensitive_tools", "flight_assistant")
graph_builder.add_edge("flight_assistant_safe_tools", "flight_assistant")
graph_builder.add_conditional_edges("flight_assistant", route_flight_assistant)

graph_builder.add_node("leave_specialized_assistant", leave_specialized_assistant)
graph_builder.add_edge("leave_specialized_assistant", "primary_assistant")

def route_flight_assistant(state: State) -> Literal["flight_assistant_safe_tools",
                                                    "flight_assistant_sensitive_tools",
                                                    "leave_specialized_assistant",
                                                    "__end__"
                                            ]:
    # tools_condition routes to ToolNode if the last message has tool calls. 
    # Otherwise, it routes to the end.
    route = tools_conditon(state)

    if route == END:
        return END
  
    # Some additional triage is needed based on WHAT tools are called.
    tool_calls = state["messages"][-1].tool_calls
    
    if any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
        return "leave_specialized_assistant"

    safe_toolnames = [t.name for t in flight_assistant_safe_tools]
    if all(tc["name"] in safe_toolnames for tc in tool_calls):
        return "flight_assistant_safe_tools"

    return "flight_assistant_sensitive_tools"


# New
#graph_builder.add_node("safe_tools", create_tool_node_with_fallback(safe_tools))
#graph_builder.add_node("sensitive_tools", create_tool_node_with_fallback(sensitive_tools))

# Starting point.
graph_builder.set_entry_point("fetch_user_info")

# New: custom tool router

def custom_tools_router(state: State) -> Literal["safe_tools", "sensitive_tools", "__end__"]:
    next_node_name = tools_condition(state)

    if next_node_name == END:
        return END

    ai_message = state["messages"][-1]
    # There can be multple tool calls -> enhance code for this e.g., via ANY statement.

    first_tool_call = ai_message.tool_calls[0]
    print("*"*80)
    print(first_tool_call["name"])
    print("*"*80)
    if first_tool_call["name"] in sensitive_tool_names:
        return "sensitive_tools"
    return "safe_tools"
    
# New.
# Edges.
#graph_builder.add_conditional_edges("chatbot", custom_tools_router)
#graph_builder.add_edge("safe_tools", "chatbot")
#graph_builder.add_edge("sensitive_tools", "chatbot")

mem = SqliteSaver.from_conn_string(":memory:")

graph = graph_builder.compile(
		checkpointer=mem,
		# New
		interrupt_before=["sensitive_tools"]
)

# Test code.

import shutil
import uuid

tutorial_questions = [
    "Hi there, what time is my flight?",
    "Am i allowed to update my flight to something sooner? I want to leave later today.",
    "Update my flight to sometime next week then",
    "The next available option is great",
    "what about lodging and transportation?",
    "Yeah i think i'd like an affordable hotel for my week-long stay (7 days). And I'll want to rent a car.",
    "OK could you place a reservation for your recommended hotel? It sounds nice.",
    "yes go ahead and book anything that's moderate expense and has availability.",
    "Now for a car, what are my options?",
    "Awesome let's just get the cheapest option. Go ahead and book for 7 days",
    "Cool so now what recommendations do you have on excursions?",
    "Are they available while I'm there?",
    "interesting - i like the museums, what options are there? ",
    "OK great pick one and book it for my second day there.",
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
