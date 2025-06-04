from dotenv import load_dotenv
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain.schema import HumanMessage, AIMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="Classify if the message requires an emotional (therapist) or logical response."
    )

class State(TypedDict):
    messages: list
    message_type: str | None

def classify_message(state: State) -> State:
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        HumanMessage(
            content="""Classify the user message as either:
- 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
- 'logical': if it asks for facts, information, logical analysis, or practical solutions
"""
        ),
        last_message
    ])
    return {
        **state,
        "message_type": result.message_type
    }

def router(state: State) -> State:
    return {
        "messages" : state["messages"],
        "message_type" : state.get("message_type", "logical")
    }

def therapist_agent(state: State) -> State:
    last_message = state["messages"][-1]

    messages = [
        HumanMessage(
            content="""You are a compassionate therapist. Focus on the emotional aspects of the user's message.
Show empathy, validate their feelings, and help them process their emotions.
Ask thoughtful questions to help them explore their feelings more deeply.
Avoid giving logical solutions unless explicitly asked."""
        ),
        last_message
    ]
    reply = llm.invoke(messages)
    return {
        "messages": add_messages(state["messages"], AIMessage(content=reply.content)),
        "message_type": state.get("message_type")
    }

def logical_agent(state: State) -> State:
    last_message = state["messages"][-1]

    messages = [
        HumanMessage(
            content="""You are a purely logical assistant. Focus only on facts and information.
Provide clear, concise answers based on logic and evidence.
Do not address emotions or provide emotional support.
Be direct and straightforward in your responses."""
        ),
        last_message
    ]
    reply = llm.invoke(messages)
    return {
        "messages": add_messages(state["messages"], AIMessage(content=reply.content)),
        "message_type": state.get("message_type")
    }

# Build the graph
graph_builder = StateGraph(State)

graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")

graph_builder.add_conditional_edges(
    "router",
    lambda state: state["message_type"],
    {"emotional": "therapist", "logical": "logical"}
)

graph_builder.add_edge("therapist", END)
graph_builder.add_edge("logical", END)

graph = graph_builder.compile()

# Chat loop
def run_chatbot():
    state: State = {"messages": [], "message_type": None}

    while True:
        user_input = input("Message: ")
        if user_input.lower().strip() == "exit":
            print("Bye")
            break

        state["messages"] = add_messages(state["messages"], HumanMessage(content=user_input))

        try:
            state = graph.invoke(state)
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")
        except Exception as e:
            print(f"Error: {e}")

run_chatbot()