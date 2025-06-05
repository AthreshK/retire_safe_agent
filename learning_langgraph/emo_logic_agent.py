from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from IPython.display import Image

# Load all the env variables from .env file, the API key
load_dotenv()

# Create the llm object for further usage
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# This is a data model used by the llm to give a structured output. 
#         This will be further used in the classify_message function in the classifier_llm wrapper creation.
#         The description here helps the llm understand what is expected from it but more clear instruction is given in the classify_message function.
class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="Classify if the message requires an emotional (therapist) or logical response."
    )

# This is the state of the agent, the memory which has all global variables, these are updated as the conversation progresses.
#         messages: list of messages exchanged between the user and the agent
#           - Noice he Annotated type, This does 2 things, ensure its of a datatype ( the list of BaseMessage)
#           - And says that whenever a node returns "messages", invoke the add_messages instead of replacing the whole thing with that output.
#           - the add_messages, which is a inbuilt method thing, adds the message to the list.
#           - This is important to keep the conversation history for context.
#           - Alternatively, we could do {"messages": add_messages(state["messages"], AIMessage/HumanMessage(content=reply.content))} everytime manually, but this is cleaner.
#         message_type: the type of message that is classified by the classifier
#         finished: boolean to indicate if the conversation is done or not

class State(TypedDict):
    messages: Annotated [list[BaseMessage], add_messages]
    message_type: str | None
    finshed: bool | None

# This is a node that takes input from the user and adds it to the messages and also checks if the text says "exit" or "quit" or "done" or "bye"
#         to end the conversation.
#         It returns the updated state with the new message and the finished flag.
# Notice the """ comment, this is a docstring for the function, it is used to describe what the function does. It also helps the llm understand the context of the function.
def human(state: State) -> State:
    """Add a human message to the state."""
    message = input("Enter message: ")
    return {
        "messages": HumanMessage(content=message),
        "message_type": state.get("message_type"),
        "finished": message.lower() in ["exit", "quit", "done","bye"]
    }

# This is a node that classifies the last user message as either emotional or logical.
#         It uses the llm to classify the message and returns the updated state with the message_type.
# Here, we use the llm.with_structured_output to create a wrapper around the llm that expects a structured output.
#         This is important because we want the llm to return a specific format that we can use in the next node.
# Notice we just use the **state to copy all the existing state variables and add the message_type to it.
# We dont add the message from AI here because it doesnt matter to history, its only for next node direction.
def classify_message(state: State) -> State:
    """Classify the last user message as emotional or logical."""
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        SystemMessage(
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

# This is a node that routes the conversation based on the message_type.
def therapist_agent(state: State) -> State:
    """Respond to the user message with emotional support."""
    instructions = [
        SystemMessage(
            content="""You are a compassionate therapist. Focus on the emotional aspects of the user's message.
Show empathy, validate their feelings, and help them process their emotions.
Ask thoughtful questions to help them explore their feelings more deeply.
Avoid giving logical solutions unless explicitly asked."""
        )
    ]
    reply = llm.invoke(instructions + state["messages"])
    print(f"Therapist agent reply: {reply.content}")
    return {
        "messages": AIMessage(content=reply.content),
        "message_type": state.get("message_type")
    }
    
# This is a node that responds to the user message with logical analysis or information.
def logical_agent(state: State) -> State:
    """Respond to the user message with logical analysis or information."""
    instructions = [
        SystemMessage(
            content="""You are a purely logical assistant. Focus only on facts and information.
Provide clear, concise answers based on logic and evidence.
Do not address emotions or provide emotional support.
Be direct and straightforward in your responses."""
        )
    ]
    reply = llm.invoke(instructions + state["messages"])
    
    print(f"Logical agent reply: {reply.content}")
    return {
        "messages": AIMessage(content=reply.content),
        "message_type": state.get("message_type")
    }

# Build the graph
# A StateGraph is a directed graph that represents the flow of the chatbot conversation.
graph_builder = StateGraph(State)

# Add nodes to the graph
graph_builder.add_node("human", human)
graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)

# Start with the human input node
graph_builder.add_edge(START, "human")

# We know that the human node will not only add the message to the state but also check for exit conditions.
# So we add a conditional edge to classifier node if the conversation is not finished.
# if it is finished, we go to END node.
graph_builder.add_conditional_edges("human", 
    lambda state: state.get("finished", False),
    {False: "classifier", True: END}
)

# The conditional edges from classifier to therapist or logical agent based on the message_type.
graph_builder.add_conditional_edges("classifier",
    lambda state: state["message_type"],
    {"emotional": "therapist", "logical": "logical"}
)

# Loop back to human after therapist or logical response
graph_builder.add_edge("therapist", "human")
graph_builder.add_edge("logical", "human")

# Compile the graph
graph = graph_builder.compile()

# Function to save the graph visualization as PNG
def save_graph_visualization(graph):
    """Save graph visualization if possible."""
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        with open("emo_logic_graph.png", "wb") as f:
            f.write(png_data)
        print("Graph visualization saved as chatbot_graph.png")
    except Exception as e:
        print(f"Could not save graph visualization: {e}")

# Function to run the chatbot
def run_chatbot():
    # Initialize the state for the chatbot, it should follow the structure defined in State
    state = State(messages=[], message_type=None, finished=False)

    try:
        # Invoke the graph to start the conversation
        print("Chatbot started. Type 'exit', 'quit', 'done', or 'bye' to end the conversation.")
        state = graph.invoke(state)
    except Exception as e:
        print(f"Error: {e}")

save_graph_visualization(graph)
run_chatbot()