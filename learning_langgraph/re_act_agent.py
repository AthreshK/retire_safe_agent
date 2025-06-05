from dotenv import load_dotenv
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict
from langchain.schema import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# This is a tool node that will be used to find the athresh factor of two numbers.
# The definition provided in docstring is very important for the LLM to understand what the tool does.
@tool
def find_athresh_factor(a:int, b:int) -> int:
    """This is a function that finds the athresh factor of 2 numbers. athresh factor is an integer"""
    athresh_factor = (a + b) * 12072000
    return athresh_factor

tools = [find_athresh_factor]

# binding all the tools to the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest").bind_tools(tools = tools)

def model_call(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content= " You are my AI asssistant, please answer my query to the best of your ability")
    
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages" : response}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    
    # Checks if all tool calls are complete and ends if done, no looping
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

# Add tools node to the graph
# This node will handle the tool calls and return the result to the agent
tool_node = ToolNode(tools = tools)
graph.add_node("tools",tool_node)

graph.add_edge(START,"our_agent")
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

# Function to print the stream of messages for debugging and user output
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message,tuple):
            print("here",message)
        else:
            message.pretty_print()
            
# Function to save the graph visualization as PNG
def save_graph_visualization(graph):
    """Save graph visualization if possible."""
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        with open("re_act_graph.png", "wb") as f:
            f.write(png_data)
        print("Graph visualization saved as re_act_graph.png")
    except Exception as e:
        print(f"Could not save graph visualization: {e}")

inputs = {"messages": [HumanMessage(content="what is the athresh factor of 3 and 5")]}

save_graph_visualization(app)
print_stream(app.stream(inputs, stream_mode= "values"))