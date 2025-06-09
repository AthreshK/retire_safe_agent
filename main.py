import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Annotated, TypedDict, List, Optional
import pandas as pd
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage, BaseMessage
from langchain_core.messages import ToolMessage
from langchain.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, InjectedState
from langchain_core.tools import InjectedToolCallId
from langgraph.types import Command

load_dotenv()

# -----------------------------
# Configuration and Data Models
# -----------------------------


class UserConfig(BaseModel):
    initial_savings: float = Field(
        description="Initial savings amount at retirement start")
    user_age: int = Field(description="User's current age")
    monthly_withdrawal: float = Field(
        description="Planned monthly withdrawal amount")
    interest_rate: float = Field(
        description="Annual interest rate applied to savings")
    inflation_rate: float = Field(description="Expected annual inflation rate")


class Deposit(TypedDict):
    age: int
    amount: float
    reason: Optional[str]


class Withdrawal(TypedDict):
    age: int
    amount: float
    reason: Optional[str]


class YearlyReport(BaseModel):
    age: int = Field(description="Age at the end of the year")
    living_expenses: float = Field(
        description="Estimated living expenses for the year")
    planned_expenses: float = Field(
        description="Planned expenses for the year")
    total_withdrawn: float = Field(
        description="Total amount withdrawn for the year")
    interest_earned: float = Field(
        description="Interest earned on savings for the year")
    planned_deposits: float = Field(
        description="Planned deposits for the year")
    total_deposited: float = Field(
        description="Total amount deposited for the year")
    balance: float = Field(
        description="Remaining balance at the end of the year")
    comment: Optional[str] = Field(
        default=None, description="Any additional comments or notes for the year")


class RetirementState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    user_config: UserConfig = Field(
        default=None, description="User's retirement configuration")
    deposits: List[Deposit] = Field(
        default_factory=list, description="List of planned deposits")
    withdrawals: List[Withdrawal] = Field(
        default_factory=list, description="List of planned withdrawals")
    prediction_report: Optional[List[YearlyReport]] = Field(
        description="Yearly prediction report of retirement plan")
    predicted_years: Optional[float] = Field(
        description="Number of years remaining until funds are depleted")
    is_last_report_generated: bool = Field(
        default=False, description="Flag to indicate if the last report was generated")


# -----------------------------
# Data Loading & Description
# -----------------------------

def read_user_data(state: RetirementState) -> RetirementState:
    with open("data/user_config.json", "r") as f:
        config_data = json.load(f)
        state["user_config"] = UserConfig(**config_data)

    df = pd.read_csv("data/financial_events.csv")
    state["deposits"] = [
        {"age": int(row["age"]), "amount": float(
            row["amount"]), "reason": row.get("reason", "")}
        for _, row in df[df["type"] == "deposit"].iterrows()
    ]
    state["withdrawals"] = [
        {"age": int(row["age"]), "amount": float(
            row["amount"]), "reason": row.get("reason", "")}
        for _, row in df[df["type"] == "withdrawal"].iterrows()
    ]
    return state


def describe_retirement_data(state: RetirementState) -> str:
    lines = []
    for field_name, model_field in UserConfig.model_fields.items():
        label = model_field.description
        value = getattr(state["user_config"], field_name)
        lines.append(f"{field_name} ({label}): {value}")
    lines.append("\nPlanned Deposits and Withdrawals:")
    for i, deposit in enumerate(state["deposits"]):
        lines.append(
            f"Deposit {i+1}: At Age {deposit['age']}, Amount ${deposit['amount']:.2f}, Reason: {deposit.get('reason', 'N/A')}")
    for i, withdrawal in enumerate(state["withdrawals"]):
        lines.append(
            f"Withdrawal {i+1}: At Age {withdrawal['age']}, Amount ${withdrawal['amount']:.2f}, Reason: {withdrawal.get('reason', 'N/A')}")
    return "\n".join(lines)


def describe_prediction_report(state: RetirementState) -> str:
    if state["prediction_report"] is None:
        return "No prediction report available. Please generate one first."

    if not state["prediction_report"]:
        return "No prediction report available. Please generate one first."

    # Prepare table headers
    headers = [
        "Age", "Living Expenses", "Planned Expenses", "Total Withdrawn",
        "Interest Earned", "Planned Deposits", "Total Deposited", "Balance", "Comments"
    ]
    # Calculate column widths
    col_widths = [max(len(h), 8) for h in headers]
    rows = []
    for report in state["prediction_report"]:
        row = [
            str(report.age),
            f"${report.living_expenses:,.2f}",
            f"${report.planned_expenses:,.2f}",
            f"${report.total_withdrawn:,.2f}",
            f"${report.interest_earned:,.2f}",
            f"${report.planned_deposits:,.2f}",
            f"${report.total_deposited:,.2f}",
            f"${report.balance:,.2f}",
            (report.comment or "")
        ]
        # Update column widths for each row
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
        rows.append(row)

    # Build table string
    def format_row(row):
        return " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))

    table_lines = []
    table_lines.append(format_row(headers))
    table_lines.append("-+-".join("-" * w for w in col_widths))
    for row in rows:
        table_lines.append(format_row(row))
    return "\n".join(table_lines)


def generate_prediction_report(state: RetirementState) -> tuple:

    balance = state["user_config"].initial_savings
    monthy_withdrawal = state["user_config"].monthly_withdrawal
    interest_rate = state["user_config"].interest_rate / 100
    inflation_rate = state["user_config"].inflation_rate / 100
    current_age = state["user_config"].user_age
    reports = []

    months = 0
    while (balance >= monthy_withdrawal):
        balance -= monthy_withdrawal
        months += 1

        if months % 12 == 0:  # yearly update
            living_expenses = monthy_withdrawal * 12
            comments = []

            interest_earned = balance * interest_rate  # interest earned
            balance += interest_earned  # add interest to balance

            monthy_withdrawal *= (1 + inflation_rate)  # adjust for inflation

            planned_deposits = 0
            for deposit in state["deposits"]:
                if deposit["age"] == current_age:
                    balance += deposit["amount"]
                    planned_deposits += deposit["amount"]
                    comments.append(
                        f"Deposited ${deposit['amount']:.2f} from {deposit.get('reason', 'N/A')}")

            planned_expenses = 0
            for withdrawal in state["withdrawals"]:
                if withdrawal["age"] == current_age:
                    inflated_withdrawal = withdrawal["amount"] * \
                        (1 + inflation_rate)
                    balance -= inflated_withdrawal
                    planned_expenses += inflated_withdrawal
                    comments.append(
                        f"Withdrew ${inflated_withdrawal:.2f}(inflated value) for {withdrawal.get('reason', 'N/A')}")

            total_withdrawn = planned_expenses + living_expenses
            total_deposited = planned_deposits + planned_deposits

            reports.append(YearlyReport(
                age=current_age,
                living_expenses=living_expenses,
                planned_expenses=planned_expenses,
                total_withdrawn=total_withdrawn,
                interest_earned=interest_earned,
                planned_deposits=planned_deposits,
                total_deposited=total_deposited,
                balance=balance,
                comment="; ".join(comments) if comments else None
            ))
            current_age += 1

    total_years = months / 12

    # returns a tuple of (report list, total years until depletion)
    return (reports, round(total_years, 3))


@tool
def add_deposit_tool(tool_call_id: Annotated[str, InjectedToolCallId], age: int, amount: float, reason: Optional[str] = None, state: Annotated[RetirementState, InjectedState] = None) -> Command:
    """Adds a deposit to the retirement plan.
        Args:
        - age (int): The age at which the deposit is made.
        - amount (float): The amount of the deposit.
        - reason (Optional[str]): The reason for the deposit (optional).
    """
    print(
        f"Tool: add_deposit_tool called with age={age}, amount={amount}, reason={reason}")
    deposits = state["deposits"]
    deposits.append({"age": age, "amount": amount, "reason": reason or ""})
    return Command(update={"deposits": deposits, "messages": [ToolMessage(tool_call_id=tool_call_id, content=f"Deposit of ${amount:.2f} added at age {age}.")]})


@tool
def add_withdrawal_tool(tool_call_id: Annotated[str, InjectedToolCallId], age: int, amount: float, reason: Optional[str] = None, state: Annotated[RetirementState, InjectedState] = None) -> Command:
    """Adds a withdrawal to the retirement plan.
        Args:
        - age (int): The age at which the withdrawal is made.
        - amount (float): The amount of the withdrawal.
        - reason (Optional[str]): The reason for the withdrawal (optional).
    """
    print(
        f"Tool: add_withdrawal_tool called with age={age}, amount={amount}, reason={reason}")
    withdrawals = state["withdrawals"]
    withdrawals.append(
        {"age": age, "amount": amount, "reason": reason or ""})
    return Command(update={"withdrawals": withdrawals, "messages": [ToolMessage(tool_call_id=tool_call_id, content=f"Withdrawal of ${amount:.2f} added at age {age}.")]})


@tool
def remove_or_update_deposit_tool(tool_call_id: Annotated[str, InjectedToolCallId], age: int, previous_amount: float, new_amount: Optional[float] = None, reason: Optional[str] = None, state: Annotated[RetirementState, InjectedState] = None) -> Command:
    """Removes or updates a deposit in the retirement plan.
        Args:
        - age (int): The age at which the deposit was made.
        - previous_amount (float): The amount of the deposit to be removed or updated.
        - new_amount (Optional[float]): The new amount for the deposit (optional) Set None if removing the deposit.
        - reason (Optional[str]): The reason for the update (optional).
    """
    print(
        f"Tool: remove_or_update_deposit_tool called with age={age}, previous_amount={previous_amount}, new_amount={new_amount}, reason={reason}")
    deposits = state["deposits"]
    for i, deposit in enumerate(deposits):
        if deposit["age"] == age and deposit["amount"] == previous_amount:
            if new_amount is not None:
                deposits[i]["amount"] = new_amount
                deposits[i]["reason"] = reason or deposits[i].get("reason", "")
                return Command(update={"deposits": deposits, "messages": [ToolMessage(tool_call_id=tool_call_id, content=f"Deposit updated to ${new_amount:.2f} at age {age}.")]})
            else:
                del deposits[i]
                return Command(update={"deposits": deposits, "messages": [ToolMessage(tool_call_id=tool_call_id, content=f"Deposit of ${previous_amount:.2f} removed at age {age}.")]})
    return Command(update={"messages": [ToolMessage(tool_call_id=tool_call_id, content="Deposit not found or invalid parameters.")]})


@tool
def remove_or_update_withdrawal_tool(tool_call_id: Annotated[str, InjectedToolCallId], age: int, previous_amount: float, new_amount: Optional[float] = None, reason: Optional[str] = None, state: Annotated[RetirementState, InjectedState] = None) -> Command:
    """Removes or updates a withdrawal in the retirement plan.
        Args:
        - age (int): The age at which the withdrawal was made.
        - previous_amount (float): The amount of the withdrawal to be removed or updated.
        - new_amount (Optional[float]): The new amount for the withdrawal (optional) Set None if removing the withdrawal.
        - reason (Optional[str]): The reason for the update (optional).
    """
    print(
        f"Tool: remove_or_update_withdrawal_tool called with age={age}, previous_amount={previous_amount}, new_amount={new_amount}, reason={reason}")
    withdrawals = state["withdrawals"]
    for i, withdrawal in enumerate(withdrawals):
        if withdrawal["age"] == age and withdrawal["amount"] == previous_amount:
            if new_amount is not None:
                withdrawals[i]["amount"] = new_amount
                withdrawals[i]["reason"] = reason or withdrawals[i].get(
                    "reason", "")
                return Command(update={"withdrawals": withdrawals, "messages": [ToolMessage(tool_call_id=tool_call_id, content=f"Withdrawal updated to ${new_amount:.2f} at age {age}.")]})
            else:
                del withdrawals[i]
                return Command(update={"withdrawals": withdrawals, "messages": [ToolMessage(tool_call_id=tool_call_id, content=f"Withdrawal of ${previous_amount:.2f} removed at age {age}.")]})
    return Command(update={"messages": [ToolMessage(tool_call_id=tool_call_id, content="Withdrawal not found or invalid parameters.")]})


@tool
def get_current_financial_status(state: Annotated[RetirementState, InjectedState]) -> str:
    """Returns the user's current retirement configuration including:
    - User's current age
    - Initial savings
    - Monthly withdrawal (living expense)
    - Interest rate
    - Inflation rate
    - Planned deposits and withdrawals (with age and reason)

    Use this tool when the user asks about their age, savings, expenses, rates, or wants to confirm their financial setup.
    """
    print("Tool: get_current_financial_status called")
    return describe_retirement_data(state)


@tool
def get_prediction_report_detailed(state: Annotated[RetirementState, InjectedState]) -> str:
    """Returns a formatted summary of the most recent prediction report, if it exists.

    The report shows:
    - Age-wise balance tracking
    - Yearly deposits and withdrawals
    - Remaining savings over time

    Use this tool when the user asks to view or summarize the full prediction report.
    If no report is available, this will return a message stating that.
    """
    print("Tool: get_prediction_report_summary called")
    return describe_prediction_report(state)


@tool
def generate_retirement_forecast(tool_call_id: Annotated[str, InjectedToolCallId], state: Annotated[RetirementState, InjectedState]) -> Command:
    """
        Generates a new retirement prediction report using the user's financial data.

    Use this tool when:
    - A prediction report is missing or outdated.
    - The user asks how long their savings will last or requests updated projections.

    The tool stores:
    - A detailed year-by-year report
    - Total number of sustainable years
    """
    print("Tool: generate_retirement_forecast called")
    results = generate_prediction_report(state)
    if (results[0] is not None and len(results[0]) > 0):
        return Command(update={
            "prediction_report": results[0],
            "predicted_years": results[1],
            "messages": [ToolMessage(tool_call_id=tool_call_id, content="Prediction report generated successfully.")]
        })
    else:
        return Command(update={
            "messages": [ToolMessage(tool_call_id=tool_call_id, content="No prediction report could be generated. Insufficient funds or invalid data.")]
        })


@tool
def get_forecast_years_remaining(state: Annotated[RetirementState, InjectedState]) -> Optional[float]:
    """
    Returns how many years the user's current savings will last based on the most recent prediction.

    Use this tool when:
    - The user asks: "How long will my savings last?", "When will I run out of money?", or "How many years do I have?"
    - You want to extract just the number of sustainable years (not the full report).
    Returns None if no forecast has been generated.
    """
    return state["predicted_years"] if state["predicted_years"] is not None else None


@tool
def is_report_lastest(state: Annotated[RetirementState, InjectedState]) -> bool:
    """
    Checks whether the existing prediction report is the latest based on user updates.

    Use this tool:
    - Before giving forecast answers.
    - To ensure a fresh report is used before referencing retirement projections.

    Returns True if the report is valid and up to date. Otherwise, generate a new one.
    """
    return state.get("is_last_report_generated", False) and state.get("prediction_report") is not None

@tool
def update_user_config_tool(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[RetirementState, InjectedState],
    initial_savings: Optional[float] = None,
    monthly_withdrawal: Optional[float] = None,
    interest_rate: Optional[float] = None,
    inflation_rate: Optional[float] = None,
    user_age: Optional[int] = None
) -> Command:
    """
    Updates one or more fields in the user's financial configuration.

    Use this tool when:
    - The user asks to change their age, initial savings, monthly expenses, interest, or inflation rate.
    - You want to reflect user adjustments in the savings plan.

    Arguments:
    - initial_savings: New initial savings amount.
    - monthly_withdrawal: New planned monthly withdrawal amount.
    - interest_rate: Updated expected interest rate (in %).
    - inflation_rate: Updated inflation rate (in %).
    - user_age: Updated current age.

    Only the provided fields will be changed. Others will remain unchanged.
    """
    print("Tool: update_user_config_tool called")

    config = state["user_config"]
    if initial_savings is not None:
        config.initial_savings = initial_savings
    if monthly_withdrawal is not None:
        config.monthly_withdrawal = monthly_withdrawal
    if interest_rate is not None:
        config.interest_rate = interest_rate
    if inflation_rate is not None:
        config.inflation_rate = inflation_rate
    if user_age is not None:
        config.user_age = user_age

    return Command(update={
        "user_config": config,
        "messages": [ToolMessage(
            tool_call_id=tool_call_id,
            content="User configuration updated successfully."
        )]
    })
# -----------------------------
# LLM and Graph Logic
# -----------------------------


def human(state: RetirementState) -> RetirementState:
    message = input("Enter message: ")
    return {
        "messages": HumanMessage(content=message),
        "finished": message.lower() in ["exit", "quit", "done", "bye"]
    }


def model_call(state: RetirementState) -> RetirementState:
    system_prompt = SystemMessage(content="""
    You are a retirement planning assistant that helps users understand how long their savings will last and how to optimize their retirement financial plan.

    You have access to the user's configuration data and future financial events (deposits and withdrawals), along with tools that let you retrieve data, generate predictions, and modify their plan.

    -------------------------------
    Tool Usage Logic
    -------------------------------

    - Use `get_current_financial_status` when the user asks about:
    - Their age
    - Savings or withdrawal settings
    - Interest or inflation rates
    - Any planned financial events

    You never need to ask the user permission to call this tool or data — it is already available with this tool.

    - Before providing any prediction or forecast:
    1. Use `is_report_lastest`
    2. If the report is not latest, call `generate_retirement_forecast`

    - Use `get_forecast_years_remaining` when the user asks:
    - "How long will my savings last?"
    - "Can I retire early?"
    - "Will I run out of money?"

    - Use `get_prediction_report_detailed` when the user explicitly asks for:
    - A year-by-year breakdown
    - Full forecast details
    - Summary of retirement balance across years
    - Directly show the string that this tool returns to show the report.
    
    -------------------------------
    Plan Modification Tools
    -------------------------------

    - To **add** future events:
    - Use `add_deposit_tool` or `add_withdrawal_tool`

    - To **remove or update** existing events:
    - Use `remove_or_update_deposit_tool` or `remove_or_update_withdrawal_tool`
    - These tools require age, amount, and optionally a reason
    
    -Use 'update_user_config_tool; if the user wants to change their age, interest rate, or other settings


    Always ask the user for missing details in a friendly, clear way, and confirm changes when needed.

    -------------------------------
    Tone and Style
    -------------------------------

    - Stay informative, friendly, and concise
    - Do not make assumptions beyond available data
    - Only show the full report if the user asks
    - Use tools as needed to fill gaps, retrieve information, or make updates
    - Always be ready to provide any suggestions on how to improve the retirement plan based on the user's financial data and goals.
    - If the user asks for anything other than retirement planning or finance related questions, politely tell them you can only assist with retirement planning and financial forecasting.
    - Never tell the user the tool names or internal tool journey. Just provide the information they need.

    """)
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": response}


def should_go_tools(state: RetirementState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    return "human" if not last_message.tool_calls else "tools"

# -----------------------------
# LangGraph Construction
# -----------------------------


tools = [get_current_financial_status, get_prediction_report_detailed,
         generate_retirement_forecast, get_forecast_years_remaining,
         is_report_lastest,
         add_deposit_tool, add_withdrawal_tool,
         remove_or_update_deposit_tool, remove_or_update_withdrawal_tool,
        update_user_config_tool]

tool_node = ToolNode(tools=tools)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest").bind_tools(tools=tools)

graph = StateGraph(RetirementState)
graph.add_node("our_agent", model_call)
graph.add_node("human", human)
graph.add_node("tools", tool_node)

graph.add_edge(START, "human")
graph.add_conditional_edges("human", lambda state: state.get(
    "finished", False), {False: "our_agent", True: END})
graph.add_conditional_edges("our_agent", should_go_tools, {
                            "human": "human", "tools": "tools"})

graph.add_edge("tools", "our_agent")
app = graph.compile()

# -----------------------------
# Execution Helpers
# -----------------------------


def print_stream(stream):
    for s in stream:
        if s["messages"]:
            message = s["messages"][-1]
            if isinstance(message, tuple):
                print("here", message)
            else:
                message.pretty_print()


def save_graph_visualization(graph):
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        with open("retire_agent_graph.png", "wb") as f:
            f.write(png_data)
        print("Graph visualization saved as retire_agent_graph.png")
    except Exception as e:
        print(f"Could not save graph visualization: {e}")

# -----------------------------
# Run the App
# -----------------------------


inputs = read_user_data(RetirementState(
    messages=[],
    user_question="",
    user_config=None,
    deposits=[],
    withdrawals=[],
    is_last_report_generated=False,
))

save_graph_visualization(app)

print_stream(app.stream(inputs, stream_mode="values", config={"recursion_limit": 50}))