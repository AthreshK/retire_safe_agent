# 💸 RetireSafe Retirement Planning Agent (LangGraph + Gemini)

This project is an interactive, tool-augmented retirement planning assistant built using [LangGraph](https://docs.langchain.com/langgraph/), [LangChain tools](https://python.langchain.com/docs/modules/agents/tools/), and [Gemini 1.5 Flash](https://ai.google.dev/gemini-api/docs/overview). It simulates a multi-turn conversation with a user to evaluate their retirement readiness based on personalized financial data.

---
## 🧠 What It Does
- Understands the user's financial situation
- Predicts how long retirement savings will last
- Allows interactive updates to retirement plans (deposits, withdrawals, etc.)
- Generates detailed yearly breakdowns of savings over time
- Uses tools and logic branching to make accurate financial recommendations

---
## 🔧 Tech Stack

| Component | Role |
|----------|------|
| **LangGraph** | Multi-turn conversational flow with conditional routing |
| **LangChain Tools** | Callable tools (e.g., generate predictions, describe financial data) |
| **Gemini 1.5 Flash** | Large Language Model for chat and decision-making |
| **Pydantic** | Typed state models and validation |
| **Pandas** | CSV parsing and event handling |
| **uv** | Fast dependency management (used instead of pip/venv) |

---
## ⚙️ Setup

Clone the repo:
```bash 
git clone https://github.com/AthreshK/retire_safe_agent.git
cd retire_safe_agent
```

Install dependencies using uv
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Configure environment variables
```bash
cp .env.example .env
# Add your Gemini API key
```

Run the app
```bash
python main.py
```
---

## 🧪 Learning Resources

The learning_langgraph/ directory includes two clean, well-commented starter scripts:

- basic_conditional_graph.py — Simple conditional routing in LangGraph
- react_agent_example.py — Shows tool usage in a ReAct-style agent

Great for anyone getting started with LangGraph!

## ✅ Example Prompts to Try

- “What is my current age?”
- “How long will my savings last?”
- “Add a withdrawal of 5000 at age 70 for medical expenses.”
- “Show me the yearly retirement report.”