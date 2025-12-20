import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
import operator

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from src.agents.manager import manager_node
from src.agents.coder import coder_node
from src.agents.tester import tester_node
from src.agents.reviewer import reviewer_node

load_dotenv()

llm = ChatOpenAI(
    model="grok-beta",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

class AgentState(TypedDict):
    tasks: List[str]
    current_task: str
    code: str
    test_results: str
    review_feedback: str
    iteration: Annotated[int, operator.add]
    messages: Annotated[list, add_messages]

def should_continue(state: AgentState):
    """
    Conditional edge: continue if iteration < 3 and not ready.
    """
    iteration = state.get("iteration", 0)
    feedback = state.get("review_feedback", "")
    if "ready" in feedback.lower() or iteration >= 3:
        return "end"
    return "manager"

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("manager", manager_node)
workflow.add_node("coder", coder_node)
workflow.add_node("tester", tester_node)
workflow.add_node("reviewer", reviewer_node)

# Edges
workflow.set_entry_point("manager")
workflow.add_edge("manager", "coder")
workflow.add_edge("coder", "tester")
workflow.add_edge("tester", "reviewer")
workflow.add_conditional_edges(
    "reviewer",
    should_continue,
    {"manager": "manager", "end": END},
)

app = workflow.compile()

if __name__ == "__main__":
    initial_state = {
        "tasks": [],
        "current_task": "",
        "code": "",
        "test_results": "",
        "review_feedback": "",
        "iteration": 0,
        "messages": [],
    }
    result = app.invoke(initial_state)
    print(result)