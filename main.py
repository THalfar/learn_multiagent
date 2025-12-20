import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
import operator
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

load_dotenv()

from src.config_loader import load_config
from src.agents.manager import Manager
from src.agents.coder import Coder
from src.agents.tester import Tester
from src.agents.reviewer import Reviewer

config = load_config()

class AgentState(TypedDict):
    tasks: List[str]
    current_task: str
    code: str
    test_results: str
    review_feedback: str
    iteration: Annotated[int, operator.add]
    messages: Annotated[list, add_messages]

def should_continue(state: AgentState):
    iteration = state.get("iteration", 0)
    feedback = state.get("review_feedback", "")
    if iteration >= config.agents.max_iterations:
        return "end"
    return "manager"

# Instantiate agents
manager = Manager("manager")
coder = Coder("coder")
tester = Tester("tester")
reviewer = Reviewer("reviewer")

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("manager", manager)
workflow.add_node("coder", coder)
workflow.add_node("tester", tester)
workflow.add_node("reviewer", reviewer)

workflow.set_entry_point("manager")
workflow.add_edge("manager", "coder")
workflow.add_edge("coder", "tester")
workflow.add_edge("tester", "reviewer")
workflow.add_conditional_edges("reviewer", should_continue, {"manager": "manager", "end": END})

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