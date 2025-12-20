from typing import TypedDict, Annotated, List
import operator
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from .config_loader import Config
from .agents.manager import Manager
from .agents.coder import Coder
from .agents.tester import Tester
from .agents.reviewer import Reviewer
import logging

logging.basicConfig(level=logging.INFO)

def create_graph(config: Config):
    class AgentState(TypedDict):
        tasks: List[str]
        current_task: str
        code: str
        test_results: str
        review_feedback: str
        iteration: Annotated[int, operator.add]
        messages: Annotated[list, add_messages]

    def should_continue(state):
        iteration = state.get("iteration", 0)
        print(f"\n=== Iteration {iteration} complete. Max: {config.agents.max_iterations} ===")
        if iteration >= config.agents.max_iterations:
            print("=== MAX ITERATIONS REACHED - END ===")
            return "end"
        print("=== CONTINUE LOOP ===")
        return "manager"

    # Instantiate agents
    manager = Manager("manager")
    coder = Coder("coder")
    tester = Tester("tester")
    reviewer = Reviewer("reviewer")

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
    return app