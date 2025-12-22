from typing import TypedDict, Annotated, List, Any
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
from rich import print

def create_graph(config: Config):
    class AgentState(TypedDict):
        tasks: List[str]
        current_task: str
        code: str
        test_results: str
        execution_stdout: str
        execution_stderr: str
        review_feedback: str
        iteration: Annotated[int, operator.add]
        messages: Annotated[list, add_messages]
        run_id: str
        video_dir: str
        stats: Any
        approved: bool
    
    def should_continue(state: AgentState) -> str:
        # Check if manager said DONE
        if state.get("current_task", "").upper() == "DONE":
            return "end"
        
        if state.get("approved", False):
            return "end"
        
        # Check iteration limit
        if state.get("iteration", 0) >= config.agents.max_iterations:
            return "end"
        
        return "manager"

    # Instantiate agents
    manager = Manager(config)
    coder = Coder(config)
    tester = Tester(config)
    reviewer = Reviewer(config)

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