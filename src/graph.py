from typing import TypedDict, Annotated, List, Any, Dict
import operator
from langgraph.graph import StateGraph, END
from .config_loader import Config
from .agents.manager import Manager
from .agents.coder import Coder
from .agents.tester import Tester
from .agents.reviewer import Reviewer
import logging

# Configure logging to be less verbose - only show warnings and errors
logging.basicConfig(
    level=logging.WARNING,
    format='%(message)s'
)
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
        review_suggestions: str
        reviewer_tester_instruction: str  # Reviewer's instruction for tester (next iteration)
        tester_reviewer_response: str  # Tester's response to reviewer's request (from previous iteration)
        environment_switch_review_feedback: str  # Reviewer's feedback from environment switch
        manager_guidance: str  # Manager's intent/guidance for reviewer (what manager wanted)
        iteration: Annotated[int, operator.add]
        conversation_history: List[Dict[str, Any]]  # Siloed conversation history per agent
        run_id: str
        video_dir: str
        stats: Any
        approved: bool
        current_env_index: int  # Index in environment_progression
        solved_environments: List[str]  # List of environment names that have been solved
        conversation_logger: Any  # Conversation logger instance
    
    def should_continue(state: AgentState) -> str:
        # Check if manager said DONE
        if state.get("current_task", "").upper() == "DONE":
            return "end"
        
        # If approved, check if there are more environments to solve
        if state.get("approved", False):
            env_progression = config.environment_progression
            current_env_index = state.get("current_env_index", 0)
            solved_environments = state.get("solved_environments", [])
            
            # If there are more environments, continue to manager (which will switch environments)
            if env_progression and current_env_index + 1 < len(env_progression):
                # Check if current environment is already in solved list
                # If not, manager will add it and switch to next
                current_env = env_progression[current_env_index]
                if current_env.name not in solved_environments:
                    # Continue to manager to handle environment switch
                    return "manager"
            
            # All environments solved or no more environments, end the run
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