"""Duo pipeline graph: Director -> Coder -> Executor -> Director.

The duo topology removes the quad pipeline's Manager and Tester LLM layers. ONE frontier
call (the Director) is strategist + judge + taskmaster; the Coder reads the Executor's RAW
stdout/stderr directly; the Executor runs the sandbox deterministically with no LLM. Selected
by `pipeline: "duo"` in the config (main.py dispatches on Config.pipeline).

The AgentState mirrors graph.py's (including the Goal-A demo-reward fields). Fields the duo
pipeline doesn't write (e.g. reviewer_tester_instruction, tester_reviewer_response) are kept
so the state shape stays compatible with the shared logger/state helpers.
"""
from typing import TypedDict, Annotated, List, Any, Dict
import operator
from langgraph.graph import StateGraph, END
from .config_loader import Config
from .agents.director import Director
from .agents.coder import Coder
from .agents.executor import Executor
import logging

logging.basicConfig(level=logging.WARNING, format='%(message)s')
from rich import print


def create_duo_graph(config: Config):
    class AgentState(TypedDict):
        tasks: List[str]
        current_task: str
        code: str
        test_results: str
        execution_stdout: str
        execution_stderr: str
        review_feedback: str
        review_suggestions: str
        reviewer_tester_instruction: str  # unused in duo (kept for state-shape compatibility)
        tester_reviewer_response: str      # unused in duo
        environment_switch_review_feedback: str
        manager_guidance: str
        iteration: Annotated[int, operator.add]
        conversation_history: List[Dict[str, Any]]
        agent_opinions: List[Dict[str, Any]]
        run_id: str
        video_dir: str
        stats: Any
        approved: bool
        current_env_index: int
        solved_environments: List[str]
        conversation_logger: Any
        current_phase: str  # "validation" | "optimization" | "demo"
        best_model_path: str
        shodan_rules: List[Dict[str, Any]]
        consecutive_failures: int
        skipped_environments: List[str]
        best_reward_this_env: Any
        best_reward_env_index: int
        recent_attempts: List[Dict[str, Any]]
        diagnosis: str
        failure_history: List[str]
        skill_store: Any
        total_env_steps: int
        metric_history: List[Any]
        measured_sps: Any
        resume_required: bool
        resume_ok: bool
        # Goal A: demo-reward gate
        demo_reward: Any
        demo_below_threshold: bool

    def should_continue(state: AgentState) -> str:
        # The Director sets current_task="DONE" when the last environment is solved.
        if state.get("current_task", "").upper() == "DONE":
            return "end"
        # Iteration cap (Director is the only node that increments `iteration`).
        if state.get("iteration", 0) >= config.agents.max_iterations:
            return "end"
        return "coder"

    director = Director(config)
    coder = Coder(config)            # no model_switcher needed; the Coder reads it defensively
    executor = Executor(config)

    workflow = StateGraph(AgentState)
    workflow.add_node("director", director)
    workflow.add_node("coder", coder)
    workflow.add_node("executor", executor)

    workflow.set_entry_point("director")
    workflow.add_conditional_edges("director", should_continue, {"coder": "coder", "end": END})
    workflow.add_edge("coder", "executor")
    workflow.add_edge("executor", "director")

    app = workflow.compile()
    return app
