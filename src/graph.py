from typing import TypedDict, Annotated, List, Any, Dict
import operator
from langgraph.graph import StateGraph, END
from .config_loader import Config
from .agents.manager import Manager
from .agents.coder import Coder
from .agents.tester import Tester
from .agents.reviewer import Reviewer
from .utils.model_switcher import ModelSwitcher
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
        agent_opinions: List[Dict[str, Any]]  # Cross-agent "dialogue" - opinions/comments shared between agents
        run_id: str
        video_dir: str
        stats: Any
        approved: bool
        current_env_index: int  # Index in environment_progression
        solved_environments: List[str]  # List of environment names that have been solved
        conversation_logger: Any  # Conversation logger instance
        # Monivaiheinen treeni - vaihe per ympäristö
        current_phase: str  # "validation" | "optimization" | "demo"
        best_model_path: str  # Polku parhaaseen malliin (optimization-vaiheesta)
        # SHODAN's Divine Codex - persistent rules for coder's prompt
        shodan_rules: List[Dict[str, Any]]  # [{"rule": "...", "iteration": N}, ...]
        # Failsafe: skip env after repeated failures
        consecutive_failures: int  # Peräkkäisten REJECTED-iteraatioiden laskuri
        last_failure_type: str  # "timeout" | "crash" | "low_reward" | ""
        # Honest scoreboard + progress-aware failsafe
        skipped_environments: List[str]  # Envs abandoned via failsafe (NOT solved)
        best_reward_this_env: Any  # Best real reward seen for the current env
        best_reward_env_index: int  # Which env index best_reward_this_env refers to
        # Manager's Playbook: ympäristöreseptit edellisistä enveistä
        playbook: List[Dict[str, Any]]  # [{"env": "CartPole-v1", "algo": "PPO", ...}]
        # A3: Coder self-memory - last few (code, diagnosis, verdict) attempts
        recent_attempts: List[Dict[str, Any]]  # [{"iter": N, "verdict": str, "diagnosis": str, "reason": str}]
        # A6: Tester's concise diagnosis of the latest run (feeds recent_attempts)
        diagnosis: str
        # A7: Manager escalation - history of failure modes for "same mode 3x -> change strategy class"
        failure_history: List[str]  # ["timeout", "timeout", "low_reward", ...]
        # PHASE B: SKILL substrate (procedural, pinned, persistent) - replaces the flat Codex
        skill_store: Any  # SkillStore instance (manages its own disk persistence)
        # C1: cumulative learning visibility (per env; reset on env switch)
        total_env_steps: int  # Total timesteps trained this env (across chunks)
        metric_history: List[Any]  # RESULT values per optimization iteration this env
        measured_sps: Any  # Measured training steps/sec (from runs >= 5000 steps)
        # C2: checkpoint-resume enforcement (Tester verifies; Reviewer gates on it)
        resume_required: bool  # True when optimization ran with an existing checkpoint
        resume_ok: bool  # True when stdout proved RESUMED: buffer_transitions=N (N>0)

    def should_continue(state: AgentState) -> str:
        # Check if manager said DONE
        if state.get("current_task", "").upper() == "DONE":
            return "end"

        current_phase = state.get("current_phase", "validation")

        # If approved, handle phase transitions
        if state.get("approved", False):
            env_progression = config.environment_progression
            current_env_index = state.get("current_env_index", 0)
            solved_environments = state.get("solved_environments", [])

            # MONIVAIHEINEN LOGIIKKA:
            # validation -> optimization -> demo -> seuraava env
            if current_phase == "validation":
                # Validation OK -> siirry optimization-vaiheeseen
                print(f"[bold green]✅ VALIDATION PASSED - Moving to OPTIMIZATION phase[/bold green]")
                return "manager"  # Manager vaihtaa vaiheen

            elif current_phase == "optimization":
                # Optimization OK (threshold saavutettu) -> siirry demo-vaiheeseen
                print(f"[bold green]✅ OPTIMIZATION COMPLETE - Moving to DEMO phase[/bold green]")
                return "manager"  # Manager vaihtaa vaiheen

            elif current_phase == "demo":
                # Demo OK -> siirry seuraavaan ympäristöön
                print(f"[bold green]✅ DEMO COMPLETE - Environment SOLVED![/bold green]")

                # If there are more environments, continue
                if env_progression and current_env_index + 1 < len(env_progression):
                    current_env = env_progression[current_env_index]
                    if current_env.name not in solved_environments:
                        return "manager"  # Manager vaihtaa ympäristön

                # All environments solved
                return "end"

        # Check iteration limit
        if state.get("iteration", 0) >= config.agents.max_iterations:
            return "end"

        return "manager"

    # Create model switcher for adaptive model switching
    model_switcher = ModelSwitcher(config)

    # Instantiate agents with model_switcher (except reviewer which stays on API)
    manager = Manager(config, model_switcher=model_switcher)
    coder = Coder(config, model_switcher=model_switcher)
    tester = Tester(config, model_switcher=model_switcher)
    reviewer = Reviewer(config)  # Reviewer stays on API - no model_switcher

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