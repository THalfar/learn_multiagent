from dotenv import load_dotenv
import os
import sys

load_dotenv()

from src.config_loader import load_config
from src.graph import create_graph
from src.duo_graph import create_duo_graph
import datetime
from src.utils.timer import RunStatistics
from src.utils.banners import print_run_banner, print_final_summary
from src.utils.conversation_logger import ConversationLogger
from rich import print

if __name__ == "__main__":
    project_path = sys.argv[1] if len(sys.argv) > 1 else "config/project.yaml"
    config = load_config(project_path=project_path)
    # Pipeline dispatch: 'duo' = Director->Coder->Executor (3 nodes, 1 LLM); 'quad' (default)
    # = Manager->Coder->Tester->Reviewer. Everything below (run_id, stats, logger, skill seed,
    # initial_state incl. demo fields, recursion_limit) is shared by both.
    app = create_duo_graph(config) if config.pipeline == "duo" else create_graph(config)

    run_id = f"{config.test_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    stats = RunStatistics(run_id=run_id)
    
    # Initialize conversation logger
    conversation_logger = ConversationLogger(run_id=run_id)
    
    # Start with first environment in progression
    env_progression = config.environment_progression
    if not env_progression:
        raise ValueError("No environment_progression defined in config!")
    
    # Build env-specific output directory: output/{run_id}/{env_name}/videos
    first_env_name = env_progression[0].name
    video_dir = os.path.abspath(os.path.normpath(f"output/{run_id}/{first_env_name}/videos"))
    os.makedirs(video_dir, exist_ok=True)

    # PHASE B: procedural skill memory (persistent across runs; replaces the flat Codex).
    from src.skills import SkillStore
    skill_store = SkillStore(skills_dir=config.skills_dir).load()
    _skills_was_empty = not skill_store.skills
    skill_store.seed_if_empty(initial_skills=config.initial_skills, initial_rules=config.initial_codex_rules)
    if _skills_was_empty and skill_store.skills:
        skill_store.save()  # only persist when seeding actually added skills; an
                            # existing store's canonical file is already on disk from load()

    initial_state = {
        "run_id": run_id,
        "video_dir": video_dir,
        "tasks": [],
        "current_task": "",
        "code": "",
        "test_results": "",
        "review_feedback": "",
        "review_suggestions": "",
        "environment_switch_review_feedback": "",  # Initialize empty
        "manager_guidance": "",  # Manager's intent/guidance for reviewer
        "iteration": 0,
        "conversation_history": [],  # Renamed from "messages" to avoid LangGraph collision
        "agent_opinions": [],  # Cross-agent "dialogue" - opinions/comments shared between agents
        "stats": stats,
        "approved": False,
        "current_env_index": 0,  # Start with first environment
        "solved_environments": [],  # No environments solved yet
        "conversation_logger": conversation_logger,  # Add logger to state
        # Monivaiheinen treeni: validation -> optimization -> demo
        "current_phase": "validation",  # Aloitetaan aina validoinnilla
        "best_model_path": "",  # Polku parhaaseen malliin (täytetään optimization-vaiheessa)
        # SHODAN's Divine Codex - persistent rules for coder's prompt.
        # Pre-seeded from config (empty by default; the seeded experiment injects an HER hint).
        "shodan_rules": [{"rule": r, "iteration": 0} for r in config.initial_codex_rules],
        # Failsafe: skip env after repeated failures
        "consecutive_failures": 0,
        # Honest scoreboard + progress-aware failsafe
        "skipped_environments": [],
        "best_reward_this_env": None,
        "best_reward_env_index": -1,
        # A3 Coder self-memory / A6 diagnosis / A7 escalation / Phase B skill store
        "recent_attempts": [],
        "diagnosis": "",
        "failure_history": [],
        "skill_store": skill_store,
        # C1 cumulative learning visibility / C2 checkpoint-resume enforcement
        "total_env_steps": 0,
        "metric_history": [],
        "measured_sps": None,
        "resume_required": False,
        "resume_ok": True,
        # Goal A: demo-reward gate (None = no demo measurement yet)
        "demo_reward": None,
        "demo_below_threshold": False,
    }
    
    # Print run start banner
    print_run_banner(config, run_id)
    
    start_time = datetime.datetime.now()
    # Minimum 50 to allow phase transitions (validation->optimization->demo) even with low max_iterations
    recursion_limit = max(config.agents.max_iterations * 5, 50)
    result = app.invoke(initial_state, config={"recursion_limit": recursion_limit})
    end_time = datetime.datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    stats.end_time = end_time
    stats.print_summary()
    os.makedirs(f"output/{run_id}", exist_ok=True)
    stats.save_to_file(f"output/{run_id}/statistics.json")
    
    # Print final summary
    iterations = result.get("iteration", 0)
    success = result.get("approved", False)
    solved_environments = result.get("solved_environments", [])
    skipped_environments = result.get("skipped_environments", [])
    print_final_summary(run_id, iterations, success, total_time, solved_environments, skipped_environments)
    print(f"[bold green]📊 Statistics saved to output/{run_id}/statistics.json[/bold green]")

    # Save final conversation log summary
    conversation_logger.log_final_summary(
        total_iterations=iterations,
        success=success,
        total_time=total_time,
        solved_environments=solved_environments,
        skipped_environments=skipped_environments,
        cost=stats.get_cost(),
    )
    print(f"[bold green]💬 Conversation log saved to {conversation_logger.get_log_path()}[/bold green]")