from dotenv import load_dotenv
import os
import sys

load_dotenv()

from src.config_loader import load_config
from src.graph import create_graph
import datetime
from src.utils.timer import RunStatistics
from src.utils.banners import print_run_banner, print_final_summary
from src.utils.conversation_logger import ConversationLogger
from rich import print

if __name__ == "__main__":
    project_path = sys.argv[1] if len(sys.argv) > 1 else "config/project.yaml"
    config = load_config(project_path=project_path)
    app = create_graph(config)

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
        # SHODAN's Divine Codex - persistent rules for coder's prompt
        "shodan_rules": [],  # Starts empty, SHODAN adds rules during review
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
    print_final_summary(run_id, iterations, success, total_time, solved_environments)
    print(f"[bold green]📊 Statistics saved to output/{run_id}/statistics.json[/bold green]")
    
    # Save final conversation log summary
    conversation_logger.log_final_summary(
        total_iterations=iterations,
        success=success,
        total_time=total_time,
        solved_environments=solved_environments
    )
    print(f"[bold green]💬 Conversation log saved to {conversation_logger.get_log_path()}[/bold green]")