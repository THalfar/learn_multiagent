from dotenv import load_dotenv
import os

load_dotenv()

from src.config_loader import load_config
from src.graph import create_graph
import datetime
from src.utils.timer import RunStatistics
from src.utils.banners import print_run_banner, print_final_summary
from rich import print

if __name__ == "__main__":
    config = load_config()
    app = create_graph(config)

    run_id = f"{config.test_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    stats = RunStatistics(run_id=run_id)
    initial_state = {
        "run_id": run_id,
        "video_dir": f"output/{run_id}/videos",
        "tasks": [],
        "current_task": "",
        "code": "",
        "test_results": "",
        "review_feedback": "",
        "iteration": 0,
        "messages": [],
        "stats": stats,
        "approved": False,
    }
    
    # Print run start banner
    print_run_banner(config, run_id)
    
    start_time = datetime.datetime.now()
    result = app.invoke(initial_state, config={"recursion_limit": config.agents.max_iterations * 5})
    end_time = datetime.datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    stats.end_time = end_time
    stats.print_summary()
    os.makedirs(f"output/{run_id}", exist_ok=True)
    stats.save_to_file(f"output/{run_id}/statistics.json")
    
    # Print final summary
    iterations = result.get("iteration", 0)
    success = result.get("approved", False)
    print_final_summary(run_id, iterations, success, total_time)
    print(f"[bold green]📊 Statistics saved to output/{run_id}/statistics.json[/bold green]")