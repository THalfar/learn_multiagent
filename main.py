from dotenv import load_dotenv
import os

load_dotenv()

from src.config_loader import load_config
from src.graph import create_graph
import datetime
from src.utils.timer import RunStatistics
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
    print(f"[bold cyan]=== RUN ID: {run_id} ===[/bold cyan]")
    print("[bold cyan]=== MULTI-AGENT RL DEV TEAM START ===[/bold cyan]")
    result = app.invoke(initial_state, config={"recursion_limit": config.agents.max_iterations * 5})
    print("[bold cyan]\n=== FINAL RESULT ===[/bold cyan]")
    print(f"[bold cyan]Final state:[/bold cyan]\\n{result}")
    print("=== END ===")
    
    stats.end_time = datetime.datetime.now()
    stats.print_summary()
    os.makedirs(f"output/{run_id}", exist_ok=True)
    stats.save_to_file(f"output/{run_id}/statistics.json")
    print(f"[bold green]📊 Statistics saved to output/{run_id}/statistics.json[/bold green]")