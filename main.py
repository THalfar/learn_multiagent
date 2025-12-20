from dotenv import load_dotenv
import os

load_dotenv()

from src.config_loader import load_config
from src.graph import create_graph

if __name__ == "__main__":
    config = load_config()
    app = create_graph(config)

    initial_state = {
        "tasks": [],
        "current_task": "",
        "code": "",
        "test_results": "",
        "review_feedback": "",
        "iteration": 0,
        "messages": [],
    }
    print("=== MULTI-AGENT RL DEV TEAM START ===")
    result = app.invoke(initial_state)
    print("\n=== FINAL RESULT ===")
    print(result)
    print("=== END ===")