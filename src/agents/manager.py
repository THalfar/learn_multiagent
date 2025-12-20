def manager_node(state):
    """
    Manager analyzes the situation, creates/updates task list, selects next task.
    """
    # Placeholder: In real impl, use LLM chain with prompt based on state
    prompt = f"""
    Current state:
    Tasks: {state.get('tasks', [])}
    Current task: {state.get('current_task', '')}
    Code: {state.get('code', '')}
    Test results: {state.get('test_results', '')}
    Review feedback: {state.get('review_feedback', '')}
    Iteration: {state.get('iteration', 0)}

    Analyze and create/update task list for developing CartPole-v1 RL solution.
    Output format:
    tasks: list of str
    current_task: str (next task or 'done')
    """
    # response = llm.invoke(prompt)
    # Parse response to update state
    return {
        "tasks": ["Initialize Gymnasium environment", "Implement basic DQN agent", "Train and evaluate"],
        "current_task": "Initialize Gymnasium environment",
        "iteration": 1
    }