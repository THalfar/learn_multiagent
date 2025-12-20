def coder_node(state):
    """
    Coder implements the current task.
    """
    prompt = f"""
    Current task: {state['current_task']}
    Existing code: {state.get('code', '')}

    Implement the task in Python code for CartPole-v1 using Gymnasium.
    Output only the full updated code.
    """
    # response = llm.invoke(prompt)
    return {
        "code": "# Placeholder code for " + state['current_task'] + "\nenv = gymnasium.make('CartPole-v1')\nobs, info = env.reset()\n",
    }