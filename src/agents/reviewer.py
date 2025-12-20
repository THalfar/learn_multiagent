def reviewer_node(state):
    """
    Reviewer evaluates code, test results, provides feedback.
    """
    prompt = f"""
    Code: {state['code']}
    Test results: {state['test_results']}

    Review for quality, efficiency, RL best practices for CartPole.
    Decide if ready (Solved: avg reward >195 over 100 episodes) or needs iteration.
    Output: feedback str, ready: bool
    """
    # response = llm.invoke(prompt)
    return {
        "review_feedback": "Placeholder feedback: Basic env setup good, but need DQN implementation.",
    }