"""Visual feedback and banner functions for multi-agent runs"""


def print_run_banner(config, run_id):
    """Print run configuration at start"""
    print("\n" + "=" * 60)
    print("🚀 MULTI-AGENT RL DEV TEAM")
    print("=" * 60)
    print(f"📁 Run ID: {run_id}")
    print(f"🎯 Environment: {config.environment.name}")
    print(f"🏆 Success threshold: {config.agents.success_threshold}")
    print(f"🔄 Max iterations: {config.agents.max_iterations}")
    print()
    print("🤖 AGENT CONFIGURATION:")
    print(f"   Manager:  {config.agent_llm.manager}")
    print(f"   Coder:    {config.agent_llm.coder}")
    print(f"   Tester:   {config.agent_llm.tester}")
    print(f"   Reviewer: {config.agent_llm.reviewer}")
    print()
    print("💻 Ollama URL:", config.ollama.base_url)
    print("=" * 60 + "\n")


def print_iteration_banner(iteration, max_iterations, task):
    """Print a nice banner for each iteration"""
    print("\n" + "🔷" * 30)
    print(f"📍 ITERATION {iteration}/{max_iterations}")
    task_display = task[:80] + "..." if len(task) > 80 else task
    print(f"📋 Task: {task_display}")
    print("🔷" * 30 + "\n")


def print_agent_transition(from_agent, to_agent):
    """Show agent transitions"""
    print(f"\n{'─' * 40}")
    print(f"  {from_agent.upper()} ➜ {to_agent.upper()}")
    print(f"{'─' * 40}\n")


def print_final_summary(run_id, iterations, success, total_time):
    """Print final run summary"""
    print("\n" + "=" * 60)
    print("🏁 RUN COMPLETE")
    print("=" * 60)
    print(f"📁 Run ID: {run_id}")
    print(f"🔄 Iterations: {iterations}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"🏆 Success: {'✅ YES!' if success else '❌ Not yet'}")
    print()
    print(f"📂 Output: output/{run_id}/")
    print(f"   📄 Code: output/{run_id}/code/")
    print(f"   🎬 Videos: output/{run_id}/videos/")
    print("=" * 60 + "\n")
