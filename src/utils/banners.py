"""Visual feedback and banner functions for multi-agent runs"""
from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns


def print_run_banner(config, run_id):
    """Print run configuration at start"""
    # Get success threshold from first environment in progression
    success_threshold = config.environment_progression[0].success_threshold if config.environment_progression else 0
    
    print("\n" + "=" * 60)
    print("🚀 MULTI-AGENT RL DEV TEAM")
    print("=" * 60)
    print(f"📁 Run ID: {run_id}")
    print(f"🎯 Environment: {config.environment.name}")
    print(f"🏆 Success threshold: {success_threshold}")
    print(f"🔄 Max iterations: {config.agents.max_iterations}")
    print()
    print("🤖 AGENT CONFIGURATION:")
    print(f"   [bold blue]Manager:[/bold blue]  {config.agent_llm.manager}")
    print(f"   [bold green]Coder:[/bold green]    {config.agent_llm.coder}")
    print(f"   [bold yellow]Tester:[/bold yellow]   {config.agent_llm.tester}")
    print(f"   [bold magenta]Reviewer:[/bold magenta] {config.agent_llm.reviewer}")
    print()
    print("💻 Ollama URL:", config.ollama.base_url)
    print("=" * 60 + "\n")


def print_iteration_banner(iteration, max_iterations, task, environment=None, success_threshold=None, solved_envs=None, total_envs=None):
    """Print a nice banner for each iteration with additional info"""
    progress = (iteration / max_iterations) * 100 if max_iterations > 0 else 0
    
    print("\n" + "═" * 70)
    print(f"ITERATION {iteration}/{max_iterations} ({progress:.0f}%)")
    print("═" * 70)
    
    if environment:
        print(f"Environment: {environment}")
    if success_threshold is not None:
        print(f"Success threshold: {success_threshold}")
    if solved_envs and total_envs:
        solved_count = len(solved_envs) if isinstance(solved_envs, list) else 0
        print(f"Progress: {solved_count}/{total_envs} environments solved")
        if solved_envs:
            print(f"Solved: {', '.join(solved_envs) if isinstance(solved_envs, list) else solved_envs}")
    
    task_display = task[:80] + "..." if len(task) > 80 else task
    print(f"Task: {task_display}")
    print("═" * 70 + "\n")


def print_agent_transition(from_agent, to_agent):
    """Show agent transitions"""
    transitions = {
        ("manager", "coder"): "Manager → Coder",
        ("coder", "tester"): "Coder → Tester",
        ("tester", "reviewer"): "Tester → Reviewer",
        ("reviewer", "manager"): "Reviewer → Manager",
    }
    
    key = (from_agent.lower(), to_agent.lower())
    message = transitions.get(key, f"{from_agent.upper()} → {to_agent.upper()}")
    
    print(f"\n{'─' * 70}")
    print(f"  {message}")
    print(f"{'─' * 70}\n")


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


def print_environment_switch_bombardment(
    current_env_name: str,
    next_env_name: str,
    solved_environments: list,
    env_progression: list,
    stats,
    tasks: list,
    iterations: int,
    test_results: str = ""
):
    """Print ADHD-style bombardment of environment statistics when switching"""
    console = Console()
    
    # Calculate stats for current environment
    current_env_iterations = iterations
    current_env_tasks = len(tasks) if tasks else 0
    
    # Agent timing stats for current environment
    agent_stats = {}
    for agent in ["manager", "coder", "tester", "reviewer"]:
        agent_timings = [t for t in stats.timings if t.agent == agent]
        if agent_timings:
            durations = [t.duration for t in agent_timings]
            agent_stats[agent] = {
                "calls": len(agent_timings),
                "total": sum(durations),
                "avg": sum(durations) / len(durations),
            }
    
    # Environment progression status
    total_envs = len(env_progression) if env_progression else 1
    solved_count = len(solved_environments)
    progress_pct = (solved_count / total_envs * 100) if total_envs > 0 else 0
    
    # Create main panel with explosion emojis
    print("\n" + "💥" * 35)
    print(" " * 20 + "[bold red]ENVIRONMENT SWITCH DETECTED![/bold red]")
    print("💥" * 35 + "\n")
    
    # Environment transition table
    env_table = Table(show_header=True, header_style="bold magenta", box=None)
    env_table.add_column("FROM", style="cyan", width=30)
    env_table.add_column("→", style="yellow", width=5, justify="center")
    env_table.add_column("TO", style="green", width=30)
    env_table.add_row(
        f"[bold]{current_env_name}[/bold]",
        "🚀",
        f"[bold]{next_env_name}[/bold]"
    )
    console.print(env_table)
    print()
    
    # Stats grid
    stats_columns = []
    
    # Current environment stats
    current_stats = Panel(
        f"[bold cyan]CURRENT ENV STATS[/bold cyan]\n\n"
        f"📊 Iterations: [yellow]{current_env_iterations}[/yellow]\n"
        f"📝 Tasks: [yellow]{current_env_tasks}[/yellow]\n"
        f"✅ Status: [green]COMPLETED[/green]",
        border_style="cyan"
    )
    stats_columns.append(current_stats)
    
    # Overall progress
    progress_stats = Panel(
        f"[bold green]OVERALL PROGRESS[/bold green]\n\n"
        f"🏆 Solved: [yellow]{solved_count}/{total_envs}[/yellow]\n"
        f"📈 Progress: [yellow]{progress_pct:.1f}%[/yellow]\n"
        f"🎯 Remaining: [yellow]{total_envs - solved_count}[/yellow]",
        border_style="green"
    )
    stats_columns.append(progress_stats)
    
    # Agent performance
    agent_perf = "\n".join([
        f"{agent.capitalize()}: {agent_stats[agent]['calls']} calls, {agent_stats[agent]['total']:.1f}s total"
        for agent in ["manager", "coder", "tester", "reviewer"]
        if agent in agent_stats
    ])
    agent_stats_panel = Panel(
        f"[bold yellow]AGENT PERFORMANCE[/bold yellow]\n\n{agent_perf}",
        border_style="yellow"
    )
    stats_columns.append(agent_stats_panel)
    
    console.print(Columns(stats_columns, equal=True, expand=True))
    print()
    
    # Environment progression timeline
    if env_progression:
        timeline_table = Table(show_header=True, header_style="bold blue", box=None)
        timeline_table.add_column("#", style="dim", width=3)
        timeline_table.add_column("Environment", style="cyan", width=20)
        timeline_table.add_column("Status", width=15)
        timeline_table.add_column("Threshold", style="yellow", width=12)
        
        for i, env in enumerate(env_progression):
            if env.name in solved_environments:
                status = "[green]✓ SOLVED[/green]"
            elif env.name == current_env_name:
                status = "[yellow]→ CURRENT[/yellow]"
            elif env.name == next_env_name:
                status = "[cyan]→ NEXT[/cyan]"
            else:
                status = "[dim]PENDING[/dim]"
            
            timeline_table.add_row(
                str(i + 1),
                env.name,
                status,
                str(env.success_threshold)
            )
        
        console.print(timeline_table)
        print()
    
    # Test results summary if available
    if test_results:
        test_summary = test_results[:300] + "..." if len(test_results) > 300 else test_results
        console.print(Panel(
            f"[bold magenta]LATEST TEST RESULTS[/bold magenta]\n\n[dim]{test_summary}[/dim]",
            border_style="magenta"
        ))
        print()


def print_manager_report(report: str):
    """Print manager's report to leadership in a fancy format"""
    console = Console()
    
    print("\n" + "📋" * 35)
    print(" " * 15 + "[bold blue]MANAGER REPORT TO LEADERSHIP[/bold blue]")
    print("📋" * 35 + "\n")
    
    # Print report in a fancy panel
    console.print(Panel(
        f"[bold white]{report}[/bold white]",
        title="[bold cyan]📊 Executive Summary[/bold cyan]",
        border_style="blue",
        padding=(1, 2)
    ))
    
    print("\n" + "─" * 70 + "\n")


def print_reviewer_cynical_report(report: str):
    """Print reviewer's cynical, sarcastic report in a fancy format"""
    console = Console()
    
    print("\n" + "⚡" * 35)
    print(" " * 10 + "[bold red]REVIEWER'S CYNICAL ASSESSMENT[/bold red]")
    print("⚡" * 35 + "\n")
    
    # Print report in a fancy panel with red/magenta theme for the cynical tone
    console.print(Panel(
        f"[italic white]{report}[/italic white]",
        title="[bold red]💀 Shodan's Verdict[/bold red]",
        border_style="red",
        padding=(1, 2),
        subtitle="[dim]A superior intellect's take on your 'achievements'[/dim]"
    ))
    
    print("\n" + "─" * 70 + "\n")
