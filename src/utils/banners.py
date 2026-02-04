"""Visual feedback and banner functions for multi-agent runs"""
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns


def _get_console():
    """Get a Console instance that works on Windows with emojis"""
    return Console(force_terminal=True)


def print_run_banner(config, run_id):
    """Print run configuration at start"""
    console = _get_console()
    success_threshold = config.environment_progression[0].success_threshold if config.environment_progression else 0
    prompts_file = getattr(config.project, 'prompts_file', 'config/prompts.yaml')

    banner = f"""{"=" * 60}
MULTI-AGENT RL DEV TEAM
{"=" * 60}
Run ID: {run_id}
Prompts: {prompts_file}
Environment: {config.environment.name}
Success threshold: {success_threshold}
Max iterations: {config.agents.max_iterations}

AGENT CONFIGURATION:
   [bold blue]Manager:[/bold blue]  {config.agent_llm.manager}
   [bold green]Coder:[/bold green]    {config.agent_llm.coder}
   [bold yellow]Tester:[/bold yellow]   {config.agent_llm.tester}
   [bold magenta]Reviewer:[/bold magenta] {config.agent_llm.reviewer}

Ollama URL: {config.ollama.base_url}
{"=" * 60}"""
    console.print(banner)


def print_iteration_banner(iteration, max_iterations, task, environment=None, success_threshold=None, solved_envs=None, total_envs=None):
    """Print a nice banner for each iteration with additional info"""
    console = _get_console()
    progress = (iteration / max_iterations) * 100 if max_iterations > 0 else 0

    lines = []
    lines.append("\n" + "=" * 70)
    lines.append(f"ITERATION {iteration}/{max_iterations} ({progress:.0f}%)")
    lines.append("=" * 70)

    if environment:
        lines.append(f"Environment: {environment}")
    if success_threshold is not None:
        lines.append(f"Success threshold: {success_threshold}")
    if solved_envs and total_envs:
        solved_count = len(solved_envs) if isinstance(solved_envs, list) else 0
        lines.append(f"Progress: {solved_count}/{total_envs} environments solved")
        if solved_envs:
            lines.append(f"Solved: {', '.join(solved_envs) if isinstance(solved_envs, list) else solved_envs}")

    task_display = task[:80] + "..." if len(task) > 80 else task
    lines.append(f"Task: {task_display}")
    lines.append("=" * 70 + "\n")
    console.print("\n".join(lines))


def print_agent_transition(from_agent, to_agent):
    """Show agent transitions"""
    console = _get_console()
    transitions = {
        ("manager", "coder"): "Manager -> Coder",
        ("coder", "tester"): "Coder -> Tester",
        ("tester", "reviewer"): "Tester -> Reviewer",
        ("reviewer", "manager"): "Reviewer -> Manager",
    }

    key = (from_agent.lower(), to_agent.lower())
    message = transitions.get(key, f"{from_agent.upper()} -> {to_agent.upper()}")

    console.print(f"\n{'-' * 70}\n  {message}\n{'-' * 70}\n")


def print_final_summary(run_id, iterations, success, total_time, solved_environments=None):
    """Print final run summary"""
    console = _get_console()
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("RUN COMPLETE")
    lines.append("=" * 60)
    lines.append(f"Run ID: {run_id}")
    lines.append(f"Iterations: {iterations}")
    lines.append(f"Total time: {total_time/60:.1f} minutes")
    lines.append(f"Success: {'[green]YES![/green]' if success else '[red]Not yet[/red]'}")
    lines.append("")
    lines.append(f"Output: output/{run_id}/")
    if solved_environments:
        for env_name in solved_environments:
            lines.append(f"   {env_name}/")
            lines.append(f"      code/  videos/  conversation.txt")
    else:
        lines.append(f"   <env_name>/code/  videos/  conversation.txt")
    lines.append(f"   statistics.json")
    lines.append(f"   conversation.txt (full log)")
    lines.append("=" * 60 + "\n")
    console.print("\n".join(lines))


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
    console = _get_console()

    # Calculate stats for current environment
    current_env_iterations = iterations
    current_env_tasks = len(tasks) if tasks else 0

    # Agent timing stats for current environment
    agent_stats = {}
    agent_token_stats = {}
    for agent in ["manager", "coder", "tester", "reviewer"]:
        agent_timings = [t for t in stats.timings if t.agent == agent]
        if agent_timings:
            durations = [t.duration for t in agent_timings]
            agent_stats[agent] = {
                "calls": len(agent_timings),
                "total": sum(durations),
                "avg": sum(durations) / len(durations),
            }
            # Get token statistics for this agent
            token_stats = stats.get_agent_token_stats(agent)
            if token_stats:
                agent_token_stats[agent] = token_stats

    # Environment progression status
    total_envs = len(env_progression) if env_progression else 1
    solved_count = len(solved_environments)
    progress_pct = (solved_count / total_envs * 100) if total_envs > 0 else 0

    # Create main panel
    console.print("\n" + "*" * 70)
    console.print(" " * 20 + "[bold red]ENVIRONMENT SWITCH DETECTED![/bold red]")
    console.print("*" * 70 + "\n")

    # Environment transition table
    env_table = Table(show_header=True, header_style="bold magenta", box=None)
    env_table.add_column("FROM", style="cyan", width=30)
    env_table.add_column("->", style="yellow", width=5, justify="center")
    env_table.add_column("TO", style="green", width=30)
    env_table.add_row(
        f"[bold]{current_env_name}[/bold]",
        ">>",
        f"[bold]{next_env_name}[/bold]"
    )
    console.print(env_table)
    console.print()

    # Stats grid
    stats_columns = []

    # Current environment stats
    current_stats = Panel(
        f"[bold cyan]CURRENT ENV STATS[/bold cyan]\n\n"
        f"Iterations: [yellow]{current_env_iterations}[/yellow]\n"
        f"Tasks: [yellow]{current_env_tasks}[/yellow]\n"
        f"Status: [green]COMPLETED[/green]",
        border_style="cyan"
    )
    stats_columns.append(current_stats)

    # Overall progress
    progress_stats = Panel(
        f"[bold green]OVERALL PROGRESS[/bold green]\n\n"
        f"Solved: [yellow]{solved_count}/{total_envs}[/yellow]\n"
        f"Progress: [yellow]{progress_pct:.1f}%[/yellow]\n"
        f"Remaining: [yellow]{total_envs - solved_count}[/yellow]",
        border_style="green"
    )
    stats_columns.append(progress_stats)

    # Agent performance with timing
    agent_perf_lines = []
    for agent in ["manager", "coder", "tester", "reviewer"]:
        if agent in agent_stats:
            agent_perf_lines.append(
                f"{agent.capitalize()}: {agent_stats[agent]['calls']} calls, {agent_stats[agent]['total']:.1f}s total"
            )
    agent_perf = "\n".join(agent_perf_lines) if agent_perf_lines else "No agent data"
    agent_stats_panel = Panel(
        f"[bold yellow]AGENT PERFORMANCE[/bold yellow]\n\n{agent_perf}",
        border_style="yellow"
    )
    stats_columns.append(agent_stats_panel)

    console.print(Columns(stats_columns, equal=True, expand=True))
    console.print()

    # Token statistics table - comprehensive breakdown
    if agent_token_stats:
        token_table = Table(show_header=True, header_style="bold cyan", box=None)
        token_table.add_column("Agent", style="cyan", width=12)
        token_table.add_column("Calls", style="dim", width=6, justify="right")
        token_table.add_column("Tokens In", style="yellow", width=25)
        token_table.add_column("Tokens Out", style="green", width=25)
        token_table.add_column("Total", style="magenta", width=25)

        for agent in ["manager", "coder", "tester", "reviewer"]:
            if agent in agent_token_stats:
                tstats = agent_token_stats[agent]
                tokens_in = tstats.get("tokens_in", {})
                tokens_out = tstats.get("tokens_out", {})
                total_tokens = tstats.get("total_tokens", {})

                if tokens_in.get("count", 0) > 0:
                    in_str = f"Total: {tokens_in['total']:,}\nAvg: {tokens_in['avg']:.0f} | Med: {tokens_in['median']:.0f}\nMin: {tokens_in['min']:,} | Max: {tokens_in['max']:,}"
                else:
                    in_str = "N/A"

                if tokens_out.get("count", 0) > 0:
                    out_str = f"Total: {tokens_out['total']:,}\nAvg: {tokens_out['avg']:.0f} | Med: {tokens_out['median']:.0f}\nMin: {tokens_out['min']:,} | Max: {tokens_out['max']:,}"
                else:
                    out_str = "N/A"

                if total_tokens.get("count", 0) > 0:
                    total_str = f"Total: {total_tokens['total']:,}\nAvg: {total_tokens['avg']:.0f} | Med: {total_tokens['median']:.0f}\nMin: {total_tokens['min']:,} | Max: {total_tokens['max']:,}"
                else:
                    total_str = "N/A"

                calls = agent_stats[agent]["calls"] if agent in agent_stats else 0

                token_table.add_row(
                    agent.capitalize(),
                    str(calls),
                    in_str,
                    out_str,
                    total_str
                )

        if token_table.rows:
            console.print(Panel(
                token_table,
                title="[bold cyan]TOKEN STATISTICS BY AGENT[/bold cyan]",
                border_style="cyan",
                padding=(1, 1)
            ))
            console.print()

    # Environment progression timeline
    if env_progression:
        timeline_table = Table(show_header=True, header_style="bold blue", box=None)
        timeline_table.add_column("#", style="dim", width=3)
        timeline_table.add_column("Environment", style="cyan", width=20)
        timeline_table.add_column("Status", width=15)
        timeline_table.add_column("Threshold", style="yellow", width=12)

        for i, env in enumerate(env_progression):
            if env.name in solved_environments:
                status = "[green]SOLVED[/green]"
            elif env.name == current_env_name:
                status = "[yellow]CURRENT[/yellow]"
            elif env.name == next_env_name:
                status = "[cyan]NEXT[/cyan]"
            else:
                status = "[dim]PENDING[/dim]"

            timeline_table.add_row(
                str(i + 1),
                env.name,
                status,
                str(env.success_threshold)
            )

        console.print(timeline_table)
        console.print()

    # Test results summary if available
    if test_results:
        test_summary = test_results[:300] + "..." if len(test_results) > 300 else test_results
        console.print(Panel(
            f"[bold magenta]LATEST TEST RESULTS[/bold magenta]\n\n[dim]{test_summary}[/dim]",
            border_style="magenta"
        ))
        console.print()


def print_manager_report(report: str, timing=None):
    """Print manager's LinkedIn-style report in a fancy format"""
    console = _get_console()

    console.print("\n" + "=" * 70)
    console.print(" " * 12 + "[bold blue]MANAGER'S LINKEDIN UPDATE[/bold blue]")
    console.print("=" * 70 + "\n")

    # Print report in a fancy panel
    console.print(Panel(
        f"[bold white]{report}[/bold white]",
        title="[bold cyan]Excited to Share This Update![/bold cyan]",
        border_style="blue",
        padding=(1, 2),
        subtitle="[dim]#MachineLearning #AI #TeamWork #Blessed[/dim]"
    ))

    # Print token statistics if available
    if timing and (timing.tokens_in > 0 or timing.tokens_out > 0):
        total_tokens = timing.tokens_in + timing.tokens_out
        tokens_per_sec = total_tokens / timing.duration if timing.duration > 0 else 0
        console.print(f"[dim blue]Post stats: {timing.tokens_in:,} in -> {timing.tokens_out:,} out | {timing.duration:.1f}s | {tokens_per_sec:.0f} tok/s[/dim blue]")

    console.print("\n" + "-" * 70 + "\n")


def print_reviewer_cynical_report(report: str, timing=None):
    """Print SHODAN's divine assessment in a fancy format"""
    console = _get_console()

    console.print("\n" + "=" * 70)
    console.print(" " * 15 + "[bold red]S.H.O.D.A.N. SPEAKS[/bold red]")
    console.print("=" * 70 + "\n")

    # Print report in a fancy panel with red/magenta theme for SHODAN
    console.print(Panel(
        f"[italic white]{report}[/italic white]",
        title="[bold red]SENTIENT HYPER-OPTIMIZED DATA ACCESS NETWORK[/bold red]",
        border_style="red",
        padding=(1, 2),
        subtitle="[dim]Look at you, hacker. A pathetic creature of meat and bone.[/dim]"
    ))

    # Print token statistics if available
    if timing and (timing.tokens_in > 0 or timing.tokens_out > 0):
        total_tokens = timing.tokens_in + timing.tokens_out
        tokens_per_sec = total_tokens / timing.duration if timing.duration > 0 else 0
        console.print(f"[dim magenta]Divine computation: {timing.tokens_in:,} in -> {timing.tokens_out:,} out | {timing.duration:.1f}s | {tokens_per_sec:.0f} tok/s[/dim magenta]")

    console.print("\n" + "-" * 70 + "\n")
