from .base import BaseAgent
from rich import print
from rich.syntax import Syntax
from rich.panel import Panel
from rich.console import Console
import os

console = Console()

class Coder(BaseAgent):
    def __init__(self, config, model_switcher=None):
        super().__init__(config, "coder", model_switcher=model_switcher)

    def _get_code_context(self, state: dict) -> str:
        """
        Get previous iteration's code for coder to see.
        Simple: if previous code exists, show it. Otherwise empty.
        Coder always writes complete scripts, no diffs needed.
        """
        iteration = state.get("iteration", 0)
        run_id = state.get("run_id", "")

        # First iteration - no previous code
        if iteration == 0 or not run_id:
            return ""

        # Try to load previous iteration's code
        prev_code_path = f"output/{run_id}/code/agent_code_iter_{iteration - 1}.py"
        if not os.path.exists(prev_code_path):
            return ""

        try:
            with open(prev_code_path, "r", encoding="utf-8") as f:
                prev_code = f.read()
        except Exception:
            return ""

        if not prev_code.strip():
            return ""

        # Show previous code (coder will write new version based on task)
        lines = prev_code.splitlines()
        line_count = len(lines)

        # For typical RL scripts (< 200 lines), show full code
        if line_count <= 200:
            return f"PREVIOUS CODE (iteration {iteration - 1}, {line_count} lines):\n{prev_code}"

        # For longer code, show first and last parts
        first_part = "\n".join(lines[:100])
        last_part = "\n".join(lines[-100:])
        return f"PREVIOUS CODE (iteration {iteration - 1}, {line_count} lines, showing first 100 + last 100):\n{first_part}\n\n... ({line_count - 200} lines omitted) ...\n\n{last_part}"

    def _print_code_summary(self, code: str, state: dict):
        """
        Print a quick visual summary of what coder produced.
        Shows stats and a snippet so you can see 'what's happening at the lathe'.
        """
        import re
        lines = code.splitlines()
        line_count = len(lines)
        iteration = state.get("iteration", 0)

        # Count imports
        import_lines = [l.strip() for l in lines if l.strip().startswith(('import ', 'from '))]
        import_count = len(import_lines)

        # Count functions and classes
        func_count = len(re.findall(r'^def \w+', code, re.MULTILINE))
        class_count = len(re.findall(r'^class \w+', code, re.MULTILINE))

        # Detect algorithm
        algo = "Unknown"
        if 'PPO' in code:
            algo = "PPO"
        elif 'DQN' in code:
            algo = "DQN"
        elif 'A2C' in code:
            algo = "A2C"
        elif 'SAC' in code:
            algo = "SAC"

        # Detect training timesteps
        timesteps_match = re.search(r'total_timesteps\s*=\s*(\d+)', code)
        timesteps = timesteps_match.group(1) if timesteps_match else "?"

        # Get first 3 imports for snippet
        import_snippet = import_lines[:3] if import_lines else ["(no imports)"]

        # Get key code snippet (first function or main block)
        snippet_lines = []
        in_main = False
        for i, line in enumerate(lines):
            if 'def ' in line or 'if __name__' in line or ('model' in line.lower() and '=' in line):
                snippet_lines = lines[i:i+3]
                break

        # Build output
        console.print("\n\n" + "─" * 70)
        console.print(f"[bold green]🔧 CODER OUTPUT - Iteration {iteration}[/bold green]")
        console.print("─" * 70)

        # Stats line
        stats_parts = [
            f"[green]{line_count}[/green] lines",
            f"[yellow]{import_count}[/yellow] imports",
            f"[blue]{func_count}[/blue] funcs",
        ]
        if class_count > 0:
            stats_parts.append(f"[magenta]{class_count}[/magenta] classes")
        stats_parts.append(f"[cyan]{algo}[/cyan]")
        if timesteps != "?":
            stats_parts.append(f"[dim]{int(timesteps):,} steps[/dim]")

        console.print("📊 " + " │ ".join(stats_parts))

        # Import health indicator
        if import_count <= 15:
            health = "[green]✓ Clean[/green]"
        elif import_count <= 25:
            health = "[yellow]~ OK[/yellow]"
        else:
            health = "[red]⚠ Bloated[/red]"
        console.print(f"📦 Imports: {health} ({', '.join(import_snippet[:2])}{'...' if len(import_lines) > 2 else ''})")

        # Code snippet
        if snippet_lines:
            snippet_preview = snippet_lines[0][:60] + ('...' if len(snippet_lines[0]) > 60 else '')
            console.print(f"[dim]📝 {snippet_preview}[/dim]")

        console.print("─" * 70)

    def _check_code_quality(self, code: str) -> str:
        """
        Check code quality and fix repetition loops.
        Deduplicates imports if model got stuck in a repetition loop.
        Triggers adaptive model switch if repetition loops persist.
        """
        lines = code.splitlines()

        # Count imports
        import_lines = [l.strip() for l in lines if l.strip().startswith(('import ', 'from '))]
        non_import_lines = [l for l in lines if l.strip() and not l.strip().startswith(('import ', 'from '))]

        # Detect repetition loop: many imports but no actual code
        if len(import_lines) > 30 and len(non_import_lines) < 5:
            console.print(f"[red]⚠️  REPETITION LOOP DETECTED: {len(import_lines)} imports, {len(non_import_lines)} code lines[/red]")
            console.print(f"[yellow]   Attempting to salvage by deduplicating imports...[/yellow]")

            # Trigger adaptive model switch if enabled
            if self.model_switcher:
                from src.utils.model_switcher import SwitchTrigger
                new_model = self.model_switcher.check_and_switch(
                    self.agent_name,
                    SwitchTrigger.REPETITION_LOOP,
                    {"import_count": len(import_lines), "code_lines": len(non_import_lines)}
                )
                if new_model:
                    self.switch_model(new_model)

            # Deduplicate imports while preserving order
            seen_imports = set()
            unique_imports = []
            for imp in import_lines:
                if imp not in seen_imports:
                    seen_imports.add(imp)
                    unique_imports.append(imp)

            console.print(f"[green]   Reduced to {len(unique_imports)} unique imports[/green]")

            # If we only have imports and no code, return a minimal error script
            if len(non_import_lines) < 3:
                console.print(f"[red]   ERROR: No actual code found after imports![/red]")
                # Return a minimal script that will fail with a clear error
                return "\n".join(unique_imports) + """

# ERROR: Model repetition loop - no actual training code was generated
# The model got stuck repeating imports and never wrote the training logic
print("ERROR: Code generation failed - model produced only imports, no training code")
raise RuntimeError("Repetition loop detected: model produced only imports")
"""

            # Reconstruct code with unique imports + remaining lines
            return "\n".join(unique_imports) + "\n" + "\n".join(non_import_lines)

        # Normal quality warnings
        if len(import_lines) > 40:
            console.print(f"[yellow]⚠️  Note: {len(import_lines)} import lines (check if intentional)[/yellow]")

        # Check for obvious repetition (same line 5+ times in a row)
        prev_line = None
        repeat_count = 0
        max_repeat = 0
        for line in lines:
            if line == prev_line and line.strip():
                repeat_count += 1
                max_repeat = max(max_repeat, repeat_count)
            else:
                repeat_count = 0
            prev_line = line

        if max_repeat >= 5:
            console.print(f"[yellow]⚠️  Note: Detected {max_repeat}x repeated line (possible loop)[/yellow]")
        else:
            # No repetition loop - clear the counter
            if self.model_switcher:
                self.model_switcher.report_success(self.agent_name)

        # Return code unchanged - let Python/tester catch actual errors
        return code

    def __call__(self, state: dict) -> dict:
        # Show what coder is working on
        iteration = state.get("iteration", 0)
        task = state.get("current_task", "")
        task_preview = task[:80] + "..." if len(task) > 80 else task
        console.print(f"\n\n[green]🔧 Coder working on: {task_preview}[/green]")

        prompt_dict = self.config.get_prompt("coder")
        
        # Normalize video_dir to absolute path (fixes Windows path issues)
        video_dir = state.get("video_dir", self.config.video.output_dir)
        video_dir = os.path.abspath(os.path.normpath(video_dir))
        
        # Get iteration for unique video subdirectory
        iteration = state.get("iteration", 0)
        
        # Get device from current environment in progression
        env_progression = self.config.environment_progression
        current_env_index = state.get("current_env_index", 0)
        current_env = env_progression[current_env_index] if env_progression and current_env_index < len(env_progression) else None

        # Get device (cpu/gpu/auto) from environment config
        device = current_env.device if current_env and hasattr(current_env, 'device') else "cpu"

        task_template = prompt_dict["task_template"].format(
            current_task=state.get("current_task", ""),
            environment=self.config.environment.name,
            video_dir=video_dir,
            iteration=iteration,
            device=device,
        )
        
        # Get code context (diff or random snippet)
        code_context = self._get_code_context(state)
        context_section = f"\n\n{code_context}" if code_context else ""

        # Add conversation history (siloed - only this agent's previous messages)
        # Note: Coder has history_window=0, so this will be empty
        history_text = self.format_conversation_history(state)

        full_prompt = prompt_dict["system"] + "\n\n" + history_text + task_template + context_section

        # Print context breakdown before LLM call (coder has no team chatter)
        prompt_tokens = self.estimate_tokens(full_prompt)
        self.print_context_breakdown(state, prompt_tokens, "")

        response = self.call_llm_timed(full_prompt, state["stats"], state.get("iteration", 0))
        
        # Print thinking process if using reasoning model (but no other output)
        self.print_thinking(response.content)
        
        # Print token statistics
        stats_obj = state["stats"]
        iteration = state.get("iteration", 0)
        agent_timings = [t for t in stats_obj.timings if t.agent == self.agent_name and t.iteration == iteration]
        if agent_timings:
            latest_timing = agent_timings[-1]
            self.print_token_stats(latest_timing)
        
        code = response.content.strip()

        # Remove any thinking tags from code (shouldn't be there, but safety check)
        import re
        code = re.sub(r'<think[^>]*>.*?</think[^>]*>', '', code, flags=re.DOTALL | re.IGNORECASE)
        code = re.sub(r'<thinking[^>]*>.*?</thinking[^>]*>', '', code, flags=re.DOTALL | re.IGNORECASE)

        # Strip markdown code blocks
        if code.startswith('```python'):
            code = code[9:].lstrip()
        elif code.startswith('```'):
            code = code[3:].lstrip()
        if code.endswith('```'):
            code = code[:-3].rstrip()
        code = code.strip()

        # Light diagnostics (no auto-fixing - trust the model)
        code = self._check_code_quality(code)

        # Always show a quick summary of what coder produced
        self._print_code_summary(code, state)

        # Show generated code if enabled (visually formatted)
        if self.config.agents.show_coder_output:
            iteration = state.get("iteration", 0)
            lines_count = len(code.splitlines())
            task = state.get("current_task", "No task specified")

            # Show task summary first
            console.print("\n\n" + "─" * 70)
            console.print(f"[bold green]🔧 CODER - Iteration {iteration}[/bold green]")
            console.print("─" * 70)
            console.print(f"[yellow]Task:[/yellow] {task[:200]}{'...' if len(task) > 200 else ''}")
            console.print(f"[green]Generating complete Python script...[/green]")
            console.print()

            # Create syntax-highlighted code
            syntax = Syntax(
                code,
                "python",
                theme="monokai",
                line_numbers=True,
                word_wrap=False,
                background_color="default"
            )

            # Create beautiful panel with the code
            panel = Panel(
                syntax,
                title=f"[bold green]✓ Generated Code[/bold green]",
                subtitle=f"[dim]{lines_count} lines • Python 3.x[/dim]",
                border_style="green",
                padding=(1, 2)
            )

            console.print(panel)
            console.print("─" * 70 + "\n")

        # Track code statistics
        lines_count = len(code.splitlines())
        iteration = state.get("iteration", 0)
        stats_obj = state.get("stats")
        if stats_obj:
            stats_obj.add_code_stats(iteration, lines_count)

        # Log to conversation file
        logger = state.get("conversation_logger")
        if logger:
            logger.log_coder(
                iteration=iteration,
                code=code,
                task=state.get("current_task", "")
            )

        # Save coder's response to conversation history
        history_update = self.save_message_to_history(state, response.content)

        result = {"code": code}
        result.update(history_update)

        return result