from .base import BaseAgent
from rich import print
from rich.syntax import Syntax
from rich.panel import Panel
from rich.console import Console
import os

console = Console()

class Coder(BaseAgent):
    def __init__(self, config):
        super().__init__(config, "coder")

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

    def _check_code_quality(self, code: str) -> str:
        """
        Light diagnostics only - no auto-fixing.
        Trust the model, let tester report actual errors.
        """
        lines = code.splitlines()

        # Diagnostic warnings (no modifications)
        import_lines = [l for l in lines if l.strip().startswith(('import ', 'from '))]
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

        # Return code unchanged - let Python/tester catch actual errors
        return code

    def __call__(self, state: dict) -> dict:
        # Coder is working
        if not self.config.agents.show_coder_output:
            # Only show simple message if not displaying code
            print("[dim]Coder is generating code... (this may take a moment if loading models)[/dim]")
        prompt_dict = self.config.get_prompt("coder")
        
        # Normalize video_dir to absolute path (fixes Windows path issues)
        video_dir = state.get("video_dir", self.config.video.output_dir)
        video_dir = os.path.abspath(os.path.normpath(video_dir))
        
        # Get iteration for unique video subdirectory
        iteration = state.get("iteration", 0)
        
        task_template = prompt_dict["task_template"].format(
            current_task=state.get("current_task", ""),
            environment=self.config.environment.name,
            video_dir=video_dir,
            iteration=iteration
        )
        
        # Get code context (diff or random snippet)
        code_context = self._get_code_context(state)
        context_section = f"\n\n{code_context}" if code_context else ""

        # Add conversation history (siloed - only this agent's previous messages)
        # Note: Coder has history_window=0, so this will be empty
        history_text = self.format_conversation_history(state)

        full_prompt = prompt_dict["system"] + "\n\n" + history_text + task_template + context_section
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

        # Show generated code if enabled (visually formatted)
        if self.config.agents.show_coder_output:
            iteration = state.get("iteration", 0)
            lines_count = len(code.splitlines())
            task = state.get("current_task", "No task specified")

            # Show task summary first
            console.print("\n" + "─" * 70)
            console.print(f"[bold cyan]🔧 CODER - Iteration {iteration}[/bold cyan]")
            console.print("─" * 70)
            console.print(f"[yellow]Task:[/yellow] {task[:200]}{'...' if len(task) > 200 else ''}")
            console.print(f"[dim]Generating complete Python script...[/dim]")
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