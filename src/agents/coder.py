from .base import BaseAgent
from rich import print
from rich.syntax import Syntax
from rich.panel import Panel
from rich.console import Console
import os
import random
import difflib

console = Console()

class Coder(BaseAgent):
    def __init__(self, config):
        super().__init__(config, "coder")

    def _get_code_context(self, state: dict) -> str:
        """
        Get code context for coder to see:
        - If previous code exists in file, show it (or diff if current code exists)
        - Otherwise, show current code if available
        - This ensures coder always sees previous version, preventing loops from scratch
        """
        current_code = state.get("code", "")
        iteration = state.get("iteration", 0)
        run_id = state.get("run_id", "")
        
        # Try to load previous iteration code from saved file
        prev_code = None
        if iteration > 0 and run_id:
            prev_code_path = f"output/{run_id}/code/agent_code_iter_{iteration - 1}.py"
            if os.path.exists(prev_code_path):
                try:
                    with open(prev_code_path, "r", encoding="utf-8") as f:
                        prev_code = f.read()
                except Exception:
                    # If reading previous code fails, continue without it
                    pass
        
        # If we have both previous and current code, show diff
        if prev_code and current_code:
            try:
                current_lines = current_code.splitlines(keepends=True)
                prev_lines = prev_code.splitlines(keepends=True)
                diff = list(difflib.unified_diff(
                    prev_lines,
                    current_lines,
                    fromfile=f"previous (iter {iteration - 1})",
                    tofile=f"current (iter {iteration})",
                    lineterm=""
                ))
                
                if diff:
                    # Show diff (skip first 2 lines which are headers)
                    diff_text = "".join(diff[2:])  # Skip "---" and "+++" headers
                    # Show full diff for learning (RL scripts are small, full context helps coder learn)
                    diff_lines = diff_text.splitlines()
                    if len(diff_lines) > 400:  # Generous limit for full visibility
                        diff_text = "\n".join(diff_lines[:400]) + "\n... (diff truncated, showing first 400 of {} lines)".format(len(diff_lines))
                    return f"Code changes (diff from previous iteration):\n{diff_text}"
            except Exception:
                # If diff generation fails, fall through to showing previous code
                pass
        
        # If we have previous code but no current code (or diff failed), show previous code
        if prev_code:
            prev_lines = prev_code.splitlines()
            # Show full context - RL scripts are small, coder benefits from seeing everything
            if len(prev_lines) <= 500:  # Most RL scripts fit here
                # Show full code - coder sees everything, learns better
                return f"Previous iteration code (iter {iteration - 1}, {len(prev_lines)} lines):\n" + "\n".join(prev_lines)
            else:
                # For very long code, show generous context from start and end
                first_part = "\n".join(prev_lines[:150])
                last_part = "\n".join(prev_lines[-150:])
                return f"Previous iteration code (iter {iteration - 1}, showing first 150 and last 150 of {len(prev_lines)} total lines):\n{first_part}\n... ({len(prev_lines) - 300} lines omitted) ...\n{last_part}"
        
        # If we have current code but no previous code, show current code
        if current_code:
            lines = current_code.splitlines()
            if len(lines) <= 30:
                # If code is short, show all
                return f"Current code ({len(lines)} lines):\n" + "\n".join(lines)
            
            # Pick random starting point
            max_start = max(0, len(lines) - 30)
            start_line = random.randint(0, max_start)
            end_line = start_line + 30
            
            # Add line numbers for context
            context_lines = []
            for i in range(start_line, min(end_line, len(lines))):
                context_lines.append(f"{i+1:4d} | {lines[i]}")
            
            return f"Code snippet (lines {start_line+1}-{end_line} of {len(lines)} total lines):\n" + "\n".join(context_lines)
        
        # No code available (first iteration)
        return ""

    def _detect_and_fix_repetition(self, code: str) -> str:
        """
        Detect and handle repetition loops in generated code.
        LLMs (especially local models) can get stuck repeating patterns.
        """
        lines = code.splitlines()

        # Check 1: Too many import lines (normal script has ~10-30 imports max)
        import_lines = [l for l in lines if l.strip().startswith(('import ', 'from '))]
        if len(import_lines) > 50:
            console.print(f"[yellow]⚠️  Detected repetition loop: {len(import_lines)} import lines (expected <50)[/yellow]")
            # Deduplicate imports and keep unique ones
            seen_imports = set()
            unique_imports = []
            for line in import_lines:
                normalized = line.strip()
                if normalized not in seen_imports:
                    seen_imports.add(normalized)
                    unique_imports.append(line)

            # Find where actual code starts (after imports)
            code_start_idx = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped and not stripped.startswith(('import ', 'from ', '#')):
                    code_start_idx = i
                    break

            # Rebuild code with unique imports + rest
            rest_of_code = lines[code_start_idx:] if code_start_idx < len(lines) else []
            # Filter out any repeated imports from rest
            rest_of_code = [l for l in rest_of_code if not l.strip().startswith(('import ', 'from '))]

            console.print(f"[green]✓ Cleaned to {len(unique_imports)} unique imports[/green]")
            code = '\n'.join(unique_imports + [''] + rest_of_code)
            lines = code.splitlines()

        # Check 2: Repeated consecutive lines (exact duplicates)
        if len(lines) > 10:
            cleaned_lines = []
            prev_line = None
            repeat_count = 0
            for line in lines:
                if line == prev_line and line.strip():  # Same non-empty line
                    repeat_count += 1
                    if repeat_count > 2:  # Allow max 2 repeats (e.g., blank lines)
                        continue  # Skip this repeated line
                else:
                    repeat_count = 0
                cleaned_lines.append(line)
                prev_line = line

            if len(cleaned_lines) < len(lines):
                removed = len(lines) - len(cleaned_lines)
                console.print(f"[yellow]⚠️  Removed {removed} consecutively repeated lines[/yellow]")
                code = '\n'.join(cleaned_lines)
                lines = cleaned_lines

        # Check 3: Unicode corruption (non-ASCII in code that shouldn't have it)
        # Thai, Chinese, etc. characters in import statements = corruption
        import re
        suspicious_unicode = re.search(r'[\u0e00-\u0e7f\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', code)
        if suspicious_unicode:
            console.print(f"[yellow]⚠️  Detected Unicode corruption in output, cleaning...[/yellow]")
            # Remove lines with suspicious unicode in import/code (not in strings)
            cleaned_lines = []
            for line in lines:
                # If line has suspicious unicode and it's not inside a string
                if re.search(r'[\u0e00-\u0e7f\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', line):
                    # Check if it's in a string (rough check)
                    if '"' not in line and "'" not in line:
                        console.print(f"[dim]  Removed corrupted line: {line[:60]}...[/dim]")
                        continue
                cleaned_lines.append(line)
            code = '\n'.join(cleaned_lines)

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

        # Detect repetition loops (common LLM failure mode)
        code = self._detect_and_fix_repetition(code)

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