from .base import BaseAgent
from rich import print
import os
import random
import difflib

class Coder(BaseAgent):
    def __init__(self, config):
        super().__init__(config, "coder")

    def _get_code_context(self, state: dict) -> str:
        """
        Get code context for coder to see:
        - If previous code exists, show diff
        - Otherwise, show 30 random lines from current code
        """
        current_code = state.get("code", "")
        if not current_code:
            return ""
        
        iteration = state.get("iteration", 0)
        run_id = state.get("run_id", "")
        
        # Try to load previous iteration code
        if iteration > 0 and run_id:
            prev_code_path = f"output/{run_id}/code/agent_code_iter_{iteration - 1}.py"
            if os.path.exists(prev_code_path):
                try:
                    with open(prev_code_path, "r", encoding="utf-8") as f:
                        prev_code = f.read()
                    
                    # Generate diff
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
                        # Limit diff to reasonable size (first 200 lines of diff)
                        diff_lines = diff_text.splitlines()
                        if len(diff_lines) > 200:
                            diff_text = "\n".join(diff_lines[:200]) + "\n... (diff truncated)"
                        return f"Code changes (diff from previous iteration):\n{diff_text}"
                except Exception:
                    # If reading previous code fails, fall through to random lines
                    pass
        
        # No previous code or reading failed - show 30 random lines
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

    def __call__(self, state: dict) -> dict:
        # Coder doesn't talk - just works silently
        prompt_dict = self.config.get_prompt("coder")
        task_template = prompt_dict["task_template"].format(
            current_task=state.get("current_task", ""),
            environment=self.config.environment.name,
            video_dir=state.get("video_dir", self.config.video.output_dir)
        )
        
        # Get code context (diff or random snippet)
        code_context = self._get_code_context(state)
        context_section = f"\n\n{code_context}" if code_context else ""
        
        full_prompt = prompt_dict["system"] + "\n\n" + task_template + context_section
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
        
        # Log to conversation file
        logger = state.get("conversation_logger")
        if logger:
            logger.log_coder(
                iteration=state.get("iteration", 0),
                code=code,
                task=state.get("current_task", "")
            )
        
        return {"code": code}