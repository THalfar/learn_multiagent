from .base import BaseAgent
from rich import print

class Coder(BaseAgent):
    def __init__(self, config):
        super().__init__(config, "coder")

    def __call__(self, state: dict) -> dict:
        # Coder doesn't talk - just works silently
        prompt_dict = self.config.get_prompt("coder")
        task_template = prompt_dict["task_template"].format(
            current_task=state.get("current_task", ""),
            environment=self.config.environment.name,
            video_dir=state.get("video_dir", self.config.video.output_dir)
        )
        full_prompt = prompt_dict["system"] + "\n\n" + task_template + f"\n\nExisting code:\n{state.get('code', '')}"
        response = self.call_llm_timed(full_prompt, state["stats"], state.get("iteration", 0))
        
        # Print thinking process if using reasoning model (but no other output)
        self.print_thinking(response.content)
        
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