from .base import BaseAgent
from rich import print

class Coder(BaseAgent):
    def __init__(self, config):
        super().__init__(config, "coder")

    def __call__(self, state: dict) -> dict:
        self.logger.info(f"Coder generating code for '{state.get('current_task', 'unknown')}'")
        prompt_dict = self.config.get_prompt("coder")
        task_template = prompt_dict["task_template"].format(
            current_task=state.get("current_task", ""),
            environment=self.config.environment.name,
            video_dir=state.get("video_dir", self.config.video.output_dir)
        )
        full_prompt = prompt_dict["system"] + "\n\n" + task_template + f"\n\nExisting code:\n{state.get('code', '')}"
        response = self.call_llm_timed(full_prompt, state["stats"], state.get("iteration", 0))
        code = response.content.strip()
        
        # Strip markdown code blocks
        if code.startswith('```python'):
            code = code[9:].lstrip()
        elif code.startswith('```'):
            code = code[3:].lstrip()
        if code.endswith('```'):
            code = code[:-3].rstrip()
        code = code.strip()
        
        print(f"[bold green]Coder output length: {len(code)} chars[/bold green]")
        print("[bold green]=== GENERATED CODE PREVIEW ===[/bold green]")
        preview = code[:1000] + "..." if len(code) > 1000 else code
        print(f"[green]{preview}[/green]")
        print("[bold green]=== END CODE PREVIEW ===[/bold green]")
        return {"code": code}