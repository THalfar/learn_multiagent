from .base import BaseAgent

class Coder(BaseAgent):
    def __call__(self, state: dict) -> dict:
        self.logger.info(f"Coder generating code for '{state.get('current_task', 'unknown')}'")
        prompt_dict = self.config.get_prompt("coder")
        task_template = prompt_dict["task_template"].format(
            current_task=state.get("current_task", ""),
            environment=self.config.environment.name,
            algorithm=self.config.algorithm.name,
            parameters=str(self.config.algorithm.parameters),
            video_dir=self.config.video.output_dir,
        )
        full_prompt = prompt_dict["system"] + "\n\n" + task_template + f"\n\nExisting code:\n{state.get('code', '')}"
        response = self.llm.invoke(full_prompt)
        code = response.content.strip()
        self.logger.info(f"Coder output length: {len(code)} chars")
        return {"code": code}