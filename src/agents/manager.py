import json
from .base import BaseAgent
from rich import print

class Manager(BaseAgent):
    def __init__(self, config):
        super().__init__(config, "manager")

    def __call__(self, state: dict) -> dict:
        print("[bold blue]Manager deciding next task[/bold blue]")
        expected_iteration = state.get("iteration", 0) + 1
        prompt_dict = self.config.get_prompt("manager")
        code_summary = (state.get("code", "")[:200] or "") + "..." if len(state.get("code", "")) > 200 else state.get("code", "")
        task_template = prompt_dict["task_template"].format(
            tasks=state.get("tasks", []),
            code_summary=code_summary,
            test_results=state.get("test_results", ""),
            review_feedback=state.get("review_feedback", ""),
            iteration=expected_iteration,
            max_iterations=self.config.agents.max_iterations,
            environment=self.config.environment.name,
            success_threshold=self.config.agents.success_threshold,
            video_dir=state.get("video_dir", self.config.video.output_dir),
        )
        system_prompt = prompt_dict["system"].format(success_threshold=self.config.agents.success_threshold)
        full_prompt = system_prompt + "\n\n" + task_template
        response = self.call_llm_timed(full_prompt, state["stats"], expected_iteration)
        try:
            parsed = json.loads(response.content.strip())
            next_task = parsed.get("next_task", "No task decided")
            print(f"[bold blue]Manager chose: {next_task}[/bold blue]")
            return {
                "tasks": state.get("tasks", []) + [next_task],
                "current_task": next_task,
                "iteration": expected_iteration,
            }
        except json.JSONDecodeError as e:
            self.logger.error(f"Manager JSON parse error: {e}")
            return {"current_task": "error: invalid JSON from LLM"}