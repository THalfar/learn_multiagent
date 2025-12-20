import json
from .base import BaseAgent

class Manager(BaseAgent):
    def __call__(self, state: dict) -> dict:
        self.logger.info("Manager deciding next task")
        prompt_dict = self.config.get_prompt("manager")
        code_summary = (state.get("code", "")[:200] or "") + "..." if len(state.get("code", "")) > 200 else state.get("code", "")
        task_template = prompt_dict["task_template"].format(
            tasks=state.get("tasks", []),
            code_summary=code_summary,
            test_results=state.get("test_results", ""),
            review_feedback=state.get("review_feedback", ""),
            iteration=state.get("iteration", 0),
            max_iterations=self.config.agents.max_iterations,
        )
        full_prompt = prompt_dict["system"] + "\n\n" + task_template
        response = self.llm.invoke(full_prompt)
        try:
            parsed = json.loads(response.content.strip())
            next_task = parsed.get("next_task", "No task decided")
            self.logger.info(f"Manager chose: {next_task}")
            return {
                "tasks": state.get("tasks", []) + [next_task],
                "current_task": next_task,
                "iteration": 1,
            }
        except json.JSONDecodeError as e:
            self.logger.error(f"Manager JSON parse error: {e}")
            return {"current_task": "error: invalid JSON from LLM"}