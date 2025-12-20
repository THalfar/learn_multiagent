import json
from .base import BaseAgent

class Reviewer(BaseAgent):
    def __call__(self, state: dict) -> dict:
        self.logger.info("Reviewer evaluating code & results")
        prompt_dict = self.config.get_prompt("reviewer")
        task_template = prompt_dict["task_template"].format(
            code=state.get("code", ""),
            test_results=state.get("test_results", ""),
            success_threshold=self.config.agents.success_threshold,
        )
        full_prompt = prompt_dict["system"] + "\n\n" + task_template
        response = self.llm.invoke(full_prompt)
        try:
            parsed = json.loads(response.content.strip())
            approved = parsed.get("approved", False)
            feedback = parsed.get("feedback", "No feedback")
            self.logger.info(f"Reviewer approved: {approved} | Feedback: {feedback[:50]}...")
            return {
                "review_feedback": feedback,
            }
        except json.JSONDecodeError as e:
            self.logger.error(f"Reviewer parse error: {e}")
            return {"review_feedback": "parse error"}