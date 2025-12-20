import json
from .base import BaseAgent

class Tester(BaseAgent):
    def __call__(self, state: dict) -> dict:
        self.logger.info("Tester evaluating (Phase 5 full exec)")
        # Dummy for Phase 4
        mean_reward = 25.0
        std_reward = 10.0
        n_episodes = 10
        video_path = f"{self.config.video.output_dir}/dummy.mp4"
        prompt_dict = self.config.get_prompt("tester")
        task_template = prompt_dict["task_template"].format(
            mean_reward=mean_reward,
            std_reward=std_reward,
            n_episodes=n_episodes,
            video_path=video_path,
        )
        full_prompt = prompt_dict["system"] + "\n\n" + task_template
        response = self.llm.invoke(full_prompt)
        try:
            parsed = json.loads(response.content.strip())
            summary = parsed.get("summary", "No summary")
            self.logger.info(f"Tester summary: {summary[:50]}...")
            return {"test_results": summary}
        except json.JSONDecodeError as e:
            self.logger.error(f"Tester parse error: {e}")
            return {"test_results": "parse error"}