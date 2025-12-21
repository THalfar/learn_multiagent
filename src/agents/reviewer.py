import json
from .base import BaseAgent
from rich import print

class Reviewer(BaseAgent):
    def __init__(self, config):
        super().__init__(config, "reviewer")

    def __call__(self, state: dict) -> dict:
        print("[bold magenta]Reviewer evaluating code & results[/bold magenta]")
        prompt_dict = self.config.get_prompt("reviewer")
        task_template = prompt_dict["task_template"].format(
            code=state.get("code", ""),
            test_results=state.get("test_results", ""),
            success_threshold=self.config.agents.success_threshold,
            video_dir=state.get("video_dir", "output/videos"),
        )
        system_prompt = prompt_dict["system"].format(
            success_threshold=self.config.agents.success_threshold,
            video_dir=state.get("video_dir", "output/videos")
        )
        full_prompt = system_prompt + "\n\n" + task_template
        response = self.call_llm_timed(full_prompt, state["stats"], state.get("iteration", 0))
        try:
            parsed = json.loads(response.content.strip())
            approved = parsed.get("approved", False)
            feedback = parsed.get("feedback", "No feedback")
            print(f"[bold magenta]Reviewer approved: {approved} | Full Feedback: {feedback}[/bold magenta]")
            
            # Live per-iteration timing
            iteration = state.get("iteration", 0)
            stats_obj = state["stats"]
            iter_stats = stats_obj.get_iteration_stats(iteration)
            total_time = iter_stats["total_time"]
            agent_times = iter_stats["agents"]
            print(f"\n⏱️  Iteration {iteration} complete in {total_time:.1f}s")
            print(f"   Manager: {agent_times.get('manager', 0):.1f}s | Coder: {agent_times.get('coder', 0):.1f}s | Tester: {agent_times.get('tester', 0):.1f}s | Reviewer: {agent_times.get('reviewer', 0):.1f}s")
            
            return {
                "review_feedback": feedback,
                "approved": approved
            }
        except json.JSONDecodeError as e:
            self.logger.error(f"Reviewer parse error: {e}")
            return {"review_feedback": "parse error"}