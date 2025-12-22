import json
from .base import BaseAgent
from rich import print

class Manager(BaseAgent):
    def __init__(self, config):
        super().__init__(config, "manager")

    def __call__(self, state: dict) -> dict:
        expected_iteration = state.get("iteration", 0) + 1
        
        # Show clear communication from reviewer
        review_feedback = state.get("review_feedback", "")
        if review_feedback:
            print("\n" + "═" * 70)
            print("[bold cyan]📨 MANAGER RECEIVING COMMUNICATION FROM REVIEWER[/bold cyan]")
            print("═" * 70)
            feedback_display = review_feedback[:500] + "..." if len(review_feedback) > 500 else review_feedback
            print(f"[cyan]Review Feedback:[/cyan] {feedback_display}")
            if len(review_feedback) > 500:
                print(f"[dim]... (full feedback: {len(review_feedback)} chars)[/dim]")
            print("═" * 70 + "\n")
        else:
            print("\n" + "─" * 70)
            print("[bold cyan]📨 MANAGER: No previous review feedback (first iteration)[/bold cyan]")
            print("─" * 70 + "\n")
        
        print("[bold blue]🤔 Manager analyzing situation and deciding next task...[/bold blue]")
        prompt_dict = self.config.get_prompt("manager")
        code_summary = (state.get("code", "")[:200] or "") + "..." if len(state.get("code", "")) > 200 else state.get("code", "")
        
        # Extract suggestions separately
        review_suggestions = state.get("review_suggestions", "No specific suggestions")
        
        # Show what manager is considering
        print(f"[dim]   📊 Considering: {len(state.get('tasks', []))} previous tasks, code ({len(state.get('code', ''))} chars), test results[/dim]")
        if review_feedback:
            print(f"[dim]   📝 Review feedback: {len(review_feedback)} chars[/dim]")
        if review_suggestions and review_suggestions != "No specific suggestions":
            print(f"[dim]   💡 Review suggestions: {len(review_suggestions)} chars[/dim]")
        
        task_template = prompt_dict["task_template"].format(
            tasks=state.get("tasks", []),
            code_summary=code_summary,
            test_results=state.get("test_results", ""),
            review_feedback=review_feedback,
            review_suggestions=review_suggestions,
            iteration=expected_iteration,
            max_iterations=self.config.agents.max_iterations,
            environment=self.config.environment.name,
            success_threshold=self.config.agents.success_threshold,
            video_dir=state.get("video_dir", self.config.video.output_dir),
        )
        system_prompt = prompt_dict["system"].format(
            environment=self.config.environment.name,
            success_threshold=self.config.agents.success_threshold
        )
        full_prompt = system_prompt + "\n\n" + task_template
        response = self.call_llm_timed(full_prompt, state["stats"], expected_iteration)

        def extract_json(content):
            """Extract and parse JSON from LLM response, handling various formats."""
            content = content.strip()

            # Try to extract from markdown code blocks first
            import re
            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1).strip()

            # Try to find JSON object in the content
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                content = content[json_start:json_end]

            return content

        try:
            json_content = extract_json(response.content)
            parsed = json.loads(json_content)
            next_task = parsed.get("next_task", "No task decided")
            reasoning = parsed.get("reasoning", "")
            
            # Print iteration banner after task is decided
            from src.utils.banners import print_iteration_banner
            print_iteration_banner(expected_iteration, self.config.agents.max_iterations, next_task)
            
            print("═" * 70)
            print("[bold green]✅ MANAGER DECISION[/bold green]")
            print("═" * 70)
            print(f"[bold green]📋 Next Task:[/bold green] {next_task}")
            if reasoning:
                print(f"[green]💭 Reasoning:[/green] {reasoning}")
            print("═" * 70 + "\n")
            return {
                "tasks": state.get("tasks", []) + [next_task],
                "current_task": next_task,
                "iteration": expected_iteration,
            }
        except json.JSONDecodeError as e:
            self.logger.error(f"Manager JSON parse error: {e}")
            self.logger.error(f"Raw response content: {response.content[:500]}...")
            return {"current_task": "error: invalid JSON from LLM"}