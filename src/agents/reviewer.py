import json
from .base import BaseAgent
from rich import print

class Reviewer(BaseAgent):
    def __init__(self, config):
        super().__init__(config, "reviewer")

    def __call__(self, state: dict) -> dict:
        from src.utils.banners import print_agent_transition
        print_agent_transition("tester", "reviewer")
        
        # Show what reviewer is evaluating
        print("\n" + "═" * 70)
        print("[bold magenta]🔍 REVIEWER EVALUATING[/bold magenta]")
        print("═" * 70)
        code = state.get("code", "")
        test_results = state.get("test_results", "")
        execution_stdout = state.get("execution_stdout", "")
        execution_stderr = state.get("execution_stderr", "")
        
        print(f"[magenta]📄 Code:[/magenta] {len(code)} characters")
        print(f"[magenta]🧪 Test Results:[/magenta] {len(test_results)} characters")
        if execution_stdout:
            stdout_preview = execution_stdout[:100] + "..." if len(execution_stdout) > 100 else execution_stdout
            print(f"[magenta]📤 Execution Output:[/magenta] {stdout_preview}")
        if execution_stderr:
            stderr_preview = execution_stderr[:100] + "..." if len(execution_stderr) > 100 else execution_stderr
            print(f"[magenta]⚠️  Execution Errors:[/magenta] {stderr_preview}")
        print("═" * 70 + "\n")
        
        print("[bold magenta]🤔 Reviewer analyzing code quality, test results, and execution...[/bold magenta]")
        prompt_dict = self.config.get_prompt("reviewer")
        task_template = prompt_dict["task_template"].format(
            code=code,
            test_results=test_results,
            execution_stdout=execution_stdout,
            execution_stderr=execution_stderr,
            success_threshold=self.config.agents.success_threshold,
            video_dir=state.get("video_dir", "output/videos"),
        )
        system_prompt = prompt_dict["system"].format(
            success_threshold=self.config.agents.success_threshold,
            video_dir=state.get("video_dir", "output/videos")
        )
        full_prompt = system_prompt + "\n\n" + task_template
        response = self.call_llm_timed(full_prompt, state["stats"], state.get("iteration", 0))

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
            approved = parsed.get("approved", False)
            feedback = parsed.get("feedback", "No feedback")
            suggestions = parsed.get("suggestions", [])
            
            # Print reviewer verdict and communication to manager
            print("\n" + "═" * 70)
            print("[bold magenta]📤 REVIEWER SENDING COMMUNICATION TO MANAGER[/bold magenta]")
            print("═" * 70)
            
            if approved:
                print("[bold green]✅ VERDICT: APPROVED! 🎉[/bold green]")
            else:
                print("[bold yellow]⚠️  VERDICT: NEEDS IMPROVEMENT[/bold yellow]")
            
            print(f"\n[magenta]📝 Feedback to Manager:[/magenta]")
            print(f"{feedback}")
            
            if suggestions:
                print(f"\n[magenta]💡 Suggestions:[/magenta]")
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"   {i}. {suggestion}")
            
            print("═" * 70)
            print(f"[dim]📊 Summary: Approved={approved}, Feedback length={len(feedback)} chars[/dim]")
            print("═" * 70 + "\n")

            # Live per-iteration timing
            iteration = state.get("iteration", 0)
            stats_obj = state["stats"]
            iter_stats = stats_obj.get_iteration_stats(iteration)
            total_time = iter_stats["total_time"]
            agent_times = iter_stats["agents"]
            print(f"\n⏱️  Iteration {iteration} complete in {total_time:.1f}s")
            print(f"   Manager: {agent_times.get('manager', 0):.1f}s | Coder: {agent_times.get('coder', 0):.1f}s | Tester: {agent_times.get('tester', 0):.1f}s | Reviewer: {agent_times.get('reviewer', 0):.1f}s")

            # Store suggestions separately for Manager to relay to Coder
            suggestions_text = "\n".join(suggestions) if suggestions else "No specific suggestions"
            
            return {
                "review_feedback": feedback,
                "review_suggestions": suggestions_text,
                "approved": approved
            }
        except json.JSONDecodeError as e:
            self.logger.error(f"Reviewer parse error: {e}")
            self.logger.error(f"Raw response content: {response.content[:500]}...")
            return {"review_feedback": "parse error"}