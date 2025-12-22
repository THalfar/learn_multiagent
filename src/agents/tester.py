import json
import os
import re
import subprocess
from .base import BaseAgent
from rich import print

ALLOWED_DIR = os.path.abspath("output/")

def is_safe_code(code: str) -> bool:
    dangerous = ["rm ", "rmdir", "shutil.rmtree", "os.remove", 
                 "subprocess", "import os\\nos.",
                 "../", "C:\\\\Windows", "C:\\\\Users"]
    return not any(d in code for d in dangerous)

class Tester(BaseAgent):
    def __init__(self, config):
        super().__init__(config, "tester")

    def __call__(self, state: dict) -> dict:
        from src.utils.banners import print_agent_transition
        print_agent_transition("coder", "tester")
        code = state.get("code", "")

        # Remove invalid 'timeout' argument from model.learn() call in generated code
        code = re.sub(
            r'timeout\\s*=\\s*[^,\\)]+(?=\\s*[,\\)]|$)',
            '',
            code,
            flags=re.IGNORECASE
        )
        if not code:
            return {
                "test_results": "No code to test",
                "execution_stdout": "",
                "execution_stderr": ""
            }

        if not is_safe_code(code):
            return {
                "test_results": "ERROR: Dangerous code detected",
                "execution_stdout": "",
                "execution_stderr": "Dangerous code detected, execution blocked"
            }

        print("\n" + "─" * 70)
        print("[bold yellow]TESTER: Executing code...[/bold yellow]")
        print("─" * 70)
        print("[dim]Note: Tester does not see the code, only execution outputs[/dim]")
        print("─" * 70 + "\n")
        
        # Get execution timeout from current environment
        env_progression = self.config.environment_progression
        current_env_index = state.get("current_env_index", 0)
        current_env = env_progression[current_env_index] if env_progression and current_env_index < len(env_progression) else None
        execution_timeout = current_env.execution_timeout if current_env else (env_progression[0].execution_timeout if env_progression else 900)
        
        run_id = state["run_id"]
        code_dir = f"output/{run_id}/code"
        video_dir = state["video_dir"]
        os.makedirs(code_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)
        code_path = f"{code_dir}/agent_code_iter_{state.get('iteration', 0)}.py"

        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            # Run in conda env with environment-specific timeout
            import time
            execution_start = time.time()
            cmd = ["python", code_path]
            result = subprocess.run(
                cmd,
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
                timeout=execution_timeout,  # Environment-specific timeout
            )
            execution_end = time.time()
            execution_duration = execution_end - execution_start

            if result.returncode != 0:
                error = result.stderr[:500] or result.stdout[:500]
                print("\n" + "─" * 70)
                print("[bold yellow]TESTER → REVIEWER[/bold yellow]")
                print("─" * 70)
                print(f"[bold red]Execution failed (return code: {result.returncode})[/bold red]")
                print(f"[red]{error}[/red]")
                print("─" * 70 + "\n")
                
                # Log to conversation file
                logger = state.get("conversation_logger")
                if logger:
                    logger.log_tester(
                        iteration=state.get("iteration", 0),
                        test_results=f"Exec failed (rc {result.returncode}): {error}",
                        execution_stdout=result.stdout,
                        execution_stderr=result.stderr,
                        execution_time=execution_duration
                    )
                
                return {
                    "test_results": f"Exec failed (rc {result.returncode}): {error}",
                    "execution_stdout": result.stdout,
                    "execution_stderr": result.stderr
                }

            stdout = result.stdout
            stderr = result.stderr
            
            # Show execution statistics
            print("\n" + "─" * 70)
            print("[bold yellow]EXECUTION COMPLETE[/bold yellow]")
            print("─" * 70)
            minutes = int(execution_duration // 60)
            seconds = int(execution_duration % 60)
            if minutes > 0:
                print(f"[green]Execution time: {minutes}m {seconds}s ({execution_duration:.1f}s total)[/green]")
            else:
                print(f"[green]Execution time: {seconds}s ({execution_duration:.1f}s total)[/green]")
            
            # Try to extract training timesteps from output if available
            timesteps_match = re.search(r'total_timesteps[:\s]*(\d+)', stdout, re.IGNORECASE)
            if timesteps_match:
                timesteps = int(timesteps_match.group(1))
                print(f"[dim]Training timesteps: {timesteps:,}[/dim]")
            
            # Try to extract algorithm info
            algo_match = re.search(r'(PPO|A2C|DQN|SAC)', stdout, re.IGNORECASE)
            if algo_match:
                print(f"[dim]Algorithm: {algo_match.group(1)}[/dim]")
            
            print("─" * 70 + "\n")

            # LLM tester analysis - let LLM analyze all outputs
            print("[dim]Analyzing execution results...[/dim]\n")
            
            # Get success threshold from current environment (execution_timeout already retrieved earlier)
            env_progression = self.config.environment_progression
            current_env_index = state.get("current_env_index", 0)
            current_env = env_progression[current_env_index] if env_progression and current_env_index < len(env_progression) else None
            success_threshold = current_env.success_threshold if current_env else (env_progression[0].success_threshold if env_progression else 0)
            
            prompt_dict = self.config.get_prompt("tester")
            task_template = prompt_dict["task_template"].format(
                execution_stdout=stdout,
                execution_stderr=stderr,
                success_threshold=success_threshold
            )
            full_prompt = prompt_dict["system"] + "\\n\\n" + task_template

            response = self.call_llm_timed(full_prompt, state["stats"], state.get("iteration", 0))
            
            # Print thinking process if using reasoning model
            self.print_thinking(response.content)
            
            # Print token statistics
            stats_obj = state["stats"]
            iteration = state.get("iteration", 0)
            agent_timings = [t for t in stats_obj.timings if t.agent == self.agent_name and t.iteration == iteration]
            if agent_timings:
                latest_timing = agent_timings[-1]
                self.print_token_stats(latest_timing)

            def extract_json(content):
                """Extract and parse JSON from LLM response, handling various formats."""
                content = content.strip()

                # Try to extract from markdown code blocks first
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
                summary = parsed.get("summary", "LLM analysis completed")
                tester_opinion = parsed.get("tester_opinion", "")
                success = parsed.get("success", None)
                metrics = parsed.get("metrics", {})
                
                # Analysis output - this is the tester's report to reviewer
                print("\n" + "─" * 70)
                print("[bold yellow]TESTER → REVIEWER[/bold yellow]")
                print("─" * 70)
                print(f"[yellow]{summary}[/yellow]")
                if success is not None:
                    if success:
                        print(f"[bold green]Result: SUCCESS[/bold green]")
                    else:
                        print(f"[bold red]Result: FAILED[/bold red]")
                if metrics:
                    mean = metrics.get("mean_reward")
                    threshold = metrics.get("meets_threshold", False)
                    if mean is not None:
                        # Get success threshold from current environment
                        env_progression = self.config.environment_progression
                        current_env_index = state.get("current_env_index", 0)
                        current_env = env_progression[current_env_index] if env_progression and current_env_index < len(env_progression) else None
                        success_threshold = current_env.success_threshold if current_env else (env_progression[0].success_threshold if env_progression else 0)
                        
                        if threshold:
                            print(f"[bold green]Mean Reward: {mean} (threshold: {success_threshold}) ✓[/bold green]")
                        else:
                            print(f"[yellow]Mean Reward: {mean} (threshold: {success_threshold}) ✗[/yellow]")
                    if metrics.get("std_reward") is not None:
                        print(f"[dim]Std Reward: {metrics.get('std_reward')}[/dim]")
                    if metrics.get("n_episodes") is not None:
                        print(f"[dim]Episodes: {metrics.get('n_episodes')}[/dim]")
                if tester_opinion:
                    print(f"\n[yellow]Assessment:[/yellow] {tester_opinion}")
                print("─" * 70 + "\n")
                
                # Combine summary and opinion for reviewer
                if tester_opinion:
                    test_results = f"{summary}\n\nTester's assessment: {tester_opinion}"
                else:
                    test_results = summary
                
                # Remove any thinking tags from test_results (shouldn't be there, but safety check)
                test_results = re.sub(r'<think[^>]*>.*?</think[^>]*>', '', test_results, flags=re.DOTALL | re.IGNORECASE)
                test_results = re.sub(r'<thinking[^>]*>.*?</thinking[^>]*>', '', test_results, flags=re.DOTALL | re.IGNORECASE)
                test_results = test_results.strip()
            except json.JSONDecodeError as e:
                print(f"[bold red]ERROR: Tester LLM JSON parse failed: {e}[/bold red]")
                test_results = f"LLM analysis failed: {str(e)[:200]}"

            # Log to conversation file
            logger = state.get("conversation_logger")
            if logger:
                logger.log_tester(
                    iteration=state.get("iteration", 0),
                    test_results=test_results,
                    execution_stdout=stdout,
                    execution_stderr=result.stderr,
                    execution_time=execution_duration
                )

            return {
                "test_results": test_results,
                "execution_stdout": stdout,
                "execution_stderr": result.stderr
            }

        except subprocess.TimeoutExpired:
            timeout_minutes = execution_timeout // 60
            timeout_seconds = execution_timeout % 60
            timeout_str = f"{timeout_minutes}m {timeout_seconds}s" if timeout_minutes > 0 else f"{timeout_seconds}s"
            
            print("\n" + "─" * 70)
            print(f"[bold red]TIMEOUT: Execution exceeded {timeout_str} ({execution_timeout}s)[/bold red]")
            print("─" * 70 + "\n")
            
            # Log to conversation file
            logger = state.get("conversation_logger")
            if logger:
                logger.log_tester(
                    iteration=state.get("iteration", 0),
                    test_results=f"Timeout: >{timeout_str} training",
                    execution_stdout="",
                    execution_stderr=f"Execution timeout after {execution_timeout} seconds",
                    execution_time=execution_timeout
                )
            
            return {
                "test_results": f"Timeout: >{timeout_str} training",
                "execution_stdout": "",
                "execution_stderr": f"Execution timeout after {execution_timeout} seconds"
            }
        except Exception as e:
            print("\n" + "─" * 70)
            print(f"[bold red]ERROR: {e}[/bold red]")
            print("─" * 70 + "\n")
            
            # Log to conversation file
            logger = state.get("conversation_logger")
            if logger:
                logger.log_tester(
                    iteration=state.get("iteration", 0),
                    test_results=f"Unexpected: {str(e)[:200]}",
                    execution_stdout="",
                    execution_stderr=str(e),
                    execution_time=0
                )
            
            return {
                "test_results": f"Unexpected: {str(e)[:200]}",
                "execution_stdout": "",
                "execution_stderr": str(e)
            }