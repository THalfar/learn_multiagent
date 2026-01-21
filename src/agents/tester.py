import json
import os
import re
import subprocess
import platform
from pathlib import Path
from .base import BaseAgent
from rich import print

ALLOWED_DIR = os.path.abspath("output/")
PROJECT_ROOT = os.path.abspath(".")

# Docker sandbox configuration
DOCKER_SANDBOX_ENABLED = True  # Set to False to run locally (for development)
DOCKER_IMAGE = "rl-sandbox:latest"


def run_in_container(code_path: str, output_dir: str, timeout: int) -> subprocess.CompletedProcess:
    """
    Execute code in Docker container for isolation.

    Args:
        code_path: Absolute path to Python code file on host
        output_dir: Absolute path to output directory on host (for videos, logs)
        timeout: Execution timeout in seconds

    Returns:
        subprocess.CompletedProcess with stdout, stderr, returncode
    """
    # Convert Windows paths to absolute paths
    code_path_abs = os.path.abspath(code_path)
    output_dir_abs = os.path.abspath(output_dir)

    # Ensure output directory exists
    os.makedirs(output_dir_abs, exist_ok=True)

    cmd = [
        "docker", "run",
        "--rm",                              # Remove container after execution
        "--network=none",                    # No network access
        "--memory=8g",                       # Memory limit (enough for parallel envs)
        "--cpus=10",                         # CPU limit: 10 of 16 threads, leaves 6 for Ollama/host
        "--pids-limit=200",                  # Process limit
        "-v", f"{code_path_abs}:/workspace/agent_script.py:ro",  # Code file (avoid 'code.py' - circular import)
        "-v", f"{output_dir_abs}:/workspace/output:rw",          # Output directory (read-write)
        DOCKER_IMAGE,
        "python", "/workspace/agent_script.py"
    ]

    # Fun Docker status messages
    print(f"[dim cyan]🐳 Docker sandbox initializing...[/dim cyan]")
    print(f"[dim]   Image: {DOCKER_IMAGE}[/dim]")
    print(f"[dim]   Memory limit: 8GB | CPUs: 10 threads[/dim]")
    print(f"[dim]   Network: disabled (sandboxed)[/dim]")
    print(f"[dim]   Timeout: {timeout}s ({timeout//60}m {timeout%60}s)[/dim]")
    print(f"[dim cyan]🚀 Launching container...[/dim cyan]")

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout
    )

def is_safe_code(code: str) -> bool:
    """
    Check if code is safe to execute.
    Blocks dangerous operations while allowing safe video directory operations.
    """
    # Critical system paths to block
    dangerous_patterns = [
        "rm -rf",           # Shell command for recursive delete
        "rmdir /s",         # Windows recursive delete
        "C:\\\\Windows\\\\",  # Windows system directory
        "C:\\\\Program Files",  # Program files directory
        "/etc/",            # Linux system config
        "/sys/",            # Linux system files
        "/proc/",           # Linux process info
        "subprocess",       # Subprocess execution
        "import os\\nos.system",  # Direct os.system calls
        "exec(",            # Dynamic code execution
        "eval(",            # Dynamic code evaluation
        "__import__",       # Dynamic imports
    ]
    
    # Block dangerous patterns
    for pattern in dangerous_patterns:
        if pattern in code:
            return False
    
    # Allow shutil.rmtree ONLY with ignore_errors=True or within try-except
    if "shutil.rmtree" in code:
        # Check if it has ignore_errors=True
        if "ignore_errors=True" not in code:
            # Check if it's in a try-except block (simple heuristic)
            if "try:" not in code or "except" not in code:
                return False
    
    # Allow os.remove only for specific file patterns (like .zip, .pkl, .csv, etc.)
    if "os.remove" in code:
        # Check if it's removing a specific file (not a directory)
        # This is a simple check - you might want to make it more sophisticated
        safe_file_patterns = [".zip", ".pkl", ".csv", ".log", ".txt", ".json"]
        has_safe_pattern = any(pattern in code for pattern in safe_file_patterns)
        if not has_safe_pattern:
            return False
    
    # Block attempts to navigate outside project with relative paths
    if "../../../" in code or "..\\..\\.\\" in code:  # More than 2 levels up
        return False
    
    return True

def check_video_files(video_dir: str) -> dict:
    """
    Check if video files exist in the video directory and validate them.
    Returns a dict with video file information.
    """
    video_info = {
        "video_dir": video_dir,
        "exists": False,
        "video_files": [],
        "total_size": 0,
        "valid_videos": 0,
        "empty_videos": 0,
        "error": None
    }
    
    try:
        video_path = Path(video_dir)
        if not video_path.exists():
            video_info["error"] = f"Video directory does not exist: {video_dir}"
            return video_info
        
        video_info["exists"] = True
        
        # Find all video files (common formats)
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        video_files = []
        
        # Check main directory
        for ext in video_extensions:
            video_files.extend(list(video_path.glob(f'*{ext}')))
        
        # Also check for rl-video files (common naming pattern, but exclude .json and .meta files)
        rl_video_files = list(video_path.glob('rl-video-*'))
        # Filter out .json, .meta.json and other non-video files
        rl_video_files = [f for f in rl_video_files if f.suffix in video_extensions]
        video_files.extend(rl_video_files)
        
        # Check iter_* subdirectories (created by coder for unique video storage)
        for iter_dir in video_path.glob('iter_*'):
            if iter_dir.is_dir():
                for ext in video_extensions:
                    video_files.extend(list(iter_dir.glob(f'*{ext}')))
                # Check for rl-video files in subdirectories too
                rl_video_files = list(iter_dir.glob('rl-video-*'))
                rl_video_files = [f for f in rl_video_files if f.suffix in video_extensions]
                video_files.extend(rl_video_files)
        
        # Remove duplicates and filter to actual video files only
        video_files = list(set([f for f in video_files if f.suffix in video_extensions]))
        
        if not video_files:
            video_info["error"] = f"No video files found in {video_dir}"
            return video_info
        
        # Check each video file
        for video_file in sorted(video_files):
            file_info = {
                "path": str(video_file),
                "name": video_file.name,
                "size": 0,
                "size_mb": 0.0,
                "is_valid": False,
                "is_empty": False
            }
            
            try:
                if video_file.exists():
                    file_size = video_file.stat().st_size
                    file_info["size"] = file_size
                    file_info["size_mb"] = file_size / (1024 * 1024)
                    
                    # Check if file is empty (likely not a real video)
                    if file_size == 0:
                        file_info["is_empty"] = True
                        video_info["empty_videos"] += 1
                    elif file_size < 1024:  # Less than 1KB is suspicious
                        file_info["is_empty"] = True
                        video_info["empty_videos"] += 1
                    else:
                        # Check if it's a valid MP4 by reading file header
                        # MP4 files start with specific bytes (ftyp box)
                        try:
                            with open(video_file, 'rb') as f:
                                header = f.read(12)
                                # MP4 files have 'ftyp' at offset 4
                                if len(header) >= 8 and header[4:8] == b'ftyp':
                                    file_info["is_valid"] = True
                                    video_info["valid_videos"] += 1
                                elif video_file.suffix == '.mp4':
                                    # If it's .mp4 but doesn't have proper header, might be corrupted
                                    file_info["is_valid"] = False
                                else:
                                    # For other formats, assume valid if size is reasonable
                                    file_info["is_valid"] = True
                                    video_info["valid_videos"] += 1
                        except Exception as e:
                            # Can't read file, mark as invalid
                            file_info["error"] = str(e)
                    video_info["total_size"] += file_size
            except Exception as e:
                file_info["error"] = str(e)
            
            video_info["video_files"].append(file_info)
        
    except Exception as e:
        video_info["error"] = f"Error checking video directory: {str(e)}"
    
    return video_info

class Tester(BaseAgent):
    def __init__(self, config):
        super().__init__(config, "tester")

    def __call__(self, state: dict) -> dict:
        # NOTE: No transition banner here - coder doesn't have opinions so nothing to show
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

        # Get execution timeout from current environment
        env_progression = self.config.environment_progression
        current_env_index = state.get("current_env_index", 0)
        current_env = env_progression[current_env_index] if env_progression and current_env_index < len(env_progression) else None
        execution_timeout = current_env.execution_timeout if current_env else (env_progression[0].execution_timeout if env_progression else 900)
        
        run_id = state["run_id"]
        code_dir = f"output/{run_id}/code"
        video_dir = state["video_dir"]
        
        # Normalize video_dir to absolute path (fixes Windows path issues)
        import os
        video_dir = os.path.abspath(os.path.normpath(video_dir))
        
        os.makedirs(code_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)
        code_path = f"{code_dir}/agent_code_iter_{state.get('iteration', 0)}.py"

        print("\n" + "─" * 70)
        print("[bold yellow]🧪 TESTER: Preparing Execution...[/bold yellow]")
        print("─" * 70)
        print("[dim]📝 Writing code to sandbox...[/dim]")
        print(f"[dim]   Code file: {code_path}[/dim]")
        print(f"[dim]   Video output: {video_dir}[/dim]")
        print(f"[dim]   Timeout: {execution_timeout}s[/dim]")
        print("[dim]🔒 Tester cannot see code - only execution results[/dim]")
        print("─" * 70 + "\n")

        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            # Run with environment-specific timeout
            import time
            execution_start = time.time()

            if DOCKER_SANDBOX_ENABLED:
                # Run in Docker container for isolation
                result = run_in_container(code_path, video_dir, execution_timeout)
            else:
                # Run locally (for development/debugging)
                cmd = ["python", code_path]
                result = subprocess.run(
                    cmd,
                    cwd=os.getcwd(),
                    capture_output=True,
                    text=True,
                    timeout=execution_timeout,
                )

            execution_end = time.time()
            execution_duration = execution_end - execution_start

            stdout = result.stdout
            stderr = result.stderr
            
            # Check for video files after execution
            video_check = check_video_files(video_dir)
            
            # Add video check information to output
            video_check_output = []
            video_check_output.append("\n" + "─" * 70)
            video_check_output.append("[bold cyan]🎬 VIDEO FILE CHECK[/bold cyan]")
            video_check_output.append("─" * 70)
            video_check_output.append(f"[dim]Scanning: {video_dir}[/dim]")
            
            if video_check["error"]:
                video_check_output.append(f"[red]❌ Error: {video_check['error']}[/red]")
            elif not video_check["exists"]:
                video_check_output.append(f"[red]📁 Video directory does not exist: {video_dir}[/red]")
                video_check_output.append(f"[dim]   (Coder needs to add RecordVideo wrapper)[/dim]")
            elif not video_check["video_files"]:
                video_check_output.append(f"[yellow]🔍 No video files found in {video_dir}[/yellow]")
                video_check_output.append(f"[dim]   (Training ran but no video recorded)[/dim]")
            else:
                video_check_output.append(f"[green]🎥 Found {len(video_check['video_files'])} video file(s)![/green]")
                video_check_output.append(f"[dim]   Total size: {video_check['total_size'] / (1024*1024):.2f} MB[/dim]")
                video_check_output.append(f"[green]   ✓ Valid videos: {video_check['valid_videos']}[/green]")
                if video_check["empty_videos"] > 0:
                    video_check_output.append(f"[red]   ✗ Empty/corrupted: {video_check['empty_videos']}[/red]")
                if video_check['valid_videos'] > 0:
                    video_check_output.append(f"[dim cyan]   🎬 Agent behavior captured on film![/dim cyan]")
                
                # Show details of first few videos
                for i, vf in enumerate(video_check["video_files"][:5]):
                    status = "✓" if vf["is_valid"] and not vf["is_empty"] else "✗"
                    color = "green" if vf["is_valid"] and not vf["is_empty"] else "red"
                    video_check_output.append(f"[{color}]{status} {vf['name']}: {vf['size_mb']:.2f} MB[/{color}]")
                
                if len(video_check["video_files"]) > 5:
                    video_check_output.append(f"[dim]... and {len(video_check['video_files']) - 5} more[/dim]")
            
            video_check_output.append("─" * 70 + "\n")
            
            # Print video check results
            for line in video_check_output:
                print(line)
            
            # Add video check info to stderr for LLM analysis
            video_info_text = "\n".join([
                "=== VIDEO FILE CHECK ===",
                f"Video directory: {video_dir}",
                f"Directory exists: {video_check['exists']}",
                f"Video files found: {len(video_check['video_files'])}",
                f"Valid videos: {video_check['valid_videos']}",
                f"Empty/corrupted: {video_check['empty_videos']}",
                f"Total size: {video_check['total_size'] / (1024*1024):.2f} MB" if video_check['total_size'] > 0 else "Total size: 0 MB",
            ])
            if video_check["error"]:
                video_info_text += f"\nError: {video_check['error']}"
            if video_check["video_files"]:
                video_info_text += "\nVideo files:"
                for vf in video_check["video_files"][:10]:  # Limit to first 10
                    status = "VALID" if vf["is_valid"] and not vf["is_empty"] else "INVALID/EMPTY"
                    video_info_text += f"\n  - {vf['name']}: {vf['size_mb']:.2f} MB ({status})"
            video_info_text += "\n"
            
            # Append video check info to stderr so LLM can see it
            stderr = stderr + "\n" + video_info_text if stderr else video_info_text
            
            # Continue with analysis even if execution failed
            execution_failed = result.returncode != 0
            
            # Show execution statistics
            if execution_failed:
                print("\n" + "─" * 70)
                print("[bold red]❌ EXECUTION FAILED[/bold red]")
                print("─" * 70)
                print(f"[red]Return code: {result.returncode}[/red]")
                print(f"[dim]Container terminated with error[/dim]")
                # Show brief error preview
                error_preview = (result.stderr[:300] or result.stdout[:300]).strip()
                if error_preview:
                    print(f"[dim]Error preview: {error_preview}...[/dim]")
            else:
                print("\n" + "─" * 70)
                print("[bold green]✅ EXECUTION COMPLETE[/bold green]")
                print("─" * 70)
                print(f"[dim]Container exited successfully (code 0)[/dim]")
            
            minutes = int(execution_duration // 60)
            seconds = int(execution_duration % 60)
            if minutes > 0:
                print(f"[green]⏱️  Execution time: {minutes}m {seconds}s ({execution_duration:.1f}s total)[/green]")
            else:
                print(f"[green]⏱️  Execution time: {seconds}s ({execution_duration:.1f}s total)[/green]")

            # Fun performance commentary
            if execution_duration < 30:
                print(f"[dim cyan]   ⚡ Lightning fast! The coder's algorithm is efficient.[/dim cyan]")
            elif execution_duration < 120:
                print(f"[dim cyan]   🏃 Good pace! Training is progressing well.[/dim cyan]")
            elif execution_duration < 300:
                print(f"[dim cyan]   🐢 Taking its time... complex environment perhaps?[/dim cyan]")
            else:
                print(f"[dim cyan]   🦥 Long training session! Hope the rewards are worth it.[/dim cyan]")
            
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

            # LLM tester analysis - let LLM analyze all outputs (including errors)
            print("[dim cyan]🤖 Tester LLM analyzing execution results...[/dim cyan]")
            print(f"[dim]   Model: {self.model_name}[/dim]")
            print(f"[dim]   Processing stdout ({len(stdout):,} chars) + stderr ({len(result.stderr):,} chars)[/dim]\n")
            
            # Get success threshold from current environment (execution_timeout already retrieved earlier)
            env_progression = self.config.environment_progression
            current_env_index = state.get("current_env_index", 0)
            current_env = env_progression[current_env_index] if env_progression and current_env_index < len(env_progression) else None
            success_threshold = current_env.success_threshold if current_env else (env_progression[0].success_threshold if env_progression else 0)
            
            prompt_dict = self.config.get_prompt("tester")

            # Get reviewer's special instruction from previous iteration (if any)
            reviewer_instruction = state.get("reviewer_tester_instruction", None)
            if reviewer_instruction:
                print(f"\n[cyan]📋 Reviewer's special request: {reviewer_instruction}[/cyan]\n")

            # Get agent opinions context (team chatter)
            agent_opinions_context = self.format_agent_opinions_context(state)

            # Format template with all available context
            template_vars = {
                "execution_stdout": stdout,
                "execution_stderr": stderr,
                "success_threshold": success_threshold,
                "manager_task": state.get("current_task", "No task description available"),
                "reviewer_instruction": reviewer_instruction or "None - no special request from reviewer",
                "agent_opinions_context": agent_opinions_context,
            }
            task_template = prompt_dict["task_template"].format(**template_vars)

            # Add conversation history (siloed - only this agent's previous messages)
            history_text = self.format_conversation_history(state)

            full_prompt = prompt_dict["system"] + "\n\n" + history_text + task_template

            # Print context breakdown before LLM call
            prompt_tokens = self.estimate_tokens(full_prompt)
            self.print_context_breakdown(state, prompt_tokens)

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
                my_opinion = parsed.get("my_opinion", "")  # Tester's personal take

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

                # Print tester's opinion if provided (team chatter)
                if my_opinion:
                    print(f"\n[yellow]💬 Tester's take:[/yellow] {my_opinion}")

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

            # Save tester's response to conversation history
            history_update = self.save_message_to_history(state, response.content)

            # Save tester's opinion to state for team chatter
            opinion_update = self.save_opinion_to_state(state, my_opinion) if my_opinion else {}

            result_dict = {
                "test_results": test_results,
                "execution_stdout": stdout,
                "execution_stderr": result.stderr
            }
            result_dict.update(history_update)
            result_dict.update(opinion_update)

            return result_dict

        except subprocess.TimeoutExpired:
            timeout_minutes = execution_timeout // 60
            timeout_seconds = execution_timeout % 60
            timeout_str = f"{timeout_minutes}m {timeout_seconds}s" if timeout_minutes > 0 else f"{timeout_seconds}s"

            print("\n" + "─" * 70)
            print(f"[bold red]⏰ TIMEOUT: Execution exceeded {timeout_str} ({execution_timeout}s)[/bold red]")
            print("─" * 70)
            print(f"[dim red]🐳 Docker container forcibly terminated[/dim red]")
            print(f"[dim]   The training took too long - try fewer timesteps or simpler approach[/dim]")
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
            print(f"[bold red]💥 UNEXPECTED ERROR[/bold red]")
            print("─" * 70)
            print(f"[red]{type(e).__name__}: {e}[/red]")
            print(f"[dim]   Something went wrong outside the sandbox[/dim]")
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