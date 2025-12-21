import json
import os
import re
import subprocess
import time
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
        code = state.get("code", "")

        # Remove invalid 'timeout' argument from model.learn() call in generated code
        code = re.sub(
            r'timeout\\s*=\\s*[^,\\)]+(?=\\s*[,\\)]|$)',
            '',
            code,
            flags=re.IGNORECASE
        )
        if not code:
            return {"test_results": "No code to test"}

        if not is_safe_code(code):
            return {"test_results": "ERROR: Dangerous code detected"}

        print("[bold yellow]Tester: saving & executing code[/bold yellow]")
        run_id = state["run_id"]
        code_dir = f"output/{run_id}/code"
        video_dir = state["video_dir"]
        os.makedirs(code_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)
        code_path = f"{code_dir}/agent_code_iter_{state.get('iteration', 0)}.py"
        print(f"[bold yellow]Run dirs - Code: {code_dir}, Videos: {video_dir}[/bold yellow]")

        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)

        print(f"[bold yellow]Code saved to {code_path}[/bold yellow]")

        try:
            # Run in conda env with timeout
            cmd = ["python", code_path]
            result = subprocess.run(
                cmd,
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
                timeout=600,  # 10 min
            )

            if result.returncode != 0:
                error = result.stderr[:500] or result.stdout[:500]
                return {"test_results": f"Exec failed (rc {result.returncode}): {error}"}

            stdout = result.stdout
            print("[bold yellow]=== FULL EXECUTION STDOUT ===[/bold yellow]")
            print(f"[yellow]{stdout}[/yellow]")
            print("[bold yellow]=== END EXECUTION STDOUT ===[/bold yellow]")

            # Parse standard metrics (coder should print these)
            mean_match = re.search(r"MEAN_REWARD[:\\s]*([\\d.-]+)", stdout, re.I)
            std_match = re.search(r"STD_REWARD[:\\s]*([\\d.-]+)", stdout, re.I)
            episodes_match = re.search(r"N_EPISODES[:\\s]*(\\d+)", stdout, re.I)
            video_match = re.search(r"VIDEO_SAVED[:\\s]*(.+?)(?:\\n|$)", stdout, re.I | re.S)

            mean_reward = float(mean_match.group(1)) if mean_match else None
            std_reward = float(std_match.group(1)) if std_match else None
            n_episodes = int(episodes_match.group(1)) if episodes_match else None
            video_path = video_match.group(1).strip() if video_match else None

            if mean_match is None:
                return {"test_results": "No MEAN_REWARD printed in stdout. Check coder."}

            # LLM tester analysis
            prompt_dict = self.config.get_prompt("tester")
            mean_str = f"{mean_reward:.2f}"
            std_str = f"{std_reward:.2f}" if std_reward is not None else "N/A"
            eps_str = str(n_episodes) if n_episodes is not None else "0"
            vid_str = video_path or "no video saved"

            task_template = prompt_dict["task_template"].format(
                mean_reward=mean_str,
                std_reward=std_str,
                n_episodes=eps_str,
                video_path=vid_str,
                success_threshold=self.config.agents.success_threshold
            )
            full_prompt = prompt_dict["system"] + "\\n\\n" + task_template

            response = self.call_llm_timed(full_prompt, state["stats"], state.get("iteration", 0))

            try:
                parsed = json.loads(response.content.strip())
                summary = parsed.get("summary", f"Mean {mean_str} ± {std_str} ({eps_str} eps), video: {vid_str}")
                print(f"[bold yellow]Tester LLM: {summary}[/bold yellow]")
            except json.JSONDecodeError as e:
                self.logger.error(f"Tester LLM JSON error: {e}")
                summary = f"Metrics: mean={mean_str} std={std_str} eps={eps_str} video={vid_str} (LLM parse failed)"

            return {"test_results": summary}

        except subprocess.TimeoutExpired:
            print("[bold red]Tester timeout[/bold red]")
            return {"test_results": "Timeout: >10min training"}
        except Exception as e:
            print(f"[bold red]Tester error: {e}[/bold red]")
            return {"test_results": f"Unexpected: {str(e)[:200]}"}