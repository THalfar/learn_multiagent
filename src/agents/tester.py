import json
import os
import re
import subprocess
import time
from .base import BaseAgent

class Tester(BaseAgent):
    def __call__(self, state: dict) -> dict:
        code = state.get("code", "")
        if not code:
            return {"test_results": "No code to test"}

        self.logger.info("Tester: saving & executing code")
        os.makedirs("output/code", exist_ok=True)
        timestamp = int(time.time())
        code_path = f"output/code/agent_code_{timestamp}.py"

        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)

        self.logger.info(f"Code saved to {code_path}")

        try:
            # Run in conda env with timeout
            cmd = ["conda", "run", "-n", "langgraph-rl", "python", code_path]
            result = subprocess.run(
                cmd,
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
                timeout=300,  # 5 min
            )

            if result.returncode != 0:
                error = result.stderr[:500] or result.stdout[:500]
                return {"test_results": f"Exec failed (rc {result.returncode}): {error}"}

            stdout = result.stdout
            self.logger.info(f"Exec stdout preview: {stdout[:200]}...")

            # Parse standard metrics (coder should print these)
            mean_match = re.search(r"MEAN_REWARD[:\s]*([\d.-]+)", stdout, re.I)
            std_match = re.search(r"STD_REWARD[:\s]*([\d.-]+)", stdout, re.I)
            episodes_match = re.search(r"N_EPISODES[:\s]*(\d+)", stdout, re.I)
            video_match = re.search(r"VIDEO_SAVED[:\s]*(.+?)(?:\n|$)", stdout, re.I | re.S)

            mean_reward = float(mean_match.group(1)) if mean_match else 0.0
            std_reward = float(std_match.group(1)) if std_match else 0.0
            n_episodes = int(episodes_match.group(1)) if episodes_match else 0
            video_path = video_match.group(1).strip() if video_match else "no video"

            summary = f"Mean reward: {mean_reward:.1f} ± {std_reward:.1f} ({n_episodes} eps). Video: {video_path}"
            success = mean_reward >= self.config.agents.success_threshold

            self.logger.info(summary)
            return {"test_results": summary}

        except subprocess.TimeoutExpired:
            self.logger.error("Tester timeout")
            return {"test_results": "Timeout: >5min training"}
        except Exception as e:
            self.logger.error(f"Tester error: {e}")
            return {"test_results": f"Unexpected: {str(e)[:200]}"}