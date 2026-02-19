"""
Conversation logger for agent interactions.
Logs all agent communications to a GitHub-friendly markdown file.
Designed to be shareable and fun to read.
"""
import os
import re
import shutil
from datetime import datetime
from typing import Optional


# CUDA banner pattern that Docker containers print on startup
_CUDA_BANNER_RE = re.compile(
    r'(?:={10}\n== CUDA ==\n={10}\n'
    r'.*?(?:for your convenience\.)\n*)',
    re.DOTALL
)

# SB3 verbose training progress tables (rollout/, train/, time/ blocks)
_SB3_PROGRESS_RE = re.compile(
    r'-{20,}\n'
    r'(?:\| [\w/\s]+\|[^\n]*\n)+'
    r'-{20,}\n?',
    re.MULTILINE
)

# SB3 wrapper messages
_SB3_WRAPPER_MSGS = [
    "Using cpu device",
    "Using cuda device",
    "Wrapping the env with a",
    "Wrapping the env in a",
]

# Common harmless Python warnings to filter from stderr
_NOISE_WARNINGS = [
    "Evaluation environment is not wrapped with a ``Monitor`` wrapper",
    "UserWarning:",
    "warnings.warn(",
]


class ConversationLogger:
    """Logs agent conversations to a GitHub-friendly markdown file."""

    def __init__(self, run_id: str, output_dir: Optional[str] = None):
        self.run_id = run_id
        if output_dir is None:
            output_dir = f"output/{run_id}"
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, "conversation.md")
        self._last_code = ""  # Store latest code for env-solved display
        self._ensure_directory()
        self._write_header()

    def _ensure_directory(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def _write_header(self):
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("# 🤖 Multi-Agent RL Training Log\n\n")
            f.write("> **Run ID:** `{}`  \n".format(self.run_id))
            f.write("> **Started:** {}  \n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            f.write("> **Pipeline:** Manager → Coder → Tester → SHODAN (Reviewer)  \n")
            f.write("> **Architecture:** LangGraph multi-agent loop with persistent memory  \n\n")
            f.write("> [!NOTE]\n")
            f.write("> This log captures the full conversation between AI agents solving\n")
            f.write("> Gymnasium RL environments. SHODAN is a frontier API model reviewing\n")
            f.write("> work done by local Ollama models. The team chatter is emergent.\n\n")
            f.write("---\n\n")

    def log_iteration_start(self, iteration: int, environment: str):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("\n## 🔄 Iteration {} — {}\n\n".format(iteration, environment))

    def log_phase_transition(self, from_phase: str, to_phase: str, environment: str):
        """Log phase transitions (validation -> optimization -> demo)."""
        phase_emoji = {"validation": "🔬", "optimization": "🚀", "demo": "🎬"}
        emoji = phase_emoji.get(to_phase, "➡️")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("> [!IMPORTANT]\n")
            f.write("> {} **Phase transition:** {} → {} ({})\n\n".format(
                emoji, from_phase.upper(), to_phase.upper(), environment))

    def log_manager(self, iteration: int, task: str, reasoning: str = "",
                    environment: str = "", success_threshold: float = 0):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("### 📊 Manager → Coder\n\n")
            if environment:
                f.write("**Environment:** `{}` | **Threshold:** {}\n\n".format(
                    environment, success_threshold))
            f.write("> **Task:** {}\n".format(task))
            if reasoning:
                f.write(">\n> **Reasoning:** {}\n".format(reasoning))
            f.write("\n")

    def log_coder(self, iteration: int, code: str, task: str = "",
                  lines: int = 0, algo: str = "", timesteps: str = "",
                  duration: float = 0, tokens_out: int = 0):
        """Store code and write a compact stats line to the log."""
        self._last_code = code
        with open(self.log_file, "a", encoding="utf-8") as f:
            parts = []
            if lines > 0:
                parts.append("{} lines".format(lines))
            if algo and algo != "Unknown":
                parts.append(algo)
            if timesteps and timesteps != "?":
                try:
                    parts.append("{:,} steps".format(int(timesteps)))
                except (ValueError, TypeError):
                    pass
            if duration > 0:
                parts.append("{:.1f}s".format(duration))
            if tokens_out > 0 and duration > 0:
                parts.append("{:.0f} tok/s".format(tokens_out / duration))
            stats_str = " | ".join(parts) if parts else "code generated"
            f.write("> **Coder** wrote: {}\n\n".format(stats_str))

    @staticmethod
    def _clean_stdout(text: str) -> str:
        """Remove Docker CUDA banner, SB3 verbose progress, and other noise from stdout."""
        if not text:
            return text
        # Remove CUDA banner block
        text = _CUDA_BANNER_RE.sub("", text)
        # Remove SB3 verbose training progress tables
        text = _SB3_PROGRESS_RE.sub("", text)
        # Line-by-line filter for remaining noise
        lines = text.splitlines()
        cleaned = []
        skip = False
        for line in lines:
            # CUDA banner fallback
            if line.strip() == "== CUDA ==":
                skip = True
                continue
            if skip:
                if "for your convenience." in line:
                    skip = False
                continue
            # SB3 wrapper messages
            if any(msg in line for msg in _SB3_WRAPPER_MSGS):
                continue
            cleaned.append(line)
        return "\n".join(cleaned).strip()

    @staticmethod
    def _clean_stderr(text: str) -> str:
        """Remove common harmless Python warnings from stderr."""
        if not text:
            return ""
        lines = text.splitlines()
        cleaned = []
        skip_next = False
        for line in lines:
            if skip_next:
                skip_next = False
                continue
            if any(noise in line for noise in _NOISE_WARNINGS):
                skip_next = True  # Also skip the continuation line (e.g. "  warnings.warn(")
                continue
            if line.strip():
                cleaned.append(line)
        return "\n".join(cleaned).strip()

    @staticmethod
    def _extract_result_line(stdout: str) -> str:
        """Extract the RESULT: line from stdout for prominent display."""
        if not stdout:
            return ""
        for line in stdout.splitlines():
            if line.strip().startswith("RESULT:"):
                return line.strip()
        return ""

    def log_tester(self, iteration: int, test_results: str,
                   execution_stdout: str = "", execution_stderr: str = "",
                   execution_time: float = 0):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("### 🧪 Tester Results\n\n")
            if execution_time > 0:
                minutes = int(execution_time // 60)
                seconds = int(execution_time % 60)
                time_str = "{}m {}s".format(minutes, seconds) if minutes > 0 else "{}s".format(seconds)
                f.write("**Execution time:** {}\n\n".format(time_str))
            # Extract and highlight the RESULT line
            result_line = self._extract_result_line(execution_stdout)
            if result_line:
                f.write("```\n{}\n```\n\n".format(result_line))
            # Summary always visible
            f.write("> {}\n\n".format(test_results.replace("\n", "\n> ")))
            # Clean stdout/stderr before logging
            clean_out = self._clean_stdout(execution_stdout)
            clean_err = self._clean_stderr(execution_stderr)
            # Raw output collapsed (only if there's meaningful content)
            if clean_out or clean_err:
                f.write("<details>\n")
                f.write("<summary>📋 Raw Output</summary>\n\n")
                if clean_out:
                    stdout_display = clean_out[:3000]
                    if len(clean_out) > 3000:
                        stdout_display += "\n... ({} chars truncated)".format(
                            len(clean_out) - 3000)
                    f.write("**stdout:**\n```\n{}\n```\n\n".format(stdout_display))
                if clean_err:
                    stderr_display = clean_err[:2000]
                    if len(clean_err) > 2000:
                        stderr_display += "\n... ({} chars truncated)".format(
                            len(clean_err) - 2000)
                    f.write("**stderr:**\n```\n{}\n```\n\n".format(stderr_display))
                f.write("</details>\n\n")

    def log_reviewer(self, iteration: int, approved: bool, feedback: str,
                     suggestions: str = ""):
        verdict = "✅ APPROVED" if approved else "❌ NEEDS IMPROVEMENT"
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("### 💀 SHODAN's Verdict: {}\n\n".format(verdict))
            f.write("> {}\n".format(feedback.replace("\n", "\n> ")))
            if suggestions and suggestions != "No specific suggestions":
                f.write(">\n> **Directives:** {}\n".format(
                    suggestions.replace("\n", "\n> ")))
            f.write("\n")

    def log_context_usage(self, agent_name: str, components: list, total_tokens: int, context_size: int):
        """Log agent's context window usage as a compact table.
        Helps monitor that conversations don't fill the context."""
        if not components or context_size <= 0:
            return
        total_pct = total_tokens / context_size * 100
        remaining = context_size - total_tokens

        # Warning indicator
        if total_pct >= 60:
            indicator = "⚠️"
        elif total_pct >= 40:
            indicator = "📦"
        else:
            indicator = "💾"

        emoji_map = {"manager": "📊", "coder": "💻", "tester": "🧪", "reviewer": "💀"}
        emoji = emoji_map.get(agent_name, "🤖")
        name_map = {"manager": "Manager", "coder": "Coder", "tester": "Tester", "reviewer": "SHODAN"}
        name = name_map.get(agent_name, agent_name.capitalize())

        with open(self.log_file, "a", encoding="utf-8") as f:
            # Compact one-line summary with component breakdown
            parts = []
            for comp in components:
                if comp["tokens"] > 0:
                    pct = comp["tokens"] / context_size * 100
                    parts.append("{} {:.0f}%".format(comp["name"].strip(), pct))

            breakdown = " | ".join(parts) if parts else "empty"
            f.write("> {} **{} context:** {:.1f}% of {:,} tokens ({}) — {} remaining\n\n".format(
                indicator, name, total_pct, context_size, breakdown, "{:,}".format(remaining)))

    def log_agent_chat(self, agent_name: str, iteration: int, opinion: str):
        """Log agent's chat opinion — the fun personality stuff."""
        if not opinion or not opinion.strip():
            return
        text = opinion.strip()
        with open(self.log_file, "a", encoding="utf-8") as f:
            if agent_name == "reviewer":
                # SHODAN gets a dramatic blockquote
                f.write("> **SHODAN speaks:**\n>\n")
                for line in text.split("\n"):
                    f.write("> *{}*\n".format(line))
                f.write("\n")
            elif agent_name == "tester":
                # Tester gets multiline support for long analyses
                f.write("> **🧪 Tester:**\n>\n")
                for line in text.split("\n"):
                    if line.strip():
                        f.write("> {}\n".format(line))
                    else:
                        f.write(">\n")
                f.write("\n")
            else:
                emoji_map = {"manager": "📊"}
                name_map = {"manager": "Manager"}
                emoji = emoji_map.get(agent_name, "💬")
                name = name_map.get(agent_name, agent_name.capitalize())
                f.write("> **{} {}:** *\"{}\"*\n\n".format(emoji, name, text))

    def log_doc_analysis(self, reviewer_instruction: str, doc_answer: str,
                         correct_params: list = None, common_mistakes: list = None):
        """Log SHODAN-requested documentation analysis to conversation."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("> [!NOTE]\n")
            f.write("> **SHODAN dispatches Tester to the Archives**\n>\n")
            f.write("> *\"{}\"*\n>\n".format(reviewer_instruction.strip()))
            if doc_answer:
                answer = doc_answer[:500] + ("..." if len(doc_answer) > 500 else "")
                f.write("> **Retrieved:** {}\n".format(answer))
            if correct_params:
                f.write("> **API:** `{}`\n".format("`, `".join(correct_params)))
            if common_mistakes:
                f.write("> **Traps:** {}\n".format(", ".join(common_mistakes)))
            f.write("\n")

    def log_codex_change(self, action: str, rule_text: str, rule_index: int = None):
        """Log SHODAN's Divine Codex changes."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            if action == "add":
                f.write("> 📜 **Codex Inscribed:** *{}*\n\n".format(rule_text))
            elif action == "remove":
                f.write("> 📜 ~~Codex Erased [{}]: {}~~\n\n".format(
                    rule_index, rule_text))

    def log_environment_switch(self, current_env: str, next_env: str,
                               manager_report: str = "", reviewer_report: str = ""):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("\n---\n\n")
            f.write("## 🎯 Environment Solved: {} ✅ → {}\n\n".format(
                current_env, next_env))
            f.write("> [!TIP]\n")
            f.write("> **Completed:** {} | **Next challenge:** {}\n\n".format(
                current_env, next_env))
            # Show the winning code that solved this environment
            if self._last_code:
                lines = len(self._last_code.splitlines())
                f.write("<details>\n")
                f.write("<summary>🏆 <b>Winning Code</b> ({} lines)</summary>\n\n".format(lines))
                f.write("```python\n{}\n```\n\n".format(self._last_code))
                f.write("</details>\n\n")
            if manager_report:
                f.write("<details>\n")
                f.write("<summary>📝 <b>Manager's LinkedIn Post</b></summary>\n\n")
                for line in manager_report.split("\n"):
                    f.write("> {}\n".format(line))
                f.write("\n</details>\n\n")
            if reviewer_report:
                f.write("### 💀 SHODAN's Assessment\n\n")
                for line in reviewer_report.split("\n"):
                    f.write("> {}\n".format(line))
                f.write("\n")
            f.write("---\n\n")

    def log_final_summary(self, total_iterations: int, success: bool,
                          total_time: float, solved_environments: list = None):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("\n---\n\n")
            status = "✅ SUCCESS" if success else "⏸️ INCOMPLETE"
            f.write("## 🏁 Final Summary: {}\n\n".format(status))
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            time_str = "{}h {}m".format(hours, minutes) if hours > 0 else "{}m".format(minutes)
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write("| Total Iterations | {} |\n".format(total_iterations))
            f.write("| Total Time | {} ({:.0f}s) |\n".format(time_str, total_time))
            if solved_environments:
                f.write("| Environments Solved | {} |\n".format(
                    len(solved_environments)))
                f.write("| Solved | {} |\n".format(
                    ', '.join(solved_environments)))
            f.write("| Status | {} |\n".format(status))
            f.write("| Ended | {} |\n\n".format(
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            f.write("---\n\n")
            f.write("*Generated by learn_multiagent — LangGraph multi-agent RL system with SHODAN oversight*\n")

    def save_environment_snapshot(self, env_name: str, run_id: str):
        env_dir = os.path.join("output", run_id, env_name)
        os.makedirs(env_dir, exist_ok=True)
        snapshot_path = os.path.join(env_dir, "conversation.md")
        try:
            shutil.copy2(self.log_file, snapshot_path)
        except Exception as e:
            print("[dim red]Warning: Could not save conversation snapshot: {}[/dim red]".format(e))

    def get_log_path(self) -> str:
        return self.log_file
