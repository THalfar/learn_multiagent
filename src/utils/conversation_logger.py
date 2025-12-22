"""
Conversation logger for agent interactions.
Logs all agent communications to a file for post-analysis.
"""
import os
from datetime import datetime
from typing import Optional


class ConversationLogger:
    """Logs agent conversations to a file."""
    
    def __init__(self, run_id: str, output_dir: Optional[str] = None):
        """
        Initialize conversation logger.
        
        Args:
            run_id: Unique identifier for this run
            output_dir: Directory to save logs (defaults to output/{run_id}/)
        """
        self.run_id = run_id
        if output_dir is None:
            output_dir = f"output/{run_id}"
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, "conversation.txt")
        self._ensure_directory()
        self._write_header()
    
    def _ensure_directory(self):
        """Ensure output directory exists."""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _write_header(self):
        """Write header to log file."""
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"AGENT CONVERSATION LOG\n")
            f.write(f"Run ID: {self.run_id}\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
    
    def log_iteration_start(self, iteration: int, environment: str):
        """Log start of a new iteration."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"ITERATION {iteration}\n")
            f.write(f"Environment: {environment}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
    
    def log_manager(self, iteration: int, task: str, reasoning: str = "", 
                    environment: str = "", success_threshold: float = 0):
        """Log manager's task assignment."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("─" * 80 + "\n")
            f.write("MANAGER → CODER\n")
            f.write("─" * 80 + "\n")
            if environment:
                f.write(f"Environment: {environment} (threshold: {success_threshold})\n")
            f.write(f"Iteration: {iteration}\n")
            f.write(f"\nTask:\n{task}\n")
            if reasoning:
                f.write(f"\nReasoning:\n{reasoning}\n")
            f.write("\n")
    
    def log_coder(self, iteration: int, code: str, task: str = ""):
        """Log coder's generated code."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("─" * 80 + "\n")
            f.write("CODER → TESTER\n")
            f.write("─" * 80 + "\n")
            if task:
                f.write(f"Task: {task}\n")
            f.write(f"Code ({len(code)} characters, {len(code.splitlines())} lines):\n")
            f.write("-" * 80 + "\n")
            f.write(code)
            f.write("\n" + "-" * 80 + "\n\n")
    
    def log_tester(self, iteration: int, test_results: str, 
                   execution_stdout: str = "", execution_stderr: str = "",
                   execution_time: float = 0):
        """Log tester's execution results."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("─" * 80 + "\n")
            f.write("TESTER → REVIEWER\n")
            f.write("─" * 80 + "\n")
            if execution_time > 0:
                minutes = int(execution_time // 60)
                seconds = int(execution_time % 60)
                if minutes > 0:
                    f.write(f"Execution time: {minutes}m {seconds}s\n")
                else:
                    f.write(f"Execution time: {seconds}s\n")
            f.write(f"\nTest Results:\n{test_results}\n")
            if execution_stdout:
                f.write(f"\nExecution Stdout:\n{execution_stdout}\n")
            if execution_stderr:
                f.write(f"\nExecution Stderr:\n{execution_stderr}\n")
            f.write("\n")
    
    def log_reviewer(self, iteration: int, approved: bool, feedback: str, 
                     suggestions: str = ""):
        """Log reviewer's decision and feedback."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("─" * 80 + "\n")
            f.write("REVIEWER → MANAGER\n")
            f.write("─" * 80 + "\n")
            f.write(f"Verdict: {'APPROVED ✓' if approved else 'NEEDS IMPROVEMENT ✗'}\n")
            f.write(f"\nFeedback:\n{feedback}\n")
            if suggestions and suggestions != "No specific suggestions":
                f.write(f"\nSuggestions:\n{suggestions}\n")
            f.write("\n")
    
    def log_environment_switch(self, current_env: str, next_env: str, 
                               manager_report: str = "", reviewer_report: str = ""):
        """Log environment switch."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("ENVIRONMENT SWITCH\n")
            f.write("=" * 80 + "\n")
            f.write(f"From: {current_env}\n")
            f.write(f"To: {next_env}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if manager_report:
                f.write(f"\nManager Report:\n{manager_report}\n")
            if reviewer_report:
                f.write(f"\nReviewer Report:\n{reviewer_report}\n")
            f.write("=" * 80 + "\n\n")
    
    def log_final_summary(self, total_iterations: int, success: bool, 
                          total_time: float, solved_environments: list = None):
        """Log final summary."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("FINAL SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total Iterations: {total_iterations}\n")
            f.write(f"Final Status: {'SUCCESS' if success else 'INCOMPLETE'}\n")
            f.write(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)\n")
            if solved_environments:
                f.write(f"Solved Environments: {', '.join(solved_environments)}\n")
            f.write(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
    
    def get_log_path(self) -> str:
        """Get the path to the log file."""
        return self.log_file
