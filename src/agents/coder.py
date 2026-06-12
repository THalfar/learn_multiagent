from .base import BaseAgent
from rich import print
from rich.syntax import Syntax
from rich.panel import Panel
from rich.console import Console
import os

console = Console()

class Coder(BaseAgent):
    def __init__(self, config, model_switcher=None):
        super().__init__(config, "coder", model_switcher=model_switcher)

    def _get_code_context(self, state: dict) -> str:
        """
        Get previous iteration's code for coder to see.
        Simple: if previous code exists, show it. Otherwise empty.
        Coder always writes complete scripts, no diffs needed.
        """
        iteration = state.get("iteration", 0)
        run_id = state.get("run_id", "")

        # First iteration - no previous code
        if iteration == 0 or not run_id:
            return ""

        # Try to load previous iteration's code. The Tester/Executor write to the
        # env-specific subdir output/{run_id}/{env}/code/ - look there (the old bare
        # output/{run_id}/code/ path never matched, so the Coder never saw its prior code).
        env_progression = self.config.environment_progression
        current_env_index = state.get("current_env_index", 0)
        current_env = env_progression[current_env_index] if env_progression and current_env_index < len(env_progression) else None
        env_name = current_env.name if current_env else self.config.environment.name
        prev_code_path = f"output/{run_id}/{env_name}/code/agent_code_iter_{iteration - 1}.py"
        if not os.path.exists(prev_code_path):
            return ""

        try:
            with open(prev_code_path, "r", encoding="utf-8") as f:
                prev_code = f.read()
        except Exception:
            return ""

        if not prev_code.strip():
            return ""

        # Show previous code (coder will write new version based on task)
        lines = prev_code.splitlines()
        line_count = len(lines)

        # For typical RL scripts (< 200 lines), show full code
        if line_count <= 200:
            return f"PREVIOUS CODE (iteration {iteration - 1}, {line_count} lines):\n{prev_code}"

        # For longer code, show first and last parts
        first_part = "\n".join(lines[:100])
        last_part = "\n".join(lines[-100:])
        return f"PREVIOUS CODE (iteration {iteration - 1}, {line_count} lines, showing first 100 + last 100):\n{first_part}\n\n... ({line_count - 200} lines omitted) ...\n\n{last_part}"

    def _format_recent_attempts(self, state: dict) -> str:
        """Show the Coder its last couple of attempts + the Tester's diagnosis and the
        Reviewer's verdict, so it does not repeat a corrected mistake. The Coder is
        otherwise stateless (history_window=0), so this is its only self-memory."""
        attempts = state.get("recent_attempts", []) or []
        if not attempts:
            return ""
        lines = ["", "=== YOUR RECENT ATTEMPTS (learn from them - do NOT repeat a corrected mistake) ==="]
        for att in attempts[-2:]:
            it = att.get("iter", "?")
            verdict = att.get("verdict", "?")
            diag = (att.get("diagnosis") or "").strip()
            reason = (att.get("reason") or "").strip()
            lines.append(f"[iter {it}] verdict: {verdict}")
            if diag:
                lines.append(f"  Tester diagnosis: {diag[:400]}")
            if reason:
                lines.append(f"  Reviewer said: {reason[:300]}")
        lines.append("=== end recent attempts ===")
        return "\n".join(lines)

    def _format_raw_output(self, state: dict) -> str:
        """DUO pipeline only: show the Coder the previous run's RAW stdout/stderr tail
        directly (plus the full AUTOMATED DIAGNOSTICS block), instead of a local model's
        paraphrase. This is the whole point of the duo topology - the Coder fixes what the
        container ACTUALLY printed, not a 'broken telephone' summary. Empty for the quad
        pipeline (the Tester paraphrases there) and on the first iteration (nothing ran yet)."""
        if getattr(self.config, "pipeline", "quad") != "duo":
            return ""
        stdout = state.get("execution_stdout", "") or ""
        stderr = state.get("execution_stderr", "") or ""
        if not stdout and not stderr:
            return ""
        import re
        stderr_tail = stderr[-3000:] if stderr else ""
        parts = ["", "=== PREVIOUS RUN - RAW EXECUTION OUTPUT (ground truth; fix what IT says, not what you assume) ==="]
        if stdout:
            parts.append("--- stdout (tail) ---")
            parts.append(stdout[-2000:])
        if stderr:
            parts.append("--- stderr (tail) ---")
            parts.append(stderr_tail)
        # Re-extract the AUTOMATED DIAGNOSTICS block in full - the tail cut above may have
        # truncated it, and it is the single highest-signal hint for the next revision.
        m = re.search(r"=== AUTOMATED DIAGNOSTICS ===.*?=== END DIAGNOSTICS ===", stderr, re.DOTALL)
        if m and m.group(0) not in stderr_tail:
            parts.append("--- automated diagnostics (full, re-extracted) ---")
            parts.append(m.group(0))
        parts.append("=== end raw execution output ===")
        return "\n".join(parts)

    def _print_code_summary(self, code: str, state: dict):
        """
        Print a quick visual summary of what coder produced.
        Shows stats and a snippet so you can see 'what's happening at the lathe'.
        """
        import re
        lines = code.splitlines()
        line_count = len(lines)
        iteration = state.get("iteration", 0)

        # Count imports
        import_lines = [l.strip() for l in lines if l.strip().startswith(('import ', 'from '))]
        import_count = len(import_lines)

        # Count functions and classes
        func_count = len(re.findall(r'^def \w+', code, re.MULTILINE))
        class_count = len(re.findall(r'^class \w+', code, re.MULTILINE))

        # Detect algorithm
        algo = "Unknown"
        if 'PPO' in code:
            algo = "PPO"
        elif 'DQN' in code:
            algo = "DQN"
        elif 'A2C' in code:
            algo = "A2C"
        elif 'SAC' in code:
            algo = "SAC"

        # Detect training timesteps
        timesteps_match = re.search(r'total_timesteps\s*=\s*(\d+)', code)
        timesteps = timesteps_match.group(1) if timesteps_match else "?"

        # Get first 3 imports for snippet
        import_snippet = import_lines[:3] if import_lines else ["(no imports)"]

        # Get key code snippet (first function or main block)
        snippet_lines = []
        in_main = False
        for i, line in enumerate(lines):
            if 'def ' in line or 'if __name__' in line or ('model' in line.lower() and '=' in line):
                snippet_lines = lines[i:i+3]
                break

        # Build output
        console.print("\n\n" + "-" * 70)
        console.print(f"[bold green]🔧 CODER OUTPUT - Iteration {iteration}[/bold green]")
        console.print("-" * 70)

        # Stats line
        stats_parts = [
            f"[green]{line_count}[/green] lines",
            f"[yellow]{import_count}[/yellow] imports",
            f"[blue]{func_count}[/blue] funcs",
        ]
        if class_count > 0:
            stats_parts.append(f"[magenta]{class_count}[/magenta] classes")
        stats_parts.append(f"[cyan]{algo}[/cyan]")
        if timesteps != "?":
            stats_parts.append(f"[dim]{int(timesteps):,} steps[/dim]")

        console.print("📊 " + " | ".join(stats_parts))

        # Import health indicator
        if import_count <= 15:
            health = "[green]✓ Clean[/green]"
        elif import_count <= 25:
            health = "[yellow]~ OK[/yellow]"
        else:
            health = "[red]⚠ Bloated[/red]"
        console.print(f"📦 Imports: {health} ({', '.join(import_snippet[:2])}{'...' if len(import_lines) > 2 else ''})")

        # Code snippet
        if snippet_lines:
            snippet_preview = snippet_lines[0][:60] + ('...' if len(snippet_lines[0]) > 60 else '')
            console.print(f"[dim]📝 {snippet_preview}[/dim]")

        console.print("-" * 70)

    def _check_code_quality(self, code: str) -> str:
        """
        Check code quality and fix repetition loops.
        Deduplicates imports if model got stuck in a repetition loop.
        Triggers adaptive model switch if repetition loops persist.
        """
        lines = code.splitlines()

        # Count imports
        import_lines = [l.strip() for l in lines if l.strip().startswith(('import ', 'from '))]
        non_import_lines = [l for l in lines if l.strip() and not l.strip().startswith(('import ', 'from '))]

        # Detect repetition loop: many imports but no actual code
        if len(import_lines) > 30 and len(non_import_lines) < 5:
            console.print(f"[red]⚠️  REPETITION LOOP DETECTED: {len(import_lines)} imports, {len(non_import_lines)} code lines[/red]")
            console.print(f"[yellow]   Attempting to salvage by deduplicating imports...[/yellow]")

            # Trigger adaptive model switch if enabled
            if self.model_switcher:
                from src.utils.model_switcher import SwitchTrigger
                new_model = self.model_switcher.check_and_switch(
                    self.agent_name,
                    SwitchTrigger.REPETITION_LOOP,
                    {"import_count": len(import_lines), "code_lines": len(non_import_lines)}
                )
                if new_model:
                    self.switch_model(new_model)

            # Deduplicate imports while preserving order
            seen_imports = set()
            unique_imports = []
            for imp in import_lines:
                if imp not in seen_imports:
                    seen_imports.add(imp)
                    unique_imports.append(imp)

            console.print(f"[green]   Reduced to {len(unique_imports)} unique imports[/green]")

            # If we only have imports and no code, return a minimal error script
            if len(non_import_lines) < 3:
                console.print(f"[red]   ERROR: No actual code found after imports![/red]")
                # Return a minimal script that will fail with a clear error
                return "\n".join(unique_imports) + """

# ERROR: Model repetition loop - no actual training code was generated
# The model got stuck repeating imports and never wrote the training logic
print("ERROR: Code generation failed - model produced only imports, no training code")
raise RuntimeError("Repetition loop detected: model produced only imports")
"""

            # Reconstruct code with unique imports + remaining lines
            return "\n".join(unique_imports) + "\n" + "\n".join(non_import_lines)

        # Normal quality warnings
        if len(import_lines) > 40:
            console.print(f"[yellow]⚠️  Note: {len(import_lines)} import lines (check if intentional)[/yellow]")

        # Check for obvious repetition (same line 5+ times in a row)
        prev_line = None
        repeat_count = 0
        max_repeat = 0
        for line in lines:
            if line == prev_line and line.strip():
                repeat_count += 1
                max_repeat = max(max_repeat, repeat_count)
            else:
                repeat_count = 0
            prev_line = line

        if max_repeat >= 5:
            console.print(f"[yellow]⚠️  Note: Detected {max_repeat}x repeated line (possible loop)[/yellow]")
        else:
            # No repetition loop - clear the counter
            if self.model_switcher:
                self.model_switcher.report_success(self.agent_name)

        # Return code unchanged - let Python/tester catch actual errors
        return code

    def __call__(self, state: dict) -> dict:
        # Show what coder is working on
        iteration = state.get("iteration", 0)
        task = state.get("current_task", "")
        task_preview = task[:80] + "..." if len(task) > 80 else task
        console.print(f"\n\n[green]🔧 Coder working on: {task_preview}[/green]")

        prompt_dict = self.config.get_prompt("coder")
        
        # Normalize video_dir to absolute path (fixes Windows path issues)
        video_dir = state.get("video_dir", self.config.video.output_dir)
        video_dir = os.path.abspath(os.path.normpath(video_dir))
        
        # Get iteration for unique video subdirectory
        iteration = state.get("iteration", 0)
        
        # Get device from current environment in progression
        env_progression = self.config.environment_progression
        current_env_index = state.get("current_env_index", 0)
        current_env = env_progression[current_env_index] if env_progression and current_env_index < len(env_progression) else None

        # Get device (cpu/gpu/auto) from environment config
        device = current_env.device if current_env and hasattr(current_env, 'device') else "cpu"

        # B4: render the procedural SKILL substrate into the Coder's prompt (replaces the
        # flat Codex). Skills are PROCEDURES ("read max_episode_steps from env.spec ...")
        # not values, and the relevant ones are selected by env/tags. Falls back to the
        # legacy flat shodan_rules list if no skill_store is present in state.
        skill_store = state.get("skill_store", None)
        env_name_now = current_env.name if current_env else self.config.environment.name
        # Prefer the env's declared tags (config); fall back to the env-id substring
        # heuristic only when none are declared. The metric tag is always derived.
        _tags = list(getattr(current_env, "tags", []) or []) if current_env else []
        if not _tags:
            _lname = env_name_now.lower()
            if "panda" in _lname or "fetch" in _lname:
                _tags = ["goal", "her", "manipulation", "robotics"]
        if current_env and getattr(current_env, "metric", "reward") == "success_rate":
            _tags = _tags + ["goal", "success_rate"]

        if skill_store is not None:
            shodan_rules_text = skill_store.render_for_coder(env_name=env_name_now, tags=_tags)
            # Also surface any flat Codex notes SHODAN inscribed via the legacy prompt_rules
            # field (tactical, per-iteration), so its learning reaches the Coder regardless of
            # which channel it used. CRITICAL for the BLIND run (empty skill library at start).
            _legacy = state.get("shodan_rules", []) or []
            if _legacy:
                _notes = "\n".join(f"  - {r['rule']}" for r in _legacy)
                shodan_rules_text = (shodan_rules_text + "\n\n=== CODEX NOTES (tactical, from SHODAN) ===\n" + _notes).strip()
            if self.config.verbose.shodan_rules and shodan_rules_text:
                _n = len(skill_store.relevant(env_name=env_name_now, tags=_tags))
                console.print(f"\n[magenta]📜 SKILLS injected ({_n} relevant) + {len(_legacy)} codex note(s):[/magenta]")
                console.print(f"[dim]{shodan_rules_text[:600]}{'...' if len(shodan_rules_text) > 600 else ''}[/dim]")
        else:
            # Legacy fallback: flat shodan_rules list
            shodan_rules = state.get("shodan_rules", [])
            shodan_rules_enabled = getattr(self.config, 'shodan_rules', None) and self.config.shodan_rules.enabled
            if shodan_rules and shodan_rules_enabled:
                rules_lines = ["", "=== CODEX RULES (learned lessons - respect them) ==="]
                for i, rule_entry in enumerate(shodan_rules):
                    rules_lines.append(f"  [{i}] {rule_entry['rule']}")
                shodan_rules_text = "\n".join(rules_lines)
            else:
                shodan_rules_text = ""

        # Optional {shodan_rules} placeholder: render_template tolerates prompt files
        # that omit it (renders empty) without a duplicated fallback format() call.
        task_template = self.render_template(
            prompt_dict["task_template"],
            current_task=state.get("current_task", ""),
            environment=self.config.environment.name,
            video_dir=video_dir,
            iteration=iteration,
            device=device,
            shodan_rules=shodan_rules_text,
        )
        
        # Get code context (previous iteration's code)
        code_context = self._get_code_context(state)
        context_section = f"\n\n{code_context}" if code_context else ""

        # A3: the Coder's own recent attempts + why they were rejected (self-memory)
        recent_section = self._format_recent_attempts(state)

        # Add conversation history (siloed - only this agent's previous messages)
        # Note: Coder has history_window=0, so this will be empty
        history_text = self.format_conversation_history(state)

        # Inject demo-specific video recording reference during demo phase
        # This prevents the infinite loop where agents guess RecordVideo parameters wrong
        current_phase = state.get("current_phase", "validation")
        if current_phase == "demo":
            env_name = self.config.environment.name
            demo_reference = f"""

=== VIDEO RECORDING - EXACT PATTERN (NO DEVIATIONS!) ===
env = gym.make("{env_name}", render_mode="rgb_array")
from gymnasium.wrappers import RecordVideo
env = RecordVideo(env, video_folder="/workspace/output/iter_{iteration}/", episode_trigger=lambda e: True, name_prefix="rl-video")
# Then: load model or train briefly, run 3-5 episodes, env.close()
# FORBIDDEN parameters (DON'T EXIST): fps, record_video_trigger
# render_mode="rgb_array" in gym.make() is MANDATORY
# Wrap SINGLE env BEFORE any DummyVecEnv
========================================================="""
            task_template += demo_reference

        # DUO: the previous run's raw stdout/stderr (+ diagnostics) goes straight to the Coder.
        raw_section = self._format_raw_output(state)

        full_prompt = prompt_dict["system"] + "\n\n" + history_text + task_template + context_section + recent_section + raw_section

        # Print context breakdown before LLM call (coder has no team chatter)
        prompt_tokens = self.estimate_tokens(full_prompt)
        self.print_context_breakdown(state, prompt_tokens, "")

        # Deterministic pre-Docker lint as a fast feedback arc ("pre-Tester"):
        # generate -> lint -> if STRUCTURAL errors (wrong env, syntax, bad imports),
        # regenerate with the lint feedback (up to K retries) BEFORE the expensive
        # Docker run. Catches the #1 time-wasters in milliseconds, not a 20-min timeout.
        # The Tester still runs the real container and does the semantic diagnosis.
        from src.utils.code_lint import lint_code
        env_name_for_lint = current_env.name if current_env else self.config.environment.name
        LINT_MAX_RETRIES = 2
        # Checkpoint-resume contract: in optimization with an existing checkpoint the
        # script MUST resume model+buffer and print RESUMED proof (the Tester gates on
        # it) - lint it here so the Coder fixes it BEFORE wasting a Docker run.
        _require_resume = bool(
            state.get("current_phase", "validation") == "optimization"
            and state.get("best_model_path", "")
        )

        def _extract_code(text: str) -> str:
            import re
            c = text.strip()
            c = re.sub(r'<think[^>]*>.*?</think[^>]*>', '', c, flags=re.DOTALL | re.IGNORECASE)
            c = re.sub(r'<thinking[^>]*>.*?</thinking[^>]*>', '', c, flags=re.DOTALL | re.IGNORECASE)
            if c.startswith('```python'):
                c = c[9:].lstrip()
            elif c.startswith('```'):
                c = c[3:].lstrip()
            if c.endswith('```'):
                c = c[:-3].rstrip()
            return c.strip()

        lint_feedback_block = ""
        response = None
        code = ""
        for _attempt in range(LINT_MAX_RETRIES + 1):
            response = self.call_llm_timed(full_prompt + lint_feedback_block, state["stats"], state.get("iteration", 0))
            self.print_thinking(response.content)
            code = _extract_code(response.content)

            lint_res = lint_code(code, env_name=env_name_for_lint, require_resume=_require_resume)
            if lint_res.ok:
                break
            if _attempt == LINT_MAX_RETRIES:
                console.print(f"[yellow]⚠ LINT still failing after {LINT_MAX_RETRIES} retries - Tester backstop will catch it[/yellow]")
                break
            console.print(f"[yellow]🔎 LINT rejected (attempt {_attempt + 1}/{LINT_MAX_RETRIES + 1}) - quick fix before Docker:[/yellow]")
            console.print(f"[dim]{lint_res.feedback()}[/dim]")
            lint_feedback_block = (
                "\n\n=== LINT FEEDBACK ===\nYOUR PREVIOUS SCRIPT (the one being rejected):\n"
                "```python\n" + code + "\n```\n"
                "It failed these structural checks - fix EXACTLY these, change nothing else:\n"
                + lint_res.feedback() + "\n=== end lint feedback ===\n"
            )

        # Print token statistics (latest call)
        stats_obj = state["stats"]
        iteration = state.get("iteration", 0)
        agent_timings = [t for t in stats_obj.timings if t.agent == self.agent_name and t.iteration == iteration]
        if agent_timings:
            latest_timing = agent_timings[-1]
            self.print_token_stats(latest_timing)

        # Light diagnostics (no auto-fixing - trust the model)
        code = self._check_code_quality(code)

        # Always show a quick summary of what coder produced
        self._print_code_summary(code, state)

        # Show generated code if enabled (visually formatted)
        if self.config.agents.show_coder_output:
            iteration = state.get("iteration", 0)
            lines_count = len(code.splitlines())
            task = state.get("current_task", "No task specified")

            # Show task summary first
            console.print("\n\n" + "-" * 70)
            console.print(f"[bold green]🔧 CODER - Iteration {iteration}[/bold green]")
            console.print("-" * 70)
            console.print(f"[yellow]Task:[/yellow] {task[:200]}{'...' if len(task) > 200 else ''}")
            console.print(f"[green]Generating complete Python script...[/green]")
            console.print()

            # Create syntax-highlighted code
            syntax = Syntax(
                code,
                "python",
                theme="monokai",
                line_numbers=True,
                word_wrap=False,
                background_color="default"
            )

            # Create beautiful panel with the code
            panel = Panel(
                syntax,
                title=f"[bold green]✓ Generated Code[/bold green]",
                subtitle=f"[dim]{lines_count} lines • Python 3.x[/dim]",
                border_style="green",
                padding=(1, 2)
            )

            console.print(panel)
            console.print("-" * 70 + "\n")

        # Track code statistics
        lines_count = len(code.splitlines())
        iteration = state.get("iteration", 0)
        stats_obj = state.get("stats")
        if stats_obj:
            stats_obj.add_code_stats(iteration, lines_count)

        # Detect algorithm and timesteps for conversation log
        import re as _re
        _algo = "Unknown"
        for _a in ["PPO", "DQN", "A2C", "SAC", "TD3", "HER"]:
            if _a in code:
                _algo = _a
                break
        _ts_match = _re.search(r'total_timesteps\s*=\s*(\d+)', code)
        _timesteps = _ts_match.group(1) if _ts_match else "?"

        # Get LLM timing for this call
        _duration = 0.0
        _tokens_out = 0
        if agent_timings:
            _duration = latest_timing.duration
            _tokens_out = latest_timing.tokens_out

        # Log to conversation file
        logger = state.get("conversation_logger")
        if logger:
            logger.log_coder(
                iteration=iteration,
                code=code,
                task=state.get("current_task", ""),
                lines=lines_count,
                algo=_algo,
                timesteps=_timesteps,
                duration=_duration,
                tokens_out=_tokens_out,
            )

        # Log context usage after all agent output
        self.log_context_to_conversation(state)

        # Save coder's response to conversation history
        history_update = self.save_message_to_history(state, response.content)

        # === CHAT CALL: team chatter AFTER the work is done. No-op for the quad pipeline
        # (opus_prompts.yaml has no coder.chat_template -> generate_chat_response returns ""
        # with no LLM call); the duo pipeline gives the Coder a voice on the team. ===
        chat_context = {
            "environment": env_name_for_lint,
            "current_task": (state.get("current_task", "") or "")[:200],
            "algo": _algo,
            "lines": lines_count,
            "iteration": iteration,
        }
        chat_opinion = self.generate_chat_response(state, chat_context, self.config.prompts)
        opinion_update = self.save_opinion_to_state(state, chat_opinion) if chat_opinion else {}
        if logger and chat_opinion:
            logger.log_agent_chat("coder", iteration, chat_opinion)

        result = {"code": code}
        result.update(history_update)
        result.update(opinion_update)

        return result