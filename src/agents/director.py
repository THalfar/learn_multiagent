"""Director — the single LLM brain of the duo pipeline (Director -> Coder -> Executor).

In the quad pipeline the Manager (local) translated feedback into a task and the Reviewer
(api/SHODAN) judged the result — two LLM hops with a Tester paraphrase in between. The duo
pipeline collapses the strategist + judge + taskmaster into ONE frontier call: the Director
judges run N-1 AND writes the task for run N in the same `__call__`. The Coder reads the
Executor's RAW stdout/stderr directly (no broken-telephone summary), and there is no Manager
and no Tester LLM.

Director subclasses Reviewer (hence BaseAgent(config, "reviewer")): it inherits the api model
(Grok), the "reviewer" timer bucket, config.get_prompt("reviewer"), and — for free — the
SHODAN environment-switch report. It does NOT reuse the Reviewer's nested extract_json (it's a
local closure); it uses the shared src.utils.json_extract.extract_json instead.

LangGraph contract (operator.add on `iteration`): the Director returns EXACTLY ONE
`"iteration": 1` on EVERY return path; the Coder and Executor never return `iteration`, so one
director->coder->executor->director cycle advances the counter by exactly one.
"""
import json

from .reviewer import Reviewer
from . import env_transitions
from src.utils.json_extract import extract_json
from src.utils.result_parser import parse_result_line
from src.utils.verdict_gates import apply_verdict_gates
from rich import print


class Director(Reviewer):
    def __init__(self, config, model_switcher=None):
        # Reviewer.__init__ pins agent_name="reviewer" (api model + timer bucket); ignore
        # model_switcher (the Director, like the Reviewer, stays on the API model).
        super().__init__(config)

    # ───────────────────────── deterministic helpers ─────────────────────────

    def _compute_chunk(self, state, current_env):
        """SPS-derived optimization chunk size (port of the Manager's logic). Returns
        (chunk_steps_or_None, note_text). Used both to instruct the LLM and to write the
        deterministic regression task."""
        opt_timeout = current_env.execution_timeout if current_env else 900
        sps = state.get("measured_sps", None)
        if sps:
            chunk = int(sps * opt_timeout * 0.8)
            chunk = max(20000, min(500000, int(round(chunk / 10000.0) * 10000)))
            note = (f"Measured speed ~{sps} steps/s -> a chunk of ~{chunk:,} steps fits the "
                    f"{opt_timeout}s timeout. Use that chunk size; do not guess.")
            return chunk, note
        return None, ("No measured speed yet - start with a moderate chunk (~50,000 steps); "
                      "the next chunk is sized from the measured steps/s.")

    def _dynamic_context(self, state, current_env, current_phase, success_threshold, metric_name):
        """The DYNAMIC facts the LLM can't infer from a static prompt: cumulative status, the
        SPS-sized chunk, the checkpoint-resume requirement, the escalation ladder, and the
        verified-skill precedence. Ported from the Manager's phase_instruction builders so the
        duo Director steers the Coder's task exactly as the quad Manager did."""
        lines = []
        tot_steps = state.get("total_env_steps", 0) or 0
        mh = state.get("metric_history", []) or []
        mh_str = ", ".join(f"{v:g}" for v in mh[-10:]) if mh else "(no completed chunks yet)"
        has_ckpt = bool(state.get("best_model_path", ""))

        if current_phase == "optimization":
            chunk, chunk_note = self._compute_chunk(state, current_env)
            lines.append(f"CUMULATIVE STATUS (this env): {tot_steps:,} steps trained so far.")
            lines.append(f"Metric per chunk: {mh_str}")
            lines.append(chunk_note)
            if has_ckpt:
                lines.append(
                    "A CHECKPOINT EXISTS (/workspace/output/best_model). The task you write MUST tell the "
                    "Coder to: (1) ALGO.load('/workspace/output/best_model', env=env); (2) load_replay_buffer "
                    "if the .pkl exists; (3) print 'RESUMED: buffer_transitions=N'; (4) learn(CHUNK) with the "
                    "DEFAULT reset_num_timesteps (never False); (5) save BOTH model and replay buffer. The "
                    "Executor REJECTS (pre-Docker) any optimization script that skips this resume chain.")
            else:
                lines.append(
                    "No checkpoint yet (first optimization chunk): train fresh, print "
                    "'RESUMED: buffer_transitions=0', and SAVE BOTH model and replay buffer at the end.")
            lines.append("EVAL: >=20 episodes with fixed seeds; print 'RESULT: mean_reward=X, std_reward=Y, "
                         f"episodes=Z' (put the {metric_name} value in the mean_reward slot).")
        elif current_phase == "validation":
            lines.append(f"VALIDATION: just confirm the code RUNS and prints a RESULT line for {metric_name}. "
                         "The threshold does not matter yet. On approval, the next task is the OPTIMIZATION "
                         "task (full training toward the threshold via checkpoint-resume).")
        elif current_phase == "demo":
            lines.append("DEMO: the Executor records video + evaluates the saved model deterministically (no "
                         "Coder code needed). The DEMO-REWARD GATE is deterministic: the measured metric must "
                         f">= {success_threshold} or the env regresses to OPTIMIZATION.")

        # Escalation ladder: same failure mode 3x -> change strategy CLASS, not the parameter.
        fh = state.get("failure_history", []) or []
        if len(fh) >= 3 and len(set(fh[-3:])) == 1:
            mode = fh[-1]
            lines.append(
                f"\n⚠️ ESCALATION: the last 3 failures were ALL '{mode}'. Change the STRATEGY CLASS, not the "
                "parameter. timeout -> checkpoint-resume in chunks; crash -> change the approach/API; "
                "low_reward -> change algorithm / add HER; resume_violation -> spell out the exact "
                "load/print/save lines verbatim in the task.")

        # Verified-skill precedence (THE PIN BINDS YOU TOO).
        if current_phase in ("validation", "optimization"):
            ss = state.get("skill_store", None)
            if ss is not None:
                try:
                    sk_txt = ss.render_for_coder(env_name=(current_env.name if current_env else ""))
                except Exception:
                    sk_txt = ""
                if sk_txt:
                    lines.append(
                        "\n⛏️ VERIFIED SKILLS TAKE PRECEDENCE (confirmed by real runs - they outrank even your "
                        "own directives). Build the task so it implements the skill's procedure:\n" + sk_txt)

        return "\n".join(lines)

    def _parse_verdict(self, response, state, previous_task):
        """Parse the Director's JSON verdict with the shared extractor + one active-parsing
        retry. Fallback keeps the run alive: reject + repeat the previous task."""
        fallback = {"approved": False, "feedback": "Parse error: could not read the verdict JSON.",
                    "next_task": previous_task, "skill_ops": None, "my_opinion": ""}
        for _attempt in range(2):
            try:
                return json.loads(extract_json(response.content))
            except json.JSONDecodeError:
                if _attempt == 0:
                    fix_prompt = (
                        "Your previous response could not be parsed as JSON. Return ONLY this object:\n"
                        '{"approved": true or false, "feedback": "...", "next_task": "...", '
                        '"skill_ops": {}, "my_opinion": "..."}\n\nYour response was:\n' + (response.content or ""))
                    try:
                        response = self.call_llm_timed(fix_prompt, state["stats"], state.get("iteration", 0))
                        self.print_thinking(response.content)
                    except Exception:
                        return fallback
        return fallback

    # ───────────────────────────── main node ─────────────────────────────

    def __call__(self, state: dict) -> dict:
        env_progression = self.config.environment_progression
        current_env_index = state.get("current_env_index", 0)
        current_env = (env_progression[current_env_index]
                       if env_progression and current_env_index < len(env_progression) else None)
        current_env_name = current_env.name if current_env else self.config.environment.name
        current_phase = state.get("current_phase", "validation")
        logger = state.get("conversation_logger")
        iteration = state.get("iteration", 0)

        # ── 1. BOOTSTRAP: no task yet -> deterministic first validation task, no LLM call ──
        if not state.get("current_task"):
            env0 = env_progression[0] if env_progression else current_env
            first_task = (env_transitions.initial_validation_task(env0) if env0
                          else "Write a minimal validation script and print RESULT: mean_reward=X.")
            print(f"\n[bold blue]🎬 DIRECTOR: bootstrapping first task for {current_env_name}[/bold blue]")
            if logger:
                logger.log_iteration_start(iteration + 1, current_env_name)
                logger.log_manager(iteration + 1, task=first_task, environment=current_env_name,
                                   success_threshold=(current_env.success_threshold if current_env else 0))
            return {
                "current_task": first_task,
                "tasks": [first_task],
                "manager_guidance": f"Task: {first_task}",
                "current_phase": "validation",
                "approved": False,
                "iteration": 1,
            }

        # ── 2. Deterministic facts (before the LLM) ──
        success_threshold = current_env.success_threshold if current_env else 0
        env_metric = getattr(current_env, "metric", "reward") if current_env else "reward"
        metric_name = "success_rate" if env_metric == "success_rate" else "mean_reward"
        stdout_real = state.get("execution_stdout", "") or ""
        real_reward = parse_result_line(stdout_real)["value"]
        previous_task = state.get("current_task", "")
        solved_environments = list(state.get("solved_environments", []))
        agent_opinions_context = self.format_agent_opinions_context(state)
        dynamic_context = self._dynamic_context(state, current_env, current_phase, success_threshold, metric_name)
        env_progression_info = (f"Environment {current_env_index + 1}/{len(env_progression)}: {current_env_name} "
                                f"(threshold {success_threshold}, metric {env_metric}). "
                                f"Solved so far: {', '.join(solved_environments) if solved_environments else 'none'}.")

        skill_store = state.get("skill_store", None)
        shodan_rules_display = (skill_store.render_summary() if skill_store is not None else "")

        # ── 3. ONE LLM call: verdict on run N-1 + task for run N ──
        prompt_dict = self.config.get_prompt("reviewer")
        system_prompt = self.render_template(
            prompt_dict.get("system", ""),
            success_threshold=success_threshold,
            video_dir=state.get("video_dir", "output/videos"),
            environment=current_env_name,
        )
        task_template = self.render_template(
            prompt_dict.get("task_template", ""),
            code=state.get("code", ""),
            test_results=state.get("test_results", ""),
            previous_task=previous_task,
            success_threshold=success_threshold,
            video_dir=state.get("video_dir", "output/videos"),
            shodan_rules_display=shodan_rules_display,
            agent_opinions_context=agent_opinions_context,
            env_progression_info=env_progression_info,
            iteration=iteration,
            max_iterations=self.config.agents.max_iterations,
        )
        history_text = self.format_conversation_history(state)
        full_prompt = (system_prompt + "\n\n" + dynamic_context + "\n\n=== CURRENT PHASE: "
                       + current_phase.upper() + " ===\n" + history_text + task_template)

        print(f"\n[bold magenta]🎬 DIRECTOR judging iter {iteration} ({current_phase}) on {current_env_name}[/bold magenta]")
        response = self.call_llm_timed(full_prompt, state["stats"], iteration)
        self.print_thinking(response.content)

        parsed = self._parse_verdict(response, state, previous_task)
        llm_approved = bool(parsed.get("approved", False))
        feedback = (parsed.get("feedback") or "").strip()
        next_task = (parsed.get("next_task") or "").strip() or previous_task
        my_opinion = (parsed.get("my_opinion") or "").strip()

        # ── 4. DETERMINISTIC GATES (override the LLM's APPROVE) ──
        gate = apply_verdict_gates(
            llm_approved, phase=current_phase, stdout=stdout_real, success_threshold=success_threshold,
            env_metric=env_metric, resume_required=state.get("resume_required", False),
            resume_ok=state.get("resume_ok", True), demo_reward=state.get("demo_reward"))
        approved = gate.approved
        if gate.feedback_prefix:
            if llm_approved and not approved:
                print(f"[yellow]⚖️  Gate '{gate.gate_fired}' overrode APPROVE -> REJECT[/yellow]")
            feedback = gate.feedback_prefix + feedback

        # ── 5. skill_ops (never crash on LLM input) ──
        skill_ops = parsed.get("skill_ops", None)
        if skill_store is not None and isinstance(skill_ops, dict):
            try:
                for _line in skill_store.apply_ops(skill_ops, iteration=iteration):
                    print(f"[magenta]🧠 {_line}[/magenta]")
            except Exception as _e:
                print(f"[dim]skill_ops failed (ignored): {_e}[/dim]")

        # ── 6. Failsafe bookkeeping (progress-aware; demo-gate rejection does NOT burn budget) ──
        best_reward = state.get("best_reward_this_env", None)
        if state.get("best_reward_env_index", -1) != current_env_index:
            best_reward = None
        improved = real_reward is not None and (best_reward is None or real_reward > best_reward)
        if improved:
            best_reward = real_reward
        last_failure_type = ""
        if approved or improved:
            consecutive_failures = 0
        elif gate.gate_fired == "demo":
            consecutive_failures = state.get("consecutive_failures", 0)  # measurement event, not a regression
        else:
            consecutive_failures = state.get("consecutive_failures", 0) + 1
            tr = state.get("test_results", "")
            if "RESUME CONTRACT FAILED" in tr or "RESUME CHECK FAILED" in tr:
                last_failure_type = "resume_violation"
            elif "TIMEOUT" in tr.upper():
                last_failure_type = "timeout"
            elif "Traceback" in (state.get("execution_stderr", "") or "") or "Error" in tr:
                last_failure_type = "crash"
            else:
                last_failure_type = "low_reward"

        recent_attempts = list(state.get("recent_attempts", []))
        recent_attempts.append({
            "iter": iteration, "verdict": "APPROVED" if approved else "REJECTED",
            "diagnosis": (state.get("diagnosis", "") or "")[:400], "reason": (feedback or "")[:300],
        })
        recent_attempts = recent_attempts[-3:]
        failure_history = list(state.get("failure_history", []))
        if not approved and last_failure_type:
            failure_history.append(last_failure_type)
        failure_history = failure_history[-8:]

        # Log the verdict now (before the task/phase bookkeeping below).
        if logger:
            t = getattr(self, "last_timing", None)
            logger.log_reviewer(iteration=iteration, approved=approved, feedback=feedback,
                                duration=(t.duration if t else 0),
                                tokens_in=(t.tokens_in if t else 0), tokens_out=(t.tokens_out if t else 0))
        opinion_update = self.save_opinion_to_state(state, my_opinion) if my_opinion else {}
        if logger and my_opinion:
            logger.log_agent_chat("reviewer", iteration, my_opinion)
        history_update = self.save_message_to_history(state, response.content)

        def _finalize(extra: dict) -> dict:
            """Attach the shared bookkeeping + history/opinion + the single iteration:1 that
            EVERY Director return path must carry (operator.add)."""
            out = {
                "review_feedback": feedback,
                "recent_attempts": recent_attempts,
                "failure_history": failure_history,
                "best_reward_this_env": best_reward,
                "best_reward_env_index": current_env_index,
                "consecutive_failures": consecutive_failures,
                "iteration": 1,
            }
            out.update(extra)
            out.update(history_update)
            out.update(opinion_update)
            return out

        # ── 6b. IMMEDIATE failsafe skip (unlike quad, the Director can switch env in the same call) ──
        failsafe_config = getattr(self.config.project, "failsafe", None)
        skip_threshold = getattr(failsafe_config, "skip_after_consecutive_failures", 8) if failsafe_config else 8
        if consecutive_failures >= skip_threshold and env_progression and current_env_index + 1 < len(env_progression):
            next_env = env_progression[current_env_index + 1]
            print(f"[bold red]⚠️  FAILSAFE: {consecutive_failures} consecutive failures on {current_env_name} -> "
                  f"skipping to {next_env.name}[/bold red]")
            new_video_dir = self._switch_config_env(state, next_env)
            skip_task = env_transitions.initial_validation_task(next_env)
            reset = env_transitions.env_switch_reset(current_env_index + 1, skip_task, new_video_dir)
            reset.update({
                "skipped_environments": list(state.get("skipped_environments", [])) + [current_env_name],
                "solved_environments": solved_environments,
                "review_feedback": f"FAILSAFE: skipped {current_env_name} after {consecutive_failures} failures.",
            })
            if logger:
                logger.log_iteration_start(iteration + 1, next_env.name)
                logger.log_manager(iteration + 1, task=skip_task, environment=next_env.name,
                                   success_threshold=next_env.success_threshold)
            reset.update(history_update)
            reset.update(opinion_update)
            return reset  # env_switch_reset already carries iteration:1

        # ── 7. Phase machine (verdict -> next phase + next task) ──
        next_phase = current_phase
        demo_reward_out = state.get("demo_reward")
        demo_below_out = False

        if approved and current_phase == "validation":
            next_phase = "optimization"
            if logger:
                logger.log_phase_transition("validation", "optimization", current_env_name)
            print("[bold green]✅ VALIDATION PASSED -> OPTIMIZATION[/bold green]")

        elif approved and current_phase == "optimization":
            next_phase = "demo"
            demo_reward_out = None  # enter demo with a clean gate
            if logger:
                logger.log_phase_transition("optimization", "demo", current_env_name)
            print("[bold green]✅ OPTIMIZATION COMPLETE -> DEMO[/bold green]")

        elif approved and current_phase == "demo":
            # ENV SOLVED. Switch to the next env (or DONE).
            return self._on_env_solved(state, current_env, current_env_index, env_progression,
                                       solved_environments, iteration, logger, _finalize)

        elif gate.demo_below_threshold:
            # REGRESSION: demo measured below threshold -> back to optimization, keep the
            # checkpoint, and OVERRIDE the LLM's task (it wrote it before the gate decided).
            next_phase = "optimization"
            demo_reward_out = None
            demo_below_out = False  # cleared now that we're acting on it
            chunk, _ = self._compute_chunk(state, current_env)
            chunk_txt = f"~{chunk:,}" if chunk else "~50,000"
            dv = state.get("demo_reward")
            next_task = (f"Demo confirmed below threshold ({dv} < {success_threshold}). Continue OPTIMIZATION "
                         f"with checkpoint-resume: load best_model + replay buffer, print "
                         f"'RESUMED: buffer_transitions=N', train {chunk_txt} steps, save BOTH model and buffer, "
                         f"evaluate >=20 fixed-seed episodes, and print RESULT: {metric_name}=X.")
            if logger:
                logger.log_phase_transition("demo", "optimization", current_env_name)
            print(f"[bold yellow]↩️  DEMO GATE: {dv} < {success_threshold} -> regress to OPTIMIZATION[/bold yellow]")

        # ── 8. Log the next task + return ──
        if logger:
            logger.log_iteration_start(iteration + 1, current_env_name)
            logger.log_manager(iteration + 1, task=next_task, environment=current_env_name,
                               success_threshold=success_threshold)
        self.log_context_to_conversation(state)
        return _finalize({
            "current_task": next_task,
            "tasks": list(state.get("tasks", [])) + [next_task],
            "manager_guidance": f"Task: {next_task}",
            "approved": approved,
            "current_phase": next_phase,
            "demo_reward": demo_reward_out,
            "demo_below_threshold": demo_below_out,
        })

    # ───────────────────────── env-solve / config switch ─────────────────────────

    def _switch_config_env(self, state, next_env):
        """Point the live config at the next env (Coder prompt, lint, Executor all agree) and
        return its fresh video_dir."""
        import os
        self.config.project.environment.name = next_env.name
        self.config.project.environment.max_episode_steps = next_env.max_episode_steps
        run_id = state.get("run_id", "")
        new_video_dir = os.path.abspath(os.path.normpath(f"output/{run_id}/{next_env.name}/videos"))
        os.makedirs(new_video_dir, exist_ok=True)
        return new_video_dir

    def _on_env_solved(self, state, current_env, current_env_index, env_progression,
                       solved_environments, iteration, logger, _finalize):
        """Demo passed the reward gate -> the env is genuinely solved. Distil a verified skill,
        emit SHODAN's switch report, switch to the next env (or finish)."""
        if current_env.name not in solved_environments:
            solved_environments = solved_environments + [current_env.name]
        print(f"[bold green]🏆 ENVIRONMENT SOLVED: {current_env.name}[/bold green]")

        # Distil a verified procedural skill from the winning code.
        winning_code = state.get("code", "")
        ss = state.get("skill_store", None)
        if winning_code and ss is not None:
            try:
                sk = env_transitions.skill_from_winning_code(winning_code, current_env.name,
                                                             getattr(current_env, "tags", None))
                sid = ss.add(created_iter=iteration, **sk)
                ss.save()
                print(f"[bold magenta]🧠 SKILL learned (verified): [{sid}] {sk['name']}[/bold magenta]")
            except Exception as _e:
                print(f"[dim]skill distil failed: {_e}[/dim]")

        # Last env? -> DONE.
        if not (env_progression and current_env_index + 1 < len(env_progression)):
            print("[bold green]🏆 ALL ENVIRONMENTS SOLVED![/bold green]")
            return _finalize({"current_task": "DONE", "solved_environments": solved_environments,
                              "approved": True, "current_phase": "demo"})

        next_env = env_progression[current_env_index + 1]

        # SHODAN's switch report (inherited from Reviewer) — chatter only, never break the run.
        manager_report = "(no Manager in duo mode)"
        reviewer_report = ""
        env_switch_reports = list(state.get("env_switch_reports", []))
        if getattr(self.config.agents, "show_env_switch_chatter", True):
            try:
                reviewer_report, _thinking, _timing = self.generate_environment_switch_report(
                    current_env_name=current_env.name, next_env_name=next_env.name,
                    manager_report=manager_report, solved_environments=solved_environments,
                    env_progression=env_progression, stats=state["stats"], tasks=state.get("tasks", []),
                    iterations=iteration, code=state.get("code", ""), test_results=state.get("test_results", ""),
                    review_feedback=state.get("review_feedback", ""),
                    previous_reports=env_switch_reports, state=state)
            except Exception as _e:
                print(f"[dim]switch report failed (ignored): {_e}[/dim]")
        if logger:
            logger.log_environment_switch(current_env=current_env.name, next_env=next_env.name,
                                          manager_report=manager_report, reviewer_report=reviewer_report)
            logger.save_environment_snapshot(current_env.name, state.get("run_id", ""))
        env_switch_reports.append({
            "environment": current_env.name, "next_environment": next_env.name,
            "manager_report": manager_report, "reviewer_report": reviewer_report,
            "iterations": iteration, "tasks_completed": len(state.get("tasks", [])),
        })

        new_video_dir = self._switch_config_env(state, next_env)
        next_task = env_transitions.initial_validation_task(next_env)
        reset = env_transitions.env_switch_reset(current_env_index + 1, next_task, new_video_dir)
        reset.update({
            "solved_environments": solved_environments,
            "env_switch_reports": env_switch_reports,
        })
        if logger:
            logger.log_iteration_start(iteration + 1, next_env.name)
            logger.log_manager(iteration + 1, task=next_task, environment=next_env.name,
                               success_threshold=next_env.success_threshold)
        # Merge the verdict-side history/opinion so they aren't lost on the switch path.
        return _finalize(reset)
