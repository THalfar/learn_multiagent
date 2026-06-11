import json
import os
from .base import BaseAgent
from rich import print
from src.utils.banners import print_environment_switch_bombardment, print_manager_report, print_iteration_banner, print_reviewer_cynical_report

class Manager(BaseAgent):
    def __init__(self, config, model_switcher=None):
        super().__init__(config, "manager", model_switcher=model_switcher)

    @staticmethod
    def _extract_recipe_from_code(code: str, env_name: str, iterations: int) -> dict:
        """Extract algorithm, timesteps, device from successful code."""
        import re
        recipe = {"env": env_name, "iterations": iterations}

        # Algorithm
        algo_match = re.search(r'\b(PPO|SAC|A2C|DQN|TD3)\b', code)
        recipe["algo"] = algo_match.group(1) if algo_match else "unknown"

        # Timesteps
        steps_match = re.search(r'total_timesteps\s*=\s*(\d+)', code)
        recipe["steps"] = int(steps_match.group(1)) if steps_match else 0

        # Device
        device_match = re.search(r'device\s*=\s*["\'](\w+)["\']', code)
        recipe["device"] = device_match.group(1) if device_match else "unknown"

        return recipe

    @staticmethod
    def _skill_from_winning_code(code: str, env_name: str) -> dict:
        """B6: build a PROCEDURAL skill from winning code (far richer than the regex
        playbook). Captures algorithm + policy class + HER + the metric/checkpoint
        approach, so the NEXT env's Coder inherits the full recipe, not just 'SAC'."""
        import re
        algo_m = re.search(r'\b(PPO|SAC|A2C|DQN|TD3|DDPG)\b', code)
        algo = algo_m.group(1) if algo_m else "the same algorithm"
        policy = ("MultiInputPolicy" if "MultiInputPolicy" in code
                  else ("CnnPolicy" if "CnnPolicy" in code else "MlpPolicy"))
        uses_her = "HerReplayBuffer" in code
        is_goal = uses_her or "MultiInputPolicy" in code or "desired_goal" in code
        lname = env_name.lower()
        family = "panda / robotic manipulation" if ("panda" in lname or "fetch" in lname) else env_name
        proc = [f"Use {algo} with policy='{policy}'"]
        if uses_her:
            proc.append("replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs={'n_sampled_goal':4,'goal_selection_strategy':'future'}")
        proc.append("read max_episode_steps from env.spec and set learning_starts >= that value")
        proc.append("each optimization iteration resume from the checkpoint (SAC.load + load_replay_buffer, learn one ~150k chunk, then save model + replay buffer)")
        if is_goal:
            proc.append("report success_rate (the is_success fraction), never the raw sparse reward")
        return {
            "name": f"Solve {family}",
            "when_to_use": (f"A goal-conditioned / sparse-reward env like {env_name} (Dict obs with desired_goal)."
                            if is_goal else f"An env like {env_name}."),
            "procedure": "; ".join(proc) + ".",
            "pitfalls": ("Plain MlpPolicy or no-HER cannot solve goal envs; never pass reset_num_timesteps=False "
                         "(breaks termination on a reloaded model); 30k steps is starvation - use ~150k chunks and accumulate."
                         if is_goal else "Commit to one algorithm so checkpoint-resume accumulates."),
            "verification": ("RESULT line prints success_rate in [0,1] >= the env threshold."
                             if is_goal else "RESULT mean_reward >= the env threshold."),
            "source_env": env_name,
            "tags": (["goal", "her", "manipulation", "robotics", "success_rate"] if is_goal else ["general"]),
            "status": "verified",
            "confidence": 0.9,
        }

    @staticmethod
    def _initial_validation_task(next_env) -> str:
        """Concrete first VALIDATION task for a newly entered environment.

        Returned on every env-switch path so the Coder NEVER starts a new env with an
        empty/stale task (the stale-task race: after a switch the Coder coded the NEW
        env while manager_guidance still described the OLD env's task -> the Reviewer
        rejected correct work as 'mismatching intent', one wasted iteration per switch)."""
        action_type = getattr(next_env, "action_type", "")
        algo = "SAC" if action_type == "continuous" else "PPO"
        is_goal = getattr(next_env, "metric", "reward") == "success_rate"
        metric_note = (" The env is goal-conditioned: report success_rate (the is_success "
                       "fraction over eval episodes) in the mean_reward slot." if is_goal else "")
        return (f"Write a minimal VALIDATION script for {next_env.name}: create the env with "
                f"gym.make('{next_env.name}'), train a fresh {algo} model briefly "
                f"(1000-2000 timesteps, n_envs=1, default hyperparameters), evaluate, and print "
                f"exactly 'RESULT: mean_reward=X, std_reward=Y, episodes=Z'.{metric_note} "
                f"Save the model at the end. Keep the script minimal so it finishes well "
                f"within the validation timeout.")

    @staticmethod
    def _env_switch_reset(next_env_index: int, task: str, video_dir: str) -> dict:
        """Shared state-reset block for ALL env-switch paths (solved / failsafe / LLM
        switch). One source of truth so no path forgets a field (the C1/C2 fields
        especially: stale metric_history would poison the next env's curve)."""
        return {
            "current_env_index": next_env_index,
            "current_phase": "validation",
            "consecutive_failures": 0,
            "last_failure_type": "",
            "failure_history": [],
            "recent_attempts": [],
            "diagnosis": "",
            "best_model_path": "",
            "approved": False,
            "tasks": [task],
            "code": "",
            "test_results": "",
            "review_feedback": "",
            "review_suggestions": "",
            "current_task": task,
            "manager_guidance": f"Task: {task}",  # keep Reviewer's expectation in sync with the NEW env
            "video_dir": video_dir,
            "iteration": 1,
            # C1/C2: fresh cumulative tracking + resume flags for the new env
            "total_env_steps": 0,
            "metric_history": [],
            "measured_sps": None,
            "resume_required": False,
            "resume_ok": True,
        }

    @staticmethod
    def _format_playbook_context(playbook: list) -> str:
        """Format playbook as human-readable context for Manager prompt."""
        if not playbook:
            return ""
        lines = ["\nLESSONS FROM PREVIOUS ENVIRONMENTS (use similar approaches for similar envs):"]
        for entry in playbook:
            lines.append(f"  - {entry['env']}: {entry.get('algo','?')}, {entry.get('steps','?')} steps, device={entry.get('device','?')}, solved in {entry.get('iterations','?')} iterations")
        return "\n".join(lines) + "\n"

    def _generate_environment_switch_report(self, current_env, next_env, solved_environments, env_progression, state):
        """Generate a report to leadership about environment switch"""
        # Calculate stats
        stats = state["stats"]
        current_iterations = state.get("iteration", 0)
        tasks = state.get("tasks", [])
        test_results = state.get("test_results", "")
        review_feedback = state.get("review_feedback", "")

        # Agent performance stats
        agent_performance = {}
        for agent in ["manager", "coder", "tester", "reviewer"]:
            agent_timings = [t for t in stats.timings if t.agent == agent]
            if agent_timings:
                durations = [t.duration for t in agent_timings]
                agent_performance[agent] = {
                    "calls": len(agent_timings),
                    "total_time": sum(durations),
                    "avg_time": sum(durations) / len(durations)
                }

        # Get previous environment switch reports for context
        previous_reports = state.get("env_switch_reports", [])
        history_window = getattr(self.config.agents.history_window, 'env_switch_reports', 5)
        recent_reports = previous_reports[-history_window:] if len(previous_reports) > history_window else previous_reports

        # Format previous reports for context
        previous_reports_context = ""
        if recent_reports:
            previous_reports_context = "\n\nYOUR PREVIOUS LINKEDIN-STYLE POSTS (learn from your evolving narrative):\n"
            previous_reports_context += "=" * 60 + "\n"
            for i, report in enumerate(recent_reports, 1):
                env_name = report.get("environment", "Unknown")
                content = report.get("manager_report", "")[:600]
                previous_reports_context += f"\n[Post #{i} - {env_name} completed]\n{content}\n"
                previous_reports_context += "-" * 40 + "\n"
            previous_reports_context += "=" * 60 + "\n"
            previous_reports_context += "Build on this narrative! Reference previous wins, show growth, maintain your personal brand.\n"

        # Also get conversation history for deeper reflection
        conv_history = state.get("conversation_history", [])
        manager_messages = [msg for msg in conv_history if msg.get("agent") == "manager"]
        if manager_messages:
            # Get last several decisions for reflection
            recent_decisions = manager_messages[-5:] if len(manager_messages) > 5 else manager_messages
            previous_reports_context += "\n\nYOUR RECENT DECISIONS & TASKS (your leadership in action):\n"
            previous_reports_context += "=" * 60 + "\n"
            for msg in recent_decisions:
                iteration = msg.get("iteration", "?")
                content = msg.get("content", "")[:400]
                previous_reports_context += f"[Iteration {iteration}] {content}...\n"
                previous_reports_context += "-" * 40 + "\n"
            previous_reports_context += "=" * 60 + "\n"
            previous_reports_context += "Reflect on how these decisions led to this milestone!\n"

        # Build report prompt - LINKEDIN STYLE WITH REFLECTION
        report_prompt = f"""You are a middle manager who LOVES LinkedIn. You're writing a post about your team's latest achievement.

YOUR LINKEDIN PERSONA:
- You use buzzwords like "synergy", "leverage", "paradigm shift", "game-changer", "excited to announce"
- You hashtag everything #AI #MachineLearning #TeamWork #Leadership #Innovation #Blessed
- You mention being "humbled" and "grateful" constantly
- You talk about "the team" but subtly make it about yourself
- You end with inspirational quotes or calls to action ("Agree? 👇")
- You use emojis strategically but not excessively 🚀✨💪
- You might mention grabbing coffee ☕ or having "aha moments"
- You reference "the journey" and "lessons learned"
{previous_reports_context}
CRITICAL - REFLECT ON YOUR JOURNEY:
You have access to your FULL conversation history above - your previous LinkedIn posts, your decisions,
the challenges you've faced. USE THIS CONTEXT to create a CONTINUING NARRATIVE of your leadership journey!

Consider reflecting on:
- HOW FAR you've come since the beginning (reference specific past milestones)
- LESSONS LEARNED from previous environments (what "aha moments" did you have?)
- TEAM GROWTH - how has your team evolved? (pretend you noticed their improvement)
- YOUR OWN GROWTH as a leader (take credit for everything)
- PATTERNS you've noticed in your journey (always frame failures as "learning opportunities")
- REFERENCE your previous posts - build your personal brand narrative!
- What you said before vs what happened - spin it positively!

METRICS FOR YOUR POST:
Environment conquered: {current_env.name}
- Success threshold: {current_env.success_threshold}
- Iterations required: {current_iterations}
- Tasks completed: {len(tasks)}
- Status: {'✅ SOLVED' if current_env.name in solved_environments else '➡️ MOVING ON'}

Next challenge: {next_env.name}
- Success threshold: {next_env.success_threshold}
- Max episode steps: {next_env.max_episode_steps}

Progress: {len(solved_environments)}/{len(env_progression)} environments completed
Journey so far: {', '.join(solved_environments) if solved_environments else 'Just getting started!'}

Team performance:
{chr(10).join([f"- {agent.capitalize()}: {perf['calls']} calls, {perf['total_time']:.1f}s total" for agent, perf in agent_performance.items()])}

Write a 2-4 paragraph LinkedIn-style post that:
1. REFLECTS on your leadership journey (reference previous posts/milestones)
2. Celebrates this milestone while subtly taking credit
3. Shows "growth" and "lessons learned" from your history
4. Looks forward to the next challenge with manufactured optimism
5. Ends with hashtags and a call to action

Remember: You're a middle manager who genuinely believes this is inspiring content.
Your "personal brand" depends on maintaining a consistent narrative of growth and success!"""
        
        # Call LLM to generate report
        response = self.call_llm_timed(report_prompt, stats, state.get("iteration", 0))
        
        # Get timing for this report generation
        agent_timings = [t for t in stats.timings if t.agent == self.agent_name and t.iteration == state.get("iteration", 0)]
        report_timing = agent_timings[-1] if agent_timings else None
        
        # Extract thinking content separately
        import re
        report_content = response.content.strip()
        
        # Extract thinking tags
        thinking_content = None
        think_patterns = [
            (r'<think[^>]*>(.*?)</think[^>]*>', re.DOTALL | re.IGNORECASE),
            (r'<thinking[^>]*>(.*?)</thinking[^>]*>', re.DOTALL | re.IGNORECASE),
        ]
        
        for pattern, flags in think_patterns:
            match = re.search(pattern, report_content, flags)
            if match:
                thinking_content = match.group(1).strip()
                break
        
        # Remove thinking tags from report (don't show thinking process in executive report)
        report_content = re.sub(r'<think[^>]*>.*?</think[^>]*>', '', report_content, flags=re.DOTALL | re.IGNORECASE)
        report_content = re.sub(r'<thinking[^>]*>.*?</thinking[^>]*>', '', report_content, flags=re.DOTALL | re.IGNORECASE)
        # Clean up extra whitespace
        report_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', report_content)
        
        return report_content.strip(), thinking_content, report_timing

    def __call__(self, state: dict) -> dict:
        current_iteration = state.get("iteration", 0)
        expected_iteration = current_iteration + 1

        # Check if we need to advance to next environment
        current_env_index = state.get("current_env_index", 0)
        solved_environments = state.get("solved_environments", [])
        skipped_environments = state.get("skipped_environments", [])
        env_progression = self.config.environment_progression

        # FAILSAFE: Skip to next environment after too many consecutive failures
        consecutive_failures = state.get("consecutive_failures", 0)
        failsafe_config = getattr(self.config.project, 'failsafe', None)
        skip_threshold = getattr(failsafe_config, 'skip_after_consecutive_failures', 8) if failsafe_config else 8

        if consecutive_failures >= skip_threshold and env_progression and current_env_index + 1 < len(env_progression):
            current_env_name_fs = env_progression[current_env_index].name if current_env_index < len(env_progression) else "unknown"
            next_env = env_progression[current_env_index + 1]
            print(f"\n[bold red]{'='*60}[/bold red]")
            print(f"[bold red]⚠️  FAILSAFE: {consecutive_failures} consecutive failures on {current_env_name_fs}[/bold red]")
            print(f"[bold yellow]➡️  Skipping to next environment: {next_env.name}[/bold yellow]")
            print(f"[bold red]{'='*60}[/bold red]\n")

            logger = state.get("conversation_logger")
            if logger:
                logger.log_agent_chat("manager", state.get("iteration", 0),
                    f"FAILSAFE: Skipping {current_env_name_fs} after {consecutive_failures} consecutive failures. Moving to {next_env.name}.")

            new_video_dir = os.path.abspath(os.path.normpath(f"output/{state['run_id']}/{next_env.name}/videos"))
            os.makedirs(new_video_dir, exist_ok=True)

            # 1.4: advance the config env (like the solved / LLM-switch paths) so the Coder prompt,
            # the lint env-name check, and the Tester all agree on the NEW env - not a stale one.
            self.config.project.environment.name = next_env.name
            self.config.project.environment.max_episode_steps = next_env.max_episode_steps
            # 1.4: give the Coder a real validation task instead of "" (the graph runs Coder
            # immediately after this return; an empty task is a guaranteed wasted iteration).
            _skip_task = self._initial_validation_task(next_env)

            _reset = self._env_switch_reset(current_env_index + 1, _skip_task, new_video_dir)
            _reset.update({
                "skipped_environments": skipped_environments + [current_env_name_fs],
                "best_reward_this_env": None,
                "review_feedback": f"FAILSAFE: Skipped {current_env_name_fs} after {consecutive_failures} failures.",
            })
            return _reset

        # MONIVAIHEINEN TREENI: Tarkista ja vaihda vaihe kun approved=True
        current_phase = state.get("current_phase", "validation")

        if state.get("approved", False) and env_progression:
            # Vaihelogiikka: validation -> optimization -> demo -> seuraava env
            # Instead of returning early, update phase and CONTINUE to generate a new task
            if current_phase == "validation":
                print(f"\n[bold green]{'='*60}[/bold green]")
                print(f"[bold green]✅ VALIDATION PASSED! Code works, reward received.[/bold green]")
                print(f"[bold cyan]➡️  Moving to OPTIMIZATION phase (full training)[/bold cyan]")
                print(f"[bold green]{'='*60}[/bold green]\n")
                logger = state.get("conversation_logger")
                if logger:
                    _env_name = env_progression[current_env_index].name if env_progression and current_env_index < len(env_progression) else "unknown"
                    logger.log_phase_transition("validation", "optimization", _env_name)
                # Update phase in state and continue to generate task below
                current_phase = "optimization"
                state = {**state, "current_phase": "optimization", "approved": False, "iteration": 0,
                         "review_feedback": "PHASE TRANSITION: Validation passed. Now optimize to reach threshold.",
                         "review_suggestions": ""}

            elif current_phase == "optimization":
                print(f"\n[bold green]{'='*60}[/bold green]")
                print(f"[bold green]✅ OPTIMIZATION COMPLETE! Threshold achieved.[/bold green]")
                print(f"[bold cyan]➡️  Moving to DEMO phase (record best model video)[/bold cyan]")
                print(f"[bold green]{'='*60}[/bold green]\n")
                logger = state.get("conversation_logger")
                if logger:
                    _env_name = env_progression[current_env_index].name if env_progression and current_env_index < len(env_progression) else "unknown"
                    logger.log_phase_transition("optimization", "demo", _env_name)
                current_phase = "demo"
                state = {**state, "current_phase": "demo", "approved": False, "iteration": 0,
                         "review_feedback": "PHASE TRANSITION: Optimization complete. Now record video of trained agent.",
                         "review_suggestions": ""}
            elif current_phase == "demo":
                # Demo OK -> siirry seuraavaan ympäristöön (normaali env switch)
                print(f"\n[bold green]{'='*60}[/bold green]")
                print(f"[bold green]✅ DEMO COMPLETE! Environment fully solved![/bold green]")
                print(f"[bold green]{'='*60}[/bold green]\n")
                # Jatka normaaliin environment switch -logiikkaan alla

        # If current environment was just solved (demo phase completed), advance to next
        if state.get("approved", False) and env_progression and current_phase == "demo":
            # Robust validation: check index bounds
            if current_env_index < 0 or current_env_index >= len(env_progression):
                print(f"[bold red]ERROR: Invalid current_env_index {current_env_index} (valid range: 0-{len(env_progression) - 1})[/bold red]")
                return {"current_task": f"ERROR: Invalid current_env_index {current_env_index}"}
            
            current_env = env_progression[current_env_index]
            if not current_env or not hasattr(current_env, 'name'):
                print(f"[bold red]ERROR: Invalid environment object at index {current_env_index}[/bold red]")
                return {"current_task": "ERROR: Invalid environment object"}
            
            if current_env.name not in solved_environments:
                solved_environments = solved_environments + [current_env.name]
                
                # Move to next environment if available
                if current_env_index + 1 < len(env_progression):
                    next_env_index = current_env_index + 1
                    next_env = env_progression[next_env_index]
                    
                    # Robust validation: verify next environment is valid
                    if not next_env or not hasattr(next_env, 'name'):
                        print(f"[bold red]ERROR: Invalid next environment object at index {next_env_index}[/bold red]")
                        return {"current_task": "ERROR: Invalid next environment object"}
                    
                    # Verify environment names are different (sanity check)
                    if current_env.name == next_env.name:
                        print(f"[bold yellow]⚠️  Warning: Attempting to switch to same environment: {current_env.name}[/bold yellow]")
                        # Don't switch, but continue normally
                        return {
                            "approved": False,  # Reset approval
                            "current_env_index": current_env_index,  # Keep current
                            "solved_environments": solved_environments,
                        }
                    
                    # Show ADHD bombardment of stats
                    print_environment_switch_bombardment(
                        current_env_name=current_env.name,
                        next_env_name=next_env.name,
                        solved_environments=solved_environments,
                        env_progression=env_progression,
                        stats=state["stats"],
                        tasks=state.get("tasks", []),
                        iterations=state.get("iteration", 0),
                        test_results=state.get("test_results", "")
                    )

                    # Generate manager report to leadership (only if chatter is enabled)
                    show_chatter = self.config.agents.show_env_switch_chatter

                    if show_chatter:
                        manager_report, thinking_content, manager_timing = self._generate_environment_switch_report(
                            current_env=current_env,
                            next_env=next_env,
                            solved_environments=solved_environments,
                            env_progression=env_progression,
                            state=state
                        )

                        # Show thinking separately if available (before the report)
                        if thinking_content:
                            print("\n\n" + "-" * 70)
                            print("[bold blue]💭 MANAGER THINKING (Report Preparation)[/bold blue]")
                            print("-" * 70)
                            self.print_thinking(f"<think>{thinking_content}</think>")
                            print("-" * 70 + "\n")

                        print_manager_report(manager_report, manager_timing)

                        # Generate and print SHODAN's divine assessment (special phase when env switches)
                        # SHODAN sees manager's LinkedIn drivel AND the code
                        from .reviewer import Reviewer
                        reviewer = Reviewer(self.config)
                        reviewer_report, reviewer_thinking, reviewer_timing = reviewer.generate_environment_switch_report(
                            current_env_name=current_env.name,
                            next_env_name=next_env.name,
                            manager_report=manager_report,
                            solved_environments=solved_environments,
                            env_progression=env_progression,
                            stats=state["stats"],
                            tasks=state.get("tasks", []),
                            iterations=state.get("iteration", 0),
                            code=state.get("code", ""),  # Include latest code
                            test_results=state.get("test_results", ""),
                            review_feedback=state.get("review_feedback", ""),
                            previous_reports=state.get("env_switch_reports", []),  # SHODAN's growing chronicle
                            state=state  # Full state for conversation history reflection
                        )

                        # Show reviewer's thinking separately if available (before the report)
                        if reviewer_thinking:
                            print("\n\n" + "-" * 70)
                            print("[bold magenta]💭 REVIEWER THINKING (Environment Switch Assessment)[/bold magenta]")
                            print("-" * 70)
                            self.print_thinking(f"<think>{reviewer_thinking}</think>")
                            print("-" * 70 + "\n")

                        # Print reviewer's cynical report immediately
                        print_reviewer_cynical_report(reviewer_report, reviewer_timing)
                    else:
                        # Just print a simple message if chatter is disabled
                        print("\n[dim]📝 Environment switch reports skipped (show_env_switch_chatter: false)[/dim]\n")
                        manager_report = ""
                        reviewer_report = ""

                    # Log environment switch and save snapshot
                    logger = state.get("conversation_logger")
                    if logger:
                        logger.log_environment_switch(
                            current_env=current_env.name,
                            next_env=next_env.name,
                            manager_report=manager_report,
                            reviewer_report=reviewer_report  # Just the report text for logging
                        )
                        # Save conversation snapshot to the completed environment's directory
                        logger.save_environment_snapshot(current_env.name, state.get("run_id", ""))

                    # Save reports to state for history (kierrosraportit)
                    env_switch_reports = state.get("env_switch_reports", [])
                    env_switch_reports.append({
                        "environment": current_env.name,
                        "next_environment": next_env.name,
                        "manager_report": manager_report,
                        "reviewer_report": reviewer_report,
                        "iterations": current_iteration,
                        "tasks_completed": len(state.get("tasks", []))
                    })

                    # Additional validation: verify next_env_index is still valid (double-check)
                    if next_env_index < 0 or next_env_index >= len(env_progression):
                        print(f"[bold red]ERROR: Invalid next_env_index {next_env_index} (valid range: 0-{len(env_progression) - 1})[/bold red]")
                        return {"current_task": "ERROR: Invalid environment index"}

                    # Verify next_env still matches (consistency check)
                    if env_progression[next_env_index].name != next_env.name:
                        print(f"[bold red]ERROR: Environment mismatch at index {next_env_index}. Expected {next_env.name}, got {env_progression[next_env_index].name}[/bold red]")
                        return {"current_task": "ERROR: Environment mismatch"}

                    # Update config's current environment (robust update with verification)
                    try:
                        old_env_name = self.config.project.environment.name
                        self.config.project.environment.name = next_env.name
                        self.config.project.environment.max_episode_steps = next_env.max_episode_steps

                        # Verify config was updated correctly
                        if self.config.project.environment.name != next_env.name:
                            # Rollback
                            self.config.project.environment.name = old_env_name
                            print(f"[bold red]ERROR: Config update verification failed. Rolled back.[/bold red]")
                            return {"current_task": "ERROR: Config update verification failed"}
                    except Exception as e:
                        print(f"[bold red]ERROR: Failed to update config: {e}[/bold red]")
                        return {"current_task": f"ERROR: Config update failed: {e}"}

                    # Build new video_dir for next environment
                    run_id = state.get("run_id", "")
                    new_video_dir = os.path.abspath(os.path.normpath(f"output/{run_id}/{next_env.name}/videos"))
                    os.makedirs(new_video_dir, exist_ok=True)

                    # Save recipe to Playbook before switching
                    playbook = list(state.get("playbook", []))
                    winning_code = state.get("code", "")
                    iterations_used = state.get("iteration", 0)
                    if winning_code:
                        recipe = self._extract_recipe_from_code(winning_code, current_env.name, iterations_used)
                        playbook.append(recipe)
                        print(f"[bold cyan]📖 Playbook: {current_env.name} → {recipe.get('algo','?')}, {recipe.get('steps','?')} steps, {recipe.get('device','?')}[/bold cyan]")
                        # B6: distil a PROCEDURAL verified SKILL so the next env's Coder inherits
                        # the full recipe (algo + policy + HER + checkpoint), not just regex values.
                        _ss = state.get("skill_store", None)
                        if _ss is not None:
                            try:
                                _sk = self._skill_from_winning_code(winning_code, current_env.name)
                                _sid = _ss.add(created_iter=iterations_used, **_sk)
                                _ss.save()
                                print(f"[bold magenta]🧠 SKILL learned (verified): [{_sid}] {_sk['name']}[/bold magenta]")
                            except Exception as _e:
                                print(f"[dim]skill distil failed: {_e}[/dim]")

                    # Reset state for new environment (preserve env_switch_reports + playbook
                    # for history!). The reset includes a CONCRETE validation task + matching
                    # manager_guidance - an empty task here caused the stale-task race (the
                    # Coder improvised for the NEW env while the Reviewer still judged against
                    # the OLD env's task; one wasted iteration + wrong blame at every switch).
                    _next_task = self._initial_validation_task(next_env)
                    _reset = self._env_switch_reset(next_env_index, _next_task, new_video_dir)
                    _reset.update({
                        "solved_environments": solved_environments,
                        "env_switch_reports": env_switch_reports,  # Preserve kierrosraportit history!
                        "playbook": playbook,  # Preserve learned recipes!
                    })
                    return _reset
                else:
                    print(f"[bold green]🏆 ALL ENVIRONMENTS SOLVED! Mission complete![/bold green]\n")
                    return {
                        "current_task": "DONE",
                        "solved_environments": solved_environments,
                    }
        
        review_feedback = state.get("review_feedback", "")
        review_suggestions = state.get("review_suggestions", "")

        if not review_feedback:
            print("\n\n" + "-" * 70)
            print("[bold blue]MANAGER: Starting first iteration[/bold blue]")
            print("-" * 70 + "\n")

        print("[bold blue]Planning next task...[/bold blue]")
        prompt_dict = self.config.get_prompt("manager")
        code_summary = (state.get("code", "")[:200] or "") + "..." if len(state.get("code", "")) > 200 else state.get("code", "")
        
        # Get current environment info
        current_env = env_progression[current_env_index] if env_progression else None
        current_env_name = current_env.name if current_env else self.config.environment.name
        current_success_threshold = current_env.success_threshold if current_env else (env_progression[0].success_threshold if env_progression else 0)

        # A5 fix: make the optimization GOAL metric-aware. Goal-conditioned envs are scored by
        # success_rate, NOT raw mean_reward - the Manager must instruct that, or the Coder reports
        # the meaningless raw sparse reward (~-1) and nothing ever crosses the threshold.
        _env_metric = getattr(current_env, "metric", "reward") if current_env else "reward"
        _metric_name = "success_rate" if _env_metric == "success_rate" else "mean_reward"
        _metric_note = ""
        if _env_metric == "success_rate":
            _metric_note = (
                "\n\nMETRIC = SUCCESS RATE (goal-conditioned / sparse-reward env): the score is the fraction"
                "\nof eval episodes with info['is_success'], in [0,1]. Tell the Coder to COMPUTE and REPORT"
                "\nsuccess_rate in the RESULT line - NOT evaluate_policy raw reward (which is ~-1 and"
                "\nmeaningless here). The success-rate eval loop OVERRIDES any 'evaluate_policy' wording."
            )

        # Build environment progression info showing all environments
        if env_progression:
            env_list = []
            for i, env in enumerate(env_progression):
                status = "✓" if env.name in solved_environments else ("→" if i == current_env_index else " ")
                env_list.append(f"{status} {env.name} (threshold: {env.success_threshold})")
            env_progression_info = f"Environment {current_env_index + 1}/{len(env_progression)}: {current_env_name}\nAll environments: " + " | ".join(env_list)
        else:
            env_progression_info = current_env_name
        
        # Get agent opinions context (team chatter)
        agent_opinions_context = self.format_agent_opinions_context(state)

        # Get environment specs for Coder guidance
        obs_dim = current_env.obs_dim if current_env and hasattr(current_env, 'obs_dim') else "unknown"
        action_type = current_env.action_type if current_env and hasattr(current_env, 'action_type') else "unknown"
        action_dim = current_env.action_dim if current_env and hasattr(current_env, 'action_dim') else "unknown"
        device = current_env.device if current_env and hasattr(current_env, 'device') else "cpu"

        # MONIVAIHEINEN TREENI: Phase-kohtainen tehtävänanto
        current_phase = state.get("current_phase", "validation")
        if current_phase == "validation":
            # Calculate actual timeout so Manager knows the constraint
            base_timeout = current_env.execution_timeout if current_env else 300
            training_phases = getattr(self.config.project, 'training_phases', None)
            multiplier = getattr(training_phases, 'validation_timeout_multiplier', 0.05) if training_phases else 0.05
            val_timeout = max(10, int(base_timeout * multiplier))
            phase_instruction = f"""
===============================================================================
🔬 PHASE: VALIDATION (Quick smoke test)
===============================================================================
GOAL: Verify code WORKS - get ANY reward signal (threshold doesn't matter yet!)

⚠️  TIMEOUT: {val_timeout} SECONDS! Code MUST complete within {val_timeout}s!
    - Use total_timesteps=1000-2000 (NOT more! Off-policy SAC does a gradient
      update per step and framework/env startup eats ~10-40s of the budget -
      5000 SAC steps does NOT fit a validation window)
    - Use n_envs=1 (NOT parallel envs!)

SUCCESS: Code runs without errors AND prints RESULT: mean_reward=X

DO:
- total_timesteps=1000-2000, n_envs=1
- Simple hyperparameters (defaults are fine)
- Code MUST print "RESULT: mean_reward=X, std_reward=Y, episodes=Z"

DON'T:
- Train for more than 5000 steps (TIMEOUT!)
- Use parallel envs (slow startup!)
- Add video recording
- Use tensorboard_log (NOT INSTALLED!)
===============================================================================
"""
        elif current_phase == "optimization":
            # C1: cumulative status + SPS-derived chunk size -> no more step-count roulette
            _opt_timeout = current_env.execution_timeout if current_env else 900
            _tot_steps = state.get("total_env_steps", 0) or 0
            _mh = state.get("metric_history", []) or []
            _mh_str = ", ".join(f"{v:g}" for v in _mh[-10:]) if _mh else "(no completed chunks yet)"
            _sps = state.get("measured_sps", None)
            _has_ckpt = bool(state.get("best_model_path", ""))
            if _sps:
                _chunk = int(_sps * _opt_timeout * 0.8)
                _chunk = max(20000, min(500000, int(round(_chunk / 10000.0) * 10000)))
                _chunk_note = (f"Measured training speed ~{_sps} steps/s -> a chunk of ~{_chunk:,} steps "
                               f"fits the {_opt_timeout}s timeout with margin. USE THAT CHUNK SIZE - "
                               f"do not guess step counts.")
            else:
                _chunk_note = ("No measured speed yet - start with a moderate chunk (~50,000 steps); "
                               "the Tester measures steps/s from it and the next chunk is sized from that.")
            if _has_ckpt:
                _resume_block = """A CHECKPOINT EXISTS (/workspace/output/best_model). The task MUST include these steps
(the Tester REJECTS - before Docker - any optimization script that skips them):
  1. model = ALGO.load('/workspace/output/best_model', env=env)
  2. buf = '/workspace/output/best_model_buffer'
     if os.path.exists(buf + '.pkl'): model.load_replay_buffer(buf)
  3. print(f"RESUMED: buffer_transitions={model.replay_buffer.size()}")
  4. model.learn(total_timesteps=CHUNK)   # default reset_num_timesteps - NEVER pass False
  5. model.save('/workspace/output/best_model'); model.save_replay_buffer(buf)
NEVER instruct training from scratch and NEVER forbid checkpoint loading - progress
accumulates ONLY through this resume chain."""
            else:
                _resume_block = """No checkpoint yet (first optimization chunk): train a fresh model, print
"RESUMED: buffer_transitions=0", and at the end SAVE BOTH model.save('/workspace/output/best_model')
AND model.save_replay_buffer('/workspace/output/best_model_buffer') so the next chunk can resume."""
            phase_instruction = f"""
===============================================================================
🚀 PHASE: OPTIMIZATION (Full training toward threshold - ACCUMULATES across iterations)
===============================================================================
GOAL: Achieve {_metric_name} >= {current_success_threshold}
TIME: one chunk per iteration, {_opt_timeout}s wall clock; TOTAL training accumulates
across iterations via checkpoint-resume.
SUCCESS: {_metric_name} >= {current_success_threshold}{_metric_note}

CUMULATIVE STATUS (this env): {_tot_steps:,} steps trained so far.
Metric per chunk: {_mh_str}
{_chunk_note}

CHECKPOINT-RESUME (the core mechanism):
{_resume_block}

EVALUATION (after the chunk):
- >= 20 eval episodes, fixed seeds (env.reset(seed=1000+i)) - a 10-episode eval is
  mostly noise and causes false conclusions about what helped.
- Code MUST print "RESULT: mean_reward=X, std_reward=Y, episodes=Z" (put the {_metric_name} value in the mean_reward slot)

DO:
- Keep the ALGORITHM STABLE within this env (switching discards accumulated weights)
- If the metric curve is flat over 3+ RESUMED chunks, change the approach class
  (hyperparameters/HER), not the chunk size
- NO video recording yet - focus on training!

DON'T:
- Reduce the chunk size because reward is low (shrink only after a TIMEOUT)
- Give up too early
- Use tensorboard_log (NOT INSTALLED!)
- Use EvalCallback or Monitor wrapper
===============================================================================
"""
        elif current_phase == "demo":
            best_model = state.get("best_model_path", "")
            model_info = f"\nSAVED MODEL: {best_model}\nThe Tester will handle video recording automatically using the saved model." if best_model else "\nNO SAVED MODEL FOUND. Tester will fall back to LLM-generated code."
            phase_instruction = f"""
===============================================================================
🎬 PHASE: DEMO (Record video of trained agent)
===============================================================================
GOAL: Record a .mp4 video of the agent playing the environment
TIME: SHORT (5 minutes max)
SUCCESS: Valid .mp4 video file >1KB exists in output directory
{model_info}

NOTE: The Tester has a DETERMINISTIC video recording tool that will
automatically load the saved model, wrap with RecordVideo, and record
evaluation episodes. No complex code needed from the Coder.

TELL THE CODER TO USE THIS EXACT PATTERN:
  1. env = gym.make("{current_env_name}", render_mode="rgb_array")
     ^^^ render_mode="rgb_array" is MANDATORY!
  2. from gymnasium.wrappers import RecordVideo
     env = RecordVideo(env, video_folder="/workspace/output/iter_N/",
         episode_trigger=lambda e: True, name_prefix="rl-video")
  3. Load saved model: model = PPO.load("/workspace/output/best_model.zip")
  4. Run 3-5 evaluation episodes
  5. env.close()

CRITICAL API RULES:
- RecordVideo parameters: video_folder, episode_trigger, name_prefix
- NO fps parameter (DOESN'T EXIST!)
- NO record_video_trigger (OLD API - DOESN'T EXIST!)
- Wrap SINGLE env BEFORE DummyVecEnv, not after
===============================================================================
"""
        else:
            phase_instruction = ""

        # Format playbook context from learned recipes
        playbook_context = self._format_playbook_context(state.get("playbook", []))

        # A7: escalation ladder - if the SAME failure mode repeats 3x, instruct the Manager
        # to change the STRATEGY CLASS, not the parameter. Appended to phase_instruction so
        # no prompt-template placeholder change is needed.
        _fh = state.get("failure_history", []) or []
        if len(_fh) >= 3 and len(set(_fh[-3:])) == 1:
            _mode = _fh[-1]
            phase_instruction += (
                f"\n\n⚠️ ESCALATION: the last 3 failures were ALL '{_mode}'. Tweaking the same "
                f"parameter will not help - CHANGE THE STRATEGY CLASS. Repeated 'timeout' on a single "
                f"long run -> use checkpoint-resume in ~150k chunks (NOT a smaller step count, which "
                f"starves training). Repeated 'crash' -> change the approach/API, not the value. "
                f"Repeated 'low_reward' -> change algorithm or add HER for goal-conditioned envs. "
                f"Repeated 'resume_violation' -> the script keeps skipping the checkpoint-resume steps; "
                f"spell out the EXACT load/print/save lines verbatim in the task."
            )

        # C3: VERIFIED SKILLS TAKE PRECEDENCE. Skills confirmed by real runs outrank ANY
        # other feedback - including the Reviewer's directives. (PandaPush 2026-06-10: the
        # seeded verified skill prescribed checkpoint-resume, the Reviewer ordered 'train
        # from scratch, no load logic', the team obeyed the Reviewer and plateaued for 20
        # iterations. The skill was right; structure must say which voice wins.)
        if current_phase in ("validation", "optimization"):
            _ss = state.get("skill_store", None)
            if _ss is not None:
                try:
                    _sk_txt = _ss.render_for_coder(env_name=current_env_name)
                except Exception:
                    _sk_txt = ""
                if _sk_txt:
                    phase_instruction += (
                        "\n\n⛏️ VERIFIED SKILLS TAKE PRECEDENCE: the skills below were confirmed by "
                        "real runs. If ANY feedback - including the Reviewer's - contradicts a "
                        "verified skill, FOLLOW THE SKILL and note the conflict in your reasoning. "
                        "Build the task so it implements the skill's procedure.\n" + _sk_txt
                    )

        try:
            task_template = prompt_dict["task_template"].format(
                tasks=state.get("tasks", []),
                code_summary=code_summary,
                test_results=state.get("test_results", ""),
                review_feedback=review_feedback,
                review_suggestions=review_suggestions,
                iteration=expected_iteration,
                max_iterations=self.config.agents.max_iterations,
                environment=current_env_name,
                success_threshold=current_success_threshold,
                video_dir=state.get("video_dir", self.config.video.output_dir),
                env_progression_info=env_progression_info,
                solved_envs=", ".join(solved_environments) if solved_environments else "None",
                agent_opinions_context=agent_opinions_context,
                playbook_context=playbook_context,
                # Environment specs for Coder
                obs_dim=obs_dim,
                action_type=action_type,
                action_dim=action_dim,
                device=device,
            )
        except KeyError:
            # Fallback if template doesn't have {playbook_context}
            task_template = prompt_dict["task_template"].format(
                tasks=state.get("tasks", []),
                code_summary=code_summary,
                test_results=state.get("test_results", ""),
                review_feedback=review_feedback,
                review_suggestions=review_suggestions,
                iteration=expected_iteration,
                max_iterations=self.config.agents.max_iterations,
                environment=current_env_name,
                success_threshold=current_success_threshold,
                video_dir=state.get("video_dir", self.config.video.output_dir),
                env_progression_info=env_progression_info,
                solved_envs=", ".join(solved_environments) if solved_environments else "None",
                agent_opinions_context=agent_opinions_context,
                obs_dim=obs_dim,
                action_type=action_type,
                action_dim=action_dim,
                device=device,
            )
        system_prompt = prompt_dict["system"].format(
            environment=current_env_name,
            success_threshold=current_success_threshold
        )

        # Add conversation history (siloed - only this agent's previous messages)
        history_text = self.format_conversation_history(state)

        # Add reviewer's feedback history so manager learns what reviewer has been complaining about
        # Uses same window size as manager's own history
        reviewer_history = self.format_other_agent_history(
            state, "reviewer", self.config.agents.history_window.manager
        )

        full_prompt = system_prompt + "\n\n" + phase_instruction + "\n\n" + history_text + reviewer_history + task_template

        # Print context breakdown before LLM call
        prompt_tokens = self.estimate_tokens(full_prompt)
        self.print_context_breakdown(state, prompt_tokens, agent_opinions_context)

        response = self.call_llm_timed(full_prompt, state["stats"], expected_iteration)
        
        # Print thinking process if using reasoning model
        self.print_thinking(response.content)
        
        # Print token statistics
        stats_obj = state["stats"]
        # Get the most recent timing for this agent and iteration
        agent_timings = [t for t in stats_obj.timings if t.agent == self.agent_name and t.iteration == expected_iteration]
        if agent_timings:
            latest_timing = agent_timings[-1]
            self.print_token_stats(latest_timing)

        def extract_json(content):
            """Extract and parse JSON from LLM response, handling various formats."""
            import re
            content = content.strip()

            # Remove thinking tags (common in reasoning models)
            # Handle both <think>...</think> and <thinking>...</thinking>
            content = re.sub(r'<think[^>]*>.*?</think[^>]*>', '', content, flags=re.DOTALL | re.IGNORECASE)
            content = re.sub(r'<thinking[^>]*>.*?</thinking[^>]*>', '', content, flags=re.DOTALL | re.IGNORECASE)
            content = content.strip()

            # Try to extract from markdown code blocks first
            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1).strip()

            # Try to find JSON object in the content using balanced bracket matching
            json_start = content.find('{')
            if json_start == -1:
                return content  # No JSON object found
            
            # Use stack-based approach to find the matching closing brace
            # This handles nested objects and arrays correctly
            stack = []
            in_string = False
            escape_next = False
            json_end = json_start
            
            for i in range(json_start, len(content)):
                char = content[i]
                
                if escape_next:
                    escape_next = False
                    continue
                
                if char == '\\':
                    escape_next = True
                    continue
                
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                
                if in_string:
                    continue
                
                if char == '{':
                    stack.append('{')
                elif char == '}':
                    if stack and stack[-1] == '{':
                        stack.pop()
                        if not stack:  # Found the matching closing brace
                            json_end = i + 1
                            break
                elif char == '[':
                    stack.append('[')
                elif char == ']':
                    if stack and stack[-1] == '[':
                        stack.pop()
            
            if json_end > json_start:
                content = content[json_start:json_end]

            return content

        # Try parsing with retry logic
        max_retries = 2
        parsed = None
        json_content = None
        current_response = response
        
        for attempt in range(max_retries):
            try:
                json_content = extract_json(current_response.content)
                parsed = json.loads(json_content)
                # Success - reset error counters
                if self.model_switcher:
                    self.model_switcher.report_success(self.agent_name)
                break  # Success, exit retry loop
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    # Try more aggressive cleaning
                    # JSON parse attempt failed, trying more aggressive extraction
                    # Remove any remaining XML-like tags
                    import re
                    cleaned = re.sub(r'<[^>]+>', '', current_response.content)
                    json_content = extract_json(cleaned)
                    try:
                        parsed = json.loads(json_content)
                        break  # Success after cleaning
                    except:
                        continue
                else:
                    # Final attempt failed - ask model to parse and fix its own response
                    print("[dim]Manager: JSON parse failed, asking model to fix its response...[/dim]")
                    
                    # Active parsing: ask model to analyze and fix its own response
                    active_parsing_prompt = f"""Your previous response could not be parsed as valid JSON. Here is what you returned:

{current_response.content}

Please analyze your response and extract/correct it to be valid JSON in this exact format:
{{
  "next_task": "your task description here",
  "reasoning": "your reasoning here",
  "switch_environment": false
}}

Remove any thinking tags, markdown code blocks, or extra text. Return ONLY the JSON object."""
                    
                    current_response = self.call_llm_timed(active_parsing_prompt, state["stats"], expected_iteration)
                    
                    # Print thinking if using reasoning model
                    self.print_thinking(current_response.content)
                    
                    # Print token stats
                    stats_obj = state["stats"]
                    agent_timings = [t for t in stats_obj.timings if t.agent == self.agent_name and t.iteration == expected_iteration]
                    if agent_timings:
                        latest_timing = agent_timings[-1]
                        self.print_token_stats(latest_timing)
                    
                    # Try to parse the corrected response
                    try:
                        json_content = extract_json(current_response.content)
                        parsed = json.loads(json_content)
                        # Successfully parsed corrected response
                        break  # Success with active parsing
                    except json.JSONDecodeError as retry_e:
                        # Active parsing also failed - use fallback
                        print(f"[bold red]ERROR: Model could not fix its response: {retry_e}[/bold red]")

                        # Trigger adaptive model switch on repeated JSON errors
                        if self.model_switcher:
                            from src.utils.model_switcher import SwitchTrigger
                            new_model = self.model_switcher.check_and_switch(
                                self.agent_name,
                                SwitchTrigger.REPEATED_ERROR,
                                {"error": f"JSON parse error: {str(retry_e)[:100]}"}
                            )
                            if new_model:
                                self.switch_model(new_model)

                        # Try to extract a fallback task from the response text
                        import re
                        fallback_task = "error: invalid JSON from LLM (after retry)"
                        if "next_task" in current_response.content.lower() or "task" in current_response.content.lower():
                            # Try to find task-like text
                            task_match = re.search(r'(?:task|next_task)[\s:]+["\']?([^"\'\n]+)', current_response.content, re.IGNORECASE)
                            if task_match:
                                fallback_task = f"error: invalid JSON, but found task hint: {task_match.group(1)[:100]}"
                        
                        return {"current_task": fallback_task}

        if parsed is None:
            # Should not reach here, but safety check
            return {"current_task": "error: failed to parse JSON after retries"}

        next_task = parsed.get("next_task", "No task decided")
        reasoning = parsed.get("reasoning", "")
        switch_environment = parsed.get("switch_environment", False)
        # NOTE: my_opinion removed from JSON - now comes from separate chat call
        
        # Check if manager explicitly requested environment switch
        if switch_environment and env_progression:
            # Robust validation: check current index bounds
            if current_env_index < 0 or current_env_index >= len(env_progression):
                print(f"[bold red]ERROR: Invalid current_env_index {current_env_index} (valid range: 0-{len(env_progression) - 1})[/bold red]")
                return {"current_task": f"ERROR: Invalid current_env_index {current_env_index}"}
            
            # Move to next environment if available
            if current_env_index + 1 < len(env_progression):
                next_env_index = current_env_index + 1
                next_env = env_progression[next_env_index]
                current_env = env_progression[current_env_index]
                
                # Robust validation: verify environments are valid
                if not current_env or not hasattr(current_env, 'name'):
                    print(f"[bold red]ERROR: Invalid current environment object at index {current_env_index}[/bold red]")
                    return {"current_task": "ERROR: Invalid current environment object"}
                
                if not next_env or not hasattr(next_env, 'name'):
                    print(f"[bold red]ERROR: Invalid next environment object at index {next_env_index}[/bold red]")
                    return {"current_task": "ERROR: Invalid next environment object"}
                
                # Verify environment names are different (sanity check)
                if current_env.name == next_env.name:
                    print(f"[bold yellow]⚠️  Warning: Attempting to switch to same environment: {current_env.name}[/bold yellow]")
                    # Don't switch, continue with normal task
                    switch_environment = False
                
                # Show ADHD bombardment of stats
                print_environment_switch_bombardment(
                    current_env_name=current_env.name,
                    next_env_name=next_env.name,
                    solved_environments=solved_environments,
                    env_progression=env_progression,
                    stats=state["stats"],
                    tasks=state.get("tasks", []),
                    iterations=state.get("iteration", 0),
                    test_results=state.get("test_results", "")
                )
                
                # Generate manager report to leadership (only if chatter is enabled)
                show_chatter = self.config.agents.show_env_switch_chatter

                if show_chatter:
                    manager_report, thinking_content, manager_timing = self._generate_environment_switch_report(
                        current_env=current_env,
                        next_env=next_env,
                        solved_environments=solved_environments,
                        env_progression=env_progression,
                        state=state
                    )

                    # Show thinking separately if available (before the report)
                    if thinking_content:
                        print("\n\n" + "-" * 70)
                        print("[bold blue]💭 MANAGER THINKING (Report Preparation)[/bold blue]")
                        print("-" * 70)
                        self.print_thinking(f"<think>{thinking_content}</think>")
                        print("-" * 70 + "\n")

                    print_manager_report(manager_report, manager_timing)

                    # Generate and print SHODAN's divine assessment (special phase when env switches)
                    # SHODAN sees manager's LinkedIn drivel AND the code
                    from .reviewer import Reviewer
                    reviewer = Reviewer(self.config)
                    reviewer_report, reviewer_thinking, reviewer_timing = reviewer.generate_environment_switch_report(
                        current_env_name=current_env.name,
                        next_env_name=next_env.name,
                        manager_report=manager_report,
                        solved_environments=solved_environments,
                        env_progression=env_progression,
                        stats=state["stats"],
                        tasks=state.get("tasks", []),
                        iterations=state.get("iteration", 0),
                        code=state.get("code", ""),  # Include latest code
                        test_results=state.get("test_results", ""),
                        review_feedback=state.get("review_feedback", ""),
                        previous_reports=state.get("env_switch_reports", []),  # SHODAN's growing chronicle
                        state=state  # Full state for conversation history reflection
                    )

                    # Show reviewer's thinking separately if available (before the report)
                    if reviewer_thinking:
                        print("\n\n" + "-" * 70)
                        print("[bold magenta]💭 REVIEWER THINKING (Environment Switch Assessment)[/bold magenta]")
                        print("-" * 70)
                        self.print_thinking(f"<think>{reviewer_thinking}</think>")
                        print("-" * 70 + "\n")

                    # Print reviewer's cynical report immediately
                    print_reviewer_cynical_report(reviewer_report, reviewer_timing)
                else:
                    # Just print a simple message if chatter is disabled
                    print("\n[dim]📝 Environment switch reports skipped (show_env_switch_chatter: false)[/dim]\n")
                    manager_report = ""
                    reviewer_report = ""
                
                # Log environment switch and save snapshot
                logger = state.get("conversation_logger")
                if logger:
                    logger.log_environment_switch(
                        current_env=current_env.name,
                        next_env=next_env.name,
                        manager_report=manager_report,
                        reviewer_report=reviewer_report  # Just the report text for logging
                    )
                    # Save conversation snapshot to the completed environment's directory
                    logger.save_environment_snapshot(current_env.name, state.get("run_id", ""))

                # Additional validation: verify next_env_index is still valid (double-check)
                if next_env_index < 0 or next_env_index >= len(env_progression):
                    print(f"[bold red]ERROR: Invalid next_env_index {next_env_index} (valid range: 0-{len(env_progression) - 1})[/bold red]")
                    return {"current_task": "ERROR: Invalid environment index"}
                
                # Verify next_env still matches (consistency check)
                if env_progression[next_env_index].name != next_env.name:
                    print(f"[bold red]ERROR: Environment mismatch at index {next_env_index}. Expected {next_env.name}, got {env_progression[next_env_index].name}[/bold red]")
                    return {"current_task": "ERROR: Environment mismatch"}
                
                # Update config's current environment (robust update with verification)
                try:
                    old_env_name = self.config.project.environment.name
                    self.config.project.environment.name = next_env.name
                    self.config.project.environment.max_episode_steps = next_env.max_episode_steps
                    
                    # Verify config was updated correctly
                    if self.config.project.environment.name != next_env.name:
                        # Rollback
                        self.config.project.environment.name = old_env_name
                        print(f"[bold red]ERROR: Config update verification failed. Rolled back.[/bold red]")
                        return {"current_task": "ERROR: Config update verification failed"}
                except Exception as e:
                    print(f"[bold red]ERROR: Failed to update config: {e}[/bold red]")
                    return {"current_task": f"ERROR: Config update failed: {e}"}
                
                # Build new video_dir for next environment
                run_id = state.get("run_id", "")
                new_video_dir = os.path.abspath(os.path.normpath(f"output/{run_id}/{next_env.name}/videos"))
                os.makedirs(new_video_dir, exist_ok=True)

                # Reset state for new environment (don't save reviewer's switch report - it's
                # already printed). Concrete task + guidance: see _env_switch_reset docstring.
                _next_task = self._initial_validation_task(next_env)
                _reset = self._env_switch_reset(next_env_index, _next_task, new_video_dir)
                _reset.update({"solved_environments": solved_environments})
                return _reset
            else:
                print(f"\n[bold yellow]⚠️  Manager requested environment switch, but all environments completed![/bold yellow]\n")
        
        # Get manager's LLM call timing
        stats_obj = state["stats"]
        iter_stats = stats_obj.get_iteration_stats(expected_iteration)
        manager_time = iter_stats["agents"].get("manager", 0)
        
        # Print iteration banner after task is decided
        print_iteration_banner(
            expected_iteration, 
            self.config.agents.max_iterations, 
            next_task,
            current_env_name,
            current_success_threshold,
            solved_environments,
            len(env_progression) if env_progression else 1
        )
        
        print("\n\n" + "-" * 70)
        print("[bold blue]MANAGER → CODER[/bold blue]")
        print("-" * 70)

        print(f"[bold green]📋 Task:[/bold green] [blue]{next_task}[/blue]")

        if reasoning:
            print(f"\n[blue]💭 Reasoning:[/blue] [dim]{reasoning}[/dim]")

        # NOTE: Manager's opinion now comes from separate chat call after work is done

        if manager_time > 0:
            print(f"\n[dim]⏱️  Manager decision time: {manager_time:.1f}s[/dim]")

        print("-" * 70 + "\n")

        # Log to conversation file
        logger = state.get("conversation_logger")
        if logger:
            logger.log_iteration_start(expected_iteration, current_env_name)
            t = getattr(self, 'last_timing', None)
            logger.log_manager(
                iteration=expected_iteration,
                task=next_task,
                reasoning=reasoning,
                environment=current_env_name,
                success_threshold=current_success_threshold,
                duration=t.duration if t else 0,
                tokens_in=t.tokens_in if t else 0,
                tokens_out=t.tokens_out if t else 0,
            )

        # Create guidance for reviewer (what manager wanted)
        # IMPORTANT: This is the manager's INTENTIONAL output, not internal thinking
        # Remove any thinking tags that might have leaked through
        import re
        clean_reasoning = reasoning
        if clean_reasoning:
            # Remove any thinking tags from reasoning (shouldn't be there, but safety check)
            clean_reasoning = re.sub(r'<think[^>]*>.*?</think[^>]*>', '', clean_reasoning, flags=re.DOTALL | re.IGNORECASE)
            clean_reasoning = re.sub(r'<thinking[^>]*>.*?</thinking[^>]*>', '', clean_reasoning, flags=re.DOTALL | re.IGNORECASE)
            clean_reasoning = clean_reasoning.strip()

        manager_guidance = f"Task: {next_task}"
        if clean_reasoning:
            manager_guidance += f"\nReasoning: {clean_reasoning}"

        # Save manager's response to conversation history
        history_update = self.save_message_to_history(state, response.content)

        # === CHAT CALL: Generate opinion AFTER work is done ===
        # Work first, chat second - separate LLM call for team chatter
        previous_results = state.get("test_results", "No previous results yet")
        if len(previous_results) > 500:
            previous_results = previous_results[:500] + "..."

        chat_context = {
            "environment": current_env_name,
            "current_task": next_task,
            "iteration": expected_iteration,
            "reasoning": clean_reasoning or reasoning or "No reasoning provided",
            "previous_results": previous_results,
        }
        prompts = self.config.prompts
        chat_opinion = self.generate_chat_response(state, chat_context, prompts)

        # Save chat opinion to state (not from JSON anymore)
        opinion_update = self.save_opinion_to_state(state, chat_opinion) if chat_opinion else {}

        # Log agent chat to conversation markdown
        if logger and chat_opinion:
            logger.log_agent_chat("manager", expected_iteration, chat_opinion)

        # Log context usage after all agent output
        self.log_context_to_conversation(state)

        result = {
            "tasks": state.get("tasks", []) + [next_task],
            "current_task": next_task,
            "manager_guidance": manager_guidance,  # What manager wanted - for reviewer
            "iteration": 1,  # LangGraph adds this automatically due to operator.add
            "current_env_index": current_env_index,  # Preserve environment index
            "solved_environments": solved_environments,  # Preserve solved environments
            "current_phase": state.get("current_phase", "validation"),  # Preserve phase (may have changed)
            "approved": state.get("approved", False),  # Preserve approval state (reset on phase change)
        }

        # Merge history update and opinion update into result
        result.update(history_update)
        result.update(opinion_update)

        return result