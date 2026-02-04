import json
from .base import BaseAgent
from rich import print
from src.utils.banners import print_environment_switch_bombardment, print_manager_report, print_iteration_banner, print_reviewer_cynical_report

class Manager(BaseAgent):
    def __init__(self, config, model_switcher=None):
        super().__init__(config, "manager", model_switcher=model_switcher)
    
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
        env_progression = self.config.environment_progression

        # MONIVAIHEINEN TREENI: Tarkista ja vaihda vaihe kun approved=True
        current_phase = state.get("current_phase", "validation")

        if state.get("approved", False) and env_progression:
            # Vaihelogiikka: validation -> optimization -> demo -> seuraava env
            if current_phase == "validation":
                # Validation OK -> siirry optimization-vaiheeseen
                print(f"\n[bold green]{'═'*60}[/bold green]")
                print(f"[bold green]✅ VALIDATION PASSED! Code works, reward received.[/bold green]")
                print(f"[bold cyan]➡️  Moving to OPTIMIZATION phase (full training)[/bold cyan]")
                print(f"[bold green]{'═'*60}[/bold green]\n")
                return {
                    "current_phase": "optimization",
                    "approved": False,  # Reset for optimization
                    "iteration": 1,
                }
            elif current_phase == "optimization":
                # Optimization OK -> siirry demo-vaiheeseen
                print(f"\n[bold green]{'═'*60}[/bold green]")
                print(f"[bold green]✅ OPTIMIZATION COMPLETE! Threshold achieved.[/bold green]")
                print(f"[bold cyan]➡️  Moving to DEMO phase (record best model video)[/bold cyan]")
                print(f"[bold green]{'═'*60}[/bold green]\n")
                return {
                    "current_phase": "demo",
                    "approved": False,  # Reset for demo
                    "iteration": 1,
                }
            elif current_phase == "demo":
                # Demo OK -> siirry seuraavaan ympäristöön (normaali env switch)
                print(f"\n[bold green]{'═'*60}[/bold green]")
                print(f"[bold green]✅ DEMO COMPLETE! Environment fully solved![/bold green]")
                print(f"[bold green]{'═'*60}[/bold green]\n")
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
                            print("\n\n" + "─" * 70)
                            print("[bold blue]💭 MANAGER THINKING (Report Preparation)[/bold blue]")
                            print("─" * 70)
                            self.print_thinking(f"<think>{thinking_content}</think>")
                            print("─" * 70 + "\n")

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
                            print("\n\n" + "─" * 70)
                            print("[bold magenta]💭 REVIEWER THINKING (Environment Switch Assessment)[/bold magenta]")
                            print("─" * 70)
                            self.print_thinking(f"<think>{reviewer_thinking}</think>")
                            print("─" * 70 + "\n")

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
                    import os
                    run_id = state.get("run_id", "")
                    new_video_dir = os.path.abspath(os.path.normpath(f"output/{run_id}/{next_env.name}/videos"))
                    os.makedirs(new_video_dir, exist_ok=True)

                    # Reset state for new environment (preserve env_switch_reports for history!)
                    return {
                        "current_env_index": next_env_index,
                        "solved_environments": solved_environments,
                        "env_switch_reports": env_switch_reports,  # Preserve kierrosraportit history!
                        "approved": False,  # Reset approval for new environment
                        "tasks": [],  # Start fresh tasks for new environment
                        "code": "",  # Reset code
                        "test_results": "",
                        "review_feedback": "",
                        "review_suggestions": "",
                        "current_task": "",
                        "video_dir": new_video_dir,  # Env-specific video directory
                        "iteration": 1,  # LangGraph adds this automatically due to operator.add
                    }
                else:
                    print(f"[bold green]🏆 ALL ENVIRONMENTS SOLVED! Mission complete![/bold green]\n")
                    return {
                        "current_task": "DONE",
                        "solved_environments": solved_environments,
                    }
        
        review_feedback = state.get("review_feedback", "")
        review_suggestions = state.get("review_suggestions", "")

        if not review_feedback:
            print("\n\n" + "─" * 70)
            print("[bold blue]MANAGER: Starting first iteration[/bold blue]")
            print("─" * 70 + "\n")

        print("[bold blue]Planning next task...[/bold blue]")
        prompt_dict = self.config.get_prompt("manager")
        code_summary = (state.get("code", "")[:200] or "") + "..." if len(state.get("code", "")) > 200 else state.get("code", "")
        
        # Get current environment info
        current_env = env_progression[current_env_index] if env_progression else None
        current_env_name = current_env.name if current_env else self.config.environment.name
        current_success_threshold = current_env.success_threshold if current_env else (env_progression[0].success_threshold if env_progression else 0)
        
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
            phase_instruction = f"""
═══════════════════════════════════════════════════════════════════════════════
🔬 PHASE: VALIDATION (Quick smoke test)
═══════════════════════════════════════════════════════════════════════════════
GOAL: Verify code WORKS - get ANY reward signal (threshold doesn't matter yet!)
TIME: SHORT run only (~50k timesteps max, 5% of normal timeout)
SUCCESS: Code runs without errors AND produces some reward (any number)

DO:
- Write minimal working code
- Use simple hyperparameters
- Focus on correctness, not performance
- NO video recording yet!

DON'T:
- Optimize hyperparameters
- Train for long
- Add video recording
═══════════════════════════════════════════════════════════════════════════════
"""
        elif current_phase == "optimization":
            phase_instruction = f"""
═══════════════════════════════════════════════════════════════════════════════
🚀 PHASE: OPTIMIZATION (Full training toward threshold)
═══════════════════════════════════════════════════════════════════════════════
GOAL: Achieve mean_reward >= {current_success_threshold}
TIME: FULL timeout available - use it wisely
SUCCESS: mean_reward >= {current_success_threshold}

DO:
- Tune hyperparameters aggressively
- Increase timesteps if needed
- Try different algorithms if stuck
- Monitor learning curves
- NO video recording yet - focus on training!

DON'T:
- Give up too early
- Waste time on video setup
═══════════════════════════════════════════════════════════════════════════════
"""
        elif current_phase == "demo":
            best_model = state.get("best_model_path", "")
            phase_instruction = f"""
═══════════════════════════════════════════════════════════════════════════════
🎬 PHASE: DEMO (Record video of best model)
═══════════════════════════════════════════════════════════════════════════════
GOAL: Generate demo video showing trained agent in action
TIME: SHORT (5 minutes max)
SUCCESS: Valid video file exists showing good performance

DO:
- Load the trained model (from optimization phase)
- Use RecordVideo wrapper
- Run evaluation episodes (5-10)
- Save video to output directory

DON'T:
- Train more - model is ready
- Overthink - just record the agent playing
═══════════════════════════════════════════════════════════════════════════════
"""
        else:
            phase_instruction = ""

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
            # Environment specs for Coder
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
        my_opinion = parsed.get("my_opinion", "")  # Manager's personal take
        
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
                        print("\n\n" + "─" * 70)
                        print("[bold blue]💭 MANAGER THINKING (Report Preparation)[/bold blue]")
                        print("─" * 70)
                        self.print_thinking(f"<think>{thinking_content}</think>")
                        print("─" * 70 + "\n")

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
                        print("\n\n" + "─" * 70)
                        print("[bold magenta]💭 REVIEWER THINKING (Environment Switch Assessment)[/bold magenta]")
                        print("─" * 70)
                        self.print_thinking(f"<think>{reviewer_thinking}</think>")
                        print("─" * 70 + "\n")

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
                import os
                run_id = state.get("run_id", "")
                new_video_dir = os.path.abspath(os.path.normpath(f"output/{run_id}/{next_env.name}/videos"))
                os.makedirs(new_video_dir, exist_ok=True)

                # Reset state for new environment (don't save reviewer's switch report - it's already printed)
                return {
                    "current_env_index": next_env_index,
                    "solved_environments": solved_environments,
                    "approved": False,  # Reset approval for new environment
                    "tasks": [],  # Start fresh tasks for new environment
                    "code": "",  # Reset code
                    "test_results": "",
                    "review_feedback": "",
                    "review_suggestions": "",
                    "current_task": "",
                    "video_dir": new_video_dir,  # Env-specific video directory
                    "iteration": 1,  # LangGraph adds this automatically due to operator.add
                }
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
        
        print("\n\n" + "─" * 70)
        print("[bold blue]MANAGER → CODER[/bold blue]")
        print("─" * 70)

        print(f"[bold green]📋 Task:[/bold green] [blue]{next_task}[/blue]")

        if reasoning:
            print(f"\n[blue]💭 Reasoning:[/blue] [dim]{reasoning}[/dim]")

        # Print manager's opinion if provided (team chatter)
        if my_opinion:
            print(f"\n[blue]💬 Manager's take:[/blue] [blue]{my_opinion}[/blue]")

        if manager_time > 0:
            print(f"\n[dim]⏱️  Manager decision time: {manager_time:.1f}s[/dim]")

        print("─" * 70 + "\n")

        # Log to conversation file
        logger = state.get("conversation_logger")
        if logger:
            logger.log_iteration_start(expected_iteration, current_env_name)
            logger.log_manager(
                iteration=expected_iteration,
                task=next_task,
                reasoning=reasoning,
                environment=current_env_name,
                success_threshold=current_success_threshold
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

        # Save manager's opinion to state for team chatter
        opinion_update = self.save_opinion_to_state(state, my_opinion) if my_opinion else {}

        result = {
            "tasks": state.get("tasks", []) + [next_task],
            "current_task": next_task,
            "manager_guidance": manager_guidance,  # What manager wanted - for reviewer
            "iteration": 1,  # LangGraph adds this automatically due to operator.add
            "current_env_index": current_env_index,  # Preserve environment index
            "solved_environments": solved_environments,  # Preserve solved environments
        }

        # Merge history update and opinion update into result
        result.update(history_update)
        result.update(opinion_update)

        return result