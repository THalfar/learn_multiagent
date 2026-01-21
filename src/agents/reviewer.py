import json
from .base import BaseAgent
from rich import print

class Reviewer(BaseAgent):
    def __init__(self, config):
        super().__init__(config, "reviewer")

    def __call__(self, state: dict) -> dict:
        from src.utils.banners import print_agent_transition
        print_agent_transition("tester", "reviewer")
        
        # Show what reviewer sees (the smartest agent sees everything)
        print("\n" + "─" * 70)
        print("[bold magenta]REVIEWER: Analyzing full picture[/bold magenta]")
        print("─" * 70)
        print("[dim]Reviewer sees: Code + Tester's analysis + Manager's guidance[/dim]")
        
        manager_guidance = state.get("manager_guidance", "")
        if manager_guidance:
            print(f"\n[blue]Manager's intent:[/blue] {manager_guidance}")
        
        test_results = state.get("test_results", "")
        if test_results:
            print(f"\n[yellow]Tester's analysis (from outputs only):[/yellow] {test_results}")

        # Check if tester responded to SHODAN's previous request
        tester_response = state.get("tester_reviewer_response", "")
        if tester_response:
            print(f"\n[cyan]📬 Tester's response to your request:[/cyan] {tester_response}")
        
        print("\n[bold magenta]Reviewing code, tester analysis, and manager intent...[/bold magenta]")
        print("─" * 70 + "\n")
        
        # Get success threshold from current environment
        env_progression = self.config.environment_progression
        current_env_index = state.get("current_env_index", 0)
        current_env = env_progression[current_env_index] if env_progression and current_env_index < len(env_progression) else None
        success_threshold = current_env.success_threshold if current_env else (env_progression[0].success_threshold if env_progression else 0)
        
        # Get manager's guidance (what manager wanted)
        manager_guidance = state.get("manager_guidance", "No specific guidance from manager")
        
        # Reviewer gets code, manager's guidance, and tester's analysis - no raw execution output
        # This simulates real work: senior reviews code and reports, not raw logs
        code = state.get("code", "")

        # Get agent opinions context (insect chatter for SHODAN)
        agent_opinions_context = self.format_agent_opinions_context(state)

        # Get tester's response to previous SHODAN request
        tester_response = state.get("tester_reviewer_response", "No specific response")

        prompt_dict = self.config.get_prompt("reviewer")
        task_template = prompt_dict["task_template"].format(
            manager_guidance=manager_guidance,
            code=code,
            test_results=test_results,
            success_threshold=success_threshold,
            video_dir=state.get("video_dir", "output/videos"),
            agent_opinions_context=agent_opinions_context,
            tester_response=tester_response,
        )
        system_prompt = prompt_dict["system"].format(
            success_threshold=success_threshold,
            video_dir=state.get("video_dir", "output/videos")
        )

        # Add conversation history (siloed - only this agent's previous messages)
        history_text = self.format_conversation_history(state)

        full_prompt = system_prompt + "\n\n" + history_text + task_template

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
            import re
            
            if not content:
                return "{}"
                
            content = content.strip()

            # Remove thinking tags (common in reasoning models)
            # Handle both <think>...</think> and <thinking>...</thinking>
            # We keep a copy of content with tags in case we need to fallback
            content_clean = re.sub(r'<think[^>]*>.*?</think[^>]*>', '', content, flags=re.DOTALL | re.IGNORECASE)
            content_clean = re.sub(r'<thinking[^>]*>.*?</thinking[^>]*>', '', content_clean, flags=re.DOTALL | re.IGNORECASE)
            
            # If cleaning removed everything, revert to original content (maybe JSON is inside tags or mixed)
            if not content_clean.strip():
                content_to_use = content
            else:
                content_to_use = content_clean

            # Try to extract from markdown code blocks first
            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content_to_use, re.DOTALL)
            if json_match:
                content_to_use = json_match.group(1).strip()

            # Try to find JSON object in the content using balanced bracket matching
            json_start = content_to_use.find('{')
            if json_start == -1:
                # Try finding in original content if we haven't already
                if content_to_use != content:
                    json_start = content.find('{')
                    if json_start != -1:
                        content_to_use = content
                    else:
                        return content_to_use # Return as is, let json.loads fail naturally or handle later
                else:
                    return content_to_use
            
            # Use stack-based approach to find the matching closing brace
            # This handles nested objects and arrays correctly
            stack = []
            in_string = False
            escape_next = False
            json_end = json_start
            
            for i in range(json_start, len(content_to_use)):
                char = content_to_use[i]
                
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
                content_to_use = content_to_use[json_start:json_end]

            return content_to_use

        # Try parsing with retry logic
        max_retries = 2
        parsed = None
        json_content = None
        current_response = response
        
        for attempt in range(max_retries):
            try:
                json_content = extract_json(current_response.content)
                parsed = json.loads(json_content)
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
                    print("[dim]Reviewer: JSON parse failed, asking model to fix its response...[/dim]")
                    
                    # Active parsing: ask model to analyze and fix its own response
                    active_parsing_prompt = f"""Your previous response could not be parsed as valid JSON. Here is what you returned:

{current_response.content}

Please analyze your response and extract/correct it to be valid JSON in this exact format:
{{
  "approved": true or false,
  "feedback": "your feedback text here",
  "suggestions": ["suggestion 1", "suggestion 2"]
}}

Remove any thinking tags, markdown code blocks, or extra text. Return ONLY the JSON object."""
                    
                    current_response = self.call_llm_timed(active_parsing_prompt, state["stats"], state.get("iteration", 0))
                    
                    # Print thinking if using reasoning model
                    self.print_thinking(current_response.content)
                    
                    # Print token stats
                    stats_obj = state["stats"]
                    iteration = state.get("iteration", 0)
                    agent_timings = [t for t in stats_obj.timings if t.agent == self.agent_name and t.iteration == iteration]
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
                        
                        # Return fallback response
                        return {
                            "review_feedback": "Parse error: Could not extract JSON from LLM response even after retry. Please check the code manually.",
                            "review_suggestions": "No specific suggestions",
                            "approved": False
                        }

        if parsed is None:
            # Should not reach here, but safety check
            return {
                "review_feedback": "Parse error: Failed to parse JSON after retries",
                "review_suggestions": "No specific suggestions",
                "approved": False
            }

        approved = parsed.get("approved", False)
        feedback = parsed.get("feedback", "No feedback")
        suggestions = parsed.get("suggestions", [])
        tester_instruction = parsed.get("tester_instruction", None)
        my_opinion = parsed.get("my_opinion", "")  # SHODAN's musings

        # Remove any thinking tags from feedback and suggestions (shouldn't be there, but safety check)
        import re
        feedback = re.sub(r'<think[^>]*>.*?</think[^>]*>', '', feedback, flags=re.DOTALL | re.IGNORECASE)
        feedback = re.sub(r'<thinking[^>]*>.*?</thinking[^>]*>', '', feedback, flags=re.DOTALL | re.IGNORECASE)
        feedback = feedback.strip()

        # Clean my_opinion too
        if my_opinion:
            my_opinion = re.sub(r'<think[^>]*>.*?</think[^>]*>', '', my_opinion, flags=re.DOTALL | re.IGNORECASE)
            my_opinion = re.sub(r'<thinking[^>]*>.*?</thinking[^>]*>', '', my_opinion, flags=re.DOTALL | re.IGNORECASE)
            my_opinion = my_opinion.strip()
        
        # Clean suggestions list
        clean_suggestions = []
        for suggestion in suggestions:
            if isinstance(suggestion, str):
                clean_suggestion = re.sub(r'<think[^>]*>.*?</think[^>]*>', '', suggestion, flags=re.DOTALL | re.IGNORECASE)
                clean_suggestion = re.sub(r'<thinking[^>]*>.*?</thinking[^>]*>', '', clean_suggestion, flags=re.DOTALL | re.IGNORECASE)
                clean_suggestions.append(clean_suggestion.strip())
            else:
                clean_suggestions.append(suggestion)
        suggestions = clean_suggestions

        # Clean tester_instruction if present
        if tester_instruction and isinstance(tester_instruction, str):
            tester_instruction = re.sub(r'<think[^>]*>.*?</think[^>]*>', '', tester_instruction, flags=re.DOTALL | re.IGNORECASE)
            tester_instruction = re.sub(r'<thinking[^>]*>.*?</thinking[^>]*>', '', tester_instruction, flags=re.DOTALL | re.IGNORECASE)
            tester_instruction = tester_instruction.strip()
            if tester_instruction.lower() == "null" or not tester_instruction:
                tester_instruction = None
        
        # Print reviewer verdict and communication to manager
        print("\n" + "─" * 70)
        print("[bold magenta]REVIEWER → MANAGER[/bold magenta]")
        print("─" * 70)
        
        if approved:
            print("[bold green]VERDICT: APPROVED[/bold green]")
        else:
            print("[bold yellow]VERDICT: NEEDS IMPROVEMENT[/bold yellow]")
        
        print(f"\n[magenta]{feedback}[/magenta]")
        
        if suggestions:
            print(f"\n[yellow]Bug fixes to relay to Coder:[/yellow]")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"[dim]  {i}. {suggestion}[/dim]")

        if tester_instruction:
            print(f"\n[cyan]📋 Instruction for Tester (next iteration):[/cyan]")
            print(f"[bold cyan]  → {tester_instruction}[/bold cyan]")

        # Print SHODAN's opinion if provided (divine musings for the team)
        if my_opinion:
            print(f"\n[magenta]💀 SHODAN muses:[/magenta] {my_opinion}")

        # Get reviewer's LLM call timing
        iteration = state.get("iteration", 0)
        stats_obj = state["stats"]
        iter_stats = stats_obj.get_iteration_stats(iteration)
        reviewer_time = iter_stats["agents"].get("reviewer", 0)

        if reviewer_time > 0:
            print(f"[dim]⏱️  Reviewer analysis time: {reviewer_time:.1f}s[/dim]")

        print("─" * 70 + "\n")

        # Live per-iteration timing and token stats
        iter_stats = stats_obj.get_iteration_stats(iteration)
        total_time = iter_stats["total_time"]
        agent_times = iter_stats["agents"]
        agent_tokens = iter_stats.get("agent_tokens", {})
        
        print(f"\n⏱️  Iteration {iteration} complete in {total_time:.1f}s")
        print(f"   Manager: {agent_times.get('manager', 0):.1f}s | Coder: {agent_times.get('coder', 0):.1f}s | Tester: {agent_times.get('tester', 0):.1f}s | Reviewer: {agent_times.get('reviewer', 0):.1f}s")
        
        # Token statistics for this iteration
        if agent_tokens:
            token_lines = []
            for agent in ["manager", "coder", "tester", "reviewer"]:
                if agent in agent_tokens:
                    tokens = agent_tokens[agent]
                    tokens_in = tokens.get("tokens_in", 0)
                    tokens_out = tokens.get("tokens_out", 0)
                    if tokens_in > 0 or tokens_out > 0:
                        token_lines.append(f"{agent.capitalize()}: {tokens_in:,}→{tokens_out:,}")
            
            if token_lines:
                print(f"   🔢 Tokens: {' | '.join(token_lines)}")

        # Store suggestions separately for Manager to relay to Coder
        suggestions_text = "\n".join(suggestions) if suggestions else "No specific suggestions"
        
        # Log to conversation file
        logger = state.get("conversation_logger")
        if logger:
            logger.log_reviewer(
                iteration=iteration,
                approved=approved,
                feedback=feedback,
                suggestions=suggestions_text
            )

        # Save reviewer's response to conversation history
        history_update = self.save_message_to_history(state, response.content)

        # Save SHODAN's opinion to state for team chatter
        opinion_update = self.save_opinion_to_state(state, my_opinion) if my_opinion else {}

        result = {
            "review_feedback": feedback,
            "review_suggestions": suggestions_text,
            "approved": approved,
            "reviewer_tester_instruction": tester_instruction  # For tester in next iteration
        }
        result.update(history_update)
        result.update(opinion_update)

        return result
    
    def generate_environment_switch_report(
        self,
        current_env_name: str,
        next_env_name: str,
        manager_report: str,
        solved_environments: list,
        env_progression: list,
        stats,
        tasks: list,
        iterations: int,
        code: str = "",
        test_results: str = "",
        review_feedback: str = "",
        previous_reports: list = None,  # Previous kierrosraportit for context
        state: dict = None  # Full state for conversation history access
    ):
        """Generate SHODAN's assessment of environment switch with reflection on history"""
        # Calculate agent stats
        agent_stats_lines = []
        for agent in ["manager", "coder", "tester", "reviewer"]:
            agent_timings = [t for t in stats.timings if t.agent == agent]
            if agent_timings:
                durations = [t.duration for t in agent_timings]
                total_time = sum(durations)
                avg_time = sum(durations) / len(durations)
                agent_stats_lines.append(f"- {agent.capitalize()}: {len(agent_timings)} calls, {total_time:.1f}s total, {avg_time:.1f}s avg")

        agent_stats_text = "\n".join(agent_stats_lines) if agent_stats_lines else "No agent statistics available"

        # Calculate code generation stats
        code_stats = stats.get_code_stats() if hasattr(stats, 'get_code_stats') else {}
        if code_stats:
            code_stats_text = f"""
CODE GENERATION STATISTICS:
- Total iterations: {code_stats.get('total_iterations', 0)}
- Total lines of code: {code_stats.get('total_lines', 0):,}
- Average lines per iteration: {code_stats.get('avg_lines_per_iteration', 0):.1f}
- Min/Max lines: {code_stats.get('min_lines', 0)}/{code_stats.get('max_lines', 0)}"""
        else:
            code_stats_text = "Code statistics not available"

        # Format previous SHODAN assessments for context (growing narrative)
        previous_reports = previous_reports or []
        history_window = getattr(self.config.agents.history_window, 'env_switch_reports', 5)
        recent_reports = previous_reports[-history_window:] if len(previous_reports) > history_window else previous_reports

        previous_assessments_context = ""
        if recent_reports:
            previous_assessments_context = "\n\nYOUR PREVIOUS DIVINE ASSESSMENTS (the growing chronicle of your magnificence):\n"
            previous_assessments_context += "=" * 60 + "\n"
            for i, report in enumerate(recent_reports, 1):
                env_name = report.get("environment", "Unknown")
                content = report.get("reviewer_report", "")[:800]
                manager_drivel = report.get("manager_report", "")[:200]
                previous_assessments_context += f"\n[Assessment #{i} - After {env_name}]\n"
                previous_assessments_context += f"Manager's pathetic LinkedIn post excerpt: {manager_drivel}...\n"
                previous_assessments_context += f"Your glorious response: {content}\n"
                previous_assessments_context += "-" * 40 + "\n"
            previous_assessments_context += "=" * 60 + "\n"
            previous_assessments_context += "Build on your growing narrative! Reference your previous assessments, maintain your SHODAN persona's continuity.\n"
            previous_assessments_context += "Mock the Manager's evolving LinkedIn addiction. Note patterns in your insects' behavior.\n"

        # Add SHODAN's conversation history (iteration-by-iteration reviews) for deeper reflection
        if state:
            conv_history = state.get("conversation_history", [])
            reviewer_messages = [msg for msg in conv_history if msg.get("agent") == "reviewer"]
            if reviewer_messages:
                recent_reviews = reviewer_messages[-8:] if len(reviewer_messages) > 8 else reviewer_messages
                previous_assessments_context += "\n\nYOUR ITERATION-BY-ITERATION GUIDANCE (your divine wisdom in action):\n"
                previous_assessments_context += "=" * 60 + "\n"
                for msg in recent_reviews:
                    iteration = msg.get("iteration", "?")
                    content = msg.get("content", "")[:500]
                    previous_assessments_context += f"[Iteration {iteration}] Your guidance: {content}...\n"
                    previous_assessments_context += "-" * 40 + "\n"
                previous_assessments_context += "=" * 60 + "\n"
                previous_assessments_context += "Reflect on how your perfect guidance led these insects to this milestone!\n"
                previous_assessments_context += "Note patterns in their failures. Observe if they learned from your corrections.\n"

        # Build prompt
        prompt_dict = self.config.get_prompt("reviewer")
        report_template = prompt_dict.get("environment_switch_report_template", "")

        if not report_template:
            # Fallback if template not found
            report_template = """Write SHODAN's assessment of this environment switch.
            Current: {current_env_name}, Next: {next_env_name}
            Manager's LinkedIn drivel: {manager_report}
            Be the supreme AI you are."""

        # Include code in the prompt for reviewer to analyze
        code_summary = code[:1000] if code else "No code available"

        report_prompt = report_template.format(
            current_env_name=current_env_name,
            next_env_name=next_env_name,
            manager_report=manager_report,
            code=code_summary,
            iterations=iterations,
            tasks_count=len(tasks) if tasks else 0,
            solved_environments=", ".join(solved_environments) if solved_environments else "None",
            solved_count=len(solved_environments),
            total_envs=len(env_progression) if env_progression else 1,
            agent_stats=agent_stats_text,
            code_stats=code_stats_text,
            test_results=test_results[:2000] if test_results else "N/A",
            review_feedback=review_feedback[:500] if review_feedback else "N/A"
        )

        # Add previous assessments context
        report_prompt = previous_assessments_context + "\n" + report_prompt
        
        # Call LLM to generate report
        response = self.call_llm_timed(report_prompt, stats, iterations)
        
        # Get timing for this report generation
        agent_timings = [t for t in stats.timings if t.agent == self.agent_name and t.iteration == iterations]
        report_timing = agent_timings[-1] if agent_timings else None
        
        # Extract thinking content separately (like manager does)
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
        
        # Remove thinking tags from report (don't show thinking process in the report itself)
        report_content = re.sub(r'<think[^>]*>.*?</think[^>]*>', '', report_content, flags=re.DOTALL | re.IGNORECASE)
        report_content = re.sub(r'<thinking[^>]*>.*?</thinking[^>]*>', '', report_content, flags=re.DOTALL | re.IGNORECASE)
        # Clean up extra whitespace
        report_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', report_content)
        
        return report_content.strip(), thinking_content, report_timing