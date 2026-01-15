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

        prompt_dict = self.config.get_prompt("reviewer")
        task_template = prompt_dict["task_template"].format(
            manager_guidance=manager_guidance,
            code=code,
            test_results=test_results,
            success_threshold=success_threshold,
            video_dir=state.get("video_dir", "output/videos"),
        )
        system_prompt = prompt_dict["system"].format(
            success_threshold=success_threshold,
            video_dir=state.get("video_dir", "output/videos")
        )
        full_prompt = system_prompt + "\n\n" + task_template
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
        
        # Remove any thinking tags from feedback and suggestions (shouldn't be there, but safety check)
        import re
        feedback = re.sub(r'<think[^>]*>.*?</think[^>]*>', '', feedback, flags=re.DOTALL | re.IGNORECASE)
        feedback = re.sub(r'<thinking[^>]*>.*?</thinking[^>]*>', '', feedback, flags=re.DOTALL | re.IGNORECASE)
        feedback = feedback.strip()
        
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
        
        return {
            "review_feedback": feedback,
            "review_suggestions": suggestions_text,
            "approved": approved
        }
    
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
        review_feedback: str = ""
    ):
        """Generate a cynical, sarcastic report about environment switch"""
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
        
        # Build prompt
        prompt_dict = self.config.get_prompt("reviewer")
        report_template = prompt_dict.get("environment_switch_report_template", "")
        
        if not report_template:
            # Fallback if template not found
            report_template = """Write a cynical, sarcastic report about this environment switch.
            Current: {current_env_name}, Next: {next_env_name}
            Manager's report: {manager_report}
            Be witty and condescending like Shodan."""
        
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
            test_results=test_results[:2000] if test_results else "N/A",  # Longer test results for context
            review_feedback=review_feedback[:500] if review_feedback else "N/A"
        )
        
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