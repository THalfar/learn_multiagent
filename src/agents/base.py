import logging
import os
import time
import requests
from src.utils.timer import AgentTiming, RunStatistics
from langchain_openai import ChatOpenAI
from rich.console import Console


def unload_ollama_models(ollama_base_url: str = "http://localhost:11434"):
    """Unload all models from Ollama to free VRAM and WAIT for unload to complete."""
    # Strip /v1 suffix if present (native API doesn't use it)
    base_url = ollama_base_url.replace("/v1", "").rstrip("/")

    try:
        response = requests.get(f"{base_url}/api/ps", timeout=10)
        if response.status_code == 200:
            running = response.json().get("models", [])
            if running:
                print(f"[dim]Unloading {len(running)} model(s) from Ollama...[/dim]")
            for model in running:
                model_name = model.get("name", "")
                if model_name:
                    print(f"[dim]  Unloading {model_name}...[/dim]")
                    # Unload by setting keep_alive to 0 (minimal prompt required by API)
                    requests.post(
                        f"{base_url}/api/generate",
                        json={"model": model_name, "prompt": "", "keep_alive": 0},
                        timeout=10
                    )

            # Wait for unload to complete by checking if models are actually gone
            if running:
                max_wait = 30  # Wait up to 30 seconds
                for i in range(max_wait):
                    time.sleep(1)
                    check_response = requests.get(f"{base_url}/api/ps", timeout=10)
                    if check_response.status_code == 200:
                        still_running = check_response.json().get("models", [])
                        if not still_running:
                            print(f"[dim]  ✓ All models unloaded ({i+1}s)[/dim]")
                            break
                        if i == max_wait - 1:
                            print(f"[yellow]  Warning: Models still loaded after {max_wait}s[/yellow]")
    except Exception as e:
        print(f"[yellow]Warning: Failed to unload models: {e}[/yellow]")
        pass

class BaseAgent:
    """
    Base class for LangGraph agent nodes.
    Loads LLM, config, logger. Subclasses implement __call__(state) -> dict.
    """
    def __init__(self, config, agent_name: str):
        self.config = config
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"agent.{agent_name}")
        
        # Get model for this agent
        model_name = getattr(config.agent_llm, agent_name, "api")
        
        if model_name == "api":
            # Use Grok API from .env
            self.llm = ChatOpenAI(
                model=os.getenv("LLM_MODEL", "grok-4-1-fast-reasoning"),
                base_url=os.getenv("OPENAI_BASE_URL"),
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0,
                timeout=120.0  # 2 minutes for API
            )
            # Using API - no log needed
        else:
            # Unload any existing models before using new one (prevents VRAM conflicts with 30B models)
            unload_ollama_models(config.ollama.base_url)

            # Preload the model explicitly so we see loading time
            print(f"[dim]Loading model {model_name}...[/dim]")
            preload_start = time.time()
            try:
                base_url = config.ollama.base_url.replace("/v1", "").rstrip("/")
                requests.post(
                    f"{base_url}/api/generate",
                    json={"model": model_name, "prompt": "hi", "keep_alive": "5m", "stream": False},
                    timeout=120  # 2 minutes for model loading
                )
                preload_time = time.time() - preload_start
                print(f"[dim]  ✓ Model loaded ({preload_time:.1f}s)[/dim]")
            except Exception as e:
                print(f"[yellow]  Warning: Model preload failed: {e}[/yellow]")

            # Use Ollama with specified model
            # Coder gets reasonable token limit (typical RL script = 100-200 lines = ~2000 tokens)
            # Too high limit causes repetition loops, too low truncates valid code
            coder_max_tokens = 2500 if agent_name == "coder" else None

            self.llm = ChatOpenAI(
                model=model_name,
                base_url=config.ollama.base_url,
                api_key=config.ollama.api_key,
                temperature=0.1,  # Small temperature to avoid deterministic repetition loops
                timeout=300.0,  # 5 minutes for generation (model already loaded)
                max_retries=2,  # Retry on transient failures
                max_tokens=coder_max_tokens,
            )
            # Using Ollama - no log needed
        
        self.model_name = model_name
    
    def is_reasoning_model(self):
        """Check if this agent uses a reasoning model (like qwq:32b)"""
        reasoning_models = ["qwq", "deepseek", "qwen", "llama-reasoning"]
        return any(rm in self.model_name.lower() for rm in reasoning_models)
    
    def print_thinking(self, content: str):
        """Extract and print thinking tags from reasoning model responses with creative formatting"""
        import re
        # Check if thinking output is enabled in config
        if not getattr(self.config.agents, 'show_thinking', False):
            return
        if not self.is_reasoning_model():
            return
        
        # Try to find thinking tags - also check for reasoning without tags
        think_patterns = [
            (r'<think[^>]*>(.*?)</think[^>]*>', re.DOTALL | re.IGNORECASE),
            (r'<thinking[^>]*>(.*?)</thinking[^>]*>', re.DOTALL | re.IGNORECASE),
        ]
        
        # Also check for reasoning text that might not be in tags
        # Some models output reasoning before the actual response
        reasoning_indicators = [
            r'(?:let me|let\'s|i need to|i should|first|thinking|reasoning)[^\.]*(?:\.|$)',
        ]
        
        thinking_found = False
        agent_colors = {
            "manager": "blue",
            "coder": "green", 
            "tester": "yellow",
            "reviewer": "magenta"
        }
        color = agent_colors.get(self.agent_name, "white")
        
        # Collect all thinking content
        all_thinking = []
        
        for pattern, flags in think_patterns:
            matches = re.finditer(pattern, content, flags)
            for match in matches:
                thinking_content = match.group(1).strip()
                if thinking_content:
                    # For coder, remove any code blocks from thinking
                    if self.agent_name == "coder":
                        # Remove code blocks from thinking content
                        thinking_content = re.sub(r'```[a-z]*\n.*?```', '', thinking_content, flags=re.DOTALL)
                        thinking_content = re.sub(r'`[^`]+`', '', thinking_content)
                        thinking_content = thinking_content.strip()
                        # Skip if it's mostly code (has many code indicators)
                        code_indicators = ['import ', 'def ', 'class ', 'from ']
                        if sum(1 for ind in code_indicators if ind in thinking_content) >= 2:
                            continue  # Skip this thinking block, it's mostly code
                    if thinking_content:
                        all_thinking.append(thinking_content)
        
        # If no thinking tags found, try to extract reasoning from content
        # Some reasoning models output thinking without tags
        if not all_thinking and len(content) > 500:
            # For coder, we need to be more careful - only extract thinking before code blocks
            if self.agent_name == "coder":
                # Find the first code block
                code_block_match = re.search(r'```', content)
                if code_block_match:
                    # Only take content before the first code block
                    content_before_code = content[:code_block_match.start()].strip()
                    if len(content_before_code) > 100:  # Substantial thinking before code
                        # Check it's not code (has code indicators)
                        code_indicators = ['import ', 'def ', 'class ', 'from ', 'gymnasium', 'stable_baselines']
                        if not any(indicator in content_before_code for indicator in code_indicators):
                            # This looks like actual thinking, not code
                            all_thinking.append(content_before_code)
                # Don't try other extraction methods for coder if we found code blocks
                if code_block_match:
                    pass  # Already handled above
                else:
                    # No code blocks found, try normal extraction but be strict
                    json_start = content.find('{')
                    structure_start = json_start if json_start > 0 else len(content)
                    if structure_start > 200:
                        potential_reasoning = content[:structure_start].strip()
                        code_indicators = ['import ', 'def ', 'class ', 'from ']
                        if not any(indicator in potential_reasoning for indicator in code_indicators):
                            cleaned = re.sub(r'^(okay|let me|let\'s|i need|i should|first|thinking|reasoning|so|therefore|now|well)[\s,]*', '', potential_reasoning, flags=re.IGNORECASE)
                            if len(cleaned.split()) > 30:
                                all_thinking.append(potential_reasoning)
            else:
                # For other agents, use normal extraction
                json_start = content.find('{')
                code_start = content.find('```')
                structure_start = min([s for s in [json_start, code_start] if s > 0] or [len(content)])
                
                if structure_start > 200:
                    potential_reasoning = content[:structure_start].strip()
                    cleaned = re.sub(r'^(okay|let me|let\'s|i need|i should|first|thinking|reasoning|so|therefore|now|well)[\s,]*', '', potential_reasoning, flags=re.IGNORECASE)
                    if len(cleaned.split()) > 30:
                        all_thinking.append(potential_reasoning)
        
        if all_thinking:
            # Print thinking with creative formatting - use Console for proper rich formatting
            console = Console()
            print()  # Empty line before thinking
            print("─" * 70)
            
            # Header with agent color and emoji
            if color == "blue":
                console.print("[bold blue]💭 MANAGER THINKING[/bold blue]")
            elif color == "green":
                console.print("[bold green]💭 CODER THINKING[/bold green]")
            elif color == "yellow":
                console.print("[bold yellow]💭 TESTER THINKING[/bold yellow]")
            elif color == "magenta":
                console.print("[bold magenta]💭 REVIEWER THINKING[/bold magenta]")
            else:
                console.print(f"[bold]💭 {self.agent_name.upper()} THINKING[/bold]")
            
            print("─" * 70)
            
            # Print all thinking content with appropriate color
            for i, thinking_content in enumerate(all_thinking, 1):
                if len(all_thinking) > 1:
                    if color == "blue":
                        console.print(f"[blue]Thinking part {i}:[/blue]")
                    elif color == "green":
                        console.print(f"[green]Thinking part {i}:[/green]")
                    elif color == "yellow":
                        console.print(f"[yellow]Thinking part {i}:[/yellow]")
                    elif color == "magenta":
                        console.print(f"[magenta]Thinking part {i}:[/magenta]")
                
                # Split into paragraphs for better readability
                paragraphs = thinking_content.split('\n\n')
                for para in paragraphs:
                    para = para.strip()
                    if para:
                        # Use italic style for thinking to differentiate from regular output
                        if color == "blue":
                            console.print(f"[italic blue]{para}[/italic blue]")
                        elif color == "green":
                            console.print(f"[italic green]{para}[/italic green]")
                        elif color == "yellow":
                            console.print(f"[italic yellow]{para}[/italic yellow]")
                        elif color == "magenta":
                            console.print(f"[italic magenta]{para}[/italic magenta]")
                        else:
                            console.print(f"[italic]{para}[/italic]")
                        print()  # Space between paragraphs
                
                if i < len(all_thinking):
                    print("─" * 70)
            
            print("─" * 70)
            print()  # Empty line after thinking

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: ~4 chars per token)"""
        return len(text) // 4
    
    def call_llm_timed(self, prompt, stats: 'RunStatistics', iteration: int):
        # Don't show model info - too verbose
        if not hasattr(self, '_model_verified'):
            self._model_verified = True

        # If using Ollama (not API), unload other models first and load this agent's model
        if self.model_name != "api":
            unload_ollama_models(self.config.ollama.base_url)

            # Preload this agent's model so it's ready
            print(f"[dim]Loading model {self.model_name}...[/dim]")
            preload_start = time.time()
            try:
                base_url = self.config.ollama.base_url.replace("/v1", "").rstrip("/")
                requests.post(
                    f"{base_url}/api/generate",
                    json={"model": self.model_name, "prompt": "hi", "keep_alive": "5m", "stream": False},
                    timeout=120
                )
                preload_time = time.time() - preload_start
                print(f"[dim]  ✓ Model loaded ({preload_time:.1f}s)[/dim]")
            except Exception as e:
                print(f"[yellow]  Warning: Model preload failed: {e}[/yellow]")

        timing = AgentTiming(agent=self.agent_name, iteration=iteration)
        timing.start_time = time.time()
        
        # Estimate input tokens
        timing.tokens_in = self.estimate_tokens(prompt)
        
        # Don't log LLM calls - they're too verbose
        result = self.llm.invoke(prompt)
        
        timing.end_time = time.time()
        timing.duration = timing.end_time - timing.start_time
        
        # Try to get token usage from response metadata
        if hasattr(result, 'response_metadata'):
            usage = result.response_metadata.get('token_usage', {})
            if usage:
                timing.tokens_in = usage.get('prompt_tokens', timing.tokens_in)
                timing.tokens_out = usage.get('completion_tokens', 0)
        
        # If no usage info, estimate output tokens
        if timing.tokens_out == 0 and hasattr(result, 'content'):
            timing.tokens_out = self.estimate_tokens(result.content)
        
        stats.add_timing(timing)
        return result
    
    def print_token_stats(self, timing: AgentTiming):
        """Print token statistics after agent response"""
        if timing.tokens_in == 0 and timing.tokens_out == 0:
            return  # No token data available

        total_tokens = timing.tokens_in + timing.tokens_out
        tokens_per_sec = total_tokens / timing.duration if timing.duration > 0 else 0

        agent_colors = {
            "manager": "blue",
            "coder": "green",
            "tester": "yellow",
            "reviewer": "magenta"
        }
        color = agent_colors.get(self.agent_name, "white")

        # Use Console for proper rich formatting
        console = Console()
        if color == "blue":
            console.print(f"[dim blue]Tokens: {timing.tokens_in:,} in → {timing.tokens_out:,} out | {tokens_per_sec:.0f} tok/s[/dim blue]")
        elif color == "green":
            console.print(f"[dim green]Tokens: {timing.tokens_in:,} in → {timing.tokens_out:,} out | {tokens_per_sec:.0f} tok/s[/dim green]")
        elif color == "yellow":
            console.print(f"[dim yellow]Tokens: {timing.tokens_in:,} in → {timing.tokens_out:,} out | {tokens_per_sec:.0f} tok/s[/dim yellow]")
        elif color == "magenta":
            console.print(f"[dim magenta]Tokens: {timing.tokens_in:,} in → {timing.tokens_out:,} out | {tokens_per_sec:.0f} tok/s[/dim magenta]")
        else:
            console.print(f"[dim]Tokens: {timing.tokens_in:,} in → {timing.tokens_out:,} out | {tokens_per_sec:.0f} tok/s[/dim]")

    def format_conversation_history(self, state: dict) -> str:
        """
        Format this agent's own conversation history (siloed memory).
        Only includes messages from THIS agent, limited by history_window config.
        """
        # Get history window for this agent
        history_window = getattr(self.config.agents.history_window, self.agent_name, 0)

        if history_window == 0:
            return ""  # No history for this agent

        # Get all messages from state
        messages = state.get("conversation_history", [])

        # Filter to only this agent's messages
        agent_messages = [msg for msg in messages if msg.get("agent") == self.agent_name]

        # Take only the last N messages
        recent_messages = agent_messages[-history_window:] if len(agent_messages) > history_window else agent_messages

        if not recent_messages:
            return ""

        # Calculate total token count for history
        total_history_text = "\n\n".join([msg.get("content", "") for msg in recent_messages])
        history_tokens = self.estimate_tokens(total_history_text)

        # Format the history
        history_lines = []
        history_lines.append("=" * 70)
        history_lines.append(f"YOUR PREVIOUS RESPONSES (last {len(recent_messages)} message{'s' if len(recent_messages) != 1 else ''}):")
        history_lines.append(f"Memory usage: ~{history_tokens:,} tokens")
        history_lines.append("=" * 70)
        history_lines.append("NOTE: Use your previous thoughts and learnings below to help with")
        history_lines.append("      your current work. Learn from past iterations.")
        history_lines.append("=" * 70)

        for i, msg in enumerate(recent_messages, 1):
            iteration = msg.get("iteration", "?")
            content = msg.get("content", "")
            history_lines.append(f"\n[Iteration {iteration}]")
            history_lines.append(content)
            if i < len(recent_messages):
                history_lines.append("\n" + "-" * 70)

        history_lines.append("=" * 70)
        history_lines.append("")  # Empty line after history

        # Print token usage for monitoring (only if history exists)
        agent_colors = {
            "manager": "blue",
            "coder": "green",
            "tester": "yellow",
            "reviewer": "magenta"
        }
        color = agent_colors.get(self.agent_name, "white")

        # Use Console for proper rich formatting
        from rich.console import Console
        console = Console()
        if color == "blue":
            console.print(f"[dim blue]💾 Manager history: {len(recent_messages)} msg, ~{history_tokens:,} tokens[/dim blue]")
        elif color == "green":
            console.print(f"[dim green]💾 Coder history: {len(recent_messages)} msg, ~{history_tokens:,} tokens[/dim green]")
        elif color == "yellow":
            console.print(f"[dim yellow]💾 Tester history: {len(recent_messages)} msg, ~{history_tokens:,} tokens[/dim yellow]")
        elif color == "magenta":
            console.print(f"[dim magenta]💾 Reviewer history: {len(recent_messages)} msg, ~{history_tokens:,} tokens[/dim magenta]")

        return "\n".join(history_lines)

    def format_other_agent_history(self, state: dict, other_agent: str, window: int) -> str:
        """
        Get another agent's history. Useful for manager to see reviewer feedback.
        """
        if window == 0:
            return ""

        messages = state.get("conversation_history", [])
        agent_messages = [msg for msg in messages if msg.get("agent") == other_agent]
        recent_messages = agent_messages[-window:] if len(agent_messages) > window else agent_messages

        if not recent_messages:
            return ""

        # Format compactly
        history_lines = []
        history_lines.append("=" * 70)
        history_lines.append(f"{other_agent.upper()}'S RECENT FEEDBACK (last {len(recent_messages)}):")
        history_lines.append("=" * 70)

        for msg in recent_messages:
            iteration = msg.get("iteration", "?")
            content = msg.get("content", "")
            # Truncate long content for other agent's history
            if len(content) > 800:
                content = content[:800] + "..."
            history_lines.append(f"\n[Iter {iteration}] {content}")
            history_lines.append("-" * 40)

        history_lines.append("=" * 70)

        total_text = "\n".join(history_lines)
        tokens = self.estimate_tokens(total_text)

        from rich.console import Console
        console = Console()
        console.print(f"[dim]📋 {other_agent.capitalize()} feedback history: {len(recent_messages)} msg, ~{tokens:,} tokens[/dim]")

        return total_text

    def save_message_to_history(self, state: dict, content: str):
        """
        Save this agent's message to the conversation history in state.
        """
        messages = state.get("conversation_history", [])

        message = {
            "agent": self.agent_name,
            "iteration": state.get("iteration", 0),
            "content": content
        }

        messages.append(message)

        # Return updated messages (will be merged into state by LangGraph)
        return {"conversation_history": messages}

    def call_llm(self, prompt):
        # Don't show model info - too verbose
        if not hasattr(self, '_model_verified'):
            self._model_verified = True

        # If using Ollama (not API), unload other models first and load this agent's model
        if self.model_name != "api":
            unload_ollama_models(self.config.ollama.base_url)

            # Preload this agent's model so it's ready
            print(f"[dim]Loading model {self.model_name}...[/dim]")
            preload_start = time.time()
            try:
                base_url = self.config.ollama.base_url.replace("/v1", "").rstrip("/")
                requests.post(
                    f"{base_url}/api/generate",
                    json={"model": self.model_name, "prompt": "hi", "keep_alive": "5m", "stream": False},
                    timeout=120
                )
                preload_time = time.time() - preload_start
                print(f"[dim]  ✓ Model loaded ({preload_time:.1f}s)[/dim]")
            except Exception as e:
                print(f"[yellow]  Warning: Model preload failed: {e}[/yellow]")

        result = self.llm.invoke(prompt)

        return result

    def __call__(self, state: dict) -> dict:
        raise NotImplementedError(f"{self.agent_name} must implement __call__")