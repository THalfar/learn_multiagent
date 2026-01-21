import logging
import os
import time
import requests
from src.utils.timer import AgentTiming, RunStatistics
from langchain_openai import ChatOpenAI
from rich.console import Console


# Known context window sizes for models (in tokens)
# Used for tracking context usage and preventing overflow
MODEL_CONTEXT_SIZES = {
    # DeepSeek models
    "deepseek-r1": 32768,
    "deepseek-r1:32b": 32768,
    "deepseek-r1:14b": 32768,
    "deepseek-r1:7b": 32768,
    "deepseek-coder": 32768,
    # Qwen models
    "qwen": 32768,
    "qwen3": 32768,
    "qwen3-coder": 32768,
    "qwen3-coder:30b": 32768,
    "qwen2.5": 32768,
    "qwq": 32768,
    "qwq:32b": 32768,
    # Llama models
    "llama3": 8192,
    "llama3.1": 131072,  # 128K
    "llama3.2": 131072,
    "llama3.3": 131072,
    # Mistral models
    "mistral": 32768,
    "mixtral": 32768,
    # API models (Claude, Grok, etc.)
    "api": 200000,  # Conservative estimate for frontier API models
    "grok": 131072,
    "claude": 200000,
    # Default fallback
    "default": 8192,
}

def get_model_context_size(model_name: str) -> int:
    """Get context window size for a model. Returns size in tokens."""
    model_lower = model_name.lower()

    # Exact match first
    if model_lower in MODEL_CONTEXT_SIZES:
        return MODEL_CONTEXT_SIZES[model_lower]

    # Try prefix matches (e.g., "deepseek-r1:32b" matches "deepseek-r1")
    for key, size in MODEL_CONTEXT_SIZES.items():
        if model_lower.startswith(key) or key in model_lower:
            return size

    return MODEL_CONTEXT_SIZES["default"]


def get_loaded_ollama_model(ollama_base_url: str = "http://localhost:11434") -> str | None:
    """Check which model is currently loaded in Ollama. Returns model name or None."""
    base_url = ollama_base_url.replace("/v1", "").rstrip("/")
    try:
        response = requests.get(f"{base_url}/api/ps", timeout=5)
        if response.status_code == 200:
            running = response.json().get("models", [])
            if running:
                # Return first loaded model (usually only one with VRAM constraints)
                return running[0].get("name", "").split(":")[0]  # Strip tag for comparison
        return None
    except Exception:
        return None


def unload_ollama_models(ollama_base_url: str = "http://localhost:11434", verbose: bool = True):
    """Unload all models from Ollama to free VRAM and WAIT for unload to complete."""
    # Strip /v1 suffix if present (native API doesn't use it)
    base_url = ollama_base_url.replace("/v1", "").rstrip("/")

    try:
        response = requests.get(f"{base_url}/api/ps", timeout=10)
        if response.status_code == 200:
            running = response.json().get("models", [])
            if running and verbose:
                print(f"[dim]Unloading {len(running)} model(s) from Ollama...[/dim]")
            for model in running:
                model_name = model.get("name", "")
                if model_name:
                    if verbose:
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
                            if verbose:
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
            # Check if model loading messages should be shown
            show_loading = getattr(config.agents, 'show_model_loading', False)

            # Check if the required model is already loaded (skip swap if same)
            current_model = get_loaded_ollama_model(config.ollama.base_url)
            model_base = model_name.split(":")[0]  # Strip tag for comparison

            if current_model and current_model == model_base:
                # Same model already loaded - skip unload/load!
                if show_loading:
                    print(f"[dim]✓ Model {model_name} already loaded, skipping swap[/dim]")
            else:
                # Different model or none loaded - need to swap
                if current_model and show_loading:
                    print(f"[dim]Swapping {current_model} → {model_name}[/dim]")

                # Unload any existing models before using new one (prevents VRAM conflicts with 30B models)
                unload_ollama_models(config.ollama.base_url, verbose=show_loading)

                # Preload the model explicitly so we see loading time
                if show_loading:
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
                    if show_loading:
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
            show_loading = getattr(self.config.agents, 'show_model_loading', False)

            # Check if same model is already loaded
            current_model = get_loaded_ollama_model(self.config.ollama.base_url)
            model_base = self.model_name.split(":")[0]

            if current_model and current_model == model_base:
                # Same model - skip swap entirely
                print(f"[dim]🔄 {self.model_name} (same model, no swap)[/dim]")
            else:
                # Different model - need to swap
                swap_start = time.time()
                unload_ollama_models(self.config.ollama.base_url, verbose=show_loading)
                unload_time = time.time() - swap_start

                # Preload this agent's model so it's ready
                if show_loading:
                    print(f"[dim]Loading model {self.model_name}...[/dim]")
                preload_start = time.time()
                try:
                    base_url = self.config.ollama.base_url.replace("/v1", "").rstrip("/")
                    requests.post(
                        f"{base_url}/api/generate",
                        json={"model": self.model_name, "prompt": "hi", "keep_alive": "5m", "stream": False},
                        timeout=120
                    )
                    load_time = time.time() - preload_start
                    if show_loading:
                        print(f"[dim]  ✓ Model loaded ({load_time:.1f}s)[/dim]")
                    # Always show brief timing summary
                    prev = current_model or "none"
                    print(f"[dim]🔄 {prev} → {self.model_name} (unload {unload_time:.1f}s, load {load_time:.1f}s)[/dim]")
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
        """Print token statistics after agent response with context window usage"""
        if timing.tokens_in == 0 and timing.tokens_out == 0:
            return  # No token data available

        total_tokens = timing.tokens_in + timing.tokens_out
        tokens_per_sec = total_tokens / timing.duration if timing.duration > 0 else 0

        # Get context window size for this model
        context_size = get_model_context_size(self.model_name)
        context_pct = (timing.tokens_in / context_size * 100) if context_size > 0 else 0
        context_remaining = context_size - timing.tokens_in

        agent_colors = {
            "manager": "blue",
            "coder": "green",
            "tester": "yellow",
            "reviewer": "magenta"
        }
        color = agent_colors.get(self.agent_name, "white")

        # Build stats line
        stats_line = f"Tokens: {timing.tokens_in:,} in → {timing.tokens_out:,} out | {tokens_per_sec:.0f} tok/s"

        # Add context usage info
        context_line = f"Context: {context_pct:.1f}% used ({timing.tokens_in:,}/{context_size:,}) | {context_remaining:,} remaining"

        # Use Console for proper rich formatting
        console = Console()

        # Print token stats with agent color
        if color == "blue":
            console.print(f"[dim blue]{stats_line}[/dim blue]")
        elif color == "green":
            console.print(f"[dim green]{stats_line}[/dim green]")
        elif color == "yellow":
            console.print(f"[dim yellow]{stats_line}[/dim yellow]")
        elif color == "magenta":
            console.print(f"[dim magenta]{stats_line}[/dim magenta]")
        else:
            console.print(f"[dim]{stats_line}[/dim]")

        # Print context usage with color based on percentage
        if context_pct >= 80:
            console.print(f"[bold red]⚠️ {context_line} - DANGER: Context nearly full![/bold red]")
        elif context_pct >= 60:
            console.print(f"[yellow]⚠️ {context_line} - Warning: High context usage[/yellow]")
        elif context_pct >= 40:
            console.print(f"[dim yellow]{context_line}[/dim yellow]")
        else:
            console.print(f"[dim green]{context_line}[/dim green]")

    def print_context_breakdown(self, state: dict, prompt_tokens: int = 0):
        """
        Print comprehensive breakdown of all context components.
        Shows: own history, team chatter, env reports, prompt, and total usage.
        Helps with context window optimization.
        """
        console = Console()
        context_size = get_model_context_size(self.model_name)

        components = []
        total_tokens = 0

        # 1. Own conversation history
        history_window = getattr(self.config.agents.history_window, self.agent_name, 0)
        if history_window > 0:
            messages = state.get("conversation_history", [])
            agent_messages = [msg for msg in messages if msg.get("agent") == self.agent_name]
            recent_messages = agent_messages[-history_window:] if len(agent_messages) > history_window else agent_messages
            if recent_messages:
                history_text = "\n\n".join([msg.get("content", "") for msg in recent_messages])
                history_tokens = self.estimate_tokens(history_text)
                total_tokens += history_tokens
                components.append({
                    "name": "Own history",
                    "count": len(recent_messages),
                    "unit": "msg",
                    "tokens": history_tokens,
                    "icon": "💾"
                })

        # 2. Team chatter (agent opinions)
        agent_opinions_config = getattr(self.config.agents, 'agent_opinions', None)
        if agent_opinions_config and getattr(agent_opinions_config, 'enabled', False):
            opinions_window = getattr(self.config.agents.history_window, 'agent_opinions', 8)
            opinions = state.get("agent_opinions", [])
            recent_opinions = opinions[-opinions_window:] if len(opinions) > opinions_window else opinions
            if recent_opinions:
                opinions_text = "\n".join([op.get("opinion", "") for op in recent_opinions])
                opinions_tokens = self.estimate_tokens(opinions_text) + 200  # overhead for formatting
                total_tokens += opinions_tokens
                components.append({
                    "name": "Team chatter",
                    "count": len(recent_opinions),
                    "unit": "opinions",
                    "tokens": opinions_tokens,
                    "icon": "💬"
                })

        # 3. Environment switch reports (kierrosraportit)
        env_reports_window = getattr(self.config.agents.history_window, 'env_switch_reports', 5)
        env_reports = state.get("env_switch_reports", [])
        recent_reports = env_reports[-env_reports_window:] if len(env_reports) > env_reports_window else env_reports
        if recent_reports:
            reports_text = "\n".join([
                f"{r.get('manager_report', '')[:600]} {r.get('reviewer_report', '')[:800]}"
                for r in recent_reports
            ])
            reports_tokens = self.estimate_tokens(reports_text) + 300  # overhead
            total_tokens += reports_tokens
            components.append({
                "name": "Env reports",
                "count": len(recent_reports),
                "unit": "reports",
                "tokens": reports_tokens,
                "icon": "📊"
            })

        # 4. Current prompt/task (if provided)
        if prompt_tokens > 0:
            total_tokens += prompt_tokens
            components.append({
                "name": "Current prompt",
                "count": 1,
                "unit": "prompt",
                "tokens": prompt_tokens,
                "icon": "📝"
            })

        # Don't print if nothing to show
        if not components:
            return

        # Print breakdown
        agent_colors = {"manager": "blue", "coder": "green", "tester": "yellow", "reviewer": "magenta"}
        color = agent_colors.get(self.agent_name, "white")

        console.print(f"[dim]┌─ Context Breakdown ({self.agent_name.upper()}) ─────────────────────[/dim]")

        for comp in components:
            pct = (comp["tokens"] / context_size * 100) if context_size > 0 else 0
            bar_width = int(pct / 2)  # Scale to ~50 chars max
            bar = "█" * min(bar_width, 25) + "░" * max(0, 5 - bar_width)

            # Color based on percentage
            if pct >= 20:
                comp_color = "yellow"
            elif pct >= 10:
                comp_color = "dim yellow"
            else:
                comp_color = "dim"

            console.print(f"[{comp_color}]│ {comp['icon']} {comp['name']:15} {comp['count']:3} {comp['unit']:8} │ {comp['tokens']:>6,} tok ({pct:4.1f}%) {bar}[/{comp_color}]")

        # Total
        total_pct = (total_tokens / context_size * 100) if context_size > 0 else 0
        remaining = context_size - total_tokens

        console.print(f"[dim]├─────────────────────────────────────────────────────[/dim]")

        # Total line with appropriate color
        if total_pct >= 60:
            console.print(f"[bold yellow]│ 📦 TOTAL CONTEXT   {total_tokens:>6,} / {context_size:,} tokens ({total_pct:.1f}%) ⚠️[/bold yellow]")
            console.print(f"[yellow]│ 🔓 Remaining: {remaining:,} tokens[/yellow]")
        elif total_pct >= 40:
            console.print(f"[yellow]│ 📦 TOTAL CONTEXT   {total_tokens:>6,} / {context_size:,} tokens ({total_pct:.1f}%)[/yellow]")
            console.print(f"[dim]│ 🔓 Remaining: {remaining:,} tokens[/dim]")
        else:
            console.print(f"[dim green]│ 📦 TOTAL CONTEXT   {total_tokens:>6,} / {context_size:,} tokens ({total_pct:.1f}%)[/dim green]")
            console.print(f"[dim]│ 🔓 Remaining: {remaining:,} tokens[/dim]")

        console.print(f"[dim]└─────────────────────────────────────────────────────[/dim]")

        return total_tokens

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

        # Print token usage for monitoring with context % (only if history exists)
        context_size = get_model_context_size(self.model_name)
        history_pct = (history_tokens / context_size * 100) if context_size > 0 else 0

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

        # Format message with context %
        history_msg = f"💾 {self.agent_name.capitalize()} history: {len(recent_messages)} msg, ~{history_tokens:,} tokens ({history_pct:.1f}% of {context_size:,} context)"

        # Color based on usage
        if history_pct >= 30:
            console.print(f"[yellow]{history_msg} ⚠️[/yellow]")
        elif color == "blue":
            console.print(f"[dim blue]{history_msg}[/dim blue]")
        elif color == "green":
            console.print(f"[dim green]{history_msg}[/dim green]")
        elif color == "yellow":
            console.print(f"[dim yellow]{history_msg}[/dim yellow]")
        elif color == "magenta":
            console.print(f"[dim magenta]{history_msg}[/dim magenta]")

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

    def format_agent_opinions_context(self, state: dict) -> str:
        """
        Format recent agent opinions for cross-agent "dialogue".
        Creates a chatter/gossip section showing what other agents have been saying.
        """
        # Check if opinions feature is enabled
        agent_opinions_config = getattr(self.config.agents, 'agent_opinions', None)
        if not agent_opinions_config or not getattr(agent_opinions_config, 'enabled', False):
            return ""

        # Get history window for opinions
        history_window = getattr(self.config.agents.history_window, 'agent_opinions', 8)

        # Get all opinions from state
        opinions = state.get("agent_opinions", [])
        if not opinions:
            return ""

        # Take recent opinions
        recent_opinions = opinions[-history_window:] if len(opinions) > history_window else opinions

        if not recent_opinions:
            return ""

        # Format based on agent type (SHODAN gets different framing)
        if self.agent_name == "reviewer":
            header = "INSECT CHATTER (your minions have opinions... how amusing):"
        else:
            header = "TEAM CHATTER (what your colleagues have been saying):"

        lines = []
        lines.append("=" * 60)
        lines.append(header)
        lines.append("=" * 60)

        for opinion in recent_opinions:
            agent = opinion.get("agent", "Unknown")
            iteration = opinion.get("iteration", "?")
            content = opinion.get("opinion", "")[:400]  # Truncate long opinions

            # Add emoji/label based on agent
            if agent == "manager":
                label = "📊 Manager"
            elif agent == "tester":
                label = "🧪 Tester"
            elif agent == "reviewer":
                label = "💀 SHODAN"
            else:
                label = agent.capitalize()

            lines.append(f"\n[Iter {iteration}] {label}:")
            lines.append(f"  \"{content}\"")
            lines.append("-" * 40)

        lines.append("=" * 60)
        lines.append("Feel free to react to what they said, agree, disagree, or ignore!")
        lines.append("")

        # Print token estimate with context %
        context_text = "\n".join(lines)
        tokens = self.estimate_tokens(context_text)
        context_size = get_model_context_size(self.model_name)
        opinions_pct = (tokens / context_size * 100) if context_size > 0 else 0

        from rich.console import Console
        console = Console()
        opinions_msg = f"💬 Team chatter: {len(recent_opinions)} opinions, ~{tokens:,} tokens ({opinions_pct:.1f}% of context)"

        if opinions_pct >= 15:
            console.print(f"[yellow]{opinions_msg} ⚠️[/yellow]")
        else:
            console.print(f"[dim cyan]{opinions_msg}[/dim cyan]")

        return context_text

    def save_opinion_to_state(self, state: dict, opinion: str):
        """
        Save this agent's opinion to state for other agents to see.
        """
        if not opinion:
            return {}

        opinions = state.get("agent_opinions", [])

        opinion_entry = {
            "agent": self.agent_name,
            "iteration": state.get("iteration", 0),
            "opinion": opinion
        }

        opinions.append(opinion_entry)

        return {"agent_opinions": opinions}

    def call_llm(self, prompt):
        # Don't show model info - too verbose
        if not hasattr(self, '_model_verified'):
            self._model_verified = True

        # If using Ollama (not API), unload other models first and load this agent's model
        if self.model_name != "api":
            show_loading = getattr(self.config.agents, 'show_model_loading', False)

            # Check if same model is already loaded
            current_model = get_loaded_ollama_model(self.config.ollama.base_url)
            model_base = self.model_name.split(":")[0]

            if current_model and current_model == model_base:
                # Same model - skip swap entirely
                print(f"[dim]🔄 {self.model_name} (same model, no swap)[/dim]")
            else:
                # Different model - need to swap
                swap_start = time.time()
                unload_ollama_models(self.config.ollama.base_url, verbose=show_loading)
                unload_time = time.time() - swap_start

                # Preload this agent's model so it's ready
                if show_loading:
                    print(f"[dim]Loading model {self.model_name}...[/dim]")
                preload_start = time.time()
                try:
                    base_url = self.config.ollama.base_url.replace("/v1", "").rstrip("/")
                    requests.post(
                        f"{base_url}/api/generate",
                        json={"model": self.model_name, "prompt": "hi", "keep_alive": "5m", "stream": False},
                        timeout=120
                    )
                    load_time = time.time() - preload_start
                    if show_loading:
                        print(f"[dim]  ✓ Model loaded ({load_time:.1f}s)[/dim]")
                    # Always show brief timing summary
                    prev = current_model or "none"
                    print(f"[dim]🔄 {prev} → {self.model_name} (unload {unload_time:.1f}s, load {load_time:.1f}s)[/dim]")
                except Exception as e:
                    print(f"[yellow]  Warning: Model preload failed: {e}[/yellow]")

        result = self.llm.invoke(prompt)

        return result

    def __call__(self, state: dict) -> dict:
        raise NotImplementedError(f"{self.agent_name} must implement __call__")