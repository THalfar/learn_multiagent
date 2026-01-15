import logging
import os
import time
import requests
from src.utils.timer import AgentTiming, RunStatistics
from langchain_openai import ChatOpenAI
from rich.console import Console


def unload_ollama_models(ollama_base_url: str = "http://localhost:11434"):
    """Unload all models from Ollama to free VRAM."""
    # Strip /v1 suffix if present (native API doesn't use it)
    base_url = ollama_base_url.replace("/v1", "").rstrip("/")
    
    try:
        response = requests.get(f"{base_url}/api/ps", timeout=10)
        if response.status_code == 200:
            running = response.json().get("models", [])
            for model in running:
                model_name = model.get("name", "")
                if model_name:
                    # Unload by setting keep_alive to 0 (minimal prompt required by API)
                    requests.post(
                        f"{base_url}/api/generate",
                        json={"model": model_name, "prompt": "", "keep_alive": 0},
                        timeout=10
                    )
    except Exception as e:
        # print(f"[dim]Warning: Failed to unload models: {e}[/dim]")
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
            # Unload any existing models before using new one
            # unload_ollama_models(config.ollama.base_url)
            
            # Use Ollama with specified model
            self.llm = ChatOpenAI(
                model=model_name,
                base_url=config.ollama.base_url,
                api_key=config.ollama.api_key,
                temperature=0,
                timeout=300.0,  # 5 minutes for model swaps + generation
                max_retries=2   # Retry on transient failures
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
        
        # If using Ollama (not API), unload other models first
        if self.model_name != "api":
            # unload_ollama_models(self.config.ollama.base_url)
            pass
        
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

    def call_llm(self, prompt):
        # Don't show model info - too verbose
        if not hasattr(self, '_model_verified'):
            self._model_verified = True
        
        # If using Ollama (not API), unload other models first
        if self.model_name != "api":
            # unload_ollama_models(self.config.ollama.base_url)
            pass
        
        result = self.llm.invoke(prompt)
        
        return result

    def __call__(self, state: dict) -> dict:
        raise NotImplementedError(f"{self.agent_name} must implement __call__")