import logging
import os
import time
from src.utils.timer import AgentTiming, RunStatistics
from langchain_openai import ChatOpenAI

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
                temperature=0
            )
            self.logger.info(f"Using API: {os.getenv('LLM_MODEL', 'grok-4-1-fast-reasoning')}")
        else:
            # Use Ollama with specified model
            self.llm = ChatOpenAI(
                model=model_name,
                base_url=config.ollama.base_url,
                api_key=config.ollama.api_key,
                temperature=0
            )
            self.logger.info(f"Using Ollama: {model_name}")
        
        self.model_name = model_name

    def call_llm_timed(self, prompt, stats: 'RunStatistics', iteration: int):
        # First call - show model info
        if not hasattr(self, '_model_verified'):
            self._model_verified = True
            self.logger.info("=" * 50)
            self.logger.info(f"🤖 [{self.agent_name.upper()}] MODEL INFO")
            display_name = os.getenv("LLM_MODEL", "grok-4-1-fast-reasoning") if self.model_name == "api" else self.model_name
            self.logger.info(f"   Model: {display_name}")
            self.logger.info(f"   Type: {'🌐 API' if self.model_name == 'api' else '💻 LOCAL (Ollama)'}")
            if self.model_name != 'api':
                # Verify Ollama model is loaded
                try:
                    import requests
                    resp = requests.get("http://localhost:11434/api/tags", timeout=5)
                    models = [m['name'] for m in resp.json().get('models', [])]
                    if any(self.model_name in m for m in models):
                        self.logger.info(f"   Status: ✅ Model available in Ollama")
                    else:
                        self.logger.warning(f"   Status: ⚠️ Model may need to be pulled")
                        self.logger.info(f"   Available: {models[:5]}...")
                except:
                    self.logger.info(f"   Status: ⏳ Ollama connection pending")
            self.logger.info("=" * 50)
        
        timing = AgentTiming(agent=self.agent_name, iteration=iteration)
        timing.start_time = time.time()
        
        display_name = os.getenv("LLM_MODEL", "grok-4-1-fast-reasoning") if self.model_name == "api" else self.model_name
        self.logger.info(f"🔄 [{self.agent_name.upper()}] Calling {display_name}...")
        
        result = self.llm.invoke(prompt)
        
        timing.end_time = time.time()
        timing.duration = timing.end_time - timing.start_time
        
        response_len = len(result.content) if hasattr(result, 'content') else len(str(result))
        tokens_approx = response_len // 4  # Rough estimate
        tokens_per_sec = tokens_approx / timing.duration if timing.duration > 0 else 0
        
        self.logger.info(f"✅ [{self.agent_name.upper()}] Done in {timing.duration:.1f}s (~{tokens_approx} tokens, {tokens_per_sec:.0f} tok/s)")
        
        stats.add_timing(timing)
        return result

    def call_llm(self, prompt):
        # First call - show model info
        if not hasattr(self, '_model_verified'):
            self._model_verified = True
            self.logger.info("=" * 50)
            self.logger.info(f"🤖 [{self.agent_name.upper()}] MODEL INFO")
            display_name = os.getenv("LLM_MODEL", "grok-4-1-fast-reasoning") if self.model_name == "api" else self.model_name
            self.logger.info(f"   Model: {display_name}")
            self.logger.info(f"   Type: {'🌐 API' if self.model_name == 'api' else '💻 LOCAL (Ollama)'}")
            if self.model_name != 'api':
                # Verify Ollama model is loaded
                try:
                    import requests
                    resp = requests.get("http://localhost:11434/api/tags", timeout=5)
                    models = [m['name'] for m in resp.json().get('models', [])]
                    if any(self.model_name in m for m in models):
                        self.logger.info(f"   Status: ✅ Model available in Ollama")
                    else:
                        self.logger.warning(f"   Status: ⚠️ Model may need to be pulled")
                        self.logger.info(f"   Available: {models[:5]}...")
                except:
                    self.logger.info(f"   Status: ⏳ Ollama connection pending")
            self.logger.info("=" * 50)
        
        start_time = time.time()
        display_name = os.getenv("LLM_MODEL", "grok-4-1-fast-reasoning") if self.model_name == "api" else self.model_name
        self.logger.info(f"🔄 [{self.agent_name.upper()}] Calling {display_name}...")
        
        result = self.llm.invoke(prompt)
        
        duration = time.time() - start_time
        response_len = len(result.content) if hasattr(result, 'content') else len(str(result))
        tokens_approx = response_len // 4  # Rough estimate
        tokens_per_sec = tokens_approx / duration if duration > 0 else 0
        
        self.logger.info(f"✅ [{self.agent_name.upper()}] Done in {duration:.1f}s (~{tokens_approx} tokens, {tokens_per_sec:.0f} tok/s)")
        
        return result

    def __call__(self, state: dict) -> dict:
        raise NotImplementedError(f"{self.agent_name} must implement __call__")