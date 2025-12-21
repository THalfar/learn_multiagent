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
        timing = AgentTiming(agent=self.agent_name, iteration=iteration)
        timing.start_time = time.time()
        
        display_name = os.getenv("LLM_MODEL", "grok-4-1-fast-reasoning") if self.model_name == "api" else self.model_name
        self.logger.info(f"[{self.agent_name.upper()}] Calling {display_name}...")
        
        result = self.llm.invoke(prompt)
        
        timing.end_time = time.time()
        timing.duration = timing.end_time - timing.start_time
        
        self.logger.info(f"[{self.agent_name.upper()}] Done in {timing.duration:.1f}s")
        
        stats.add_timing(timing)
        return result

    def call_llm(self, prompt):
        llm_type = 'API' if self.model_name == 'api' else 'Ollama'
        self.logger.info(f"[{self.agent_name.upper()}] Calling {llm_type} LLM...")
        return self.llm.invoke(prompt)

    def __call__(self, state: dict) -> dict:
        raise NotImplementedError(f"{self.agent_name} must implement __call__")