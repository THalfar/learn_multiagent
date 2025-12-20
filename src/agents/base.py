import logging
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ..config_loader import load_config

class BaseAgent:
    """
    Base class for LangGraph agent nodes.
    Loads LLM, config, logger. Subclasses implement __call__(state) -> dict.
    """
    def __init__(self, name: str):
        self.name = name
        load_dotenv()
        self.llm = ChatOpenAI(
            model="grok-beta",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        self.config = load_config()
        self.logger = logging.getLogger(f"agent.{name}")

    def __call__(self, state: dict) -> dict:
        raise NotImplementedError(f"{self.name} must implement __call__")