import yaml
import os
from typing import Any, Dict
from pydantic import BaseModel, Field, ConfigDict

class Environment(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(..., description="Gymnasium environment name")
    max_episode_steps: int = Field(default=500)

class EnvironmentStep(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(..., description="Gymnasium environment name")
    max_episode_steps: int = Field(default=500)
    success_threshold: int = Field(..., description="Reward threshold to consider solved")
    execution_timeout: int = Field(default=900, description="Maximum execution time in seconds for tester (default: 900 = 15 minutes)")

class Algorithm(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(default="PPO", description="Stable-Baselines3 algorithm")
    parameters: Dict[str, Any] = Field(default_factory=dict)

class Training(BaseModel):
    model_config = ConfigDict(extra='forbid')
    total_timesteps: int
    eval_frequency: int
    n_eval_episodes: int

class Video(BaseModel):
    model_config = ConfigDict(extra='forbid')
    enabled: bool
    output_dir: str
    record_frequency: int

class Agents(BaseModel):
    model_config = ConfigDict(extra='forbid')
    max_iterations: int
    show_thinking: bool = Field(default=False, description="Show agent thinking process output")

class Llm(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: str

class OllamaConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    base_url: str
    api_key: str

class AgentLLMConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    manager: str
    coder: str
    tester: str
    reviewer: str

class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    environment: Environment
    environment_progression: list[EnvironmentStep] = Field(default_factory=list, description="List of environments from easy to hard")
    training: Training
    video: Video
    agents: Agents
    llm: Llm
    agent_llm: AgentLLMConfig
    ollama: OllamaConfig
    test_name: str

def load_project_config(path: str = 'config/project.yaml') -> ProjectConfig:
    """Load and validate project.yaml with Pydantic."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Project config not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return ProjectConfig.model_validate(data)

def load_prompts(path: str = 'config/prompts.yaml') -> Dict[str, Dict[str, str]]:
    """Load prompts.yaml as dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompts config not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data  # {'manager': {'system': str, 'task_template': str}, ...}

class Config:
    """Unified config with easy getters."""
    def __init__(self, project_config: ProjectConfig, prompts: Dict[str, Dict[str, str]]):
        self.project = project_config
        self.prompts = prompts

    @property
    def environment(self) -> Environment:
        return self.project.environment
    
    @property
    def environment_progression(self) -> list[EnvironmentStep]:
        return self.project.environment_progression


    @property
    def training(self) -> Training:
        return self.project.training

    @property
    def video(self) -> Video:
        return self.project.video

    @property
    def agents(self) -> Agents:
        return self.project.agents

    @property
    def llm(self) -> Llm:
        return self.project.llm

    @property
    def agent_llm(self) -> AgentLLMConfig:
        return self.project.agent_llm

    @property
    def ollama(self) -> OllamaConfig:
        return self.project.ollama
            
    @property
    def test_name(self) -> str:
        return self.project.test_name

    def get_prompt(self, agent_name: str) -> Dict[str, str]:
        """Get prompt dict for agent (e.g. 'manager')."""
        return self.prompts.get(agent_name, {})

def load_config(project_path: str = 'config/project.yaml',
                prompts_path: str = 'config/prompts.yaml') -> Config:
    """Load full config."""
    project = load_project_config(project_path)
    prompts = load_prompts(prompts_path)
    return Config(project, prompts)

if __name__ == "__main__":
    config = load_config()
    print(f"Environment: {config.environment.name} (max steps: {config.environment.max_episode_steps})")
    print(f"LLM Model: {config.llm.model}")
    print(f"Max iterations: {config.agents.max_iterations}")
    mgr_prompt = config.get_prompt("manager")
    print(f"Manager system preview: {mgr_prompt.get('system', '')[:100]}...")