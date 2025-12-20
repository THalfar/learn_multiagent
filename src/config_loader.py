import yaml
import os
from typing import Any, Dict
from pydantic import BaseModel, Field, ConfigDict

class Environment(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(..., description="Gymnasium environment name")
    max_episode_steps: int = Field(default=500)

class Algorithm(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(..., description="Stable-Baselines3 algorithm")
    parameters: Dict[str, Any]

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
    success_threshold: int

class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    environment: Environment
    algorithm: Algorithm
    training: Training
    video: Video
    agents: Agents

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
    def algorithm(self) -> Algorithm:
        return self.project.algorithm

    @property
    def training(self) -> Training:
        return self.project.training

    @property
    def video(self) -> Video:
        return self.project.video

    @property
    def agents(self) -> Agents:
        return self.project.agents

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
    print(f"Algorithm: {config.algorithm.name}")
    print(f"Max iterations: {config.agents.max_iterations}")
    mgr_prompt = config.get_prompt("manager")
    print(f"Manager system prompt preview: {mgr_prompt.get('system', '')[:100]}...")
