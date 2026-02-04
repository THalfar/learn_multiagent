import yaml
import os
from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

# Device selection: cpu is faster for small MLPs, gpu for large networks
DeviceType = Literal["cpu", "gpu", "auto"]
# Action space type
ActionType = Literal["discrete", "continuous"]

class Environment(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(..., description="Gymnasium environment name")
    max_episode_steps: int = Field(default=500)
    # Environment specs for Coder/Tester (optional for backwards compatibility)
    obs_dim: Optional[int] = Field(default=None, description="Observation space dimension")
    action_type: Optional[ActionType] = Field(default=None, description="Action space type: discrete or continuous")
    action_dim: Optional[int] = Field(default=None, description="Action space dimension")
    device: DeviceType = Field(default="cpu", description="Training device: cpu (fast for small MLPs), gpu (large networks), auto (system decides)")

class EnvironmentStep(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(..., description="Gymnasium environment name")
    max_episode_steps: int = Field(default=500)
    success_threshold: float = Field(..., description="Reward threshold to consider solved")
    execution_timeout: int = Field(default=900, description="Maximum execution time in seconds for tester (default: 900 = 15 minutes)")
    # Environment specs for Coder/Tester
    obs_dim: Optional[int] = Field(default=None, description="Observation space dimension")
    action_type: Optional[ActionType] = Field(default=None, description="Action space type: discrete or continuous")
    action_dim: Optional[int] = Field(default=None, description="Action space dimension")
    device: DeviceType = Field(default="cpu", description="Training device: cpu (fast for small MLPs), gpu (large networks), auto (system decides)")

class Algorithm(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(default="PPO", description="Stable-Baselines3 algorithm")
    parameters: Dict[str, Any] = Field(default_factory=dict)

class Video(BaseModel):
    model_config = ConfigDict(extra='forbid')
    enabled: bool
    output_dir: str
    record_frequency: int

class HistoryWindow(BaseModel):
    model_config = ConfigDict(extra='forbid')
    manager: int = Field(default=0, description="Number of own previous messages manager can see")
    coder: int = Field(default=0, description="Number of own previous messages coder can see")
    tester: int = Field(default=0, description="Number of own previous messages tester can see")
    reviewer: int = Field(default=0, description="Number of own previous messages reviewer can see")
    env_switch_reports: int = Field(default=5, description="Number of previous environment switch reports to include")
    agent_opinions: int = Field(default=8, description="Number of previous agent opinions to show for team chatter")

class AgentOpinionsConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    enabled: bool = Field(default=False, description="Master switch for agent opinions feature")
    manager: bool = Field(default=True, description="Manager shares opinions")
    coder: bool = Field(default=False, description="Coder shares opinions (usually false)")
    tester: bool = Field(default=True, description="Tester shares opinions")
    reviewer: bool = Field(default=True, description="Reviewer (SHODAN) shares opinions")

class Agents(BaseModel):
    model_config = ConfigDict(extra='forbid')
    max_iterations: int
    show_thinking: bool = Field(default=False, description="Show agent thinking process output")
    show_coder_output: bool = Field(default=True, description="Show the code that Coder generates (syntax highlighted)")
    show_model_loading: bool = Field(default=False, description="Show Ollama model loading/unloading debug messages")
    show_env_switch_chatter: bool = Field(default=True, description="Show Manager's LinkedIn post and SHODAN's divine assessment when switching environments")
    history_window: HistoryWindow = Field(default_factory=HistoryWindow, description="Siloed conversation history per agent")
    agent_opinions: AgentOpinionsConfig = Field(default_factory=AgentOpinionsConfig, description="Agent opinions/chatter configuration")

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

class GpuConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    enabled: bool = Field(default=True, description="Enable GPU for RL training in Docker")
    max_vram_gb: float = Field(default=4.0, description="Max VRAM for RL model (reject if exceeds)")
    warn_vram_gb: float = Field(default=2.0, description="Warn if VRAM usage exceeds this")

# ═══════════════════════════════════════════════════════════════════════════════
# MONIVAIHEINEN TREENI - Validation -> Optimization -> Demo
# ═══════════════════════════════════════════════════════════════════════════════

class TrainingPhases(BaseModel):
    """Monivaiheinen treeni: ei tuhlata 2h jos koodi ei toimi"""
    model_config = ConfigDict(extra='forbid')
    enabled: bool = Field(default=False, description="Enable multi-phase training")
    validation_timeout_multiplier: float = Field(default=0.05, description="Validation phase timeout as fraction of base (5% = 0.05)")
    demo_timeout_seconds: int = Field(default=300, description="Demo phase timeout in seconds")

# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE MODEL SWITCHING - Vaihtaa mallia satunnaisesti kun agentti jumittuu
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveTriggers(BaseModel):
    """Triggerit mallin vaihdolle"""
    model_config = ConfigDict(extra='forbid')
    repeated_error_threshold: int = Field(default=2, description="Sama virhe N kertaa -> vaihda")
    no_improvement_iterations: int = Field(default=5, description="Reward ei parane N iteraatiossa -> vaihda")
    repetition_loop_threshold: int = Field(default=2, description="N peräkkäistä repetition loopia -> vaihda")
    timeout_threshold: int = Field(default=2, description="N peräkkäistä timeoutia -> vaihda")

class AgentModelPool(BaseModel):
    """Mallipooli yhdelle agentille"""
    model_config = ConfigDict(extra='forbid')
    models: list[str] = Field(..., description="List of models in pool - chosen randomly when stuck")

class AdaptiveModelSwitching(BaseModel):
    """Adaptiivinen mallinvaihto konfiguraatio"""
    model_config = ConfigDict(extra='forbid', protected_namespaces=())  # Allow model_pools field
    enabled: bool = Field(default=False, description="Enable adaptive model switching")
    chaos_mode: bool = Field(default=False, description="🎲 CHAOS MODE - random model EVERY call!")
    triggers: AdaptiveTriggers = Field(default_factory=AdaptiveTriggers, description="Triggers for model switching")
    model_pools: Dict[str, AgentModelPool] = Field(default_factory=dict, description="Model pools per agent")

class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    environment: Environment
    environment_progression: list[EnvironmentStep] = Field(default_factory=list, description="List of environments from easy to hard")
    video: Video
    agents: Agents
    llm: Llm
    agent_llm: AgentLLMConfig
    ollama: OllamaConfig
    gpu: GpuConfig = Field(default_factory=GpuConfig, description="GPU/VRAM settings for RL training")
    training_phases: TrainingPhases = Field(
        default_factory=TrainingPhases,
        description="Multi-phase training: validation -> optimization -> demo"
    )
    adaptive_model_switching: AdaptiveModelSwitching = Field(
        default_factory=AdaptiveModelSwitching,
        description="Adaptive model switching - switch model randomly when agent gets stuck"
    )
    prompts_file: str = Field(
        default="config/prompts.yaml",
        description="Path to prompts YAML file - swap to use different prompt sets"
    )
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
    def gpu(self) -> GpuConfig:
        return self.project.gpu

    @property
    def test_name(self) -> str:
        return self.project.test_name

    @property
    def adaptive_model_switching(self) -> AdaptiveModelSwitching:
        return self.project.adaptive_model_switching

    def get_prompt(self, agent_name: str) -> Dict[str, str]:
        """Get prompt dict for agent (e.g. 'manager')."""
        return self.prompts.get(agent_name, {})

def load_config(project_path: str = 'config/project.yaml') -> Config:
    """Load full config. Prompts path is read from project config's prompts_file field."""
    project = load_project_config(project_path)
    prompts = load_prompts(project.prompts_file)
    return Config(project, prompts)

if __name__ == "__main__":
    config = load_config()
    print(f"Environment: {config.environment.name} (max steps: {config.environment.max_episode_steps})")
    print(f"LLM Model: {config.llm.model}")
    print(f"Max iterations: {config.agents.max_iterations}")
    mgr_prompt = config.get_prompt("manager")
    print(f"Manager system preview: {mgr_prompt.get('system', '')[:100]}...")