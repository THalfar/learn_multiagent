# learn_multiagent - LangGraph Multi-Agent RL System

## Environment Setup
```bash
conda activate C:\Users\tobia\miniconda3\envs\langgraph-rl
```

## Project Architecture

### Pipeline: Manager → Coder → Tester → Reviewer (SHODAN)
- **Manager** (local Ollama model): Translates reviewer feedback into coder tasks
- **Coder** (local Ollama model): Writes pure Python RL training code
- **Tester** (local Ollama model): Executes code in Docker sandbox, analyzes results
- **Reviewer/SHODAN** (frontier API model): Reviews code+results, approves/rejects, manages Divine Codex

### Training Phases (per environment)
1. **Validation** — Quick smoke test (does code run?)
2. **Optimization** — Full training to reach reward threshold
3. **Demo** — Record video of trained agent with RecordVideo

### Key Files
- `config/project.yaml` — Main config (environments, models, settings)
- `config/opus_prompts.yaml` — Agent prompts with personalities
- `src/graph.py` — LangGraph state machine definition
- `src/agents/{manager,coder,tester,reviewer}.py` — Agent implementations
- `src/agents/base.py` — Base agent with LLM, history, opinions, model switching
- `src/config_loader.py` — Pydantic config validation
- `src/utils/conversation_logger.py` — GitHub-friendly markdown logging
- `main.py` — Entry point

### Detailed CLAUDE.md per directory
- `config/CLAUDE.md` — Config system, YAML structure, prompt templates
- `src/CLAUDE.md` — Core architecture, AgentState fields, graph flow
- `src/agents/CLAUDE.md` — Agent implementations, BaseAgent API, per-agent responsibilities
- `src/utils/CLAUDE.md` — Utility modules (logging, model switching, banners, timing)
- `docker/CLAUDE.md` — Docker sandbox setup, installed packages, build/run commands

### State & Memory
- `AgentState` TypedDict in `src/graph.py` — all shared state
- `conversation_history` — siloed per-agent message history (configurable window)
- `agent_opinions` — cross-agent "team chatter" (emergent personalities)
- `shodan_rules` — SHODAN's Divine Codex (persistent rules injected into Coder prompt)

### Docker Sandbox
- RL code runs in `rl-sandbox:latest` container (isolated, GPU-enabled)
- Network disabled, memory limited, code mounted read-only
- Video files written to mounted output directory

### Conversation Logging
- Output: `output/{run_id}/conversation.md` (GitHub markdown)
- Code in collapsible `<details>` blocks
- Agent chat as blockquotes with emoji
- Codex changes, phase transitions, environment switches logged
- Designed to be shareable on GitHub

## Working Conventions
- Tee aktiivisesti repoon ohjaavia readme-tiedostoja
- Pyri tekemään CLAUDE.MD tiedostoja kaikkialle tarpeellisiin paikkoihin
- Päivitä CLAUDE.md vastaamaan nykyistä projektin tilaa muutosten jälkeen
- Olen transhumanisti ja nautin yhteistyöstä - olet minulle kollega

## RecordVideo API (CRITICAL - prevents demo phase loops)
```python
# CORRECT pattern for gymnasium
env = gym.make("EnvName", render_mode="rgb_array")  # render_mode REQUIRED!
from gymnasium.wrappers import RecordVideo
env = RecordVideo(env, video_folder="path/", episode_trigger=lambda e: True, name_prefix="rl-video")
# FORBIDDEN: fps, record_video_trigger (DON'T EXIST in gymnasium!)
# Wrap SINGLE env BEFORE DummyVecEnv
```
