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
1. **Validation** — Quick smoke test (does code run? ~2% timeout, 5000 steps max)
2. **Optimization** — Full training to reach reward threshold + model saved to `best_model.zip`
3. **Demo** — Tester's deterministic video recording (loads saved model, no LLM code generation)

Phase transitions happen INSIDE Manager's `__call__` — Manager updates the phase, clears old feedback, and immediately generates a new task for the new phase in the same call. This ensures Coder always gets a phase-appropriate task.

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
- RL code runs in `citadel-rl:latest` container (isolated, GPU-enabled)
- Network disabled, memory limited, code mounted read-only
- Video files written to mounted output directory

### Conversation Logging
- Output: `output/{run_id}/conversation.md` (GitHub markdown)
- Code in collapsible `<details>` blocks
- Agent chat as blockquotes with emoji
- Codex changes, phase transitions, environment switches logged
- Designed to be shareable on GitHub

### Deterministic Demo Video Recording
The demo phase does NOT rely on LLM-generated code. Instead:
1. Coder is prompted to always `model.save("/workspace/output/best_model.zip")` after optimization
2. Tester finds the `.zip` model file and stores `best_model_path` in state
3. In demo phase, Tester generates a hardcoded Python script (`generate_video_script()`) that:
   - Auto-detects SB3 algorithm (tries PPO/SAC/A2C/DQN/TD3)
   - Wraps env with RecordVideo
   - Runs 5 evaluation episodes
   - Falls back to LLM flow only if deterministic script fails

## Working Conventions
- Tee aktiivisesti repoon ohjaavia readme-tiedostoja
- Pyri tekemään CLAUDE.MD tiedostoja kaikkialle tarpeellisiin paikkoihin
- Päivitä CLAUDE.md JA README.md vastaamaan nykyistä projektin tilaa AINA kun teet muutoksia
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
