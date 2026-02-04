# LangGraph RL Dev Team

Multi-agent RL system: LLM agents collaborate to solve Gymnasium environments autonomously using Stable-Baselines3.

## Architecture

```
Manager --> Coder --> Tester --> Reviewer --+
  ^                                        |
  +--- (approved? next env : retry) -------+
```

**Agents:**
- **Manager** - Assigns tasks, picks algorithms/hyperparameters, drives strategy
- **Coder** - Generates pure Python RL training scripts
- **Tester** - Runs code in Docker (GPU), extracts metrics, validates results
- **Reviewer** - Reviews code quality + results, approves/rejects

**Key features:**
- Environment progression from easy to hard (CartPole -> Humanoid)
- Multi-phase training: validation (smoke test) -> optimization -> demo (video)
- Swappable prompt configurations for different agent personalities
- Agent opinions system for cross-agent "dialogue"
- Adaptive model switching when agents get stuck
- Docker sandbox with GPU (CUDA 12.8)

## Setup

```bash
conda create -n langgraph-rl python=3.11 -y
conda activate langgraph-rl
pip install -r requirements.txt
```

**.env** (API key for frontier model):
```
OPENAI_API_KEY=xai-your-key
OPENAI_BASE_URL=https://api.x.ai/v1
```

Docker (for code execution):
```bash
docker build -t rl-sandbox -f docker/Dockerfile .
```

## Usage

```bash
python main.py                       # default: config/project.yaml
python main.py config/my_run.yaml    # custom project config
```

Output goes to `output/{test_name}/{timestamp}/` with videos, statistics, and conversation logs.

## Quick Start - Opus Prompts (free agents)

`opus_prompts.yaml` gives agents maximum freedom - no assigned personas, no scripted behavior. Only rule: get the work done.

```bash
python main.py config/project.yaml
```

The project config points to prompt file via `prompts_file` field. Default `config/project.yaml` is already set up for opus prompts. Results in `output/{test_name}/{timestamp}/`.

## Prompt Configurations

The system loads agent prompts from a separate YAML file, configured in `config/project.yaml`:

```yaml
prompts_file: "config/prompts.yaml"     # Original - structured personas (SHODAN reviewer)
prompts_file: "config/opus_prompts.yaml" # Opus - maximum agent freedom
```

### Available prompt sets

| File | Philosophy | Reviewer persona |
|------|-----------|-----------------|
| `config/prompts.yaml` | Detailed instructions, assigned personas | SHODAN - god-like AI, mocks "inferior processors" |
| `config/opus_prompts.yaml` | Minimal constraints, emergent personality | Open - "who you are is up to you" |

### Running with opus_prompts

1. Set the prompt file in `config/project.yaml`:
   ```yaml
   prompts_file: "config/opus_prompts.yaml"
   test_name: "opus_prompts"
   ```

2. Run:
   ```bash
   python main.py
   ```

3. Results appear in `output/opus_prompts/{timestamp}/`

### Comparing prompt sets

Run the same environments with different prompts and compare:

```yaml
# Run 1: original prompts
prompts_file: "config/prompts.yaml"
test_name: "original_prompts"

# Run 2: opus prompts
prompts_file: "config/opus_prompts.yaml"
test_name: "opus_prompts"
```

Results are organized by test_name:
```
output/
  original_prompts/
    20250204_143000/
      statistics.json
      videos/
    20250204_160000/
      ...
  opus_prompts/
    20250204_150000/
      statistics.json
      videos/
```

### Creating your own prompt set

1. Copy an existing prompt file:
   ```bash
   cp config/prompts.yaml config/my_prompts.yaml
   ```

2. Edit the prompts. Each agent needs:
   - `system` - System prompt (personality + instructions)
   - `task_template` - Per-iteration template with `{placeholders}`

   Required agents: `manager`, `coder`, `tester`, `reviewer`

   Reviewer also needs: `environment_switch_report_template`

3. Point to it:
   ```yaml
   prompts_file: "config/my_prompts.yaml"
   test_name: "my_experiment"
   ```

## Configuration

All in `config/project.yaml`:

| Section | What it controls |
|---------|-----------------|
| `environment_progression` | List of Gymnasium envs from easy to hard |
| `environment` | Current env (auto-set from progression) |
| `agents` | Max iterations, history windows, agent opinions |
| `llm` / `agent_llm` | Model selection per agent (api or ollama) |
| `adaptive_model_switching` | Auto-switch models when stuck |
| `training_phases` | Validation -> optimization -> demo pipeline |
| `prompts_file` | Which prompt YAML to load |
| `test_name` | Groups output runs under this name |

## Project Structure

```
config/
  project.yaml          # Main configuration
  prompts.yaml          # Original prompt set (SHODAN)
  opus_prompts.yaml     # Opus prompt set (free agents)
src/
  agents/
    base.py             # BaseAgent - LLM wrapper, history, model switching
    manager.py          # Task assignment + environment progression
    coder.py            # Code generation (pure Python output)
    tester.py           # Docker execution + metric extraction
    reviewer.py         # Code review + approval/rejection
  utils/
    banners.py          # Console output formatting
    conversation_logger.py  # Logs agent interactions
    model_switcher.py   # Adaptive model switching
    timer.py            # Runtime statistics
  config_loader.py      # Pydantic models + YAML loading
  graph.py              # LangGraph workflow orchestration
docker/
  Dockerfile            # GPU sandbox (CUDA 12.8, SB3, Gymnasium, MuJoCo)
main.py                 # Entry point
```
