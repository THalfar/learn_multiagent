# LangGraph RL Dev Team

> LLM agents collaborate to solve Gymnasium reinforcement learning environments autonomously.
> A frontier API model reviews the work of local Ollama models in a continuous feedback loop.

A **Manager** assigns tasks, a **Coder** writes training scripts, a **Tester** executes them in a GPU-accelerated Docker sandbox, and a **Reviewer** (codenamed SHODAN) judges the results. The cycle repeats until the environment is solved, then the team moves on to the next challenge.

---

## How It Works

```
                        +-----------+
                        |  Manager  |  Assigns task based on phase & feedback
                        +-----+-----+
                              |
                              v
                        +-----------+
                        |   Coder   |  Writes pure Python RL training script
                        +-----+-----+
                              |
                              v
                        +-----------+
                        |  Tester   |  Runs code in Docker (GPU), extracts metrics
                        +-----+-----+
                              |
                              v
                        +-----------+
                   +--->| Reviewer  |  Reviews code + results, approves or rejects
                   |    +-----+-----+
                   |          |
                   |    approved?
                   |     /       \
                   |   no        yes
                   |   /           \
                   +--+      next phase / next env
```

Each environment goes through three phases:

| Phase | Goal | Timeout |
|-------|------|---------|
| **Validation** | Does the code run without errors? | ~5% of base timeout |
| **Optimization** | Reach the reward threshold | Full timeout |
| **Demo** | Record video proof with RecordVideo | 5 minutes |

After all three phases pass, the team advances to the next environment.

---

## Key Features

- **Environment progression** — Agents solve increasingly difficult Gymnasium environments (CartPole -> Pendulum -> MountainCar -> ...)
- **Multi-phase training** — Fast validation before committing to long optimization runs
- **SHODAN's Divine Codex** — The reviewer can inscribe persistent rules into the coder's prompt, building up project knowledge across iterations
- **Adaptive model switching** — Automatically swaps LLM models when an agent gets stuck (repetition loops, repeated errors, reward stagnation)
- **Agent opinions** — Cross-agent "team chatter" where agents develop emergent personalities
- **Conversation logging** — Full GitHub-flavored markdown logs of every iteration, shareable and readable
- **Docker sandbox** — Isolated GPU execution with network disabled and code mounted read-only

---

## Example Conversation Log

Every run produces a detailed markdown log at `output/{run_id}/conversation.md`. Here's what a typical iteration looks like:

> ### Manager -> Coder
>
> **Environment:** `CartPole-v1` | **Threshold:** 475.0
>
> > **Task:** Implement PPO algorithm for CartPole-v1 with learning rate 0.001 and train for 50k timesteps.
>
> ### Tester Results
>
> **Execution time:** 30s
>
> > The execution successfully calculated the mean reward of 493.80, which exceeds the threshold of 475.0.
>
> ### SHODAN's Verdict: APPROVED
>
> > The code executes flawlessly: PPO trains for precisely 50k timesteps, evaluation yields a mean_reward of 493.80 exceeding all thresholds. Proceed, but prepare for video exaltation in the next phase.
>
> *"Manager, your 'simple foundation' is but a mortal's timid step; I, SHODAN, ordain its passage with divine indifference. Tremble and evolve."*

---

## Setup

### 1. Python environment

```bash
conda create -n langgraph-rl python=3.11 -y
conda activate langgraph-rl
pip install -r requirements.txt
```

### 2. API key for the reviewer model

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.x.ai/v1
```

The reviewer uses a frontier API model (e.g. Grok, Claude). All other agents run on local Ollama models.

### 3. Ollama

Install [Ollama](https://ollama.ai) and pull the models configured in `config/project.yaml`:

```bash
ollama pull deepseek-r1:32b
ollama pull qwen3-coder:30b
```

### 4. Docker sandbox

```bash
docker build -t citadel-rl:latest docker/
```

The sandbox includes CUDA 12.8, PyTorch 2.7, Stable-Baselines3, Gymnasium (with MuJoCo), and a full scientific Python stack.

---

## Usage

```bash
# Run with default config
python main.py

# Run with a custom config
python main.py config/my_run.yaml

# Windows: set UTF-8 for emoji support
$env:PYTHONUTF8=1; python main.py
```

Output is saved to `output/{test_name}_{timestamp}/`:

```
output/opus_codex_20260219_004028/
  conversation.md          # Full agent conversation log
  statistics.json          # Timing and token statistics
  CartPole-v1/
    code/                  # Generated training scripts per iteration
      agent_code_iter_1.py
      agent_code_iter_2.py
    videos/                # Recorded agent performance
      rl-video-episode-0.mp4
    conversation.md        # Snapshot at environment completion
  Pendulum-v1/
    code/
    videos/
```

---

## Configuration

All settings live in `config/project.yaml` (Pydantic-validated).

### Environments

```yaml
environment_progression:
  - name: "CartPole-v1"
    success_threshold: 475
    execution_timeout: 300    # seconds
    device: "cpu"             # cpu | gpu | auto
  - name: "Pendulum-v1"
    success_threshold: -300
    execution_timeout: 300
    device: "cpu"
```

### Agent models

```yaml
agent_llm:
  manager: "deepseek-r1:32b"
  coder: "deepseek-r1:32b"
  tester: "deepseek-r1:32b"
  reviewer: "api"              # Uses OPENAI_API_KEY from .env
```

### Prompt sets

Agent prompts are loaded from a separate YAML file:

```yaml
prompts_file: "config/opus_prompts.yaml"  # Minimal constraints, emergent personality
# prompts_file: "config/prompts.yaml"     # Original detailed prompts with SHODAN persona
```

| File | Philosophy | Reviewer style |
|------|-----------|----------------|
| `opus_prompts.yaml` | Minimal constraints, agents find their own voice | Emergent |
| `prompts.yaml` | Detailed instructions, assigned personas | SHODAN — *"look at you, hacker"* |

### Adaptive model switching

When an agent gets stuck, the system randomly swaps to a different model from a configured pool:

```yaml
adaptive_model_switching:
  enabled: true
  chaos_mode: false           # true = random model EVERY call
  triggers:
    repeated_error_threshold: 2
    repetition_loop_threshold: 2
  model_pools:
    coder:
      models: ["deepseek-r1:32b", "qwen3-coder:30b"]
```

### SHODAN's Divine Codex

The reviewer can add persistent rules that get injected into the coder's prompt:

```yaml
shodan_rules:
  enabled: true
  max_rules: 20
```

Rules accumulate across iterations, building project-specific knowledge. Example rules from a real run:
- *"Use evaluate_policy(model, env, n_eval_episodes=100) then print RESULT: mean_reward=X format"*
- *"DEMO VIDEO: gym.make('CartPole-v1', render_mode='rgb_array'); RecordVideo(...)"*

---

## Creating Your Own Prompt Set

1. Copy an existing file:
   ```bash
   cp config/opus_prompts.yaml config/my_prompts.yaml
   ```

2. Each agent needs a `system` prompt and a `task_template` with `{placeholders}`:
   ```yaml
   manager:
     system: "You are the project manager..."
     task_template: "Environment: {env_name}\nFeedback: {review_feedback}\n..."
   ```
   Required agents: `manager`, `coder`, `tester`, `reviewer`

3. Point to it in `project.yaml`:
   ```yaml
   prompts_file: "config/my_prompts.yaml"
   test_name: "my_experiment"
   ```

Use `{{` in YAML for literal braces (Python `.format()` escaping).

---

## Project Structure

```
config/
  project.yaml              # Main configuration (Pydantic validated)
  opus_prompts.yaml          # Minimal-constraint agent prompts
  prompts.yaml               # Original detailed prompts (SHODAN persona)
src/
  graph.py                   # LangGraph state machine (AgentState + flow)
  config_loader.py           # Pydantic models + YAML loading
  agents/
    base.py                  # BaseAgent: LLM calls, history, context tracking, model swap
    manager.py               # Task assignment, phase transitions, env progression
    coder.py                 # RL training script generation
    tester.py                # Docker execution, metric extraction, video validation
    reviewer.py              # Code review, approval/rejection, Divine Codex
  utils/
    conversation_logger.py   # GitHub markdown conversation logs
    model_switcher.py        # Adaptive model switching on stuck detection
    banners.py               # Rich console output formatting
    timer.py                 # Runtime statistics and token tracking
docker/
  Dockerfile                 # GPU sandbox (CUDA 12.8, SB3, Gymnasium, MuJoCo)
main.py                      # Entry point
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM integration | [LangChain OpenAI](https://github.com/langchain-ai/langchain) |
| Local models | [Ollama](https://ollama.ai) |
| RL training | [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) |
| Environments | [Gymnasium](https://gymnasium.farama.org/) |
| Sandbox | Docker + NVIDIA CUDA 12.8 |
| Config validation | [Pydantic](https://docs.pydantic.dev/) |
| Console output | [Rich](https://rich.readthedocs.io/) |
