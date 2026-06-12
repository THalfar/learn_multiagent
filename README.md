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

**Two pipeline topologies** (pick with `pipeline: quad|duo` in the config):
- **quad** (above, default) — `Manager → Coder → Tester → Reviewer`: four LLM roles, the Tester paraphrases the run for the Reviewer.
- **duo** — `Director → Coder → Executor`: collapses the Manager+Reviewer into ONE frontier call (the **Director** judges the last run *and* writes the next task) and replaces the Tester LLM with a deterministic **Executor**, so the Coder reads the container's **raw stdout/stderr** directly — no "broken telephone". Run it with `config/duo_robot_seeded.yaml`. Both pipelines share the same gates, skills, checkpoint-resume and demo machinery.

Each environment goes through three phases:

| Phase | Goal | Timeout |
|-------|------|---------|
| **Validation** | Does the code run without errors? | ~2% of base timeout |
| **Optimization** | Reach the reward threshold, save model | Full timeout |
| **Demo** | Record video **+ confirm the measured metric clears the threshold** | 5 minutes |

The Coder always saves the trained model (`best_model.zip`) after optimization. In the demo phase, the pipeline bypasses LLM code generation entirely and runs a **deterministic recording+eval script** that auto-detects the SB3 algorithm, loads the saved model, evaluates **20 fixed-seed episodes** (recording video for the first 5), and prints a **metric-aware** RESULT (`success_rate` for goal envs, else `mean_reward`). A **demo-reward gate** then requires that measured metric to clear the threshold too — a convincing-looking video whose policy still misses the goal no longer counts as solved; instead the env **regresses to optimization** (the checkpoint keeps training). This both eliminates the old demo-phase loops *and* closes the "passed on video alone" hole.

After all three phases pass (demo metric included), the team advances to the next environment.

---

## Key Features

- **Environment progression** — Agents solve increasingly difficult Gymnasium environments (CartPole -> Pendulum -> MountainCar -> ...)
- **Multi-phase training** — Fast validation before committing to long optimization runs
- **SKILL memory** — Procedural skills (when-to-use / procedure / pitfalls / verification), not flat values, are injected into the **Coder** and the **Manager**, and persist to disk across runs. SHODAN adds/improves/verifies/removes them; `verified` skills are pinned AND take precedence over any other feedback — including SHODAN's own directives. Replaces the old flat "Divine Codex".
- **Pre-Docker lint** — A fast deterministic check (env name / imports / syntax / **SB3 kwarg validation against curated signatures** / the checkpoint-resume contract) gives the Coder instant feedback *before* the expensive Docker run — a "pre-Tester" that catches the #1 time-wasters in milliseconds (it does not replace the Tester's semantic diagnosis).
- **Chunked checkpoint-resume (ENFORCED)** — Each optimization iteration resumes the saved model + replay buffer and trains one more wall-clock-sized chunk, so progress ACCUMULATES across iterations (lets hard envs needing ~1M steps solve under a per-run timeout). The resume is *verified*, not assumed: the script must print `RESUMED: buffer_transitions=N` (N>0) when a checkpoint exists — the Tester rejects violations before Docker, and the Reviewer has a deterministic resume gate. Chunk size is computed from measured training speed (steps/s × timeout × 0.8). Goal-conditioned envs are scored by **success_rate** (`metric: success_rate`) over ≥20 fixed-seed eval episodes, not raw sparse reward.
- **Cumulative status** — `total_env_steps` + the per-chunk metric curve are shown to the Manager and the Reviewer every optimization iteration, so "flat curve with resume verified" (change approach) is distinguishable from "flat curve without resume" (fix the mechanism).
- **Adaptive model switching** — Automatically swaps LLM models when an agent gets stuck (repetition loops, repeated errors, reward stagnation)
- **Agent opinions** — Cross-agent "team chatter" where agents develop emergent personalities
- **Conversation logging** — Full GitHub-flavored markdown logs of every iteration, shareable and readable, with demo videos embedded inline
- **Live view** — Optional read-only browser dashboard that re-renders the conversation log in real time (great for presentations) — see `scripts/live_view.py`
- **Docker sandbox** — Isolated GPU execution with network disabled and code mounted read-only
- **Procedural skills** — On env-solve the Manager distils a verified, persistent procedural skill (algo + policy + HER + checkpoint + metric) into the SkillStore, injected into the next env's Coder (this replaced the old regex "playbook")
- **Automated diagnostics** — Rule-based, phase-aware error detection catches common failures before LLM analysis
- **Failsafe skip** — Automatically advances to next environment after too many consecutive failures

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

Install [Ollama](https://ollama.ai) and pull the models configured in `config/project.yaml`.

**Recommended set for an RTX 5090 (32GB VRAM, ~26GB free) — one coder + one general reasoner:**

```bash
ollama pull qwen3-coder:30b     # Coder    — 19GB, 256K ctx, MoE (3B active = fast)
ollama pull qwen3:14b           # Reasoner — 9.3GB, co-resides with the coder -> ~0 model swaps
# Optional max-quality reasoner (swaps with the coder, 2x/iter):
ollama pull qwen3:30b-thinking  #            19GB, thinking mode
```

| Role | Model | Size | Why |
|------|-------|------|-----|
| Coder | `qwen3-coder:30b` | 19GB | Strong + fast MoE coder |
| Manager + Tester | `qwen3:14b` | 9.3GB | Light reasoner, co-resident with coder |
| Reviewer (SHODAN) | `api` (Grok) | cloud | Best verdict drama — or run fully local, see `config/demo_local.yaml` |

> The two 30B models are MoE with only ~3B active parameters, so they run fast on Blackwell despite their size. Requires a recent Ollama (≥0.19, sm_120 / CUDA 12.8+). The model that an agent uses, its context window (`num_ctx`), and how long it stays in VRAM (`keep_alive`) are all set in config — see [Configuration](#agent-models).

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

### Ready-made configs

| Config | Use it for |
|--------|-----------|
| `config/project.yaml` | Full environment progression (CartPole → … → BipedalWalker) |
| `config/demo.yaml` | **AI-evening demo** — 2 fast envs, short run, predictable (adaptive switching off), cloud SHODAN |
| `config/demo_local.yaml` | Same demo, but **fully local** — SHODAN runs on the GPU too (no cloud) |
| `config/test_single.yaml` | **Smoke test** — one already-installed model for all local agents, quick end-to-end check |
| `config/night_run.yaml` | **Overnight run** — full hard env progression, honest scoreboard (solved vs skipped) |
| `config/robot_arm_blind.yaml` | **Intelligence test** — panda-gym manipulation (sparse + goal-conditioned). No HER hint: does the team discover the structural fix itself? |
| `config/robot_arm_seeded.yaml` | **Capability demo** — same, but the Codex is pre-seeded with the HER recipe → solves PandaPush/PickAndPlace |
| `config/duo_robot_seeded.yaml` | **Duo pipeline** — the seeded panda progression run as Director → Coder → Executor (1 LLM role; Coder reads raw output) |
| `config/duo_smoke.yaml` | **Duo 2-iteration sanity** — checks the duo wiring (needs Docker + Ollama) |

```bash
# Reliable short demo for a presentation
$env:PYTHONUTF8=1; python main.py config/demo.yaml
```

### Live view (for presentations)

While a run is going, open a browser dashboard that re-renders the conversation log every few seconds — agent chatter and SHODAN's verdicts appear live, and recorded demo videos play right in the page. It is read-only, so it cannot disturb the run. In a **second terminal**:

```bash
python scripts/live_view.py output/<run_id>     # then open http://localhost:8000
# `pip install markdown` for nicer rendering (optional)
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
  - name: "PandaPush-v3"
    success_threshold: 0.7
    metric: "success_rate"    # reward | success_rate (goal-conditioned envs use the is_success fraction in [0,1])
    execution_timeout: 1800
    device: "auto"
    tags: ["goal", "her", "manipulation", "robotics"]  # optional env-family tags for SKILL retrieval
```

### Agent models

```yaml
agent_llm:
  manager: "qwen3:14b"          # Light general reasoner (co-resident with the coder)
  coder: "qwen3-coder:30b"      # Specialized code generation (fast MoE)
  tester: "qwen3:14b"           # Same as manager = zero swaps Coder<->Tester
  reviewer: "api"                # Uses OPENAI_API_KEY from .env (or a local tag for fully-local)
```

Per-model context window and VRAM dwell time are config-driven, so adding a new model is a config edit, not a code change:

```yaml
ollama:
  keep_alive: "10m"             # How long a model stays in VRAM after a call (fewer reloads)
  options: { num_gpu: 999, num_thread: 8 }   # Applied to all models
  model_options:                # Per-model overrides (merged over the globals)
    "qwen3-coder:30b": { num_ctx: 32768 }
    "qwen3:14b":       { num_ctx: 16384 }
```

> On a 32GB RTX 5090, a ~9GB reasoner (`qwen3:14b`) and the ~19GB coder fit **at the same time**, so the pipeline runs with essentially no model swapping. Swap `manager`/`tester` to `qwen3:30b-thinking` for a stronger (but swapping) reasoner. When a swap does happen, a console spinner shows progress so it never looks frozen.

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

### SKILL memory (procedural) & the legacy Codex

The reviewer (SHODAN) inscribes **procedural skills** — *when to use / procedure / pitfalls / verification* — that are injected into the Coder and **persist to disk** (`skills/<dir>/skills.json` + browsable `<id>.SKILL.md`), so the team's knowledge accumulates across runs. `verified` skills are pinned (a later wrong guess can't overwrite them). When an env is solved the Manager distils a verified skill from the winning code.

```yaml
skills_dir: "skills"          # where the SkillStore persists (point runs at different dirs for a clean A/B)
initial_skills:               # optional structured seed (each: name/when_to_use/procedure/pitfalls/verification/tags)
  - name: "Goal-conditioned envs need HER"
    procedure: "SAC + MultiInputPolicy + HerReplayBuffer; chunked checkpoint-resume; report success_rate"
    tags: ["goal", "her"]
shodan_rules: { enabled: true, max_rules: 20 }   # legacy flat Codex (still works, surfaced as "CODEX NOTES")
```

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
  demo.yaml                  # Short, predictable presentation run (cloud SHODAN)
  demo_local.yaml            # Same demo, fully local (SHODAN on the GPU)
  test_single.yaml           # Smoke test with one already-installed model
  opus_prompts.yaml          # Minimal-constraint agent prompts
  prompts.yaml               # Original detailed prompts (SHODAN persona)
scripts/
  live_view.py               # Auto-refreshing browser view of conversation.md (presentations)
src/
  graph.py                   # LangGraph state machine (AgentState + flow)
  config_loader.py           # Pydantic models + YAML loading
  agents/
    base.py                  # BaseAgent: LLM calls, history, context tracking, model swap
    manager.py               # Task assignment, phase transitions, env progression
    coder.py                 # RL training script generation
    tester.py                # Docker execution, metric extraction, deterministic video recording
    reviewer.py              # Code review, approval/rejection, Divine Codex
  utils/
    conversation_logger.py   # GitHub markdown conversation logs
    model_switcher.py        # Adaptive model switching on stuck detection
    code_lint.py             # Deterministic pre-Docker lint (env name / imports / syntax / SB3 kwargs / resume contract)
    banners.py               # Rich console output formatting
    timer.py                 # Runtime statistics and token tracking
  skills/
    skill_store.py           # Procedural skill memory (pinned, persistent, semantic-search-ready)
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
