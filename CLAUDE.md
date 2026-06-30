# learn_multiagent - LangGraph Multi-Agent RL System

## Environment Setup
```bash
conda activate C:\Users\tobia\miniconda3\envs\langgraph-rl
```

## Project Architecture

### Two pipeline topologies (config `pipeline: quad|duo`, dispatched in `main.py`)

**Quad** (default) — `Manager → Coder → Tester → Reviewer (SHODAN)`:
- **Manager** (local Ollama model): Translates reviewer feedback into coder tasks
- **Coder** (local Ollama model): Writes pure Python RL training code
- **Tester** (local Ollama model): Executes code in Docker sandbox, analyzes results
- **Reviewer/SHODAN** (frontier API model): Reviews code+results, approves/rejects, manages Divine Codex

**Duo** — `Director → Coder → Executor` (1 LLM role): collapses the Manager+Reviewer into ONE frontier call and replaces the Tester LLM with a deterministic node:
- **Director** (frontier API model, `class Director(Reviewer)`): strategist + judge + taskmaster in ONE `__call__` — judges run N−1 AND writes the task for run N. Keeps the SHODAN persona, the deterministic gates, skill_ops, the failsafe, and the full phase machine.
- **Coder** (local Ollama model): same Coder, but it reads the Executor's **RAW stdout/stderr** directly (`_format_raw_output`) — no Tester paraphrase ("broken telephone"). Gets a team-chatter voice via its `chat_template`.
- **Executor** (`src/agents/executor.py`, NO LLM — not a BaseAgent): runs the sandbox deterministically and emits a factual report (raw output + AUTOMATED DIAGNOSTICS + parsed RESULT + resume/cumulative tracking + deterministic demo recording). It reuses the Tester's container/diagnostic/video helpers so both pipelines run code identically.
- LangGraph contract: only the Director returns `iteration: 1` (operator.add); Coder/Executor never do, so one director→coder→executor cycle advances by exactly one.
- Run it: `python main.py config/duo_robot_seeded.yaml` (night run) or `config/duo_smoke.yaml` (2-iter sanity). Uses the Grok `api` path (no Anthropic).

### Training Phases (per environment)
1. **Validation** — Quick smoke test (does code run? timeout = `max(validation_timeout_floor, multiplier × base)` — the floor exists because framework/env startup eats a FIXED ~10–40 s the multiplier ignores)
2. **Optimization** — Full training to reach reward threshold + model saved to `best_model.zip`
3. **Demo** — deterministic video recording (loads saved model, no LLM code generation) **+ a reward gate**: the demo evaluates the saved model over 20 fixed-seed episodes and prints a metric-aware RESULT (`success_rate` for goal envs, else `mean_reward`); the **demo-reward gate** requires that measured metric to ALSO clear the threshold. If it doesn't (e.g. PandaPush optimization hit 0.5 but the demo measured 0.40), the env does NOT solve — it **regresses to optimization** (checkpoint preserved, training continues), instead of being passed on videos alone. A **below-threshold** demo rejection does NOT burn the failsafe budget (the model already proved progress, and the regression keeps training); but a demo that yields **no measurement** (`demo_reward=None` — crash / timeout / no saved model) does NOT regress, so it DOES advance `consecutive_failures` — otherwise a deterministically broken demo (the duo Executor has no LLM fallback) loops forever; advancing the counter lets the failsafe bound it. The demo timeout scales with the env (`max(demo_timeout_seconds, 0.5 × base)`) since the deterministic demo evaluates 20 fixed-seed episodes (a flat 300 s could time out a long-horizon env into the None path). State: `demo_reward` / `demo_below_threshold`.

Phase transitions happen INSIDE Manager's `__call__` (quad) or the Director's `__call__` (duo) — Manager updates the phase, clears old feedback, and immediately generates a new task for the new phase in the same call. This ensures Coder always gets a phase-appropriate task. Environment switches (solved / failsafe / LLM-requested) all go through `Manager._env_switch_reset()`, which sets a CONCRETE validation task + matching `manager_guidance` for the new env — an empty/stale task here caused the "stale-task race" (Coder coded the new env, Reviewer judged against the old env's task; one wasted iteration per switch).

### Checkpoint-Resume & Metrics (CRITICAL for hard envs)
- **Chunked checkpoint-resume**: each optimization iteration = one wall-clock-bounded CHUNK that resumes the saved model, so progress ACCUMULATES across iterations (the only way envs like PandaPickAndPlace, needing ~1M steps, solve under a per-run timeout): `SAC.load(path, env=env)` → `load_replay_buffer` (off-policy/HER needs this or it silently forgets everything) → `print(f"RESUMED: buffer_transitions={model.replay_buffer.size()}")` → `learn(N)` (DEFAULT reset_num_timesteps — **NEVER `=False`**, it breaks episode termination on a reloaded model) → save model + `save_replay_buffer`.
- **Resume is ENFORCED, not advisory** (lesson from PandaPush 2026-06-10: 24 iterations of "fresh 50k chunks" while everyone believed training accumulated): when a checkpoint exists in optimization, (1) the Coder's lint requires the resume contract (`check_resume_contract`), (2) the Tester skips Docker and rejects a script that violates it, (3) after the run the Tester verifies `RESUMED: buffer_transitions=N` with N>0 in stdout, and (4) the Reviewer has a deterministic **resume gate** (like the threshold gate) that overrides APPROVE if the proof is missing. State: `resume_required` / `resume_ok`.
- **Chunk sizing from measured SPS**: the Tester measures steps/s from every run ≥5000 steps (`measured_sps` state); the Manager's optimization instruction computes chunk ≈ `SPS × timeout × 0.8` — no more step-count roulette against the timeout.
- **Cumulative visibility**: `total_env_steps` + `metric_history` (per env, reset on switch) are injected into both the Manager's and SHODAN's prompts, so a flat curve over resumed chunks reads as "change approach" and a flat curve WITHOUT verified resume reads as "fix the mechanism".
- **success_rate metric**: goal-conditioned envs (Dict obs with `desired_goal`) are scored by the `is_success` fraction in [0,1], not the raw sparse reward. Set `metric: "success_rate"` on the env step; the reviewer threshold gate then also rejects a raw-reward report that drifts outside [0,1]. Optimization evals use ≥20 episodes with fixed seeds (a 10-episode success_rate is mostly variance — the Tester emits an EVAL NOISE diagnostic below 20).

### Learning Mechanisms
- **SKILL substrate** (`src/skills/skill_store.py`) — the primary memory. Procedural skills (when_to_use / procedure / pitfalls / verification), NOT flat values. Persistent on disk (`skills/<dir>/skills.json` + `<id>.SKILL.md`), so learning accumulates across runs. Injected into the **Coder** AND the **Manager** (phase instruction). SHODAN manages them via `skill_ops` {add, improve, verify, remove}; `verified` skills are PINNED (a weaker write can't clobber them). **Precedence rule**: verified skills outrank ANY other feedback — including SHODAN's directives; the Manager is told to follow the skill and flag the conflict, and SHODAN's prompt forbids directives that contradict a verified skill ("THE PIN BINDS YOU TOO"). Lesson: PandaPush 2026-06-10, where SHODAN's "train from scratch, no load logic" overrode the correct seeded skill for 20 iterations. When an env is solved the Manager distils a verified procedural skill from the winning code (`_skill_from_winning_code`). Seeded via config `initial_skills`; `skills_dir` isolates runs (seeded vs blind A/B).
- **Pre-Docker lint** (`src/utils/code_lint.py`) — a fast "pre-Tester" feedback arc: ast syntax + exact env name + import whitelist + **SB3 kwarg validation** (curated signatures for SAC/PPO/DQN/TD3/A2C/DDPG/HerReplayBuffer/`.learn()`; a hallucinated kwarg like `online_sample_strategy` is a hard error — it burned 7 iterations in one night run) + the **checkpoint-resume contract** (`require_resume=True` in optimization when a checkpoint exists), in ~ms. The Coder lint-retries (K=2) before the expensive Docker run; the Tester also lint-backstops (skips Docker on structural errors). NOT a bypass — the Tester still runs the container and does the semantic diagnosis.
- **Coder self-memory** (`recent_attempts` state) — the Coder sees its last ~2 attempts + the Tester's diagnosis + the Reviewer's verdict, so it doesn't repeat a corrected mistake.
- **Manager escalation ladder** (`failure_history` state) — if the same failure mode repeats 3×, the Manager is told to change the strategy CLASS (e.g. 3× timeout → checkpoint-resume; 3× resume_violation → spell out the exact load/print/save lines in the task), not the parameter value.
- **Tester's Pattern Library** — rule-based `diagnose_common_issues(..., phase=...)` catches known failures BEFORE LLM analysis (phase-aware: the "no RESULT / no MODEL_SAVED" rules are skipped in the demo phase, where a correct script legitimately neither trains nor saves). Findings appended to stderr as `=== AUTOMATED DIAGNOSTICS ===`.
- *(Removed) Manager's Playbook* — the legacy regex recipe (algo/steps/device) was fully superseded by the SKILL substrate (`_skill_from_winning_code` distils a richer, persistent, Coder-injected procedural skill on env-solve) and deleted.
- **Failsafe** — progress-aware: the consecutive-failure counter RESETS on a new best metric, so an env that keeps improving is never skipped; only a stuck (non-improving) env is abandoned after N (`failsafe.skip_after_consecutive_failures`).

### Key Files
- `config/project.yaml` — Main config (environments, models, settings)
- `config/demo.yaml` — Short, predictable presentation run (cloud SHODAN)
- `config/demo_local.yaml` — Same demo, fully local (SHODAN on the GPU, reviewer = local tag)
- `config/test_single.yaml` — Smoke test: one already-installed model for all local agents
- `config/opus_prompts.yaml` — Quad agent prompts with personalities (LEGACY: still contains env→algo recipe tables)
- `config/duo_prompts.yaml` — Duo prompts: the SHODAN Director (no Manager/Tester, four-gate JSON contract) + the Coder (raw-output paragraph + chat_template). **RECIPE-FREE by design** ("general intelligence" principle): prompts teach METHOD (discovery → measure → escalate → inscribe skills) + harness FACTS (container inventory, RESULT/RESUMED contracts), but contain ZERO env names, algorithm names, or step-count tables — `smoke_test_duo.py` has a no-recipe guard that fails if one leaks back in. Env-specific knowledge lives ONLY in the SkillStore (seeded via `initial_skills` or EARNED by runs). Validation tasks demand a discovery printout (observation_space / action_space / env.spec / one step's info dict) as the evidence base for choosing the approach.
- `config/duo_robot_seeded.yaml` / `config/duo_smoke.yaml` — Duo night run / 2-iter sanity
- `src/graph.py` — Quad LangGraph state machine (`AgentState` lives here)
- `src/duo_graph.py` — Duo LangGraph state machine (`create_duo_graph`: director/coder/executor)
- `src/agents/{manager,coder,tester,reviewer}.py` — Quad agent implementations
- `src/agents/director.py` — Duo Director (`class Director(Reviewer)`): verdict + task + phase machine in one call
- `src/agents/executor.py` — Duo Executor: deterministic sandbox node (NO LLM, not a BaseAgent)
- `src/agents/env_transitions.py` — `initial_validation_task` / `env_switch_reset` / `skill_from_winning_code` as module fns shared by Manager (delegates) and Director
- `src/agents/base.py` — Base agent with LLM, history, opinions, model switching
- `src/config_loader.py` — Pydantic config validation (`pipeline: quad|duo`, env `metric: reward|success_rate`, env `tags`, `skills_dir`, `initial_skills`)
- `src/skills/skill_store.py` — SKILL substrate (procedural, pinned, persistent, semantic-search-ready)
- `src/utils/code_lint.py` — deterministic pre-Docker lint (env name / imports / syntax / SB3 kwargs / resume contract)
- `src/utils/result_parser.py` — `parse_result_line()`: the single canonical parser for the Coder's `RESULT:` line (Tester/Executor + Reviewer/Director share it; no drifting copies)
- `src/utils/verdict_gates.py` — `apply_verdict_gates()`: the deterministic threshold / metric-lock / resume / **demo** gates that override an LLM APPROVE (LLM-free → unit-testable; shared by Reviewer + Director)
- `src/utils/json_extract.py` — shared stack-based `extract_json` (Director only; quad Manager/Reviewer keep their inner copies = zero risk)
- `src/utils/conversation_logger.py` — GitHub-friendly markdown logging (incl. `log_video`)
- `scripts/live_view.py` — Read-only auto-refresh browser view of conversation.md (presentations)
- `scripts/smoke_test_fixes.py` / `scripts/smoke_test_duo.py` — deterministic smoke suites (quad regression guard / duo pipeline + gates)
- `main.py` — Entry point (dispatches quad vs duo on `config.pipeline`)

### Models & RTX 5090 (32GB VRAM)
- Local models run via Ollama (OpenAI-compatible endpoint); reviewer uses a frontier API by default.
- Default set: **Coder** `qwen3-coder:30b` (19GB MoE, fast) + **Manager/Tester** `qwen3:14b` (9.3GB) → co-resident in VRAM, ~0 model swaps. Reviewer `api` (Grok) or a local tag.
- Model assignment is config-only (`agent_llm`). Per-model context (`ollama.model_options.<tag>.num_ctx`) and VRAM dwell time (`ollama.keep_alive`, default "5m") are config-driven too — adding a model needs no code change. `get_model_context_size()`/`get_ollama_options()` in `base.py` read these.
- A `rich` console spinner wraps model swaps and LLM generation so long pauses never look like a hang.

### Detailed CLAUDE.md per directory
- `config/CLAUDE.md` — Config system, YAML structure, prompt templates
- `src/CLAUDE.md` — Core architecture, AgentState fields, graph flow
- `src/agents/CLAUDE.md` — Agent implementations, BaseAgent API, per-agent responsibilities
- `src/utils/CLAUDE.md` — Utility modules (logging, model switching, banners, timing, code_lint)
- `src/skills/CLAUDE.md` — SkillStore (procedural memory substrate)
- `docker/CLAUDE.md` — Docker sandbox setup, installed packages, build/run commands

### State & Memory
- `AgentState` TypedDict in `src/graph.py` — all shared state
- `conversation_history` — siloed per-agent message history (configurable window)
- `agent_opinions` — cross-agent "team chatter" (emergent personalities)
- `skill_store` — the SkillStore instance (procedural memory, persists to disk); rendered into the Coder prompt
- `recent_attempts` — Coder self-memory (last ~2 attempts + Tester diagnosis + Reviewer verdict)
- `failure_history` — failure-mode history for the Manager's escalation ladder
- `shodan_rules` — legacy flat Codex (still inscribable; now surfaced to the Coder as "CODEX NOTES" alongside skills)

### Docker Sandbox
- RL code runs in `citadel-rl:latest` container (isolated, GPU-enabled)
- Network disabled, memory limited, code mounted read-only
- Video files written to mounted output directory

### Conversation Logging
- Output: `output/{run_id}/conversation.md` (GitHub markdown)
- Code in collapsible `<details>` blocks
- Agent chat as blockquotes with emoji
- Codex changes, phase transitions, environment switches logged
- Demo videos embedded inline via `logger.log_video()` (relative-path `<video>` + link; plays locally and serves through `scripts/live_view.py`)
- Designed to be shareable on GitHub

### Deterministic Demo Video Recording
The demo phase does NOT rely on LLM-generated code. Instead:
1. Coder is prompted to always `model.save("/workspace/output/best_model.zip")` after optimization
2. Tester/Executor finds the `.zip` model file and stores `best_model_path` in state
3. In demo phase, `generate_video_script(env, model, out, metric=..., n_eval_episodes=20, n_video_episodes=5)` builds a hardcoded script that:
   - Auto-detects SB3 algorithm (tries PPO/SAC/A2C/DQN/TD3)
   - Evaluates **20 episodes with FIXED seeds** (`reset(seed=2000+ep)`) so the demo metric is reproducible; records video for only the first 5 (`episode_trigger=lambda e: e < 5`)
   - Prints a **metric-aware** RESULT: `success_rate=` (the `is_success` fraction) for goal envs, else `mean_reward=` — this was the root-cause fix; the old script always printed raw `mean_reward` and was compared against a `success_rate` threshold
4. The **demo-reward gate** (in `verdict_gates.apply_verdict_gates`) requires that measured metric to clear the threshold; below → env regresses to optimization (see Training Phases). The quad Tester falls back to LLM flow if the deterministic script fails; the duo Executor is deterministic-only (no fallback).

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
