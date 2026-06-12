# src/agents/ - Agent Implementations

## Architecture
All agents extend `BaseAgent` (base.py). Each agent is a callable: `__call__(state: dict) -> dict`.

```
BaseAgent (base.py)
  |-- Manager (manager.py)   -- local Ollama model      [quad]
  |-- Coder (coder.py)       -- local Ollama model      [quad + duo]
  |-- Tester (tester.py)     -- local Ollama model      [quad]
  |-- Reviewer (reviewer.py) -- frontier API model      [quad]
        |-- Director (director.py) -- frontier API model [duo]  (subclasses Reviewer)

Executor (executor.py)       -- NOT a BaseAgent (no LLM) [duo]
env_transitions.py           -- pure fns shared by Manager (delegates) + Director
```

The **duo** pipeline (`pipeline: duo`) runs Director → Coder → Executor. The Director is
strategist+judge+taskmaster in one call; the Executor is deterministic (no LLM). See the
duo sections below; the quad agents are unchanged.

## base.py — BaseAgent (~1050 lines)
The foundation class. Handles:

- **LLM initialization**: `ChatOpenAI` (langchain_openai) pointing at Ollama or API (no preload at init — lazy loading via `_ensure_model_loaded()`)
- **Model swapping**: `_ensure_model_loaded()` checks Ollama `/api/ps`, unloads old model if different
- **Context tracking**: Estimates token usage per prompt component, tracks fill percentage
- **Conversation history**: Siloed per-agent with configurable window (`history_window`)
- **Agent opinions**: Cross-agent "team chatter" (`format_agent_opinions_context()`)
- **Model switching**: Integrates with `ModelSwitcher` for chaos mode and stuck detection
- **Timing**: Records `AgentTiming` with token counts to `RunStatistics`

Key methods:
- `_call_llm(system_prompt, user_prompt)` — main LLM call with retry, timing, token tracking
- `_ensure_model_loaded()` — Ollama model swap (skip if same model already loaded); single
  source of truth for the unload+preload+spinner sequence (was copy-pasted in call_llm_timed/call_llm)
- `render_template(template, **kwargs)` — `format_map` with empty-default for missing keys, so
  OPTIONAL placeholders (e.g. `{shodan_rules}`) need no per-call try/except KeyError fallback
- `_estimate_tokens(text)` — rough token count (chars/3.5)
- `log_context_to_conversation(state)` — writes context usage to conversation logger
- `format_agent_opinions_context(state)` — formats team chatter for prompt injection

Constants: `MODEL_CONTEXT_SIZES` dict maps model names to context window sizes.

## manager.py — Manager (~1000 lines)
Orchestrates the pipeline. Responsibilities:
- Assigns tasks to Coder based on phase and feedback
- Handles phase transitions (validation -> optimization -> demo) **inline** — does NOT return early, generates new task in same call
- Handles environment switches (generates LinkedIn-style reports)
- Generates SHODAN environment switch assessments (via API call)
- Resets agent state between environments
- Validation phase: injects concrete timeout (seconds) into task so Manager/Coder know the constraint

Key: Manager checks `current_phase` and `approved` to decide next action. On phase transition, it updates state in-place and continues to task generation.

Learning features:
- `_skill_from_winning_code(code, env_name, env_tags)` — on env-solve, distils a verified
  PROCEDURAL skill (algo + policy + HER + checkpoint + metric) into the SkillStore. This
  REPLACED the old regex "playbook" (`_extract_recipe_from_code`/`_format_playbook_context`,
  removed) which captured only algo/steps/device, reached only the Manager, and printed
  'unknown' whenever its regex missed. Env family comes from `EnvironmentStep.tags` (the
  env-id substring heuristic is only a fallback).
- Failsafe: skips to next env after N consecutive failures (configurable)
- `_env_switch_reset()` — shared state reset for ALL env-switch paths (solved/failsafe/LLM);
  sets a concrete validation task (`_initial_validation_task`) + matching `manager_guidance`
  so the first iteration of a new env is never judged against the old env's task
- Optimization phase instruction is built from state: cumulative status (`total_env_steps`,
  `metric_history`), SPS-sized chunk (`measured_sps × timeout × 0.8`), the checkpoint-resume
  contract (RESUMED proof), eval discipline (≥20 episodes, fixed seeds), and the relevant
  skills with the precedence rule (verified skills outrank Reviewer directives)

## coder.py — Coder (~400 lines)
Writes pure Python RL training scripts.
- `_get_code_context()` — loads previous iteration's code from `output/{run_id}/code/`
- `_print_code_summary()` — visual stats of generated code
- Extracts code from LLM response (handles markdown blocks)
- Saves code to `output/{run_id}/code/agent_code_iter_{N}.py`
- Divine Codex rules injected via `{shodan_rules}` placeholder in prompt

## tester.py — Tester (~1550 lines, largest agent)
Executes code in Docker sandbox and analyzes results.
- `run_in_container()` — builds `docker run` command with GPU, mounts, timeouts
- `validate_gpu_in_container()` — checks CUDA availability in container
- `check_video_files()` — validates RecordVideo output (recursive search, MP4 header check)
- `extract_json()` — robust JSON extraction from LLM responses
- Sends analysis to Reviewer, can respond to Reviewer's `reviewer_tester_instruction`

**Deterministic demo video recording:**
- `_find_saved_model(output_dir)` — `@staticmethod` (so the duo Executor can reuse it), searches for `.zip` model files, prefers `best_model.zip`
- `generate_video_script(env_name, model_path, output_dir, metric="reward", n_eval_episodes=20, n_video_episodes=5)` — `@staticmethod`. Evaluates 20 FIXED-seed episodes (`reset(seed=2000+ep)`), records video for the first 5 (`episode_trigger=lambda e: e < 5`), and prints a **metric-aware** RESULT (`success_rate` via the `is_success` fraction for goal envs, else `mean_reward`). Root-cause fix for the demo-reward gate.
- In demo phase `__call__`: passes `metric=current_env.metric`; sets `demo_reward` on EVERY return path (value or None); bypasses LLM code, falls back to LLM flow if the script fails.
- After optimization: finds saved model and sets `best_model_path` in returned state
- `compute_execution_timeout(config, base_timeout, phase)` — module-level (shared with the Executor): validation floor/multiplier, optimization full, demo `demo_timeout_seconds`

**Rule-based diagnostics:**
- `diagnose_common_issues()` — catches known failures (timeout+wrong device, wrong algo for action space, missing model.save, callback crashes, double .zip, EVAL NOISE: <20 eval episodes on a success_rate env) BEFORE LLM analysis
- Findings appended to stderr as `=== AUTOMATED DIAGNOSTICS ===` block for LLM to reference

**Checkpoint-resume enforcement (optimization):**
- Pre-Docker: if a checkpoint exists (`_find_saved_model`), the script must satisfy
  `check_resume_contract()` (load model + buffer, RESUMED print, save both) — otherwise
  Docker is skipped and the iteration fails fast (test_results carries "RESUME CONTRACT FAILED",
  which the Reviewer classifies into `failure_history` for the escalation ladder)
- Post-run: stdout must contain `RESUMED: buffer_transitions=N` with N>0, else
  `resume_ok=False` and test_results is prefixed with RESUME CHECK FAILED
- Cumulative tracking: appends to `metric_history`, adds parsed steps to `total_env_steps`,
  measures `measured_sps` from runs ≥5000 steps
- Validation timeout: `max(validation_timeout_floor, multiplier × base)` — the floor covers
  fixed startup cost (imports, CUDA, pybullet)

Docker config: `DOCKER_IMAGE = "citadel-rl:latest"`, `ALLOWED_DIR = output/`

## reviewer.py — Reviewer/SHODAN (~670 lines)
Frontier API model that reviews code + results.
- Phase-aware criteria (validation: "does it run?", optimization: "meets threshold?", demo: "video works?")
- Deterministic gates that override an LLM APPROVE — now via `src/utils/verdict_gates.apply_verdict_gates()`
  (one shared, unit-testable implementation): threshold gate, metric lock (success_rate in [0,1]),
  resume gate (`resume_required` without `resume_ok`), and the **demo gate** (the demo's measured
  metric must clear the threshold; below → `demo_below_threshold` → Manager regresses demo→optimization).
  A demo-gate rejection does NOT increment `consecutive_failures`. Returns `demo_below_threshold`.
- Optimization criteria include the cumulative status (total steps, metric curve) + structural
  truths: timeout bounds ONE CHUNK (never order from-scratch / smaller chunks for low reward),
  and verified skills are pinned (SHODAN may not contradict them)
- Manages Divine Codex: parses `prompt_rules: {add: [...], remove: [idx]}` from own output
- Generates `reviewer_tester_instruction` for next iteration's Tester
- No `model_switcher` — stays on API model always

## director.py — Director (duo pipeline; `class Director(Reviewer)`)
Frontier API model; the ONLY LLM in the duo pipeline. Subclasses Reviewer to inherit the api
model, the "reviewer" timer bucket, `config.get_prompt("reviewer")`, and `generate_environment_switch_report`.
One `__call__` does verdict(N−1) + task(N):
1. **Bootstrap** — no task yet → deterministic first validation task (`env_transitions.initial_validation_task`), no LLM call.
2. Deterministic context (ported from the Manager): cumulative status, SPS-sized chunk, resume block, escalation ladder, verified-skill precedence.
3. ONE `call_llm_timed` → JSON `{approved, feedback, next_task, skill_ops, my_opinion}` (shared `json_extract.extract_json` + retry, fallback = reject + repeat task).
4. `apply_verdict_gates` (same four gates). 5. `skill_ops` (never crash). 6. Progress-aware failsafe **+ immediate env-skip** (the Director can switch env in the same call). 7. Phase machine (validation→optimization→demo→solved/DONE; **demo-reward regression deterministically overrides the LLM's next_task**). 8. Logs as both reviewer (verdict) and manager (next task); returns **exactly one `iteration: 1`** per path.

## executor.py — Executor (duo pipeline; NOT a BaseAgent — no LLM)
Deterministic sandbox node mirroring `Tester.__call__` minus the LLM analysis/chat. Reuses the
Tester's `run_in_container` / `diagnose_common_issues` / `check_video_files` / `auto_fix_common_issues` /
`is_safe_code` / `compute_execution_timeout` / `Tester.generate_video_script` / `Tester._find_saved_model`.
Emits a factual report: raw stdout/stderr + `=== AUTOMATED DIAGNOSTICS ===` (the Coder's raw-revision
input), parsed RESULT → deterministic `test_results`, resume pre/post gates, cumulative tracking,
`best_model_path`. Demo path is deterministic-ONLY (no LLM fallback) and sets `demo_reward`. NEVER
returns `iteration`/`approved`/`current_task` or history/opinion keys.

## env_transitions.py — shared env-transition helpers (pure functions, no LLM)
`initial_validation_task(env)`, `env_switch_reset(idx, task, video_dir)` (incl. the demo-field reset),
`skill_from_winning_code(code, env_name, env_tags)`. The Manager keeps 1-line delegating staticmethods
(exercised by `smoke_test_fixes.py`); the Director imports the module functions directly.

## Patterns
- Each agent's `__call__` follows: build prompt -> call LLM -> parse response -> update state -> return partial state
- Prompt templates loaded via `self.config.get_prompt(self.agent_name)`
- `self.render_template(template, **kwargs)` with named placeholders; optional vars (e.g.
  `{shodan_rules}`) render empty when a prompt file omits them — no try/except KeyError needed
- All agents log to `conversation_logger` from state
- `iteration` returned as 1 (auto-added by LangGraph's Annotated[int, operator.add])
