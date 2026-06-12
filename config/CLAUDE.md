# config/ - Configuration Files

## Overview
All YAML configuration lives here. Two separate concerns: **project settings** and **agent prompts**.

## Files

### project.yaml — Main config
Pydantic-validated via `src/config_loader.py` (`ProjectConfig` model).

Key sections:
- `pipeline: quad|duo` — topology (default `quad`). `quad` = Manager→Coder→Tester→Reviewer; `duo` = Director→Coder→Executor (1 LLM role). `main.py` dispatches on it.
- `environment` / `environment_progression` — Gymnasium env specs, thresholds, timeouts, device (cpu/gpu)
  - `metric: reward|success_rate` — goal-conditioned envs are scored by the `is_success` fraction in [0,1]
  - `tags: [..]` — optional env-family tags (e.g. `['goal','her','manipulation']`) for SKILL retrieval; declaring them avoids the Coder/Manager hard-coding `panda`/`fetch` substring checks (which remain a fallback)
- `agents` — `max_iterations`, `history_window` (per-agent siloed history), `agent_opinions` (team chatter)
- `llm` / `agent_llm` / `ollama` — Model names per agent, Ollama base URL, runtime options
  - `ollama.options` — Global Ollama options (num_gpu, num_thread, etc.) applied to all models
  - `ollama.model_options` — Per-model overrides merged with global (e.g. num_ctx for large models; quote tags with colons, e.g. `"qwen3-coder:30b": { num_ctx: 32768 }`)
  - `ollama.keep_alive` — How long a model stays in VRAM after a call (default "5m"; raise to "10m"+ so co-resident models aren't evicted between iterations)

### Demo / test configs (alternates to project.yaml)
- `demo.yaml` — short, predictable presentation run (2 fast envs, adaptive switching off, cloud SHODAN)
- `demo_local.yaml` — same, but reviewer is a local Ollama tag (fully local, no cloud)
- `test_single.yaml` — smoke test: one already-installed model for all local agents
- `duo_robot_seeded.yaml` — **duo pipeline** night run (panda progression + HER seed; `pipeline: duo`, `prompts_file: duo_prompts.yaml`, `skills_dir: skills/duo_seeded`)
- `duo_smoke.yaml` — duo 2-iteration wiring sanity (`max_iterations: 2`, `skills_dir: skills/duo_smoke`)
- All are full configs (Pydantic `extra='forbid'`); `main.py config/<name>.yaml` selects one, and each names its own `prompts_file`.
- `gpu` — VRAM limits for RL training in Docker
- `training_phases` — Multi-phase: validation -> optimization -> demo.
  `validation_timeout_floor` (default 60 s) sets the minimum validation timeout —
  framework/env startup (imports, CUDA init, pybullet) is a FIXED cost the
  `validation_timeout_multiplier` ignores; panda-gym runs use 90 s.
- `verbose` — Granular console output control (tester banners, GPU stats, SHODAN rules display etc.)
- `shodan_rules` — Enable SHODAN's Divine Codex (persistent rules injected into Coder prompt)
- `adaptive_model_switching` — Switch LLM model randomly when agent gets stuck (chaos_mode = every call)
- `prompts_file` — Path to prompts YAML (swap between prompt sets)
- `test_name` — Prefix for output directory naming

### opus_prompts.yaml — Agent prompts (current)
Minimal-constraint prompts with agent personalities. Contains `{placeholder}` template vars.

Structure per agent (e.g. `manager:`, `coder:`, `tester:`, `reviewer:`):
- `system` — System prompt (personality, role)
- `task_template` — User message template with `{placeholders}` filled at runtime

Special placeholders in coder's prompt:
- `{shodan_rules}` — Active Divine Codex rules (only in opus_prompts.yaml)
- `{shodan_rules_display}` — Human-readable rules display

### prompts.yaml — Original detailed prompts
Larger, more constrained prompts. Same structure as opus_prompts.yaml but without `{shodan_rules}` placeholders — `BaseAgent.render_template()` renders an absent optional placeholder as empty, so no per-call try/except KeyError fallback is needed.

### duo_prompts.yaml — Duo pipeline prompts
Only two sections (the duo pipeline has no Manager/Tester nodes):
- `reviewer:` — the **Director** (SHODAN). "There is no Manager and no Tester. YOU are strategist, judge, and taskmaster." Carries the four-gate awareness, the resume contract, success_rate semantics, the skill_ops contract + THE PIN BINDS YOU TOO, and a JSON `task_template` contract of `{{approved, feedback, next_task, skill_ops, my_opinion}}` (no `tester_instruction`). Rendered via `render_template` (format_map), so literal braces are doubled. Inherits `environment_switch_report_template`.
- `coder:` — the opus coder + a "RAW EXECUTION OUTPUT" paragraph (fix what the container actually printed) + a short `chat_template` (team chatter). Its `system` is concatenated raw (braces single); its `task_template` is formatted (braces doubled).

## Conventions
- YAML uses `{{` for literal braces (Python `.format()` escaping)
- All config validated strictly with `extra='forbid'` in Pydantic models
- New config field: add Pydantic model in `config_loader.py` -> add to `ProjectConfig` -> add `Config` property
- Prompt templates rendered with `.format(**kwargs)` in each agent's `__call__`
