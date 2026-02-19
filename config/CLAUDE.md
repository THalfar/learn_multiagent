# config/ - Configuration Files

## Overview
All YAML configuration lives here. Two separate concerns: **project settings** and **agent prompts**.

## Files

### project.yaml — Main config
Pydantic-validated via `src/config_loader.py` (`ProjectConfig` model).

Key sections:
- `environment` / `environment_progression` — Gymnasium env specs, thresholds, timeouts, device (cpu/gpu)
- `agents` — `max_iterations`, `history_window` (per-agent siloed history), `agent_opinions` (team chatter)
- `llm` / `agent_llm` / `ollama` — Model names per agent, Ollama base URL, runtime options
  - `ollama.options` — Global Ollama options (num_gpu, num_thread, etc.) applied to all models
  - `ollama.model_options` — Per-model overrides merged with global (e.g. num_ctx for large models)
- `gpu` — VRAM limits for RL training in Docker
- `training_phases` — Multi-phase: validation -> optimization -> demo
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
Larger, more constrained prompts. Same structure as opus_prompts.yaml but without `{shodan_rules}` placeholders (fallback via try/except KeyError in coder.py).

## Conventions
- YAML uses `{{` for literal braces (Python `.format()` escaping)
- All config validated strictly with `extra='forbid'` in Pydantic models
- New config field: add Pydantic model in `config_loader.py` -> add to `ProjectConfig` -> add `Config` property
- Prompt templates rendered with `.format(**kwargs)` in each agent's `__call__`
