# src/utils/ - Utility Modules

## Files

### conversation_logger.py — Markdown conversation log
Writes `output/{run_id}/conversation.md` (GitHub-flavored markdown).

Key design decisions:
- Coder's code is NOT logged during iterations, only on environment solved ("Winning Code")
- stdout/stderr cleaned: CUDA banner, SB3 progress tables, wrapper messages, harmless warnings filtered
- Context usage logged as compact one-liner after each agent
- Collapsible `<details>` blocks for raw output and code

Methods:
- `log_iteration_start()`, `log_phase_transition()`
- `log_manager()`, `log_coder()` (stores code, doesn't write), `log_tester()`, `log_reviewer()`
- `log_context_usage()` — compact context fill percentage
- `log_agent_chat()` — personality opinions as blockquotes
- `log_codex_change()` — Divine Codex add/remove
- `log_environment_switch()` — shows Winning Code + LinkedIn post + SHODAN assessment
- `log_final_summary()` — metrics table

### model_switcher.py — Adaptive model switching
Switches LLM model when agent gets stuck. All in Finnish comments.

Classes:
- `SwitchTrigger` (Enum) — REPETITION_LOOP, REPEATED_ERROR, NO_REWARD_IMPROVEMENT, TIMEOUT, MANUAL
- `AgentStuckState` — tracks errors, rewards, repetitions, timeouts per agent
- `ModelSwitcher` — main class, loads model pools from config, picks random model on trigger

Key methods:
- `check_and_switch(agent_name, trigger, context)` -> new model name or None
- `get_chaos_model(agent_name)` -> random model every call (chaos_mode)
- `report_success(agent_name)` — resets stuck counters

### banners.py — Console output formatting
Rich-based visual feedback. All functions use `_get_console()` (force_terminal=True for Windows).

Functions:
- `print_run_banner()` — config summary at start
- `print_iteration_banner()` — iteration progress with env info
- `print_agent_transition()` — "Manager -> Coder" dividers
- `print_final_summary()` — end-of-run stats
- `print_environment_switch_bombardment()` — detailed stats tables when switching envs
- `print_manager_report()` — Manager's LinkedIn-style post
- `print_reviewer_cynical_report()` — SHODAN's assessment

### timer.py — Timing and statistics
- `AgentTiming` — per-call timing with token counts
- `RunStatistics` — aggregates all timings, generates per-agent and per-iteration breakdowns
- `save_to_file()` — JSON export to `output/{run_id}/statistics.json`
- `get_agent_token_stats()` — token in/out/total with min/max/avg/median
- `get_code_stats()` — lines of code per iteration tracking
