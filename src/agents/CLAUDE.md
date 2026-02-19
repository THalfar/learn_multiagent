# src/agents/ - Agent Implementations

## Architecture
All agents extend `BaseAgent` (base.py). Each agent is a callable: `__call__(state: dict) -> dict`.

```
BaseAgent (base.py)
  |-- Manager (manager.py)   -- local Ollama model
  |-- Coder (coder.py)       -- local Ollama model
  |-- Tester (tester.py)     -- local Ollama model
  |-- Reviewer (reviewer.py) -- frontier API model (no model_switcher)
```

## base.py — BaseAgent (~1050 lines)
The foundation class. Handles:

- **LLM initialization**: `ChatOpenAI` (langchain_openai) pointing at Ollama or API
- **Model swapping**: `_ensure_model_loaded()` checks Ollama `/api/ps`, unloads old model if different
- **Context tracking**: Estimates token usage per prompt component, tracks fill percentage
- **Conversation history**: Siloed per-agent with configurable window (`history_window`)
- **Agent opinions**: Cross-agent "team chatter" (`format_agent_opinions_context()`)
- **Model switching**: Integrates with `ModelSwitcher` for chaos mode and stuck detection
- **Timing**: Records `AgentTiming` with token counts to `RunStatistics`

Key methods:
- `_call_llm(system_prompt, user_prompt)` — main LLM call with retry, timing, token tracking
- `_ensure_model_loaded()` — Ollama model swap (skip if same model already loaded)
- `_estimate_tokens(text)` — rough token count (chars/3.5)
- `log_context_to_conversation(state)` — writes context usage to conversation logger
- `format_agent_opinions_context(state)` — formats team chatter for prompt injection

Constants: `MODEL_CONTEXT_SIZES` dict maps model names to context window sizes.

## manager.py — Manager (~980 lines)
Orchestrates the pipeline. Responsibilities:
- Assigns tasks to Coder based on phase and feedback
- Handles phase transitions (validation -> optimization -> demo)
- Handles environment switches (generates LinkedIn-style reports)
- Generates SHODAN environment switch assessments (via API call)
- Resets agent state between environments

Key: Manager checks `current_phase` and `approved` to decide next action.

## coder.py — Coder (~400 lines)
Writes pure Python RL training scripts.
- `_get_code_context()` — loads previous iteration's code from `output/{run_id}/code/`
- `_print_code_summary()` — visual stats of generated code
- Extracts code from LLM response (handles markdown blocks)
- Saves code to `output/{run_id}/code/agent_code_iter_{N}.py`
- Divine Codex rules injected via `{shodan_rules}` placeholder in prompt

## tester.py — Tester (~1440 lines, largest agent)
Executes code in Docker sandbox and analyzes results.
- `_run_in_docker()` — builds `docker run` command with GPU, mounts, timeouts
- `_validate_gpu_output()` — checks CUDA availability in container
- `_check_video_files()` — validates RecordVideo output in demo phase
- `_extract_metrics()` — parses reward, episode length from stdout
- `extract_json()` — robust JSON extraction from LLM responses
- Sends analysis to Reviewer, can respond to Reviewer's `reviewer_tester_instruction`

Docker config: `DOCKER_IMAGE = "citadel-rl:latest"`, `ALLOWED_DIR = output/`

## reviewer.py — Reviewer/SHODAN (~670 lines)
Frontier API model that reviews code + results.
- Phase-aware criteria (validation: "does it run?", optimization: "meets threshold?", demo: "video works?")
- Manages Divine Codex: parses `prompt_rules: {add: [...], remove: [idx]}` from own output
- Generates `reviewer_tester_instruction` for next iteration's Tester
- No `model_switcher` — stays on API model always

## Patterns
- Each agent's `__call__` follows: build prompt -> call LLM -> parse response -> update state -> return partial state
- Prompt templates loaded via `self.config.get_prompt(self.agent_name)`
- `.format()` with named placeholders; try/except KeyError for optional vars
- All agents log to `conversation_logger` from state
- `iteration` returned as 1 (auto-added by LangGraph's Annotated[int, operator.add])
