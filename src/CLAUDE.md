# src/ - Core Source Code

## Architecture

```
main.py (entry point)
  |
  v
src/config_loader.py  -- loads project.yaml + prompts YAML -> Config object
  |
  v
src/graph.py          -- LangGraph StateGraph: Manager -> Coder -> Tester -> Reviewer loop
  |
  v
src/agents/           -- Agent implementations (all extend BaseAgent)
src/utils/            -- Banners, logging, model switching, timing
```

## Key Files

### config_loader.py — Config system
- `load_config()` -> `Config` object (wraps `ProjectConfig` + prompts dict)
- `ProjectConfig` — Pydantic model with strict validation (`extra='forbid'`)
- `Config.get_prompt(agent_name)` -> `{"system": str, "task_template": str}`
- Property accessors: `config.environment`, `config.agents`, `config.llm`, etc.

### graph.py — LangGraph state machine
- `AgentState` TypedDict — all shared state between agents
- `create_graph(config)` -> compiled LangGraph app
- Flow: Manager -> Coder -> Tester -> Reviewer -> conditional edge (continue/end)
- `should_continue()` — handles phase transitions (validation -> optimization -> demo) and environment progression
- `ModelSwitcher` created here, passed to local agents (not Reviewer who stays on API)

## State Fields (AgentState)
| Field | Type | Purpose |
|-------|------|---------|
| `tasks` | `List[str]` | Task list from Manager |
| `current_task` | `str` | Active task ("DONE" = end) |
| `code` | `str` | Coder's latest Python script |
| `test_results` | `str` | Tester's analysis summary |
| `execution_stdout/stderr` | `str` | Raw Docker output |
| `review_feedback` | `str` | Reviewer's verdict |
| `reviewer_tester_instruction` | `str` | Reviewer -> Tester message (next iter) |
| `tester_reviewer_response` | `str` | Tester -> Reviewer response |
| `manager_guidance` | `str` | Manager's intent for Reviewer |
| `iteration` | `int` | Auto-incrementing (Annotated with operator.add) |
| `conversation_history` | `List[Dict]` | Siloed per-agent message history |
| `agent_opinions` | `List[Dict]` | Cross-agent "team chatter" |
| `current_phase` | `str` | "validation" / "optimization" / "demo" |
| `shodan_rules` | `List[Dict]` | Divine Codex rules `[{"rule": str, "iteration": int}]` |
| `current_env_index` | `int` | Index in environment_progression |
| `solved_environments` | `List[str]` | Completed environment names |

## Conventions
- `iteration` uses `Annotated[int, operator.add]` — LangGraph auto-adds return values
- Agents return partial state dicts; LangGraph merges them
- `recursion_limit = max(max_iterations * 5, 50)` — minimum 50 for phase transitions
