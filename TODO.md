# TODO: LangGraph RL Dev Team

## Project Structure
````
learn_multiagent/
├── config/
│   ├── project.yaml      # RL project settings
│   └── prompts.yaml      # Agent prompts
├── src/
│   ├── agents/
│   │   ├── manager.py
│   │   ├── coder.py
│   │   ├── tester.py
│   │   └── reviewer.py
│   ├── graph.py          # LangGraph definition
│   └── config_loader.py  # YAML loader
├── output/
│   ├── code/             # Generated code
│   └── videos/           # Gymnasium videos
├── main.py
├── TODO.md
├── requirements.txt
└── .env
````

---

## Phase 1: Project Structure
- [x] Create folder structure (src/, config/, output/)
  > Created `config/`, `src/agents/`, `output/code/`, `output/videos/` using PowerShell `mkdir ..., -Force`.
  > **Why**: Implements modular architecture - configs isolated, source organized by agent, outputs separated to avoid clutter.
  > **How**: PowerShell `mkdir` creates nested directories recursively in one command.
  > **Learning**: Clean structure essential for multi-agent projects; aligns with phases 2-7.
- [x] Move agent logic to separate files
  > Extracted placeholder nodes (`manager_node`, `coder_node`, `tester_node`, `reviewer_node`) from `main.py` to `src/agents/*.py`.
  > Added imports in `main.py`; fixed iteration increment (`"iteration": 1` in manager) to prevent infinite loop.
  > **Why**: SRP - each module handles one agent; easier to extend with LLM prompts (Phase 4).
  > **How LangGraph connects**: Nodes are pure functions → state dict updates → graph edges define flow.
  > **Learning**: Use `Annotated[int, operator.add]` + return `{"iteration": 1}` for counters; conditional edges route dynamically.

## Phase 2: YAML Configuration

### config/project.yaml
- [x] Create project.yaml:
  > Copied exact YAML from TODO.md spec to `config/project.yaml`.
  > **Why**: Centralizes tunable RL params (CartPole-v1 env, PPO hyperparams, training/video settings).
  > **How**: Nested YAML dict; e.g., `agents.max_iterations: 3` syncs with current graph loop.
  > **Learning**: Config-first design; Phase 3 `config_loader.py` will parse to Python attrs/dot access.
````yaml
# Gymnasium environment
environment:
  name: "CartPole-v1"
  max_episode_steps: 500

# Stable-Baselines3 algorithm
algorithm:
  name: "PPO"
  parameters:
    learning_rate: 0.0003
    n_steps: 2048
    batch_size: 64
    n_epochs: 10
    gamma: 0.99

# Training settings
training:
  total_timesteps: 50000
  eval_frequency: 10000
  n_eval_episodes: 10

# Video recording
video:
  enabled: true
  output_dir: "output/videos"
  record_frequency: 10000

# Agent loop settings
agents:
  max_iterations: 3
  success_threshold: 195
````

### config/prompts.yaml
- [x] Create prompts.yaml:
  > Copied exact multi-line templates/system prompts to `config/prompts.yaml`.
  > **Why**: Externalizes LLM behavior; iterate prompts without code changes.
  > **How**: YAML `|` blocks preserve newlines; {placeholders} injected at runtime (e.g. {tasks}, {code}).
  > **Learning**: JSON-only outputs for parsing; structured prompts reduce hallucinations.
````yaml
manager:
  system: |
    You are an RL project manager. You analyze the current state and decide the next task.
    Be concise and focused on Stable-Baselines3 + Gymnasium best practices.
  task_template: |
    Current state:
    - Tasks completed: {tasks}
    - Code status: {code_summary}
    - Test results: {test_results}
    - Review feedback: {review_feedback}
    - Iteration: {iteration}/{max_iterations}
    
    Decide the next task. Respond in JSON format only:
    {{"next_task": "task description", "reasoning": "why this task"}}

coder:
  system: |
    You are an RL engineer. You write clean Python code using Stable-Baselines3 and Gymnasium.
    Always include proper imports, error handling, and video recording setup.
  task_template: |
    Task: {current_task}
    Environment: {environment}
    Algorithm: {algorithm}
    Parameters: {parameters}
    Video output: {video_dir}
    
    Write complete, runnable Python code. Output only code, no explanations.

tester:
  system: |
    You are a code tester. You analyze test results and report metrics.
  task_template: |
    Code executed with results:
    - Mean reward: {mean_reward}
    - Std reward: {std_reward}
    - Episodes: {n_episodes}
    - Video saved: {video_path}
    
    Summarize the test results. Respond in JSON format:
    {{"success": true/false, "summary": "brief summary", "metrics": {{}}}}

reviewer:
  system: |
    You are a code reviewer specializing in reinforcement learning.
    You evaluate code quality, RL best practices, and training results.
  task_template: |
    Code:
```python
    {code}
```
    
    Test results: {test_results}
    Success threshold: {success_threshold} average reward
    
    Review and respond in JSON format only:
    {{"approved": true/false, "feedback": "detailed feedback", "suggestions": ["suggestion1", "suggestion2"]}}
````

## Phase 3: Config Loader
- [x] Create src/config_loader.py
  > Implemented `src/config_loader.py` with Pydantic models for project.yaml, dict for prompts.yaml.
- [x] Load and validate YAML files
  > `load_project_config()` uses Pydantic BaseModel.model_validate() for type/schema validation; `load_prompts()` simple yaml.safe_load.
  > FileNotFoundError if missing; extra='forbid' rejects unknown keys.
- [x] Provide easy getters: `config.environment`, `config.get_prompt("manager")`
  > `Config` class with @property for `environment`, `algorithm` etc.; `get_prompt(agent)` returns dict{'system', 'task_template'}.

## Phase 4: Refactor Agents
- [x] src/agents/base.py - BaseAgent class with LLM call + logging
  > BaseAgent loads config, LLM (.env), logger. `__call__(state)` callable for LangGraph.
- [x] src/agents/manager.py - uses prompts.yaml
  > Manager class: formats template with state/config, LLM → parse JSON next_task/reasoning, updates tasks.
- [x] src/agents/coder.py - uses prompts.yaml + project.yaml
  > Coder class: formats with task/env/algo/params/video_dir + existing code, LLM → extracts full code string.
- [x] src/agents/tester.py - executes code, records video
  > Tester class: dummy metrics → LLM summarize/parse JSON (Phase 5 real exec/subprocess/video).
- [x] src/agents/reviewer.py - uses prompts.yaml
  > Reviewer class: formats code/tests/threshold, LLM → parse JSON feedback/approved.

## Phase 5: Tester Agent (Real Implementation)
- [x] Save generated code to output/code/
  > Tester saves state["code"] to `output/code/agent_code_{timestamp}.py`
- [x] Execute code via subprocess
  > `subprocess.run(["conda", "run", "-n", "langgraph-rl", "python", code_path], timeout=300)` in env.
- [x] Configure Gymnasium RecordVideo wrapper
  > Coder prompt enforces "video recording setup"; tester parses VIDEO_SAVED from stdout.
- [x] Save video to output/videos/
  > Code uses config.video.output_dir; tester logs path from stdout.
- [x] Return eval metrics (avg reward, success rate)
  > Parses stdout RE MEAN_REWARD/STD_REWARD/N_EPISODES; summary str with success check vs threshold.

## Phase 6: LangGraph Integration
- [x] Update graph.py to use new agents
  > Created `src/graph.py` create_graph(config): instantiates agent classes, wires StateGraph, uses config max_iters.
- [x] State reads max_iterations and success_threshold from YAML
  > should_continue in graph.py uses config.agents.max_iterations; threshold in reviewer prompt.
- [x] Verbose logging at each step with clear separators
  > logging.INFO in agents/base; print "=== Iteration X ===" in should_continue; main separators.

## Phase 7: Testing & Documentation
- [ ] Test with CartPole-v1 + PPO
- [ ] Test with different algorithm (DQN)
- [ ] Update README.md with usage instructions

---

## Out of Scope (for now)
- No Optuna integration yet
- No complex environments (classic control only)
- No custom reward functions

## Goal
````bash
python main.py --config config/project.yaml
````
→ Agents read YAML configs
→ Discuss and generate code
→ Training runs
→ Video saved
→ Results reported