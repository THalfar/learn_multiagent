# LangGraph RL Dev Team

Multi-agent RL code generator & optimizer using Stable-Baselines3 + Gymnasium.

## Features
- **Agents**: Manager (tasks), Coder (gen PPO/DQN code), Tester (train/eval/video), Reviewer (feedback)
- **Config-driven**: YAML prompts/hyperparams (algos, timesteps, thresholds)
- **Loop**: Until avg reward >195 (CartPole solved) or max_iters=3
- **Outputs**: Code (`output/code/`), videos (`output/videos/`)

## Setup
```bash
conda create -n langgraph-rl python=3.11 -y
conda activate langgraph-rl
pip install -r requirements.txt
```

**.env** (xAI/Grok):
```
OPENAI_API_KEY=xai-your-key
OPENAI_BASE_URL=https://api.x.ai/v1
```

## Usage
```bash
python main.py
```

Custom config:
```bash
# Edit config/project.yaml (PPO→DQN, timesteps...)
python main.py
```

## Graph (src/graph.py)
```
Manager → Coder → Tester → Reviewer → (iter < max_iters? loop : END)
```
Verbose logs + separators.

## Testing
- **PPO**: Default, 50k steps CartPole-v1
- **DQN**: Edit `config/project.yaml`:
  ```yaml
  algorithm:
    name: "DQN"
    parameters:
      buffer_size: 100000
      learning_starts: 100
      batch_size: 32
      tau: 1.0
      gamma: 0.99
      train_freq: 4
      gradient_steps: 1
      target_update_interval: 1000
  ```
  Run again.

Videos auto-saved every `record_frequency`.

## Structure
```
config/     YAML
src/agents/ Classes
src/graph.py LangGraph
output/     Artifacts
```

## Learnings
- LangGraph: StateGraph + callable nodes + conditional edges + Annotated reducers.
- Pydantic: YAML → validated objects.
- Subprocess: Safe RL training exec.

Enjoy your RL dev team! 🎮