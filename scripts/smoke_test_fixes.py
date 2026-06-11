# Smoke test for the 2026-06-11 fix batch (lint kwargs, resume contract, config load).
# Run: python scripts/smoke_test_fixes.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.code_lint import lint_code, check_resume_contract

failures = []

def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failures.append(name)

# 1. Hallucinated HER kwarg (the iteration-12-18 killer) -> hard error
bad_her = """
import os
from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
import gymnasium as gym
env = gym.make("PandaPush-v3")
model = SAC("MultiInputPolicy", env, replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs={'n_sampled_goal': 4, 'online_sample_strategy': 'future'})
"""
res = lint_code(bad_her, env_name="PandaPush-v3")
check("hallucinated replay_buffer_kwargs key -> error", not res.ok and any("online_sample_strategy" in e for e in res.errors))

# 2. Hallucinated constructor kwarg
bad_ctor = """
from stable_baselines3 import SAC
model = SAC("MlpPolicy", "Pendulum-v1", sample_goal_strategy="future")
"""
res = lint_code(bad_ctor, env_name="Pendulum-v1")
check("hallucinated SAC kwarg -> error", not res.ok and any("sample_goal_strategy" in e for e in res.errors))

# 3. reset_num_timesteps=False -> error
bad_learn = """
from stable_baselines3 import SAC
model = SAC("MlpPolicy", "Pendulum-v1")
model.learn(total_timesteps=1000, reset_num_timesteps=False)
"""
res = lint_code(bad_learn, env_name="Pendulum-v1")
check("reset_num_timesteps=False -> error", not res.ok)

# 4. Valid code passes
good = """
import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
env = gym.make("PandaPush-v3")
model = SAC("MultiInputPolicy", env, verbose=0, device="auto",
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs={'n_sampled_goal': 4, 'goal_selection_strategy': 'future'})
model.learn(total_timesteps=50000)
model.save("/workspace/output/best_model")
"""
res = lint_code(good, env_name="PandaPush-v3")
check("valid SAC+HER passes", res.ok)

# 5. Resume contract: fresh-training script with checkpoint present -> 4 violations
v = check_resume_contract(good)
check("resume contract catches fresh-training script", len(v) >= 3)

# 6. Resume contract satisfied
resumed = """
import os
import gymnasium as gym
from stable_baselines3 import SAC
env = gym.make("PandaPush-v3")
model = SAC.load("/workspace/output/best_model", env=env)
buf = "/workspace/output/best_model_buffer"
if os.path.exists(buf + ".pkl"):
    model.load_replay_buffer(buf)
print(f"RESUMED: buffer_transitions={model.replay_buffer.size()}")
model.learn(total_timesteps=50000)
model.save("/workspace/output/best_model")
model.save_replay_buffer(buf)
"""
check("resume contract satisfied -> no violations", check_resume_contract(resumed) == [])
res = lint_code(resumed, env_name="PandaPush-v3", require_resume=True)
check("lint(require_resume) passes on resumed script", res.ok)

# 7. Config + prompts load end-to-end (validates YAML, Pydantic, skill store JSON)
from src.config_loader import load_config
cfg = load_config("config/robot_arm_seeded.yaml")
check("seeded config loads", cfg is not None)
check("validation_timeout_floor present", cfg.project.training_phases.validation_timeout_floor == 90)
check("max_iterations generous", cfg.agents.max_iterations == 300)
mgr_p = cfg.get_prompt("manager"); cod_p = cfg.get_prompt("coder"); rev_p = cfg.get_prompt("reviewer")
check("coder prompt has RESUMED contract", "RESUMED: buffer_transitions" in cod_p["system"])
check("reviewer prompt has pin-binds rule", "THE PIN BINDS YOU TOO" in rev_p["system"])

cfg2 = load_config("config/robot_arm_blind.yaml")
check("blind config loads", cfg2 is not None)

from src.skills import SkillStore
ss = SkillStore(skills_dir="skills/robot_seeded").load()
check("seeded skill store loads", len(ss.skills) >= 2)
check("skill procedure has RESUMED proof", "RESUMED: buffer_transitions" in list(ss.skills.values())[0].procedure)

# 8. Manager helpers
from src.config_loader import Config
from src.agents.manager import Manager
env0 = cfg.environment_progression[0]
task = Manager._initial_validation_task(env0)
check("initial validation task names the env", env0.name in task and "RESULT:" in task)
reset = Manager._env_switch_reset(1, task, "output/x")
check("env switch reset syncs guidance", reset["manager_guidance"] == f"Task: {task}" and reset["metric_history"] == [])

print()
if failures:
    print(f"{len(failures)} FAILURE(S): {failures}")
    sys.exit(1)
print("ALL SMOKE TESTS PASSED")
