# Smoke test for the DUO pipeline + demo-reward gate (2026-06-12).
# Run: python scripts/smoke_test_duo.py
# No API key / Docker / Ollama needed: a dummy OPENAI_API_KEY lets the Director construct,
# and every check below is deterministic (gates, prompts, video-script text, graph compile).
import os
import sys

os.environ.setdefault("OPENAI_API_KEY", "dummy")
os.environ.setdefault("LLM_BASE_URL", "http://localhost:11434/v1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

failures = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failures.append(name)


# ── 1. Duo config loads; prompts carry the duo contract ──
from src.config_loader import load_config

cfg = load_config("config/duo_robot_seeded.yaml")
check("duo config loads", cfg is not None)
check("pipeline == duo", cfg.pipeline == "duo")
check("duo prompts_file wired", cfg.project.prompts_file == "config/duo_prompts.yaml")
check("separate duo skills_dir", cfg.skills_dir == "skills/duo_seeded")
check("env tags declared", cfg.environment_progression[0].tags == ["goal", "her", "manipulation", "robotics"])

rev_p = cfg.get_prompt("reviewer")
cod_p = cfg.get_prompt("coder")
check("director JSON contract has next_task", '"next_task"' in rev_p["task_template"])
check("director JSON contract has skill_ops", '"skill_ops"' in rev_p["task_template"])
check("director contract drops tester_instruction", "tester_instruction" not in rev_p["task_template"])
check("director knows the four gates", "DEMO gate" in rev_p["system"] and "THRESHOLD gate" in rev_p["system"])
check("director: no Manager/Tester", "There is no Manager and no Tester" in rev_p["system"])
check("THE PIN BINDS YOU TOO survives", "THE PIN BINDS YOU TOO" in rev_p["system"])
check("coder prompt has RESUMED contract", "RESUMED: buffer_transitions" in cod_p["system"])
check("coder has a chat_template", bool(cod_p.get("chat_template")))
check("coder sees RAW EXECUTION OUTPUT", "RAW EXECUTION OUTPUT" in cod_p["system"])

# ── 1b. NO-RECIPE GUARD: duo prompts must teach METHOD + FACTS, never env recipes ──
# (General-intelligence principle: env->algorithm tables, step-count tables and ready-made
# solutions like the HER kwargs gave the answer away in every run - including "blind" ones.
# Recipes belong ONLY in the SkillStore, seeded explicitly per experiment or EARNED by runs.)
import re as _re

_duo_text = open("config/duo_prompts.yaml", encoding="utf-8").read()
_RECIPE_TOKENS = [
    # env names (any mention in prompts = leaked benchmark knowledge)
    "CartPole", "Pendulum", "MountainCar", "LunarLander", "BipedalWalker", "Acrobot",
    "PandaReach", "PandaPush", "PandaPickAndPlace", "aviary",
    # ready-made solution markers
    "HerReplayBuffer", "MultiInputPolicy", "n_sampled_goal", "150k",
    # algorithm names = steering; ALGO placeholder is the allowed generic form
    "SAC", "PPO", "DQN", "TD3", "A2C", "DDPG",
]
_leaks = [t for t in _RECIPE_TOKENS if _re.search(r"\b" + _re.escape(t), _duo_text)]
check(f"duo prompts are recipe-free (leaks: {_leaks or 'none'})", not _leaks)
check("coder prompt demands discovery", "observation_space" in cod_p["system"])
check("director demands discovery printout", "observation_space" in rev_p["system"])
check("director skill example is method-shaped", "Inspect the env before choosing budgets" in rev_p["system"])

# initial_validation_task must also be algorithm-free + demand discovery
from src.agents import env_transitions as _et
_task0 = _et.initial_validation_task(cfg.environment_progression[0])
check("initial validation task is algorithm-free",
      not _re.search(r"\b(SAC|PPO|DQN|TD3|A2C|DDPG)\b", _task0))
check("initial validation task demands discovery", "observation_space" in _task0)

# ── 2. Both graphs compile (duo new; quad regression guard) ──
from src.duo_graph import create_duo_graph
from src.graph import create_graph

check("create_duo_graph compiles", create_duo_graph(cfg) is not None)
qcfg = load_config("config/robot_arm_seeded.yaml")
check("create_graph (quad) still compiles", create_graph(qcfg) is not None)
check("quad config defaults pipeline=quad", qcfg.pipeline == "quad")

# ── 3. apply_verdict_gates unit cases (the deterministic math) ──
from src.utils.verdict_gates import apply_verdict_gates

g = apply_verdict_gates(True, phase="optimization", stdout="RESULT: mean_reward=10.0",
                        success_threshold=100, env_metric="reward", resume_required=False,
                        resume_ok=True, demo_reward=None)
check("threshold gate rejects below-threshold", (not g.approved) and g.gate_fired == "threshold")

g = apply_verdict_gates(True, phase="optimization", stdout="RESULT: mean_reward=-45.0",
                        success_threshold=0.5, env_metric="success_rate", resume_required=False,
                        resume_ok=True, demo_reward=None)
check("metric-lock rejects raw reward in success_rate env", (not g.approved) and g.gate_fired == "metric_lock")

g = apply_verdict_gates(True, phase="optimization", stdout="RESULT: mean_reward=150.0",
                        success_threshold=100, env_metric="reward", resume_required=True,
                        resume_ok=False, demo_reward=None)
check("resume gate rejects unresumed chunk", (not g.approved) and g.gate_fired == "resume")

g = apply_verdict_gates(True, phase="demo", stdout="", success_threshold=0.5,
                        env_metric="success_rate", resume_required=False, resume_ok=True,
                        demo_reward=0.40)
check("demo gate rejects 0.40<0.5 + sets regression flag", (not g.approved) and g.demo_below_threshold and g.gate_fired == "demo")

g = apply_verdict_gates(True, phase="demo", stdout="", success_threshold=0.5,
                        env_metric="success_rate", resume_required=False, resume_ok=True,
                        demo_reward=0.60)
check("demo gate passes 0.60>=0.5", g.approved and (not g.demo_below_threshold) and g.gate_fired == "")

g = apply_verdict_gates(True, phase="demo", stdout="", success_threshold=0.5,
                        env_metric="success_rate", resume_required=False, resume_ok=True,
                        demo_reward=None)
check("demo None rejects WITHOUT regression flag", (not g.approved) and (not g.demo_below_threshold) and g.gate_fired == "demo")

# ── 4. env_switch_reset clears the demo-gate fields (all switch paths inherit this) ──
from src.agents import env_transitions

reset = env_transitions.env_switch_reset(1, "task", "output/x")
check("env_switch_reset clears demo_reward", "demo_reward" in reset and reset["demo_reward"] is None)
check("env_switch_reset clears demo_below_threshold", reset.get("demo_below_threshold") is False)
check("env_switch_reset syncs guidance", reset["manager_guidance"] == "Task: task" and reset["metric_history"] == [])

# ── 5. Metric-aware demo script (the root-cause fix) ──
from src.agents.tester import Tester

vs = Tester.generate_video_script("PandaReach-v3", "/workspace/output/best_model.zip",
                                  "/workspace/output/iter_0/", metric="success_rate")
check("video script computes is_success", "is_success" in vs)
check("video script records only first N episodes", "episode_trigger=lambda e: e <" in vs)
check("video script prints success_rate RESULT", "success_rate=" in vs)
check("video script uses fixed eval seeds", "seed=2000" in vs)

vs_r = Tester.generate_video_script("CartPole-v1", "/m", "/o", metric="reward")
check("reward-metric script prints mean_reward RESULT", "mean_reward=" in vs_r)

# ── 6. Canonical parser reads the success_rate RESULT line ──
from src.utils.result_parser import parse_result_line

pr = parse_result_line("RESULT: success_rate=0.40, std_reward=0.00, episodes=20")
check("parse_result_line reads success_rate value", pr["value"] == 0.40 and pr["metric"] == "success_rate")

# ── 7. Executor instantiates WITHOUT an LLM ──
from src.agents.executor import Executor
from src.agents.base import BaseAgent

ex = Executor(cfg)
check("Executor instantiates", ex is not None)
check("Executor is NOT a BaseAgent (no LLM client)", not isinstance(ex, BaseAgent))

print()
if failures:
    print(f"{len(failures)} FAILURE(S): {failures}")
    sys.exit(1)
print("ALL DUO SMOKE TESTS PASSED")
