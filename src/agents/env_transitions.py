"""Environment-transition helpers — pure functions shared by the quad Manager and the
duo Director.

These were `@staticmethod`s on the Manager. The duo Director is a Reviewer subclass (it
can't inherit the Manager's methods), so the bodies live here as plain module functions
and the Manager keeps one-line delegating staticmethods (smoke_test_fixes.py exercises
those delegates). No LLM, no state machine — just the deterministic task/reset/skill
construction that both pipelines need to agree on.
"""
from __future__ import annotations

import re
from typing import Optional


def initial_validation_task(next_env) -> str:
    """Concrete first VALIDATION task for a newly entered environment.

    Returned on every env-switch path so the Coder NEVER starts a new env with an
    empty/stale task (the stale-task race: after a switch the Coder coded the NEW env
    while manager_guidance still described the OLD env's task -> the Reviewer rejected
    correct work as 'mismatching intent', one wasted iteration per switch)."""
    is_goal = getattr(next_env, "metric", "reward") == "success_rate"
    metric_note = (" The env is goal-conditioned: report success_rate (the is_success "
                   "fraction over eval episodes) in the mean_reward slot." if is_goal else "")
    # Deliberately algorithm-free: choosing the algorithm IS the intelligence being tested.
    # The discovery printout gives the team (and the Director) the evidence to choose from.
    return (f"Write a minimal VALIDATION script for {next_env.name}. First DISCOVER: create "
            f"the env with gym.make('{next_env.name}') and print its observation_space, "
            f"action_space, env.spec.max_episode_steps, env.spec.reward_threshold, and the "
            f"info dict from one env.step(). Then train a fresh model briefly with an "
            f"algorithm YOU choose based on those observations (1000-2000 timesteps, n_envs=1, "
            f"default hyperparameters), evaluate, and print exactly "
            f"'RESULT: mean_reward=X, std_reward=Y, episodes=Z'.{metric_note} "
            f"Save the model at the end. Keep the script minimal so it finishes well "
            f"within the validation timeout.")


def env_switch_reset(next_env_index: int, task: str, video_dir: str) -> dict:
    """Shared state-reset block for ALL env-switch paths (solved / failsafe / LLM switch).
    One source of truth so no path forgets a field (the C1/C2 cumulative + resume fields
    especially: a stale metric_history would poison the next env's curve; stale demo flags
    would mis-fire the demo-reward gate on the new env's first demo)."""
    return {
        "current_env_index": next_env_index,
        "current_phase": "validation",
        "consecutive_failures": 0,
        "failure_history": [],
        "recent_attempts": [],
        "diagnosis": "",
        "best_model_path": "",
        "approved": False,
        "tasks": [task],
        "code": "",
        "test_results": "",
        # The duo Coder renders execution_stdout/stderr verbatim as "PREVIOUS RUN - RAW EXECUTION
        # OUTPUT (ground truth)". Without clearing them, the NEW env's first Coder prompt embeds the
        # OLD env's output (e.g. a passing demo's success_rate / MODEL_LOADED lines), telling the
        # model its previous run on THIS env already succeeded - the stale-context bug this reset exists to prevent.
        "execution_stdout": "",
        "execution_stderr": "",
        "review_feedback": "",
        "review_suggestions": "",
        "current_task": task,
        "manager_guidance": f"Task: {task}",  # keep the Reviewer's expectation in sync with the NEW env
        "video_dir": video_dir,
        "iteration": 1,
        # C1/C2: fresh cumulative tracking + resume flags for the new env
        "total_env_steps": 0,
        "metric_history": [],
        "measured_sps": None,
        "resume_required": False,
        "resume_ok": True,
        # Goal A: fresh demo-reward gate for the new env
        "demo_reward": None,
        "demo_below_threshold": False,
    }


def skill_from_winning_code(code: str, env_name: str, env_tags: Optional[list] = None) -> dict:
    """Build a PROCEDURAL skill from winning code (far richer than a regex recipe).
    Captures algorithm + policy class + HER + the metric/checkpoint approach, so the NEXT
    env's Coder inherits the full recipe, not just 'SAC'. env_tags (from EnvironmentStep.tags)
    declare the family; the env-id substring heuristic is only a fallback when none declared."""
    env_tags = [t.lower() for t in (env_tags or [])]
    algo_m = re.search(r'\b(PPO|SAC|A2C|DQN|TD3|DDPG)\b', code)
    algo = algo_m.group(1) if algo_m else "the same algorithm"
    policy = ("MultiInputPolicy" if "MultiInputPolicy" in code
              else ("CnnPolicy" if "CnnPolicy" in code else "MlpPolicy"))
    uses_her = "HerReplayBuffer" in code
    is_goal = (uses_her or "MultiInputPolicy" in code or "desired_goal" in code
               or "goal" in env_tags or "success_rate" in env_tags)
    lname = env_name.lower()
    is_manip = ("manipulation" in env_tags or "robotics" in env_tags
                or "panda" in lname or "fetch" in lname)
    family = "panda / robotic manipulation" if is_manip else env_name
    proc = [f"Use {algo} with policy='{policy}'"]
    if uses_her:
        proc.append("replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs={'n_sampled_goal':4,'goal_selection_strategy':'future'}")
    proc.append("read max_episode_steps from env.spec and set learning_starts >= that value")
    proc.append(f"each optimization iteration resume from the checkpoint ({algo}.load + load_replay_buffer, learn one wall-clock-sized chunk computed from measured steps/s, then save model + replay buffer)")
    if is_goal:
        proc.append("report success_rate (the is_success fraction), never the raw sparse reward")
    return {
        "name": f"Solve {family}",
        "when_to_use": (f"A goal-conditioned / sparse-reward env like {env_name} (Dict obs with desired_goal)."
                        if is_goal else f"An env like {env_name}."),
        "procedure": "; ".join(proc) + ".",
        "pitfalls": ((("Plain MlpPolicy or no-HER did NOT work here; " if uses_her else "")
                      + "never pass reset_num_timesteps=False (breaks termination on a reloaded model); "
                      "starving the chunk stalls progress - size it from measured steps/s and accumulate.")
                     if is_goal else "Commit to one algorithm so checkpoint-resume accumulates."),
        "verification": ("RESULT line prints success_rate in [0,1] >= the env threshold."
                         if is_goal else "RESULT mean_reward >= the env threshold."),
        "source_env": env_name,
        "tags": (sorted(set(env_tags) | {"goal", "her", "manipulation", "robotics", "success_rate"}) if is_goal
                 else (env_tags or ["general"])),
        "status": "verified",
        "confidence": 0.9,
    }
