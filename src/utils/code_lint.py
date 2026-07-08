"""Deterministic pre-Docker lint for Coder-generated RL scripts.

This is NOT a bypass of the Tester — it is a fast (~milliseconds) feedback arc
INSIDE the agent loop (a "pre-Tester"). It catches gross STRUCTURAL errors cheaply
so the Coder gets immediate correction before a 5-20 minute Docker run is wasted:

  1. syntax errors                (ast.parse)
  2. wrong / substituted env name (the single biggest time-waster)
  3. imports outside the known container stack (likely ModuleNotFoundError)
  4. obvious bloat vs the task's line budget (optional)
  5. hallucinated SB3 kwargs      (e.g. HerReplayBuffer(online_sample_strategy=...)
     burned 7 iterations in one night run - caught here in ~ms instead)
  6. the checkpoint-resume contract (optional, optimization phase): when a checkpoint
     exists the script MUST load it + the replay buffer and print
     "RESUMED: buffer_transitions=<n>" so the Tester can verify accumulation.

The Tester still runs the real container and does the SEMANTIC diagnosis on the
real stdout — lint only adds a cheap feedback channel, it does not replace a role.

Usage:
    res = lint_code(code, env_name="PandaPush-v3")
    if not res.ok:
        # hand res.feedback() back to the Coder for a quick regeneration
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import List, Optional, Set

# Modules known to be importable in the citadel-rl container (see docker/Dockerfile).
# Anything outside this set is flagged as a WARNING (likely ModuleNotFoundError).
DEFAULT_ALLOWED_IMPORTS: Set[str] = {
    # stdlib commonly used by RL scripts
    "os", "sys", "time", "math", "random", "json", "re", "warnings",
    "collections", "typing", "functools", "itertools", "pathlib", "dataclasses",
    "argparse", "copy", "glob", "shutil",
    # scientific / RL stack installed in the container
    "numpy", "scipy", "pandas", "torch", "torchvision",
    "gymnasium", "gym", "stable_baselines3", "sb3_contrib", "panda_gym",
    "gym_pybullet_drones", "transforms3d",
    "matplotlib", "seaborn", "optuna", "cv2", "tqdm", "rich",
    "tensorboard", "imageio", "moviepy",
}


# ── Known SB3 2.x signatures (curated; superset across recent versions). ──────
# Hardcoded on purpose: the host env does not have stable_baselines3 installed,
# and a curated superset gives deterministic, version-stable feedback. An unknown
# kwarg on these constructors is almost always an LLM hallucination that would
# crash the container after minutes of startup.
_SB3_COMMON = {
    "policy", "env", "learning_rate", "gamma", "tensorboard_log", "policy_kwargs",
    "verbose", "seed", "device", "_init_setup_model", "stats_window_size",
}
_SB3_OFFPOLICY = _SB3_COMMON | {
    "buffer_size", "learning_starts", "batch_size", "tau", "train_freq",
    "gradient_steps", "action_noise", "replay_buffer_class", "replay_buffer_kwargs",
    "optimize_memory_usage",
}
SB3_ALGO_KWARGS = {
    "SAC": _SB3_OFFPOLICY | {"ent_coef", "target_update_interval", "target_entropy",
                             "use_sde", "sde_sample_freq", "use_sde_at_warmup"},
    "TD3": _SB3_OFFPOLICY | {"policy_delay", "target_policy_noise", "target_noise_clip"},
    "DDPG": _SB3_OFFPOLICY,
    "DQN": _SB3_OFFPOLICY | {"target_update_interval", "exploration_fraction",
                             "exploration_initial_eps", "exploration_final_eps",
                             "max_grad_norm"},
    "PPO": _SB3_COMMON | {"n_steps", "batch_size", "n_epochs", "gae_lambda", "clip_range",
                          "clip_range_vf", "normalize_advantage", "ent_coef", "vf_coef",
                          "max_grad_norm", "use_sde", "sde_sample_freq",
                          "rollout_buffer_class", "rollout_buffer_kwargs", "target_kl"},
    "A2C": _SB3_COMMON | {"n_steps", "gae_lambda", "normalize_advantage", "ent_coef",
                          "vf_coef", "max_grad_norm", "rms_prop_eps", "use_rms_prop",
                          "use_sde", "sde_sample_freq",
                          "rollout_buffer_class", "rollout_buffer_kwargs"},
    "HerReplayBuffer": {"buffer_size", "observation_space", "action_space", "env",
                        "device", "n_envs", "optimize_memory_usage",
                        "handle_timeout_termination", "n_sampled_goal",
                        "goal_selection_strategy", "copy_info_dict"},
}
SB3_LEARN_KWARGS = {"total_timesteps", "callback", "log_interval", "tb_log_name",
                    "reset_num_timesteps", "progress_bar"}
# Valid keys of replay_buffer_kwargs={...} when replay_buffer_class=HerReplayBuffer
HER_RBK_KEYS = {"n_sampled_goal", "goal_selection_strategy", "copy_info_dict",
                "handle_timeout_termination", "optimize_memory_usage"}

# ── Hyperparameter sanity FLOOR (guardrail for LLM-governed tuning). ──────────
# NOT a tuner and NOT an opinion about good values - only rejects literals that are
# physically broken (learning_rate=10, gamma=2, batch_size=0) and would waste a
# container run or silently train a useless policy. A value the LLM cannot justify
# but is *in range* is left alone (the metric curve judges it). Only literal numbers
# are checked; a variable / schedule fn / 'auto' string is skipped (can't be sure).
_HP_BOUNDS = {
    "learning_rate":   (1e-7, 1.0),
    "ent_coef":        (0.0, 1e3),         # numeric only; ent_coef='auto' is a string -> skipped
    "vf_coef":         (0.0, 1e3),
    "clip_range":      (1e-4, 1.0),
    "batch_size":      (1, 1_000_000),
    "buffer_size":     (1, 100_000_000),
    "learning_starts": (0, 50_000_000),
    "n_steps":         (1, 10_000_000),
    "n_epochs":        (1, 1000),
    "gradient_steps":  (-1, 1_000_000),
    "n_sampled_goal":  (1, 64),
}
# Probabilities / rates that must live in (0, 1].
_HP_UNIT_INTERVAL = {"gamma", "tau", "gae_lambda"}


def _const_num(node: ast.AST):
    """Return the float value of a numeric literal (incl. a unary-minus literal),
    or None for anything non-literal (variable, call, 'auto' string, schedule fn)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _const_num(node.operand)
        return -inner if inner is not None else None
    return None


def check_hyperparam_sanity(tree: ast.AST) -> List[str]:
    """Reject clearly-insane hyperparameter LITERALS on SB3 algo constructors.

    A sanity floor for the LLM-governed tuning loop: it fires only on values outside
    physically-reasonable ranges (e.g. learning_rate=10, gamma=2, batch_size=0), never
    on a merely-suboptimal-but-valid value. Returns a list of violations (empty = sane).
    """
    out: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node.func) not in SB3_ALGO_KWARGS:   # only algo constructors carry these
            continue
        for kw in node.keywords:
            if kw.arg is None:
                continue
            v = _const_num(kw.value)
            if v is None:
                continue
            if kw.arg in _HP_UNIT_INTERVAL:
                if not (0.0 < v <= 1.0):
                    out.append(f"{kw.arg}={v:g} is out of range - it must be in (0, 1].")
            elif kw.arg in _HP_BOUNDS:
                lo, hi = _HP_BOUNDS[kw.arg]
                if v < lo or v > hi:
                    out.append(f"{kw.arg}={v:g} is insane - expected within [{lo:g}, {hi:g}].")
    return out


@dataclass
class LintResult:
    """Outcome of a lint pass. `errors` must be fixed (regenerate / skip Docker);
    `warnings` are advisory and surfaced to the Coder as feedback."""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when there are no hard errors (warnings still allow the run)."""
        return not self.errors

    def feedback(self) -> str:
        """Human/LLM-readable feedback block (empty string if fully clean)."""
        lines: List[str] = []
        for e in self.errors:
            lines.append(f"  [ERROR] {e}")
        for w in self.warnings:
            lines.append(f"  [WARN]  {w}")
        return "\n".join(lines)


def _make_env_literals(tree: ast.AST) -> List[str]:
    """Collect string literals passed as the first arg to any `*.make(...)` call
    (e.g. gym.make("X"), gymnasium.make("X")). These are the env names the script
    actually instantiates."""
    envs: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "make" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    envs.append(first.value)
    return envs


def _imported_top_modules(tree: ast.AST) -> Set[str]:
    """Top-level module name of every import (e.g. `from stable_baselines3.her import X`
    -> `stable_baselines3`)."""
    mods: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import (never valid for a standalone script)
            if node.module and node.level == 0:
                mods.add(node.module.split(".")[0])
    return mods


def _call_name(func: ast.AST) -> str:
    """Resolve the called name: SAC(...) -> 'SAC', sb3.SAC(...) -> 'SAC',
    model.learn(...) -> 'learn'."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _check_sb3_kwargs(tree: ast.AST, res: LintResult, code: str) -> None:
    """Validate kwarg NAMES against the known SB3 signatures. An unknown kwarg on
    SAC/PPO/.../HerReplayBuffer/.learn() is a hard error (it is a guaranteed
    TypeError after a wasted container start). `.load()` is NOT checked - it
    legitimately accepts arbitrary attribute overrides via **kwargs."""
    uses_her = "HerReplayBuffer" in code
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        allowed: Optional[Set[str]] = None
        if name in SB3_ALGO_KWARGS:
            allowed = SB3_ALGO_KWARGS[name]
        elif name == "learn":
            allowed = SB3_LEARN_KWARGS
        if allowed is None:
            continue
        for kw in node.keywords:
            if kw.arg is None:  # **kwargs splat - can't analyze
                continue
            if kw.arg not in allowed:
                res.errors.append(
                    f"Unknown kwarg '{kw.arg}' for {name}() - it does NOT exist in the "
                    f"stable-baselines3 API and will raise TypeError. "
                    f"Valid kwargs: {', '.join(sorted(allowed))}."
                )
            # reset_num_timesteps=False breaks episode termination on a reloaded model
            if (name == "learn" and kw.arg == "reset_num_timesteps"
                    and isinstance(kw.value, ast.Constant) and kw.value.value is False):
                res.errors.append(
                    "reset_num_timesteps=False breaks episode termination on a reloaded "
                    "model - remove it (the DEFAULT True still continues the loaded weights)."
                )
            # replay_buffer_kwargs dict keys (HER): catches hallucinated strategies
            if kw.arg == "replay_buffer_kwargs" and uses_her and isinstance(kw.value, ast.Dict):
                for key_node in kw.value.keys:
                    if (isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)
                            and key_node.value not in HER_RBK_KEYS):
                        res.errors.append(
                            f"Unknown HerReplayBuffer kwarg '{key_node.value}' in "
                            f"replay_buffer_kwargs - valid keys: {', '.join(sorted(HER_RBK_KEYS))}."
                        )


def check_resume_contract(code: str) -> List[str]:
    """The checkpoint-resume contract for OPTIMIZATION iterations when a checkpoint
    already exists. Returns a list of violations (empty = contract satisfied).

    Required so training ACCUMULATES across iterations instead of silently
    restarting from scratch (the failure mode that burned PandaPush for 24
    iterations: model loaded without its replay buffer -> off-policy SAC+HER
    forgot everything each chunk):
      1. load the checkpoint        (ALGO.load(..., env=env))
      2. load the replay buffer     (model.load_replay_buffer(...))
      3. print the resume proof     (print(f"RESUMED: buffer_transitions={n}"))
      4. save model + buffer after the chunk (model.save + save_replay_buffer)
    """
    violations: List[str] = []
    if ".load(" not in code:
        violations.append(
            "A checkpoint EXISTS but the script never loads it - use "
            "ALGO.load('/workspace/output/best_model', env=env) so training accumulates. "
            "Do NOT initialize a fresh model when a checkpoint exists."
        )
    if "load_replay_buffer" not in code:
        violations.append(
            "Off-policy resume without model.load_replay_buffer(...) silently forgets "
            "all collected experience - load '/workspace/output/best_model_buffer' "
            "(guard with os.path.exists(buf + '.pkl'))."
        )
    if "RESUMED:" not in code:
        violations.append(
            "Missing the resume proof line. Immediately after loading, print exactly: "
            "print(f\"RESUMED: buffer_transitions={model.replay_buffer.size()}\") "
            "- the Tester REJECTS the run without it."
        )
    if "save_replay_buffer" not in code:
        violations.append(
            "Missing model.save_replay_buffer('/workspace/output/best_model_buffer') "
            "after training - without it the NEXT chunk cannot resume the buffer."
        )
    return violations


def lint_code(
    code: str,
    env_name: Optional[str] = None,
    allowed_imports: Optional[Set[str]] = None,
    max_lines: Optional[int] = None,
    require_resume: bool = False,
) -> LintResult:
    """Run the deterministic structural checks. Returns a LintResult; `res.ok` is
    True when there are no hard errors.

    Args:
        code: the Python source the Coder produced.
        env_name: the EXACT environment the task requires (e.g. "PandaPush-v3").
            If the script instantiates a different env, that is a hard error.
        allowed_imports: override the default container import whitelist.
        max_lines: if given, more than this many non-blank/non-comment lines is
            flagged as bloat (a hard error). Leave None to skip the bloat check.
        require_resume: enforce the checkpoint-resume contract (optimization phase
            with an existing checkpoint) - see check_resume_contract().
    """
    res = LintResult()

    if not code or not code.strip():
        res.errors.append("Empty code — no script was produced.")
        return res

    # 1. Syntax — also yields the AST used by the remaining checks.
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        res.errors.append(f"SyntaxError at line {e.lineno}: {e.msg}")
        return res  # cannot analyze further without a parse tree

    # 2. Environment name — the #1 waste (substituted / wrong / dropped -v3).
    if env_name:
        env_literals = _make_env_literals(tree)
        if env_literals:
            if env_name not in env_literals:
                res.errors.append(
                    f"Wrong environment: the task requires '{env_name}' but the "
                    f"script creates {env_literals}. Use EXACTLY '{env_name}'."
                )
        elif env_name not in code:
            res.warnings.append(
                f"Environment '{env_name}' not found as a literal in the script — "
                f"make sure the gym.make call uses exactly '{env_name}'."
            )

    # 3. Import whitelist — unknown modules are almost always a ModuleNotFoundError
    #    that would only surface after a wasted container start.
    allowed = allowed_imports if allowed_imports is not None else DEFAULT_ALLOWED_IMPORTS
    for mod in sorted(_imported_top_modules(tree)):
        if mod not in allowed:
            res.warnings.append(
                f"Unusual import '{mod}' — not in the known container stack; "
                f"verify it is installed or the run will fail with ModuleNotFoundError."
            )

    # 3b. SB3 kwarg names — hallucinated kwargs are a guaranteed TypeError that
    #     would only surface after a wasted container start (or 7 wasted iterations).
    _check_sb3_kwargs(tree, res, code)

    # 3b-2. Hyperparameter sanity floor — the LLM governs hyperparameters now, so
    #       catch physically-broken literals (learning_rate=10, gamma=2) before Docker.
    for v in check_hyperparam_sanity(tree):
        res.errors.append(f"[HYPERPARAM] {v}")

    # 3c. Checkpoint-resume contract (optimization phase, checkpoint exists).
    if require_resume:
        for v in check_resume_contract(code):
            res.errors.append(f"[RESUME CONTRACT] {v}")

    # 4. Bloat vs the task's stated line budget (optional).
    if max_lines is not None:
        code_lines = [
            ln for ln in code.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        if len(code_lines) > max_lines:
            res.errors.append(
                f"Bloat: {len(code_lines)} code lines but the task budget is "
                f"~{max_lines}. Remove unrequested code (extra prints, callbacks, "
                f"model.save, etc.) and keep it minimal."
            )

    return res
