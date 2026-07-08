"""Single source of truth for parsing the Coder's canonical RESULT line.

The training scripts print exactly:

    RESULT: mean_reward=X, std_reward=Y, episodes=Z

(goal-conditioned envs may use `success_rate=` as the metric key — the value is
still carried in the same slot). Three agents read this line — the Tester (ground-truth
reconciliation + cumulative tracking + video caption) and the Reviewer (threshold/metric
gate). They MUST agree, so the regex lives here once instead of being copy-pasted (a
format drift fixed in one file but not another silently force-rejected every iteration).
"""
from __future__ import annotations

import re
from typing import Optional, TypedDict

# Strict: only the explicit "RESULT:" line counts (never loose prose like "Mean reward
# over 5 episodes" elsewhere in stdout, which the old loose video-caption regex matched).
_RESULT_RE = re.compile(
    r"RESULT:\s*(mean_reward|success_rate)\s*=\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE
)
_STD_RE = re.compile(r"std_reward\s*=\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_EPS_RE = re.compile(r"episodes\s*=\s*(\d+)", re.IGNORECASE)


class ResultLine(TypedDict):
    value: Optional[float]   # the metric value, or None when no RESULT line is present
    metric: Optional[str]    # "mean_reward" | "success_rate" | None
    std: Optional[float]
    episodes: Optional[int]


def parse_result_line(stdout: Optional[str]) -> ResultLine:
    """Parse the canonical RESULT line from a run's stdout.

    Returns a dict with `value` None when there is no RESULT line (crash / timeout /
    no evaluation) — callers treat that as 'there is no reward', never as 0.
    """
    text = stdout or ""
    m = _RESULT_RE.search(text)
    if not m:
        return {"value": None, "metric": None, "std": None, "episodes": None}
    std_m = _STD_RE.search(text)
    eps_m = _EPS_RE.search(text)
    return {
        "value": float(m.group(2)),
        "metric": m.group(1).lower(),
        "std": float(std_m.group(1)) if std_m else None,
        "episodes": int(eps_m.group(1)) if eps_m else None,
    }


# ── Optional telemetry lines: "STATS:" and "PARAMS:" ─────────────────────────
# The Coder may print, alongside RESULT, two diagnostic lines the Manager/Reviewer
# requested so the LLM can GOVERN hyperparameters from data instead of guessing:
#   PARAMS: learning_rate=3e-4, batch_size=256, gamma=0.99   (the hyperparameters it chose)
#   STATS:  ep_len_mean=180.5, action_sat=0.31, success_last100=0.6   (run diagnostics)
# Both are flat "key=number" lists (the same shape as RESULT) so they parse with one
# regex and never need fragile JSON. Non-numeric values (e.g. ent_coef=auto) are simply
# skipped — only measured numbers carry signal. Absent line -> empty dict (a no-op).
_KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def parse_kv_line(stdout: Optional[str], label: str) -> dict:
    """Parse a 'LABEL: k=v, k=v, ...' telemetry line into {k: float}.

    Scans the FIRST line whose stripped text starts with '<label>:' (case-insensitive)
    and pulls every key=number pair from it. Returns {} when the line is absent, so a
    caller can treat 'no telemetry' the same as 'no keys'.
    """
    text = stdout or ""
    prefix = label.upper() + ":"
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith(prefix):
            return {k: float(v) for k, v in _KV_RE.findall(s[len(label) + 1:])}
    return {}


def parse_stats_line(stdout: Optional[str]) -> dict:
    """Run-diagnostics the Coder reported (ep_len_mean, action_sat, losses, ...)."""
    return parse_kv_line(stdout, "STATS")


def parse_params_line(stdout: Optional[str]) -> dict:
    """The numeric hyperparameters the Coder chose this run (learning_rate, ...)."""
    return parse_kv_line(stdout, "PARAMS")
