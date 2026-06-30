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
