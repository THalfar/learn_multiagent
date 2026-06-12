"""Deterministic verdict gates — the math that overrides an LLM reviewer's APPROVE.

The frontier reviewer (SHODAN / the duo Director) decides approve/reject in prose, but a
handful of verdicts are NOT negotiable and must not depend on the model's mood:

  - threshold gate   : optimization passes ONLY if the real reward clears the threshold.
  - metric lock      : a success_rate env must report a fraction in [0,1], not the raw
                       sparse reward (e.g. -45) — otherwise we'd judge apples vs oranges.
  - resume gate      : if a checkpoint existed but the chunk didn't provably resume it
                       (no 'RESUMED: buffer_transitions=N'), training did NOT accumulate,
                       so a passing reward is a fluke and must not be approved.
  - demo gate (NEW)  : the demo phase's MEASURED metric must ALSO clear the threshold —
                       videos alone are not proof. Below threshold => regress to
                       optimization (demo_below_threshold) so the checkpoint keeps
                       training instead of looping on the same model forever.

These were inline in reviewer.py; extracting them here makes them LLM-free and therefore
unit-testable without an API key, and lets BOTH pipelines (quad Reviewer + duo Director)
share one implementation instead of drifting copies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from src.utils.result_parser import parse_result_line


@dataclass
class GateResult:
    approved: bool          # the verdict AFTER applying the gates (may override True->False)
    feedback_prefix: str    # text to prepend to the reviewer's feedback (empty if no gate fired)
    demo_below_threshold: bool  # demo measured the metric below threshold -> regress to optimization
    gate_fired: str         # "" | "threshold" | "metric_lock" | "resume" | "demo"


def apply_verdict_gates(
    llm_approved: bool,
    *,
    phase: str,
    stdout: Optional[str],
    success_threshold: float,
    env_metric: str,
    resume_required: bool,
    resume_ok: bool,
    demo_reward: Any,
) -> GateResult:
    """Apply the deterministic gates to an LLM verdict.

    `stdout` is parsed for the optimization gates (single source of truth for the real
    reward); `demo_reward` is the demo phase's already-measured metric (None = no RESULT
    line / crash). Returns a GateResult; callers do `feedback = result.feedback_prefix + feedback`.
    """
    approved = bool(llm_approved)
    prefix = ""
    demo_below = False
    gate = ""

    real = parse_result_line(stdout)["value"]

    if phase == "optimization":
        # Metric lock + threshold (combined, as in the original inline gate): an env passes
        # optimization ONLY if the real reward meets the threshold, and a success_rate env's
        # value must live in [0,1].
        wrong_metric = (env_metric == "success_rate" and real is not None
                        and (real < 0.0 or real > 1.0))
        if real is None or real < success_threshold or wrong_metric:
            if wrong_metric:
                gate = "metric_lock"
                prefix += ("[Metric lock] This environment is scored by SUCCESS RATE in [0,1], but the "
                           "RESULT line reported {}. Report the is_success fraction over the eval "
                           "episodes, NOT the raw sparse reward.\n\n".format(real))
            else:
                gate = "threshold"
                prefix += ("[Threshold gate] measured {} is below the {} optimization threshold - "
                           "not approved yet, keep training.\n\n".format(real, success_threshold))
            approved = False

        # Resume gate: deterministic, like the threshold gate. A checkpoint existed but the
        # chunk did not verifiably resume it -> the run did not accumulate.
        if resume_required and not resume_ok:
            if not gate:
                gate = "resume"
            prefix += ("[Resume gate] A checkpoint exists but this chunk did not verifiably resume it "
                       "(missing/zero 'RESUMED: buffer_transitions=N'). Training did NOT accumulate. "
                       "Fix the resume (ALGO.load + load_replay_buffer + RESUMED print) before anything "
                       "else.\n\n")
            approved = False

    elif phase == "demo":
        # Demo-reward gate: videos alone are not proof of solving. The measured metric must
        # clear the threshold; otherwise regress to optimization (the checkpoint keeps training).
        if demo_reward is not None and demo_reward < success_threshold:
            gate = "demo"
            demo_below = True
            approved = False
            prefix += ("[Demo gate] The demo evaluation measured {:.3f}, below the {} threshold. "
                       "A convincing-looking video is not proof - returning to OPTIMIZATION to keep "
                       "training the checkpoint.\n\n".format(float(demo_reward), success_threshold))
        elif demo_reward is None:
            # No measurement (crash / no RESULT line): reject but do NOT regress - a retry may
            # fix a transient crash, and we have no evidence the policy is actually below par.
            gate = "demo"
            approved = False
            prefix += ("[Demo gate] The demo run produced no RESULT line (crash or no evaluation), so "
                       "the policy's demo metric could not be confirmed against the threshold. Rejecting "
                       "this demo attempt; retry the deterministic recording.\n\n")

    return GateResult(approved=approved, feedback_prefix=prefix,
                      demo_below_threshold=demo_below, gate_fired=gate)
