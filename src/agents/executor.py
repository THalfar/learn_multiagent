"""Executor — the deterministic execution node of the duo pipeline (Director -> Coder -> Executor).

It runs the Coder's script in the Docker sandbox and emits a FACTUAL, LLM-FREE report:
the raw stdout/stderr (with the rule-based AUTOMATED DIAGNOSTICS block appended), the parsed
RESULT, resume verification, cumulative tracking, and deterministic demo video+eval. This
replaces the quad pipeline's Tester LLM layer — the Coder reads the raw output directly
(no 'broken telephone' through a local model's paraphrase) and the Director judges from the
same ground truth.

It is deliberately NOT a BaseAgent: a BaseAgent would build an LLM client, and the whole
point of the Executor is that it never calls a model. It reuses the Tester's container,
diagnostics, video and timeout helpers so both pipelines run code identically, and it mirrors
Tester.__call__ minus the LLM analysis/chat and the reviewer-doc-inspection branch.

LangGraph contract (duo): the Executor NEVER returns `iteration`, `approved`, `current_task`,
or any history/opinion key — only the Director advances the iteration counter and the verdict.
(The quad Tester's demo path returns `iteration: 1`; that double-increment is a quad quirk and
is intentionally NOT copied here.)
"""
import os
import re
import time
import subprocess

from .tester import (
    run_in_container,
    diagnose_common_issues,
    check_video_files,
    auto_fix_common_issues,
    is_safe_code,
    run_diagnostic_in_container,
    compute_execution_timeout,
    strip_invalid_timeout_kwarg,
    DOCKER_SANDBOX_ENABLED,
    Tester,
)
from src.utils.result_parser import parse_result_line
from rich import print


class Executor:
    def __init__(self, config):
        self.config = config

    def _log(self, logger, iteration, test_results, stdout, stderr, exec_time):
        """log_tester with the LLM fields zeroed (the Executor runs no model)."""
        if logger:
            logger.log_tester(
                iteration=iteration, test_results=test_results,
                execution_stdout=stdout, execution_stderr=stderr,
                execution_time=exec_time, llm_duration=0, tokens_in=0, tokens_out=0,
            )

    def __call__(self, state: dict) -> dict:
        code = state.get("code", "")
        current_phase = state.get("current_phase", "validation")
        _in_demo = current_phase == "demo"
        _demo_none = {"demo_reward": None} if _in_demo else {}
        iteration = state.get("iteration", 0)
        logger = state.get("conversation_logger")

        # Strip the invalid model.learn(timeout=...) kwarg LLMs sometimes emit (shared,
        # newline-safe helper; the previous inline regex matched across lines and could delete
        # whole statements around a plain `timeout = N` assignment).
        code = strip_invalid_timeout_kwarg(code)
        # The early-return paths below run no optimization chunk, so clear any resume verdict left
        # in state from a PREVIOUS iteration - otherwise the Director's resume gate fires on stale
        # flags and misattributes this iteration's (lint/no-code) failure to the resume contract.
        _no_resume = {"resume_required": False, "resume_ok": True}

        if not code:
            return {"test_results": "No code to test", "execution_stdout": "",
                    "execution_stderr": "", "diagnosis": "No code produced", **_no_resume, **_demo_none}
        if not is_safe_code(code):
            return {"test_results": "ERROR: Dangerous code detected", "execution_stdout": "",
                    "execution_stderr": "Dangerous code detected, execution blocked",
                    "diagnosis": "Dangerous code blocked", **_no_resume, **_demo_none}

        env_progression = self.config.environment_progression
        current_env_index = state.get("current_env_index", 0)
        current_env = (env_progression[current_env_index]
                       if env_progression and current_env_index < len(env_progression) else None)
        base_timeout = (current_env.execution_timeout if current_env
                        else (env_progression[0].execution_timeout if env_progression else 900))
        current_env_name = current_env.name if current_env else "unknown"
        execution_timeout = compute_execution_timeout(self.config, base_timeout, current_phase)

        run_id = state["run_id"]
        code_dir = f"output/{run_id}/{current_env_name}/code"
        video_dir = os.path.abspath(os.path.normpath(state["video_dir"]))
        os.makedirs(code_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)
        code_path = f"{code_dir}/agent_code_iter_{iteration}.py"

        # ── Lint backstop (non-demo): skip Docker on structural errors ──
        if not _in_demo:
            from src.utils.code_lint import lint_code
            _lint = lint_code(code, env_name=(current_env.name if current_env else None))
            if not _lint.ok:
                print("[yellow]🔎 LINT BACKSTOP: structural errors - skipping Docker (fast fail)[/yellow]")
                fb = _lint.feedback()
                self._log(logger, iteration, "LINT FAILED:\n" + fb, "", "LINT FAILED:\n" + fb, 0)
                return {
                    "test_results": "LINT FAILED (Docker skipped - fix these structural errors):\n" + fb,
                    "execution_stdout": "", "execution_stderr": "LINT FAILED:\n" + fb,
                    "diagnosis": ("LINT FAILED: " + fb)[:300],
                    **_no_resume,  # this iteration ran no chunk -> clear any stale resume verdict
                }

        # ── Resume pre-gate (optimization): a checkpoint must be resumed, not retrained fresh ──
        resume_required = False
        resume_ok = True
        _buffer_existed = False
        if current_phase == "optimization":
            _ckpt = Tester._find_saved_model(video_dir)
            resume_required = bool(_ckpt)
            _buffer_existed = os.path.isfile(os.path.join(video_dir, "best_model_buffer.pkl"))
            if resume_required:
                from src.utils.code_lint import check_resume_contract
                _violations = check_resume_contract(code)
                if _violations:
                    _fb = "\n".join(f"  [ERROR] [RESUME CONTRACT] {v}" for v in _violations)
                    print("[yellow]🔗 RESUME GATE: checkpoint exists but the script does not resume it - skipping Docker[/yellow]")
                    self._log(logger, iteration, "RESUME CONTRACT FAILED:\n" + _fb, "", "RESUME CONTRACT FAILED:\n" + _fb, 0)
                    return {
                        "test_results": ("RESUME CONTRACT FAILED (Docker skipped). A checkpoint exists at "
                                         "/workspace/output/best_model - the optimization script MUST resume it "
                                         "(ALGO.load + load_replay_buffer + RESUMED print + save both) so training "
                                         "accumulates:\n" + _fb),
                        "execution_stdout": "", "execution_stderr": "RESUME CONTRACT FAILED:\n" + _fb,
                        "diagnosis": ("RESUME CONTRACT FAILED: " + "; ".join(_violations))[:400],
                        "resume_required": True, "resume_ok": False,
                    }

        # ── DEMO: deterministic-only video+eval (NO LLM fallback) ──
        if _in_demo:
            return self._run_demo(state, current_env, current_env_name, code_dir,
                                  video_dir, execution_timeout, logger, iteration)

        # ── Train / validation: write code, run in the sandbox ──
        code = auto_fix_common_issues(code)
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)

        gpu_enabled = self.config.gpu.enabled
        try:
            execution_start = time.time()
            if DOCKER_SANDBOX_ENABLED:
                result, _vram = run_in_container(
                    code_path, video_dir, execution_timeout, gpu_enabled,
                    verbose_gpu_validation=self.config.verbose.gpu_validation,
                    verbose_docker_info=self.config.verbose.docker_sandbox_info,
                )
            else:
                result = subprocess.run(
                    ["python", code_path], cwd=os.getcwd(), capture_output=True, text=True,
                    encoding='utf-8', errors='replace', timeout=execution_timeout,
                )
            execution_duration = time.time() - execution_start
        except subprocess.TimeoutExpired:
            timeout_str = f"{execution_timeout // 60}m {execution_timeout % 60}s"
            print(f"[bold red]⏰ TIMEOUT: execution exceeded {timeout_str}[/bold red]")
            tr = (f"TIMEOUT: Execution exceeded {timeout_str}. No metrics or videos were generated. "
                  f"Reduce timesteps (or the resumed chunk size) so it finishes within the timeout.")
            self._log(logger, iteration, tr, "", f"Execution timeout after {execution_timeout} seconds", execution_timeout)
            return {
                "test_results": tr, "execution_stdout": "",
                "execution_stderr": f"Execution timeout after {execution_timeout} seconds",
                "diagnosis": f"Timeout after {timeout_str}",
                "resume_required": resume_required, "resume_ok": False if resume_required else True,
            }
        except Exception as e:
            print(f"[bold red]💥 UNEXPECTED ERROR: {type(e).__name__}: {e}[/bold red]")
            tr = f"Unexpected error outside the sandbox: {str(e)[:200]}"
            self._log(logger, iteration, tr, "", str(e), 0)
            return {
                "test_results": tr, "execution_stdout": "", "execution_stderr": str(e),
                "diagnosis": f"{type(e).__name__}: {str(e)[:150]}",
                "resume_required": resume_required, "resume_ok": False if resume_required else True,
            }

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        # ImportError -> automatic container diagnostics (what's actually installed).
        if "ImportError" in stderr or "ModuleNotFoundError" in stderr:
            print("[bold yellow]🔍 ImportError detected - running container diagnostics...[/bold yellow]")
            diag_output = run_diagnostic_in_container()
            stderr = stderr + "\n\n=== CONTAINER DIAGNOSTICS ===\n" + diag_output

        # Rule-based diagnostics appended to stderr — THIS is the heart of the Coder's raw
        # revision block (it reads the AUTOMATED DIAGNOSTICS the same way SHODAN's Tester did).
        auto_diagnostics = diagnose_common_issues(
            stdout=stdout, stderr=stderr, execution_timeout=execution_timeout,
            execution_time=execution_duration, env_config=current_env, code=code, phase=current_phase,
        )
        if auto_diagnostics:
            diag_block = "\n=== AUTOMATED DIAGNOSTICS ===\n"
            for i, finding in enumerate(auto_diagnostics, 1):
                diag_block += f"[{i}] {finding}\n"
                print(f"[bold yellow]🔍 [{i}] {finding}[/bold yellow]")
            diag_block += "=== END DIAGNOSTICS ===\n"
            stderr = (stderr or "") + diag_block

        # ── Parse RESULT (single source of truth) ──
        _parsed = parse_result_line(stdout)
        _val = _parsed["value"]
        success_threshold = current_env.success_threshold if current_env else 0
        _env_metric = getattr(current_env, "metric", "reward") if current_env else "reward"

        # ── Resume post-verify: stdout must show RESUMED: buffer_transitions=N (N>0 if a
        # buffer file existed before the run; N=0 is fine on the first chunk). ──
        if resume_required:
            _resume_m = re.search(r"RESUMED:\s*buffer_transitions\s*=\s*(\d+)", stdout or "")
            resume_ok = bool(_resume_m and (int(_resume_m.group(1)) > 0 or not _buffer_existed))
            if resume_ok:
                print(f"[bold green]🔗 RESUME VERIFIED: buffer_transitions={_resume_m.group(1)}[/bold green]")
            else:
                print("[bold red]🔗 RESUME CHECK FAILED: training did not provably accumulate[/bold red]")

        # ── Deterministic test_results (no LLM paraphrase) ──
        if _val is not None:
            meets = _val >= float(success_threshold)
            test_results = (f"EXECUTION OK. RESULT: {_env_metric}={_val} episodes={_parsed['episodes']} "
                            f"| meets_threshold={meets} (threshold {success_threshold}) | {execution_duration:.0f}s")
        else:
            test_results = (f"NO RESULT LINE: the script printed no 'RESULT: {_env_metric}=...' line "
                            f"(crash / timeout / no evaluation). Ran {execution_duration:.0f}s. See stderr.")
        if resume_required and not resume_ok:
            test_results = ("RESUME CHECK FAILED: a checkpoint exists but this chunk did not verifiably resume "
                            "it (missing/zero 'RESUMED: buffer_transitions=N'). Training did NOT accumulate - "
                            "fix the load/print before anything else.\n\n") + test_results

        # ── Deterministic diagnosis (feeds the Coder's recent-attempts memory) ──
        if auto_diagnostics:
            _diagnosis = "; ".join(auto_diagnostics)[:400]
        elif _val is None:
            _err_lines = [ln for ln in (stderr or "").splitlines() if ln.strip()]
            _err_last = _err_lines[-1][:200] if _err_lines else ""
            _diagnosis = (f"No RESULT line.{(' Last stderr: ' + _err_last) if _err_last else ''}").strip()
        else:
            _diagnosis = test_results[:300]

        result_dict = {
            "test_results": test_results,
            "execution_stdout": stdout,
            "execution_stderr": stderr,
            "diagnosis": _diagnosis,
            "resume_required": resume_required,
            "resume_ok": resume_ok,
        }

        # ── Cumulative tracking (deterministic; makes (non-)accumulation visible) ──
        # Only count steps that genuinely accumulated: skip resume-violated runs (resume_required
        # but resume_ok=False) because those trained fresh from scratch — counting them would
        # inflate total_env_steps/metric_history and could reset consecutive_failures via the
        # Director's improved check even though no real progress was made on the checkpoint.
        _steps_m = (re.search(r"total_timesteps\s*=\s*(\d+)", code)
                    or re.search(r"\.learn\(\s*(\d+)", code))
        _accumulation_valid = not resume_required or resume_ok
        if _val is not None and _accumulation_valid:
            if current_phase == "optimization":
                result_dict["metric_history"] = (state.get("metric_history") or []) + [_val]
            if _steps_m:
                _steps_done = int(_steps_m.group(1))
                result_dict["total_env_steps"] = (state.get("total_env_steps") or 0) + _steps_done
                if _steps_done >= 5000 and execution_duration > 1:
                    _sps = round(_steps_done / execution_duration, 1)
                    result_dict["measured_sps"] = _sps
                    print(f"[dim]📈 Cumulative: {result_dict['total_env_steps']:,} steps this env | ~{_sps} steps/s[/dim]")

        # ── Locate the saved model (validation OR optimization) ──
        # Validation also saves a model (initial_validation_task: "Save the model at the end"),
        # so recording best_model_path here keeps the Director's checkpoint signal (it reads
        # best_model_path) in sync with this Executor's resume pre-gate (which scans the
        # filesystem). Otherwise the validation .zip makes the gate DEMAND a resume while the
        # Director still tells the Coder to "train fresh" - a guaranteed wasted iteration.
        if current_phase in ("validation", "optimization"):
            mp = Tester._find_saved_model(video_dir)
            if mp:
                result_dict["best_model_path"] = mp
                print(f"[bold green]💾 MODEL FOUND: {mp}[/bold green]")
            elif current_phase == "optimization":
                print(f"[yellow]⚠️  No saved model (.zip) found in {video_dir}[/yellow]")

        self._log(logger, iteration, test_results, stdout, stderr, execution_duration)
        return result_dict

    def _run_demo(self, state, current_env, current_env_name, code_dir, video_dir,
                  execution_timeout, logger, iteration):
        """Deterministic demo: load the saved model, eval over fixed-seed episodes, record
        video. NO LLM fallback (the quad Tester falls back to LLM code; the duo Executor must
        not). Always sets demo_reward (value or None) for the Director's demo-reward gate."""
        best_model_path = state.get("best_model_path", "")
        host_model_path = best_model_path if (best_model_path and os.path.isfile(best_model_path)) else ""
        if not host_model_path:
            tr = ("DEMO FAILED: no saved model (.zip) found. The optimization phase must "
                  "model.save('/workspace/output/best_model') before the demo can record.")
            print(f"[yellow]⚠️  {tr}[/yellow]")
            self._log(logger, iteration, tr, "", tr, 0)
            return {"test_results": tr, "execution_stdout": "", "execution_stderr": tr,
                    "diagnosis": "No saved model for demo", "demo_reward": None}

        demo_metric = getattr(current_env, "metric", "reward")
        docker_model_path = "/workspace/output/best_model.zip"
        docker_output_dir = f"/workspace/output/iter_{iteration}/"
        video_script = Tester.generate_video_script(
            current_env_name, docker_model_path, docker_output_dir, metric=demo_metric)
        demo_code_path = f"{code_dir}/demo_video_iter_{iteration}.py"
        with open(demo_code_path, "w", encoding="utf-8") as f:
            f.write(video_script)

        print("\n" + "=" * 70)
        print("[bold magenta]🎬 DETERMINISTIC DEMO (eval + video, no LLM)[/bold magenta]")
        print(f"[magenta]   Env: {current_env_name} | metric: {demo_metric}[/magenta]")
        print("=" * 70)

        gpu_enabled = self.config.gpu.enabled
        try:
            execution_start = time.time()
            if DOCKER_SANDBOX_ENABLED:
                result, _vram = run_in_container(
                    demo_code_path, video_dir, execution_timeout, gpu_enabled,
                    verbose_gpu_validation=False,
                    verbose_docker_info=self.config.verbose.docker_sandbox_info,
                )
            else:
                result = subprocess.run(
                    ["python", demo_code_path], cwd=os.getcwd(), capture_output=True, text=True,
                    encoding='utf-8', errors='replace', timeout=execution_timeout,
                )
            execution_duration = time.time() - execution_start
        except subprocess.TimeoutExpired:
            tr = f"DEMO TIMEOUT: the deterministic recording exceeded {execution_timeout}s."
            print(f"[bold red]⏰ {tr}[/bold red]")
            self._log(logger, iteration, tr, "", tr, execution_timeout)
            return {"test_results": tr, "execution_stdout": "", "execution_stderr": tr,
                    "diagnosis": "Demo timeout", "demo_reward": None}
        except Exception as e:
            tr = f"DEMO ERROR: {str(e)[:200]}"
            print(f"[bold red]💥 {tr}[/bold red]")
            self._log(logger, iteration, tr, "", str(e), 0)
            return {"test_results": tr, "execution_stdout": "", "execution_stderr": str(e),
                    "diagnosis": "Demo error", "demo_reward": None}

        stdout = result.stdout or ""
        stderr = result.stderr or ""
        video_check = check_video_files(video_dir)
        _demo_val = parse_result_line(stdout)["value"]
        _demo_val_txt = f"{_demo_val:.2f}" if _demo_val is not None else "N/A (no RESULT line)"
        valid = video_check.get("valid_videos", 0)
        test_results = (f"DEMO: deterministic eval over fixed-seed episodes. Measured {demo_metric}={_demo_val_txt}. "
                        f"{valid} valid video(s). Execution {execution_duration:.1f}s.")
        print(f"[bold magenta]🎬 Demo: {valid} video(s), {demo_metric}={_demo_val_txt}, {execution_duration:.1f}s[/bold magenta]")

        self._log(logger, iteration, test_results, stdout, stderr, execution_duration)
        if logger and valid > 0:
            try:
                valid_videos = [vf for vf in video_check["video_files"]
                                if vf.get("is_valid") and not vf.get("is_empty")]
                logger.log_video(current_env_name, valid_videos, mean_reward=_demo_val)
            except Exception as _e:
                print(f"[dim]Could not embed video in log: {_e}[/dim]")

        # NOTE: no "iteration" key — only the Director increments in the duo pipeline.
        return {"test_results": test_results, "execution_stdout": stdout, "execution_stderr": stderr,
                "diagnosis": test_results[:300], "demo_reward": _demo_val}
