from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COLD_START_MAX_THRESHOLD_SECONDS = 50.0


class ColdStartThresholdExceeded(RuntimeError):
    pass


def default_batch() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    return f"proposed_x1_no_preflight_gate_coldstart50_{stamp}"


def build_runner_argv(batch: str) -> list[str]:
    return [
        "--batch", batch,
        "--methods", "proposed",
        "--repetitions", "1",
        "--start-row", "11708",
        "--duration-minutes", "180",
        "--ci-window-seconds", "480",
        "--ci-time-scale", "8.0",
        "--initial-pods", "2",
        "--initial-placement-nodes", "llm-node-1,llm-node2",
        "--initial-placement-recovery-attempts", "4",
        "--initial-placement-recovery-delay-seconds", "10",
        "--pre-warmup-settle-seconds", "20",
        "--warmup-duration-seconds", "60",
        "--warmup-targeted-requests-per-pod", "8",
        "--warmup-targeted-concurrency", "2",
        "--warmup-targeted-min-cpu-delta", "0.05",
        "--drain-seconds", "30",
        "--cooldown-seconds", "300",
        "--formal-start-delay-seconds", "30",
        "--api-outage-grace-seconds", "180",
        "--ready-latency-p95-seconds", "40",
        "--api-health-checks", "1",
        "--api-health-interval-seconds", "0",
        "--api-health-timeout-seconds", "15",
        "--pod-sample-seconds", "60",
        "--legacy-metrics-poll-seconds", "60",
        "--prometheus-step-seconds", "10",
        "--allow-workload-slice-overrides",
    ]


def enforce_cold_start_threshold(
    method_result: dict[str, Any],
    threshold_seconds: float = COLD_START_MAX_THRESHOLD_SECONDS,
) -> float | None:
    metrics = method_result.get("run_metrics") or {}
    value = metrics.get("cold_start_latency_max_seconds")
    if value is None:
        prom = ((method_result.get("checks") or {}).get("prometheus_summary") or {})
        value = prom.get("cold_start_latency_max_seconds")
    if value is None:
        return None

    max_seconds = float(value)
    if max_seconds > threshold_seconds:
        run_id = method_result.get("run_id", "<unknown>")
        raise ColdStartThresholdExceeded(
            f"cold_start_latency_max_seconds={max_seconds} exceeds "
            f"threshold={threshold_seconds} for run_id={run_id}"
        )
    return max_seconds


def skipped_prelaunch_steps() -> list[str]:
    return [
        "yaml_contract",
        "yaml_image_precheck",
        "image_precheck",
        "controller_mu_dry_run",
        "batch_start_hpa_metrics_preflight",
        "initial_health",
        "initial_cleanup",
        "initial_cooldown",
        "final_health",
    ]


def parse_runner_args(runner: Any, batch: str) -> argparse.Namespace:
    old_argv = sys.argv[:]
    try:
        sys.argv = ["run_stage2_baselines_x3.py", *build_runner_argv(batch)]
        return runner.parse_args()
    finally:
        sys.argv = old_argv


def load_runner() -> Any:
    here = Path(__file__).resolve().parent
    project_root = here if (here / "stage2").is_dir() else here.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    import stage2.run_stage2_baselines_x3 as runner

    return runner


def run_batch(batch: str) -> int:
    runner = load_runner()
    args = parse_runner_args(runner, batch)
    args.run_attempt_pre_health_checks = False
    args.run_post_health_checks = False
    args.collect_extra_prometheus_monitoring = False
    args.collect_hpa_diagnostics_on_success = False

    runner.base.normalize_initial_placement_nodes(args)
    runner.configure_child_pythonpath()
    runner.base.LOGS.mkdir(parents=True, exist_ok=True)
    runner.base.RESULTS.mkdir(parents=True, exist_ok=True)

    workload = Path(args.workload)
    if not workload.exists():
        raise FileNotFoundError(workload)

    label = runner.duration_label(args.duration_minutes)
    batch_dir = (
        runner.base.RESULTS
        / f"stage2_all_baselines_x3_s{args.start_row}_{label}_{args.batch}"
    )
    batch_dir.mkdir(parents=True, exist_ok=True)
    summary_path = batch_dir / "stage2_state.json"
    lock_handle = runner.acquire_lock()
    try:
        if summary_path.exists():
            summary: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
            resumed_at = runner.base.utc_now()
            summary["resumed_at_utc"] = resumed_at
            summary["resume_count"] = int(summary.get("resume_count") or 0) + 1
            summary.setdefault("resume_history", []).append(
                {"resumed_at_utc": resumed_at, "args": vars(args)}
            )
            summary.pop("error", None)
            summary.pop("failed_at_utc", None)
        else:
            summary = {
                "batch": args.batch,
                "started_at_utc": runner.base.utc_now(),
                "args": vars(args),
                "workload": str(workload),
                "selected_methods": args.methods,
                "repetitions": args.repetitions,
                "formal_repetition_seeds": {
                    str(rep): runner.seed_for_repetition(rep, args)
                    for rep in range(1, args.repetitions + 1)
                },
                "formal_repetition_order_indexes": runner.REPETITION_ORDER_INDEXES,
                "passed": False,
                "complete": False,
                "resume_count": 0,
                "events": [],
            }

        summary["formal_repetition_seeds"] = {
            str(rep): runner.seed_for_repetition(rep, args)
            for rep in range(1, args.repetitions + 1)
        }
        summary["formal_repetition_order_indexes"] = runner.REPETITION_ORDER_INDEXES
        summary["no_preflight_gate_override"] = {
            "user_override": "skip preflight/gate preliminary tests for this run",
            "skipped_prelaunch_steps": skipped_prelaunch_steps(),
            "preserved_runner_steps": [
                "initial placement recovery inside run_method",
                "targeted pod-IP warmup",
                "service warmup",
                "drain before formal replay",
                "formal replay",
                "per-attempt cleanup",
                "per-attempt cooldown",
            ],
            "cold_start_restart_policy": {
                "metric": "cold_start_latency_max_seconds",
                "threshold_seconds": COLD_START_MAX_THRESHOLD_SECONDS,
                "action": "mark attempt invalid and retry through runner attempt loop",
                "max_attempts_per_run": args.max_attempts_per_run,
            },
        }
        runner.write_json(summary_path, summary)

        try:
            runner.validate_formal_resume_args(summary, args)
            current_frozen_inputs = runner.validate_frozen_inputs(args)
            if summary.get("frozen_inputs"):
                runner.assert_frozen_inputs_unchanged(
                    summary["frozen_inputs"],
                    current_frozen_inputs,
                )
            else:
                summary["frozen_inputs"] = current_frozen_inputs
            summary["config_snapshot"] = runner.write_config_snapshot(
                batch_dir,
                args,
                summary["frozen_inputs"],
            )
            for step in skipped_prelaunch_steps():
                summary[step] = {
                    "skipped": True,
                    "reason": "user_no_preflight_gate_override",
                }
            runner.write_json(summary_path, summary)

            for repetition in range(1, args.repetitions + 1):
                repetition_order = runner.order_for_repetition(repetition, args.methods)
                for method in repetition_order:
                    method_slot = batch_dir / f"rep{repetition:02d}" / method
                    valid_summary_path = method_slot / "valid_run_summary.json"
                    if valid_summary_path.exists():
                        existing = json.loads(valid_summary_path.read_text(encoding="utf-8"))
                        if existing.get("passed") is True:
                            print(
                                f"SKIP completed repetition={repetition} method={method}",
                                flush=True,
                            )
                            continue

                    success = False
                    last_error = ""
                    start_attempt = runner.next_attempt_number(method_slot)
                    stop_attempt = start_attempt + args.max_attempts_per_run
                    for attempt in range(start_attempt, stop_attempt):
                        attempt_dir = method_slot / f"attempt{attempt:02d}"
                        attempt_dir.mkdir(parents=True, exist_ok=True)
                        print(
                            "========== Stage-2 "
                            f"repetition={repetition} method={method} attempt={attempt} "
                            "no-preflight-gate ==========",
                            flush=True,
                        )
                        event: dict[str, Any] = {
                            "repetition": repetition,
                            "method": method,
                            "attempt": attempt,
                            "seed": runner.seed_for_repetition(repetition, args),
                            "repetition_order": repetition_order,
                            "started_at_utc": runner.base.utc_now(),
                            "pre_health": {
                                "skipped": True,
                                "reason": "attempt_pre_health_check_disabled_by_user_override",
                            },
                        }
                        try:
                            method_result = runner.run_method(
                                method,
                                repetition,
                                attempt,
                                args,
                                attempt_dir,
                                workload,
                            )
                            cold_start_max = enforce_cold_start_threshold(
                                method_result,
                                COLD_START_MAX_THRESHOLD_SECONDS,
                            )
                            runner.write_json(valid_summary_path, method_result)
                            event["passed"] = True
                            event["run_id"] = method_result["run_id"]
                            event["cold_start_latency_max_seconds"] = cold_start_max
                            event["cold_start_threshold_seconds"] = (
                                COLD_START_MAX_THRESHOLD_SECONDS
                            )
                            success = True
                        except ColdStartThresholdExceeded as exc:
                            last_error = str(exc)
                            event["passed"] = False
                            event["failure_type"] = "cold_start_threshold_failure"
                            event["error"] = last_error
                            event["cold_start_threshold_seconds"] = (
                                COLD_START_MAX_THRESHOLD_SECONDS
                            )
                            runner.write_json(
                                attempt_dir / "failed_attempt.json",
                                {**event, "failed_at_utc": runner.base.utc_now()},
                            )
                            print(
                                f"COLD START THRESHOLD FAILED repetition={repetition} "
                                f"method={method} attempt={attempt}: {exc}",
                                flush=True,
                            )
                        except runner.base.KubeApiUnavailable as exc:
                            last_error = str(exc)
                            event["passed"] = False
                            event["failure_type"] = "infra_failure"
                            event["error"] = last_error
                            runner.write_json(
                                attempt_dir / "failed_attempt.json",
                                {**event, "failed_at_utc": runner.base.utc_now()},
                            )
                            print(
                                f"INFRA FAILURE repetition={repetition} "
                                f"method={method} attempt={attempt}: {exc}",
                                flush=True,
                            )
                        except Exception as exc:  # noqa: BLE001
                            last_error = str(exc)
                            event["passed"] = False
                            event["failure_type"] = "method_failure"
                            event["error"] = last_error
                            runner.write_json(
                                attempt_dir / "failed_attempt.json",
                                {**event, "failed_at_utc": runner.base.utc_now()},
                            )
                            print(
                                f"ATTEMPT FAILED repetition={repetition} "
                                f"method={method} attempt={attempt}: {exc}",
                                flush=True,
                            )
                        finally:
                            event["finished_at_utc"] = runner.base.utc_now()
                            summary.setdefault("events", []).append(event)
                            runner.write_json(summary_path, summary)
                            try:
                                event["cleanup"] = runner.cleanup_qwen_with_wait(
                                    args,
                                    f"rep{repetition:02d}_{method}_attempt{attempt:02d}",
                                )
                                runner.base.cooldown(
                                    args.cooldown_seconds,
                                    f"rep{repetition:02d}_{method}_attempt{attempt:02d}",
                                )
                            except runner.base.KubeApiUnavailable as cleanup_exc:
                                event["cleanup_failure_type"] = "infra_failure"
                                event["cleanup_error"] = str(cleanup_exc)
                                event["failure_type"] = "infra_failure"
                                last_error = str(cleanup_exc)
                                print(
                                    "cleanup after attempt hit infra failure: "
                                    f"{cleanup_exc}",
                                    flush=True,
                                )
                            except Exception as cleanup_exc:  # noqa: BLE001
                                event["cleanup_failure_type"] = "cleanup_failure"
                                event["cleanup_error"] = str(cleanup_exc)
                                last_error = str(cleanup_exc)
                                print(
                                    f"cleanup after attempt failed: {cleanup_exc}",
                                    flush=True,
                                )
                            event["post_health"] = {
                                "skipped": True,
                                "reason": "post_attempt_health_check_disabled_by_user_override",
                            }
                            runner.write_json(summary_path, summary)
                        if event.get("failure_type") == "infra_failure":
                            raise RuntimeError(
                                "infrastructure failure; stopping batch before retrying "
                                f"repetition={repetition} method={method}: {last_error}"
                            )
                        if success:
                            aggregate = runner.write_aggregate_artifacts(
                                batch_dir,
                                args.repetitions,
                                args.methods,
                            )
                            summary["aggregate"] = aggregate
                            runner.write_json(summary_path, summary)
                            break
                    if not success:
                        raise RuntimeError(
                            f"no valid run for repetition={repetition} method={method} "
                            f"after attempts {start_attempt}..{stop_attempt - 1}: {last_error}"
                        )

            aggregate = runner.write_aggregate_artifacts(
                batch_dir,
                args.repetitions,
                args.methods,
            )
            summary["aggregate"] = aggregate
            if not aggregate["complete"]:
                raise RuntimeError(f"Stage-2 run matrix incomplete: {aggregate['counts']}")
            summary["passed"] = True
            summary["complete"] = True
            summary["finished_at_utc"] = runner.base.utc_now()
            summary["final_health"] = {
                "skipped": True,
                "reason": "user_no_preflight_gate_override",
            }
            runner.write_json(summary_path, summary)
            print("stage2_state=" + str(summary_path), flush=True)
            return 0
        except BaseException as exc:
            summary["passed"] = False
            summary["complete"] = False
            summary["failed_at_utc"] = runner.base.utc_now()
            summary["error"] = str(exc)
            runner.write_json(summary_path, summary)
            print(f"ERROR: {exc}", flush=True)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return 1
    finally:
        try:
            lock_handle.seek(0)
            lock_handle.truncate()
            lock_handle.flush()
            runner.fcntl.flock(lock_handle.fileno(), runner.fcntl.LOCK_UN)
        finally:
            lock_handle.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default=default_batch())
    args = parser.parse_args(argv)
    return run_batch(args.batch)


if __name__ == "__main__":
    raise SystemExit(main())
