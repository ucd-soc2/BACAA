from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ColdStartThresholdExceeded(RuntimeError):
    message: str
    observed_seconds: float
    threshold_seconds: float
    method_result: dict[str, Any]

    def __str__(self) -> str:
        return self.message


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")


def load_runner() -> Any:
    here = Path(__file__).resolve().parent
    project_root = here if (here / "stage2").is_dir() else here.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    import stage2.run_stage2_baselines_x3 as runner

    return runner


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


def preserved_runner_steps() -> list[str]:
    return [
        "initial placement recovery inside run_method",
        "targeted pod-IP warmup",
        "service warmup",
        "drain before formal replay",
        "formal replay",
        "per-attempt cleanup",
        "per-attempt cooldown",
    ]


def parse_runner_args(runner: Any, runner_argv: list[str]) -> argparse.Namespace:
    old_argv = sys.argv[:]
    try:
        sys.argv = ["run_stage2_baselines_x3.py", *runner_argv]
        return runner.parse_args()
    finally:
        sys.argv = old_argv


def cold_start_latency_max(method_result: dict[str, Any]) -> float | None:
    metrics = method_result.get("run_metrics") or {}
    value = metrics.get("cold_start_latency_max_seconds")
    if value is None:
        prom = ((method_result.get("checks") or {}).get("prometheus_summary") or {})
        value = prom.get("cold_start_latency_max_seconds")
    if value is None:
        return None
    return float(value)


def enforce_cold_start_threshold(
    method_result: dict[str, Any],
    threshold_seconds: float,
) -> float | None:
    observed = cold_start_latency_max(method_result)
    if observed is None or observed <= threshold_seconds:
        return observed
    run_id = method_result.get("run_id", "<unknown>")
    raise ColdStartThresholdExceeded(
        message=(
            f"cold_start_latency_max_seconds={observed} exceeds "
            f"threshold={threshold_seconds} for run_id={run_id}"
        ),
        observed_seconds=observed,
        threshold_seconds=threshold_seconds,
        method_result=method_result,
    )


def is_infra_stop_error(exc: BaseException) -> bool:
    text = str(exc)
    needles = [
        "Kube API",
        "Kubernetes API",
        "API unavailable",
        "KubeApiUnavailable",
        "NodeNotReady",
        "NodeStatusUnknown",
        "node_not_ready",
        "infra_failure",
        "another Stage-2 runner holds",
        ".stage2_baselines_x3.lock",
    ]
    return any(needle in text for needle in needles)


def base_runner_argv(
    batch: str,
    repetitions: int,
    *,
    pod_capacity_per_window: float = 12.5,
    proposed_queue_mode: str = "ready",
    controller_k_ci: float | None = None,
    controller_q_plan_mode: str = "confidence_blend",
    allow_mu_sensitivity_overrides: bool = False,
) -> list[str]:
    argv = [
        "--batch", batch,
        "--methods", "proposed",
        "--repetitions", str(repetitions),
        "--max-attempts-per-run", "3",
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
        "--pod-capacity-per-window", str(pod_capacity_per_window),
        "--proposed-queue-mode", proposed_queue_mode,
        "--allow-workload-slice-overrides",
    ]
    if allow_mu_sensitivity_overrides:
        argv.append("--allow-mu-sensitivity-overrides")
    if controller_k_ci is not None:
        argv.extend(["--controller-k-ci", str(controller_k_ci)])
    if controller_q_plan_mode != "confidence_blend":
        argv.extend(["--controller-q-plan-mode", controller_q_plan_mode])
    return argv


def ensure_batch_summary(
    runner: Any,
    summary_path: Path,
    args: argparse.Namespace,
    workload: Path,
) -> dict[str, Any]:
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
    return summary


def release_lock(runner: Any, lock_handle: Any) -> None:
    if lock_handle is None:
        return
    try:
        lock_handle.seek(0)
        lock_handle.truncate()
        lock_handle.flush()
        runner.fcntl.flock(lock_handle.fileno(), runner.fcntl.LOCK_UN)
    finally:
        lock_handle.close()


def run_no_preflight_batch(
    *,
    runner: Any,
    runner_argv: list[str],
    cold_start_threshold_seconds: float,
    summary_manifest_key: str | None = None,
    summary_manifest: dict[str, Any] | None = None,
    manifest_filename: str | None = None,
    manifest_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    args = parse_runner_args(runner, runner_argv)
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
    summary = ensure_batch_summary(runner, summary_path, args, workload)
    summary["no_preflight_gate_override"] = {
        "protocol_override": "skip preflight/gate preliminary tests for this run",
        "skipped_prelaunch_steps": skipped_prelaunch_steps(),
        "preserved_runner_steps": preserved_runner_steps(),
        "cold_start_restart_policy": {
            "metric": "cold_start_latency_max_seconds",
            "threshold_seconds": cold_start_threshold_seconds,
            "action": "mark attempt invalid and retry the same method",
            "max_attempts_per_run": args.max_attempts_per_run,
        },
    }
    if summary_manifest_key and summary_manifest is not None:
        summary[summary_manifest_key] = summary_manifest
    runner.write_json(summary_path, summary)

    if manifest_filename and manifest_payload is not None:
        runner.write_json(batch_dir / manifest_filename, manifest_payload)

    lock_handle = None
    try:
        lock_handle = runner.acquire_lock()
    except BaseException as exc:
        summary["passed"] = False
        summary["complete"] = False
        summary["failed_at_utc"] = runner.base.utc_now()
        summary["failure_type"] = "infra_failure"
        summary["error"] = str(exc)
        runner.write_json(summary_path, summary)
        raise

    try:
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
                    stop_attempt = int(args.max_attempts_per_run) + 1
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
                                "reason": "attempt_pre_health_check_disabled_by_protocol_override",
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
                                cold_start_threshold_seconds,
                            )
                            runner.write_json(valid_summary_path, method_result)
                            event["passed"] = True
                            event["run_id"] = method_result["run_id"]
                            event["cold_start_latency_max_seconds"] = cold_start_max
                            event["cold_start_threshold_seconds"] = cold_start_threshold_seconds
                            success = True
                        except ColdStartThresholdExceeded as exc:
                            last_error = str(exc)
                            event["passed"] = False
                            event["failure_type"] = "cold_start_threshold_violation"
                            event["error"] = last_error
                            event["cold_start_threshold_seconds"] = exc.threshold_seconds
                            event["observed_cold_start_latency_max_seconds"] = exc.observed_seconds
                            event["rerun_reason"] = (
                                "cold_start_latency_max_seconds exceeded threshold"
                            )
                            invalid_summary = dict(exc.method_result)
                            invalid_summary["passed"] = False
                            invalid_summary["failure_type"] = event["failure_type"]
                            invalid_summary["cold_start_threshold_seconds"] = exc.threshold_seconds
                            invalid_summary["observed_cold_start_latency_max_seconds"] = (
                                exc.observed_seconds
                            )
                            runner.write_json(attempt_dir / "invalid_run_summary.json", invalid_summary)
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
                            event["failure_type"] = (
                                "infra_failure" if is_infra_stop_error(exc) else "method_failure"
                            )
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
                                "reason": "post_attempt_health_check_disabled_by_protocol_override",
                            }
                            event["finished_at_utc"] = runner.base.utc_now()
                            summary.setdefault("events", []).append(event)
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
                            f"within attempts 1..{args.max_attempts_per_run}: {last_error}"
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
            return {
                "batch": args.batch,
                "batch_dir": str(batch_dir),
                "summary_path": str(summary_path),
                "aggregate": aggregate,
                "args": vars(args),
            }
        except BaseException as exc:
            summary["passed"] = False
            summary["complete"] = False
            summary["failed_at_utc"] = runner.base.utc_now()
            summary["failure_type"] = (
                "infra_failure" if is_infra_stop_error(exc) else summary.get("failure_type")
            )
            summary["error"] = str(exc)
            runner.write_json(summary_path, summary)
            print(f"ERROR: {exc}", flush=True)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise
    finally:
        release_lock(runner, lock_handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], preferred_fields: list[str] | None = None) -> None:
    fields: list[str] = []
    for field in preferred_fields or []:
        if field not in fields:
            fields.append(field)
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def stddev_or_zero(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    group_field: str,
    metric_names: list[str],
    metadata_fields: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = []
    for row in rows:
        value = row.get(group_field)
        if value not in groups:
            groups.append(value)
    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []
    for group in groups:
        group_rows = [row for row in rows if row.get(group_field) == group]
        meta = {}
        for field in metadata_fields or []:
            values = [row.get(field) for row in group_rows if row.get(field) not in (None, "")]
            meta[field] = values[0] if values else ""
        wide_row: dict[str, Any] = {
            group_field: group,
            "valid_runs": len(group_rows),
            **meta,
        }
        for metric in metric_names:
            values = [
                parsed
                for parsed in (parse_float(row.get(metric)) for row in group_rows)
                if parsed is not None
            ]
            average = statistics.fmean(values) if values else ""
            stddev = stddev_or_zero(values) if values else ""
            long_rows.append({
                group_field: group,
                **meta,
                "metric": metric,
                "valid_runs": len(values),
                "min": min(values) if values else "",
                "max": max(values) if values else "",
                "average": average,
                "stddev": stddev,
            })
            wide_row[f"{metric}_mean"] = average
            wide_row[f"{metric}_stddev"] = stddev
        wide_rows.append(wide_row)
    return long_rows, wide_rows


def combine_per_run_metrics(
    *,
    total_dir: Path,
    run_records: list[dict[str, Any]],
    output_filename: str,
    output_method_field: str,
    extra_fields: list[str],
) -> Path:
    rows: list[dict[str, Any]] = []
    for record in run_records:
        batch_dir = Path(record["batch_dir"])
        per_run_path = batch_dir / "per_run_metrics.csv"
        for row in read_csv(per_run_path):
            row["method"] = record[output_method_field]
            row["source_batch"] = record["batch"]
            row["source_batch_dir"] = str(batch_dir)
            for field in extra_fields:
                if field in record:
                    row[field] = record[field]
            rows.append(row)
    output_path = total_dir / output_filename
    write_csv(output_path, rows)
    return output_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_directory(total_dir: Path) -> dict[str, Any]:
    export_dir = total_dir.parent / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    archive_path = export_dir / f"{total_dir.name}_deref.tar.gz"
    with tarfile.open(archive_path, "w:gz", dereference=True) as tar:
        tar.add(total_dir, arcname=total_dir.name, recursive=True)
    sha = sha256_file(archive_path)
    sha_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    sha_path.write_text(f"{sha}  {archive_path.name}\n", encoding="utf-8")
    return {
        "archive": str(archive_path),
        "sha256": sha,
        "sha256_path": str(sha_path),
    }


def write_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
