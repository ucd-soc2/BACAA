from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from no_preflight_experiment_common import (
    base_runner_argv,
    load_runner,
    read_csv,
    run_no_preflight_batch,
    write_json,
)


COLD_START_THRESHOLD_SECONDS = 55.0
BATCH_PREFIX = "main_proposed_x3_cold55_no_preflight_gate"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")


def default_batch() -> str:
    return f"{BATCH_PREFIX}_{utc_stamp()}"


def build_runner_argv(batch: str) -> list[str]:
    return base_runner_argv(
        batch,
        3,
        pod_capacity_per_window=12.5,
        proposed_queue_mode="ready",
    )


def batch_dir_for(runner: Any, batch: str) -> Path:
    label = runner.duration_label(180.0)
    return runner.base.RESULTS / f"stage2_all_baselines_x3_s11708_{label}_{batch}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_batch(runner: Any, batch: str, batch_dir: Path) -> dict[str, Any]:
    export_dir = runner.base.RESULTS / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive = export_dir / f"{batch}_export_{stamp}_deref.tar.gz"
    sha_rollup = export_dir / f"{batch}_export.sha256"
    manifest = {
        "batch": batch,
        "batch_dir": str(batch_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(archive),
        "included_paths": [],
        "logs_included": "logs files whose basename contains the batch id",
        "cold_start_threshold_seconds": COLD_START_THRESHOLD_SECONDS,
    }
    if not batch_dir.exists():
        manifest["skipped"] = True
        manifest["reason"] = "batch_dir_missing"
        write_json(export_dir / f"{batch}_package_manifest.json", manifest)
        return manifest

    paths = [str(batch_dir)]
    logs_dir = runner.base.LOGS
    if logs_dir.exists():
        paths.extend(str(path) for path in sorted(logs_dir.glob(f"*{batch}*")) if path.is_file())
    manifest["included_paths"] = paths
    try:
        subprocess.run(
            ["tar", "--dereference", "-czf", str(archive), *paths],
            cwd=str(runner.base.ROOT),
            check=True,
            timeout=900,
        )
    except Exception as exc:  # noqa: BLE001
        manifest["package_error"] = str(exc)
        write_json(export_dir / f"{batch}_package_manifest.json", manifest)
        return manifest

    sha = sha256_file(archive)
    sha_path = archive.with_suffix(archive.suffix + ".sha256")
    sha_path.write_text(f"{sha}  {archive.name}\n", encoding="utf-8")
    sha_rollup.write_text(f"{sha}  {archive.name}\n", encoding="utf-8")
    manifest["sha256"] = sha
    manifest["sha256_path"] = str(sha_path)
    manifest["sha256_rollup_path"] = str(sha_rollup)
    write_json(export_dir / f"{batch}_package_manifest.json", manifest)
    print(f"proposed_archive={archive}", flush=True)
    print(f"proposed_archive_sha256={sha}", flush=True)
    return manifest


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def validate_proposed_x3(batch_dir: Path) -> None:
    required_paths = [
        batch_dir / "stage2_state.json",
        batch_dir / "config_snapshot" / "effective_args.json",
        batch_dir / "config_snapshot" / "frozen_inputs.json",
        batch_dir / "per_run_metrics.csv",
        batch_dir / "method_aggregate_metrics.csv",
        batch_dir / "method_aggregate_metrics_wide.csv",
        batch_dir / "FINAL_REPORT.md",
    ]
    for path in required_paths:
        if not path.exists():
            raise RuntimeError(f"missing required artifact: {path}")

    state = json.loads((batch_dir / "stage2_state.json").read_text(encoding="utf-8"))
    if state.get("complete") is not True or state.get("passed") is not True:
        raise RuntimeError(f"stage2_state is not complete/passed at {batch_dir}")
    if state.get("selected_methods") != ["proposed"]:
        raise RuntimeError(f"selected_methods mismatch: {state.get('selected_methods')}")
    aggregate = state.get("aggregate") or {}
    if int(aggregate.get("valid_run_count") or 0) != 3:
        raise RuntimeError(f"expected aggregate valid_run_count=3, got {aggregate}")

    override = state.get("no_preflight_gate_override") or {}
    if not override:
        raise RuntimeError("missing no_preflight_gate_override in stage2_state")
    if float((override.get("cold_start_restart_policy") or {}).get("threshold_seconds") or 0) != 55.0:
        raise RuntimeError("cold_start_restart_policy threshold is not 55")

    effective = json.loads(
        (batch_dir / "config_snapshot" / "effective_args.json").read_text(encoding="utf-8")
    )
    expected = {
        "methods": ["proposed"],
        "repetitions": 3,
        "max_attempts_per_run": 3,
        "start_row": 11708,
        "duration_minutes": 180.0,
        "ci_window_seconds": 480.0,
        "ci_time_scale": 8.0,
        "initial_pods": 2,
        "initial_placement_nodes": ["llm-node-1", "llm-node2"],
        "cooldown_seconds": 300.0,
        "api_outage_grace_seconds": 180.0,
        "api_health_checks": 1,
        "api_health_interval_seconds": 0.0,
        "api_health_timeout_seconds": 15.0,
        "pod_sample_seconds": 60.0,
        "legacy_metrics_poll_seconds": 60.0,
        "prometheus_step_seconds": 10.0,
        "pod_capacity_per_window": 12.5,
        "proposed_queue_mode": "ready",
    }
    for key, expected_value in expected.items():
        actual = effective.get(key)
        if isinstance(expected_value, float):
            actual_float = parse_float(actual)
            if actual_float is None or abs(actual_float - expected_value) > 1e-9:
                raise RuntimeError(f"effective arg {key} mismatch: {actual!r}")
        else:
            if actual != expected_value:
                raise RuntimeError(f"effective arg {key} mismatch: {actual!r}")
    for field in [
        "run_attempt_pre_health_checks",
        "run_post_health_checks",
        "collect_extra_prometheus_monitoring",
        "collect_hpa_diagnostics_on_success",
    ]:
        if effective.get(field) is not False:
            raise RuntimeError(f"{field} is not false")
    for forbidden_field in ["disable_queue", "decision_mode"]:
        actual = effective.get(forbidden_field)
        if forbidden_field == "disable_queue" and actual is True:
            raise RuntimeError("disable_queue is true")
        if forbidden_field == "decision_mode" and actual == "direct":
            raise RuntimeError("decision_mode is direct")

    rows = read_csv(batch_dir / "per_run_metrics.csv")
    if len(rows) != 3:
        raise RuntimeError(f"expected exactly 3 proposed valid rows, got {len(rows)}")
    seen_reps = sorted(int(row.get("repetition") or 0) for row in rows)
    if seen_reps != [1, 2, 3]:
        raise RuntimeError(f"expected repetitions [1, 2, 3], got {seen_reps}")
    for row in rows:
        if row.get("method") != "proposed":
            raise RuntimeError(f"unexpected method in per_run_metrics: {row.get('method')}")
        cold = parse_float(row.get("cold_start_latency_max_seconds"))
        if cold is not None and cold > COLD_START_THRESHOLD_SECONDS:
            raise RuntimeError(f"cold_start_latency_max_seconds={cold} exceeds threshold")
        queue_samples = parse_float(row.get("queue_sample_count"))
        if queue_samples is not None and queue_samples <= 0:
            raise RuntimeError(f"queue_sample_count is not positive for {row.get('run_id')}")

    wide_rows = read_csv(batch_dir / "method_aggregate_metrics_wide.csv")
    if len(wide_rows) != 1 or wide_rows[0].get("method") != "proposed":
        raise RuntimeError("method_aggregate_metrics_wide does not contain one proposed row")
    if parse_float(wide_rows[0].get("valid_runs")) != 3.0:
        raise RuntimeError("aggregate wide valid_runs is not 3")


def run(batch: str) -> int:
    runner = load_runner()
    runner_argv = build_runner_argv(batch)
    batch_dir = batch_dir_for(runner, batch)
    manifest = {
        "selected_methods": ["proposed"],
        "valid_runs_required": 3,
        "checkpoint": "checkpoints_transformer_h2",
        "yaml": "stage2/yamls/qwen-transformer.yaml",
        "decision_mode_expected": "proposed",
        "proposed_decision_layer_expected": True,
        "queue_mode_expected": "ready",
        "queue_aware_correction_expected": True,
        "no_transformer_direct": True,
        "no_smoke": True,
        "no_preflight_gate": True,
        "cold_start_threshold_seconds": COLD_START_THRESHOLD_SECONDS,
        "restart_supervisor": "outer monitor restarts this same failed repetition/batch under the recorded retry limits",
    }
    try:
        result = run_no_preflight_batch(
            runner=runner,
            runner_argv=runner_argv,
            cold_start_threshold_seconds=COLD_START_THRESHOLD_SECONDS,
            summary_manifest_key="proposed_no_preflight_audit",
            summary_manifest=manifest,
            manifest_filename="PROPOSED_NO_PREFLIGHT_MANIFEST.json",
            manifest_payload=manifest,
        )
        validate_proposed_x3(Path(result["batch_dir"]))
        print("proposed_batch_dir=" + result["batch_dir"], flush=True)
        return 0
    except BaseException as exc:
        print(f"ERROR: {exc}", flush=True)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return 1
    finally:
        package_batch(runner, batch, batch_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default=default_batch())
    args = parser.parse_args(argv)
    return run(args.batch)


if __name__ == "__main__":
    raise SystemExit(main())
