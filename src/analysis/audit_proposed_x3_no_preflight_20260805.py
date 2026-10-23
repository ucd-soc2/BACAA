#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


REQUIRED_TOP_LEVEL = [
    "stage2_state.json",
    "per_run_metrics.csv",
    "method_aggregate_metrics.csv",
    "method_aggregate_metrics_wide.csv",
    "FINAL_REPORT.md",
    "TEACHER_REQUIREMENTS_SUMMARY.json",
    "TEACHER_REQUIREMENTS_SUMMARY.md",
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--results-dir", default="/home/artifact/online_experiment/results")
    parser.add_argument("--cold-start-threshold", type=float, default=55.0)
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir)
    results_dir = Path(args.results_dir)
    failures = []

    if not batch_dir.is_dir():
        failures.append(f"missing batch dir: {batch_dir}")

    required = {}
    for name in REQUIRED_TOP_LEVEL:
        path = batch_dir / name
        required[name] = {"exists": path.exists(), "size": path.stat().st_size if path.exists() else None}
        if not path.exists() or path.stat().st_size <= 0:
            failures.append(f"missing or empty required file: {name}")

    state = load_json(batch_dir / "stage2_state.json") if (batch_dir / "stage2_state.json").exists() else {}
    aggregate = state.get("aggregate") or {}
    if state.get("complete") is not True:
        failures.append("state.complete is not true")
    if state.get("passed") is not True:
        failures.append("state.passed is not true")
    if state.get("selected_methods") != ["proposed"]:
        failures.append(f"selected_methods mismatch: {state.get('selected_methods')}")
    if state.get("repetitions") != 3:
        failures.append(f"repetitions mismatch: {state.get('repetitions')}")
    if aggregate.get("valid_run_count") != 3:
        failures.append(f"valid_run_count mismatch: {aggregate.get('valid_run_count')}")
    if (aggregate.get("counts") or {}).get("proposed") != 3:
        failures.append(f"proposed count mismatch: {aggregate.get('counts')}")

    per_run_rows = read_csv(batch_dir / "per_run_metrics.csv") if (batch_dir / "per_run_metrics.csv").exists() else []
    proposed_rows = [row for row in per_run_rows if row.get("method") == "proposed"]
    if len(proposed_rows) != 3:
        failures.append(f"expected 3 proposed per-run rows, got {len(proposed_rows)}")

    per_run_summary = []
    for row in proposed_rows:
        rep = row.get("repetition")
        attempt = row.get("attempt")
        run_id = row.get("run_id")
        cold = float(row.get("cold_start_latency_max_seconds") or "nan")
        if not (cold <= args.cold_start_threshold):
            failures.append(f"cold-start threshold exceeded for {run_id}: {cold}")
        metadata_path = results_dir / f"{run_id}_replay_metadata.json"
        metadata = load_json(metadata_path) if metadata_path.exists() else {}
        if metadata.get("artifact_complete") is not True:
            failures.append(f"metadata incomplete for {run_id}")
        runtime = metadata.get("runtime_snapshot") or {}
        if runtime.get("completed_requests") != 4118:
            failures.append(f"completed_requests mismatch for {run_id}: {runtime.get('completed_requests')}")
        per_run_summary.append(
            {
                "repetition": int(float(rep)) if rep not in (None, "") else rep,
                "attempt": int(float(attempt)) if attempt not in (None, "") else attempt,
                "run_id": run_id,
                "cold_start_latency_max_seconds": cold,
                "successful_requests": row.get("successful_requests"),
                "failed_requests": row.get("failed_requests"),
                "timeout_requests": row.get("timeout_requests"),
                "p95_latency_seconds": row.get("p95_latency_seconds"),
                "p99_latency_seconds": row.get("p99_latency_seconds"),
                "energy_joules": row.get("energy_joules"),
                "carbon_gco2e": row.get("carbon_gco2e"),
                "metadata_status_counts": runtime.get("status_counts"),
            }
        )

    failed_attempt_files = sorted(str(path.relative_to(batch_dir)) for path in batch_dir.glob("rep*/proposed/attempt*/failed_attempt.json"))
    method_summary_files = sorted(str(path.relative_to(batch_dir)) for path in batch_dir.glob("rep*/proposed/attempt*/method_summary.json"))

    out = {
        "batch_dir": str(batch_dir),
        "passed": not failures,
        "failures": failures,
        "required_top_level": required,
        "state": {
            "complete": state.get("complete"),
            "passed": state.get("passed"),
            "selected_methods": state.get("selected_methods"),
            "repetitions": state.get("repetitions"),
            "event_count": state.get("event_count"),
            "last_event": state.get("last_event"),
            "aggregate": aggregate,
        },
        "per_run_summary": per_run_summary,
        "failed_attempt_files": failed_attempt_files,
        "method_summary_files": method_summary_files,
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
