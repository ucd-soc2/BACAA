from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from no_preflight_experiment_common import (
    aggregate_rows,
    base_runner_argv,
    combine_per_run_metrics,
    load_runner,
    read_csv,
    run_no_preflight_batch,
    utc_stamp,
    write_csv,
    write_json,
)


BASE_MU = 12.5
COLD_START_THRESHOLD_SECONDS = 55.0

MU_CASES = [
    {
        "sensitivity_level": "mu10p00_neg20",
        "batch_prefix": "mu_sensitivity_proposed_mu10p00_neg20_cold55_no_preflight_gate",
        "mu": 10.0,
        "tested_mu": 10.0,
        "relative_change_pct": -20.0,
        "sensitivity_parameter": "mu",
        "valid_under_cold_start_threshold": True,
        "reason": "robustness analysis only; not main protocol tuning; -20 pct from Stage-2 base_mu=12.5",
    },
    {
        "sensitivity_level": "mu15p00_pos20",
        "batch_prefix": "mu_sensitivity_proposed_mu15p00_pos20_cold55_no_preflight_gate",
        "mu": 15.0,
        "tested_mu": 15.0,
        "relative_change_pct": 20.0,
        "sensitivity_parameter": "mu",
        "valid_under_cold_start_threshold": True,
        "reason": "robustness analysis only; not main protocol tuning; +20 pct from Stage-2 base_mu=12.5",
    },
]

REPORT_METRICS = [
    "successful_requests",
    "failed_requests",
    "timeout_requests",
    "success_rate_pct",
    "failure_rate_pct",
    "timeout_rate_pct",
    "p50_latency_seconds",
    "p95_latency_seconds",
    "p99_latency_seconds",
    "estimated_total_energy_j",
    "estimated_total_carbon_gco2e",
    "energy_per_successful_request_j",
    "carbon_per_successful_request_gco2e",
    "mean_ready_pods_observed",
    "mean_running_pods_observed",
    "max_ready_pods_observed",
    "max_running_pods_observed",
    "peak_queue_depth",
    "mean_queue_depth",
    "peak_pending_requests",
    "mean_pending_requests",
    "scale_to_ready_latency_p95_seconds",
    "scale_to_ready_latency_max_seconds",
    "cold_start_latency_mean_seconds",
    "cold_start_latency_p95_seconds",
    "cold_start_latency_max_seconds",
    "timeout_window_count",
]


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean_value(rows: list[dict[str, Any]], field: str) -> float | str:
    values = [value for value in (parse_float(row.get(field)) for row in rows) if value is not None]
    return statistics.fmean(values) if values else ""


def stddev_value(rows: list[dict[str, Any]], field: str) -> float | str:
    values = [value for value in (parse_float(row.get(field)) for row in rows) if value is not None]
    return statistics.stdev(values) if len(values) > 1 else (0.0 if values else "")


def summary_row_from_rows(
    *,
    level: str,
    mu: float,
    relative_change_pct: float,
    source: str,
    source_batch: str,
    source_batch_dir: str,
    rows: list[dict[str, Any]],
    metric_names: list[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sensitivity_level": level,
        "mu": mu,
        "tested_mu": mu,
        "relative_change_pct": relative_change_pct,
        "sensitivity_parameter": "mu",
        "source": source,
        "source_batch": source_batch,
        "source_batch_dir": source_batch_dir,
        "valid_runs": len(rows),
    }
    for metric in metric_names:
        row[f"{metric}_mean"] = mean_value(rows, metric)
        row[f"{metric}_stddev"] = stddev_value(rows, metric)
    return row


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_complete_main_proposed_candidate(batch_dir: Path) -> bool:
    name = batch_dir.name.lower()
    excluded = [
        "mu_sensitivity",
        "decision_ablation",
        "oracle",
        "smoke",
        "live_preflight",
    ]
    if any(part in name for part in excluded):
        return False
    state_path = batch_dir / "stage2_state.json"
    args_path = batch_dir / "config_snapshot" / "effective_args.json"
    per_run_path = batch_dir / "per_run_metrics.csv"
    if not (state_path.exists() and args_path.exists() and per_run_path.exists()):
        return False
    try:
        state = load_json(state_path)
        effective = load_json(args_path)
    except Exception:
        return False
    if state.get("complete") is not True or state.get("passed") is not True:
        return False
    methods = effective.get("methods") or []
    if "proposed" not in methods:
        return False
    required = {
        "start_row": 11708,
        "duration_minutes": 180.0,
        "ci_window_seconds": 480.0,
        "ci_time_scale": 8.0,
        "pod_capacity_per_window": BASE_MU,
    }
    for key, expected in required.items():
        actual = parse_float(effective.get(key))
        if actual is None or abs(actual - float(expected)) > 1e-9:
            return False
    return True


def find_reusable_baseline_summary(
    runner: Any,
    metric_names: list[str],
) -> dict[str, Any] | None:
    candidates = sorted(
        (path for path in runner.base.RESULTS.glob("stage2_all_baselines_x3_s11708_180min_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for batch_dir in candidates:
        if not is_complete_main_proposed_candidate(batch_dir):
            continue
        rows = [
            row
            for row in read_csv(batch_dir / "per_run_metrics.csv")
            if row.get("method") == "proposed"
        ]
        valid_rows = [
            row
            for row in rows
            if parse_float(row.get("cold_start_latency_max_seconds")) is None
            or parse_float(row.get("cold_start_latency_max_seconds")) <= COLD_START_THRESHOLD_SECONDS
        ]
        if not valid_rows:
            continue
        return summary_row_from_rows(
            level="mu12p50_baseline",
            mu=BASE_MU,
            relative_change_pct=0.0,
            source="reused_main_proposed_or_control",
            source_batch=batch_dir.name.replace("stage2_all_baselines_x3_s11708_180min_", "", 1),
            source_batch_dir=str(batch_dir),
            rows=valid_rows,
            metric_names=metric_names,
        )
    return None


def validate_pm20_per_run(path: Path) -> None:
    rows = read_csv(path)
    if len(rows) != 2:
        raise RuntimeError(f"expected exactly 2 valid pm20 rows, got {len(rows)} at {path}")
    observed = sorted(round(float(row["mu"]), 8) for row in rows)
    if observed != [10.0, 15.0]:
        raise RuntimeError(f"expected mu values [10.0, 15.0], got {observed}")
    for row in rows:
        cold = parse_float(row.get("cold_start_latency_max_seconds"))
        if cold is not None and cold > COLD_START_THRESHOLD_SECONDS:
            raise RuntimeError(f"invalid cold_start_latency_max_seconds={cold} in {path}")
        capacity = parse_float(row.get("pod_capacity_per_window"))
        mu = parse_float(row.get("mu"))
        if capacity is None or mu is None or abs(capacity - mu) > 1e-9:
            raise RuntimeError(
                f"pod_capacity_per_window={capacity} does not match mu={mu} in {path}"
            )


def package_paths(
    runner: Any,
    *,
    total_dir: Path,
    run_records: list[dict[str, Any]],
) -> dict[str, Any]:
    export_dir = runner.base.RESULTS / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive = export_dir / f"{total_dir.name}_export_{stamp}_deref.tar.gz"
    sha_rollup = export_dir / f"{total_dir.name}_export.sha256"
    paths = [str(total_dir)]
    for record in run_records:
        paths.append(record["batch_dir"])
    if runner.base.LOGS.exists():
        for record in run_records:
            batch = record["batch"]
            paths.extend(str(path) for path in sorted(runner.base.LOGS.glob(f"*{batch}*")) if path.is_file())
    manifest: dict[str, Any] = {
        "total_dir": str(total_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(archive),
        "included_paths": paths,
    }
    try:
        subprocess.run(
            ["tar", "--dereference", "-czf", str(archive), *paths],
            cwd=str(runner.base.ROOT),
            check=True,
            timeout=1800,
        )
    except Exception as exc:  # noqa: BLE001
        manifest["package_error"] = str(exc)
        write_json(total_dir / "package_manifest.json", manifest)
        return manifest
    sha = sha256_file(archive)
    sha_path = archive.with_suffix(archive.suffix + ".sha256")
    sha_path.write_text(f"{sha}  {archive.name}\n", encoding="utf-8")
    sha_rollup.write_text(f"{sha}  {archive.name}\n", encoding="utf-8")
    manifest["sha256"] = sha
    manifest["sha256_path"] = str(sha_path)
    manifest["sha256_rollup_path"] = str(sha_rollup)
    write_json(total_dir / "package_manifest.json", manifest)
    return manifest


def report_value(row: dict[str, Any], metric: str) -> str:
    value = row.get(f"{metric}_mean", "")
    if value == "":
        return ""
    parsed = parse_float(value)
    if parsed is None:
        return str(value)
    return f"{parsed:.6g}"


def write_report(
    report_path: Path,
    *,
    summary_rows: list[dict[str, Any]],
    per_run_path: Path,
    summary_path: Path,
    result_dirs_path: Path,
    package: dict[str, Any],
) -> None:
    lines = [
        "# Proposed mu Sensitivity Supplement pm20",
        "",
        "- User hard override: run exactly two supplemental Proposed runs with mu=10.0 (-20%) and mu=15.0 (+20%).",
        f"- Generated: `{utc_stamp()}`",
        f"- Cold-start validity threshold: `{COLD_START_THRESHOLD_SECONDS}s`",
        "- Prelaunch gates, smoke tests, post-health checks, HPA diagnostics, and extra Prometheus monitoring were skipped by no-preflight override.",
        "- Runner internal placement recovery, targeted warmup, service warmup, drain, formal replay, cleanup, cooldown, and standard Prometheus accounting were preserved.",
        "",
        "## Summary",
        "",
        "| Level | mu | rel pct | source | valid | success pct | timeout pct | P95 s | P99 s | energy J | carbon gCO2e | J/success | gCO2e/success | cold max s | scale-ready P95 s | peak queue | timeout windows |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {level} | {mu} | {rel} | {source} | {valid} | {success} | {timeout} | {p95} | {p99} | {energy} | {carbon} | {jps} | {cps} | {cold} | {ready} | {queue} | {timeouts} |".format(
                level=row.get("sensitivity_level", ""),
                mu=row.get("mu", row.get("tested_mu", "")),
                rel=row.get("relative_change_pct", ""),
                source=row.get("source", ""),
                valid=row.get("valid_runs", ""),
                success=report_value(row, "success_rate_pct"),
                timeout=report_value(row, "timeout_rate_pct"),
                p95=report_value(row, "p95_latency_seconds"),
                p99=report_value(row, "p99_latency_seconds"),
                energy=report_value(row, "estimated_total_energy_j"),
                carbon=report_value(row, "estimated_total_carbon_gco2e"),
                jps=report_value(row, "energy_per_successful_request_j"),
                cps=report_value(row, "carbon_per_successful_request_gco2e"),
                cold=report_value(row, "cold_start_latency_max_seconds"),
                ready=report_value(row, "scale_to_ready_latency_p95_seconds"),
                queue=report_value(row, "peak_queue_depth"),
                timeouts=report_value(row, "timeout_window_count"),
            )
        )
    baseline = next((row for row in summary_rows if row.get("mu") == BASE_MU), None)
    lines.extend(["", "## Interpretation", ""])
    if baseline is None:
        lines.append(
            "- Baseline mu=12.5 was not found as a complete reusable Proposed/control result; compare the -20% and +20% rows directly."
        )
    else:
        base_p95 = parse_float(baseline.get("p95_latency_seconds_mean"))
        for row in summary_rows:
            mu = parse_float(row.get("mu"))
            if mu is None or mu == BASE_MU or base_p95 in (None, 0.0):
                continue
            p95 = parse_float(row.get("p95_latency_seconds_mean"))
            if p95 is None:
                continue
            delta = ((p95 - base_p95) / base_p95) * 100.0
            lines.append(f"- mu={mu:g} P95 latency delta vs reused mu=12.5 baseline: {delta:.2f}%.")
    lines.extend([
        "",
        "## Artifacts",
        "",
        f"- Per-run metrics: `{per_run_path}`",
        f"- Summary: `{summary_path}`",
        f"- Result dirs: `{result_dirs_path}`",
        f"- Archive: `{package.get('archive', '')}`",
        f"- Archive sha256: `{package.get('sha256', package.get('package_error', ''))}`",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp", default=utc_stamp())
    args = parser.parse_args(argv)

    runner = load_runner()
    total_dir = (
        runner.base.RESULTS
        / f"mu_sensitivity_proposed_pm20_cold55_no_preflight_gate_{args.stamp}"
    )
    total_dir.mkdir(parents=True, exist_ok=True)

    run_records: list[dict[str, Any]] = []
    result_dirs_path = total_dir / "mu_sensitivity_result_dirs.json"

    for case in MU_CASES:
        batch = f"{case['batch_prefix']}_{args.stamp}"
        manifest = {
            "sensitivity_parameter": "mu",
            "base_mu": BASE_MU,
            "tested_mu": case["tested_mu"],
            "relative_change_pct": case["relative_change_pct"],
            "reason": case["reason"],
            "cold_start_threshold_seconds": COLD_START_THRESHOLD_SECONDS,
            "capacity_sweep_configuration": "plus/minus 20 percent supplemental capacity sweep",
        }
        runner_argv = base_runner_argv(
            batch,
            1,
            pod_capacity_per_window=case["tested_mu"],
            proposed_queue_mode="ready",
            allow_mu_sensitivity_overrides=True,
        )
        result = run_no_preflight_batch(
            runner=runner,
            runner_argv=runner_argv,
            cold_start_threshold_seconds=COLD_START_THRESHOLD_SECONDS,
            summary_manifest_key="mu_sensitivity_manifest",
            summary_manifest=manifest,
            manifest_filename="mu_sensitivity_manifest.json",
            manifest_payload=manifest,
        )
        record = {**result, **case, "source": "current_pm20_supplement", "mu_sensitivity_manifest": manifest}
        run_records.append(record)
        write_json(result_dirs_path, {
            "cold_start_threshold_seconds": COLD_START_THRESHOLD_SECONDS,
            "total_dir": str(total_dir),
            "runs": run_records,
        })

    per_run_path = combine_per_run_metrics(
        total_dir=total_dir,
        run_records=run_records,
        output_filename="mu_sensitivity_per_run_metrics.csv",
        output_method_field="sensitivity_level",
        extra_fields=[
            "mu",
            "tested_mu",
            "relative_change_pct",
            "sensitivity_parameter",
            "valid_under_cold_start_threshold",
            "sensitivity_level",
            "source",
        ],
    )
    validate_pm20_per_run(per_run_path)

    per_run_rows = read_csv(per_run_path)
    _, summary_rows = aggregate_rows(
        per_run_rows,
        group_field="sensitivity_level",
        metric_names=list(runner.AGGREGATE_METRICS),
        metadata_fields=[
            "mu",
            "tested_mu",
            "relative_change_pct",
            "sensitivity_parameter",
            "source",
        ],
    )
    baseline = find_reusable_baseline_summary(runner, list(runner.AGGREGATE_METRICS))
    if baseline is not None:
        summary_rows.insert(1, baseline)
    summary_path = total_dir / "mu_sensitivity_summary.csv"
    write_csv(
        summary_path,
        summary_rows,
        [
            "sensitivity_level",
            "mu",
            "tested_mu",
            "relative_change_pct",
            "sensitivity_parameter",
            "source",
            "source_batch",
            "source_batch_dir",
            "valid_runs",
        ],
    )

    package = package_paths(runner, total_dir=total_dir, run_records=run_records)
    write_json(result_dirs_path, {
        "cold_start_threshold_seconds": COLD_START_THRESHOLD_SECONDS,
        "total_dir": str(total_dir),
        "runs": run_records,
        "baseline": baseline,
        "package": package,
    })
    report_path = total_dir / "MU_SENSITIVITY_REPORT.md"
    write_report(
        report_path,
        summary_rows=summary_rows,
        per_run_path=per_run_path,
        summary_path=summary_path,
        result_dirs_path=result_dirs_path,
        package=package,
    )
    print("mu_sensitivity_total_dir=" + str(total_dir), flush=True)
    print("mu_sensitivity_per_run_metrics=" + str(per_run_path), flush=True)
    print("mu_sensitivity_summary=" + str(summary_path), flush=True)
    print("mu_sensitivity_archive=" + str(package.get("archive", "")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
