from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from no_preflight_experiment_common import (
    aggregate_rows,
    base_runner_argv,
    combine_per_run_metrics,
    load_runner,
    package_directory,
    read_csv,
    run_no_preflight_batch,
    sha256_file,
    skipped_prelaunch_steps,
    preserved_runner_steps,
    utc_stamp,
    write_csv,
    write_json,
)


COLD_START_THRESHOLD_SECONDS = 55.0
ABLATION_REPETITIONS = 2


ABLATION_CASES = [
    {
        "ablation_method": "proposed_wo_carbon_aware_fusion",
        "batch_prefix": "decision_ablation_wo_carbon_fusion_x2_cold55_no_preflight_gate",
        "description": "Disable carbon-aware fusion only by setting k_ci=0.",
        "controller_k_ci": 0.0,
        "proposed_queue_mode": "ready",
        "controller_q_plan_mode": "confidence_blend",
        "changed_params": {
            "controller_k_ci": 0.0,
        },
    },
    {
        "ablation_method": "proposed_wo_readiness_aware_pressure",
        "batch_prefix": "decision_ablation_wo_readiness_pressure_x2_cold55_no_preflight_gate",
        "description": "Use current queue backlog and remove Ready-latency lead pressure from planning.",
        "controller_k_ci": None,
        "proposed_queue_mode": "current",
        "controller_q_plan_mode": "confidence_blend",
        "changed_params": {
            "proposed_queue_mode": "current",
            "queue_lead_arrivals": "0 or not participating by queue_mode=current",
            "queue_lead_capacity": "0 or not participating by queue_mode=current",
        },
    },
    {
        "ablation_method": "proposed_wo_confidence_upper_quantile_planning",
        "batch_prefix": "decision_ablation_wo_confidence_quantile_x2_cold55_no_preflight_gate",
        "description": "Use q50-only Q_plan while keeping q90/confidence logged.",
        "controller_k_ci": None,
        "proposed_queue_mode": "ready",
        "controller_q_plan_mode": "q50_only",
        "changed_params": {
            "controller_q_plan_mode": "q50_only",
            "uncertainty_planning_enabled": False,
        },
    },
]


UNCHANGED_CONFIG = {
    "methods": ["proposed"],
    "repetitions": ABLATION_REPETITIONS,
    "max_attempts_per_run": 3,
    "start_row": 11708,
    "duration_minutes": 180,
    "ci_window_seconds": 480,
    "ci_time_scale": 8.0,
    "initial_pods": 2,
    "initial_placement_nodes": "llm-node-1,llm-node2",
    "initial_placement_recovery_attempts": 4,
    "initial_placement_recovery_delay_seconds": 10,
    "pre_warmup_settle_seconds": 20,
    "warmup_duration_seconds": 60,
    "warmup_targeted_requests_per_pod": 8,
    "warmup_targeted_concurrency": 2,
    "warmup_targeted_min_cpu_delta": 0.05,
    "drain_seconds": 30,
    "cooldown_seconds": 300,
    "formal_start_delay_seconds": 30,
    "api_outage_grace_seconds": 180,
    "ready_latency_p95_seconds": 40,
    "api_health_checks": 1,
    "api_health_interval_seconds": 0,
    "api_health_timeout_seconds": 15,
    "pod_sample_seconds": 60,
    "legacy_metrics_poll_seconds": 60,
    "prometheus_step_seconds": 10,
    "pod_capacity_per_window": 12.5,
    "proposed_queue_mode": "ready",
    "allow_workload_slice_overrides": True,
}


def code_hashes(runner: Any) -> dict[str, str]:
    project_root = Path(runner.__file__).resolve().parents[1]
    return {
        "stage2_runner": sha256_file(Path(runner.__file__).resolve()),
        "stage2_controller": sha256_file(project_root / "stage2/controller_stage2.py"),
        "common_launcher_module": sha256_file(Path(__file__).resolve().with_name("no_preflight_experiment_common.py")),
        "ablation_launcher_x2": sha256_file(Path(__file__).resolve()),
    }


def write_report(
    report_path: Path,
    *,
    run_records: list[dict[str, Any]],
    per_run_path: Path,
    aggregate_path: Path,
    aggregate_wide_path: Path,
    package: dict[str, Any],
) -> None:
    lines = [
        "# Proposed Decision Ablation X2",
        "",
        f"- Generated: `{utc_stamp()}`",
        f"- Repetitions per ablation: `{ABLATION_REPETITIONS}`",
        f"- Cold-start validity threshold: `{COLD_START_THRESHOLD_SECONDS}s`",
        "- User override for this run: each ablation is run two times in this supplemental batch.",
        "- Prelaunch gates, health probes, smoke tests, HPA diagnostics, and extra Prometheus monitoring were skipped by user override.",
        "- Runner internal placement recovery, targeted warmup, drain, formal replay, cleanup, cooldown, and standard Prometheus energy metrics were preserved.",
        "- proposed_control was not rerun by this launcher, under the recorded protocol.",
        "- proposed_wo_historical_smoothing was not run.",
        "",
        "## Runs",
        "",
        "| Ablation method | Valid runs | Batch dir |",
        "|---|---:|---|",
    ]
    for record in run_records:
        batch_dir = Path(record["batch_dir"])
        rows = read_csv(batch_dir / "per_run_metrics.csv")
        lines.append(
            f"| {record['ablation_method']} | {len(rows)} | `{batch_dir}` |"
        )
    lines.extend([
        "",
        "## Artifacts",
        "",
        f"- Per-run metrics: `{per_run_path}`",
        f"- Aggregate metrics: `{aggregate_path}`",
        f"- Aggregate wide metrics: `{aggregate_wide_path}`",
        f"- Archive: `{package['archive']}`",
        f"- Archive sha256: `{package['sha256']}`",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp", default=utc_stamp())
    args = parser.parse_args(argv)

    runner = load_runner()
    hashes = code_hashes(runner)
    run_records: list[dict[str, Any]] = []
    total_dir = (
        runner.base.RESULTS
        / f"decision_ablation_proposed_3items_x2_cold55_no_preflight_gate_{args.stamp}"
    )
    total_dir.mkdir(parents=True, exist_ok=True)

    for case in ABLATION_CASES:
        batch = f"{case['batch_prefix']}_{args.stamp}"
        runner_argv = base_runner_argv(
            batch,
            ABLATION_REPETITIONS,
            pod_capacity_per_window=12.5,
            proposed_queue_mode=case["proposed_queue_mode"],
            controller_k_ci=case["controller_k_ci"],
            controller_q_plan_mode=case["controller_q_plan_mode"],
            allow_mu_sensitivity_overrides=False,
        )
        manifest = {
            "ablation_method": case["ablation_method"],
            "ablation_description": case["description"],
            "changed_parameters_or_flags": case["changed_params"],
            "unchanged_base_config": UNCHANGED_CONFIG,
            "user_override": {
                "repetitions_per_ablation": ABLATION_REPETITIONS,
                "local_save_dir": "C:\\Users\\artifact\\Desktop\\\u6d88\u878d\u8865\u6d4b\u4e24\u6b21",
            },
            "code_sha256": hashes,
            "config_snapshot": {
                "runner_argv": runner_argv,
                "cold_start_threshold_seconds": COLD_START_THRESHOLD_SECONDS,
            },
            "skipped_prelaunch_steps": skipped_prelaunch_steps(),
            "preserved_runner_steps": preserved_runner_steps(),
        }
        result = run_no_preflight_batch(
            runner=runner,
            runner_argv=runner_argv,
            cold_start_threshold_seconds=COLD_START_THRESHOLD_SECONDS,
            summary_manifest_key="ablation_manifest",
            summary_manifest=manifest,
            manifest_filename="ablation_manifest.json",
            manifest_payload=manifest,
        )
        record = {
            **result,
            **case,
            "ablation_manifest": manifest,
        }
        run_records.append(record)
        write_json(total_dir / "ablation_result_dirs.json", {
            "cold_start_threshold_seconds": COLD_START_THRESHOLD_SECONDS,
            "total_dir": str(total_dir),
            "runs": run_records,
        })

    per_run_path = combine_per_run_metrics(
        total_dir=total_dir,
        run_records=run_records,
        output_filename="decision_ablation_per_run_metrics.csv",
        output_method_field="ablation_method",
        extra_fields=["ablation_method", "description"],
    )
    per_run_rows = read_csv(per_run_path)
    long_rows, wide_rows = aggregate_rows(
        per_run_rows,
        group_field="method",
        metric_names=list(runner.AGGREGATE_METRICS),
        metadata_fields=["ablation_method", "description"],
    )
    aggregate_path = total_dir / "decision_ablation_method_aggregate_metrics.csv"
    aggregate_wide_path = total_dir / "decision_ablation_method_aggregate_metrics_wide.csv"
    write_csv(aggregate_path, long_rows, [
        "method",
        "ablation_method",
        "description",
        "metric",
        "valid_runs",
        "min",
        "max",
        "average",
        "stddev",
    ])
    write_csv(aggregate_wide_path, wide_rows, [
        "method",
        "ablation_method",
        "description",
        "valid_runs",
    ])
    package = package_directory(total_dir)
    write_json(total_dir / "ablation_result_dirs.json", {
        "cold_start_threshold_seconds": COLD_START_THRESHOLD_SECONDS,
        "total_dir": str(total_dir),
        "runs": run_records,
        "package": package,
    })
    report_path = total_dir / "DECISION_ABLATION_REPORT.md"
    write_report(
        report_path,
        run_records=run_records,
        per_run_path=per_run_path,
        aggregate_path=aggregate_path,
        aggregate_wide_path=aggregate_wide_path,
        package=package,
    )
    print("decision_ablation_total_dir=" + str(total_dir), flush=True)
    print("decision_ablation_archive=" + str(package["archive"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
