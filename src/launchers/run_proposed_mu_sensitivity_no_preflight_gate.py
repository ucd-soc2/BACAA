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
    utc_stamp,
    write_csv,
    write_json,
)


BASE_MU = 12.5
COLD_START_THRESHOLD_SECONDS = 55.0


MU_CASES = [
    {
        "sensitivity_level": "mu11p25_neg10",
        "batch_prefix": "mu_sensitivity_proposed_mu11p25_neg10_cold55_no_preflight_gate",
        "tested_mu": 11.25,
        "relative_change_pct": -10.0,
        "reason": "mu sensitivity -10 pct from Stage-2 base_mu=12.5",
    },
    {
        "sensitivity_level": "mu13p75_pos10",
        "batch_prefix": "mu_sensitivity_proposed_mu13p75_pos10_cold55_no_preflight_gate",
        "tested_mu": 13.75,
        "relative_change_pct": 10.0,
        "reason": "mu sensitivity +10 pct from Stage-2 base_mu=12.5",
    },
]


def write_report(
    report_path: Path,
    *,
    run_records: list[dict[str, Any]],
    per_run_path: Path,
    summary_path: Path,
    package: dict[str, Any],
) -> None:
    lines = [
        "# Proposed mu Sensitivity",
        "",
        f"- Generated: `{utc_stamp()}`",
        f"- Cold-start validity threshold: `{COLD_START_THRESHOLD_SECONDS}s`",
        "- Prelaunch gates, health probes, smoke tests, HPA diagnostics, and extra Prometheus monitoring were skipped by user override.",
        "- Runner internal placement recovery, targeted warmup, drain, formal replay, cleanup, cooldown, and standard Prometheus energy metrics were preserved.",
        "- Base mu=12.5 was not rerun by this launcher, under the recorded protocol.",
        "",
        "## Runs",
        "",
        "| Sensitivity | mu | Relative change pct | Valid runs | Batch dir |",
        "|---|---:|---:|---:|---|",
    ]
    for record in run_records:
        batch_dir = Path(record["batch_dir"])
        rows = read_csv(batch_dir / "per_run_metrics.csv")
        lines.append(
            f"| {record['sensitivity_level']} | {record['tested_mu']} | "
            f"{record['relative_change_pct']} | {len(rows)} | `{batch_dir}` |"
        )
    lines.extend([
        "",
        "## Artifacts",
        "",
        f"- Per-run metrics: `{per_run_path}`",
        f"- Summary: `{summary_path}`",
        f"- Archive: `{package['archive']}`",
        f"- Archive sha256: `{package['sha256']}`",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp", default=utc_stamp())
    args = parser.parse_args(argv)

    runner = load_runner()
    run_records: list[dict[str, Any]] = []
    total_dir = (
        runner.base.RESULTS
        / f"mu_sensitivity_proposed_pm10_cold55_no_preflight_gate_{args.stamp}"
    )
    total_dir.mkdir(parents=True, exist_ok=True)

    for case in MU_CASES:
        batch = f"{case['batch_prefix']}_{args.stamp}"
        manifest = {
            "sensitivity_parameter": "mu",
            "base_mu": BASE_MU,
            "tested_mu": case["tested_mu"],
            "relative_change_pct": case["relative_change_pct"],
            "reason": case["reason"],
            "cold_start_threshold_seconds": COLD_START_THRESHOLD_SECONDS,
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
        record = {
            **result,
            **case,
            "mu_sensitivity_manifest": manifest,
        }
        run_records.append(record)
        write_json(total_dir / "mu_sensitivity_result_dirs.json", {
            "cold_start_threshold_seconds": COLD_START_THRESHOLD_SECONDS,
            "total_dir": str(total_dir),
            "runs": run_records,
        })

    per_run_path = combine_per_run_metrics(
        total_dir=total_dir,
        run_records=run_records,
        output_filename="mu_sensitivity_per_run_metrics.csv",
        output_method_field="sensitivity_level",
        extra_fields=["tested_mu", "relative_change_pct", "sensitivity_level"],
    )
    per_run_rows = read_csv(per_run_path)
    _, summary_rows = aggregate_rows(
        per_run_rows,
        group_field="sensitivity_level",
        metric_names=list(runner.AGGREGATE_METRICS),
        metadata_fields=["tested_mu", "relative_change_pct"],
    )
    summary_path = total_dir / "mu_sensitivity_summary.csv"
    write_csv(summary_path, summary_rows, [
        "sensitivity_level",
        "tested_mu",
        "relative_change_pct",
        "valid_runs",
    ])
    package = package_directory(total_dir)
    write_json(total_dir / "mu_sensitivity_result_dirs.json", {
        "cold_start_threshold_seconds": COLD_START_THRESHOLD_SECONDS,
        "total_dir": str(total_dir),
        "runs": run_records,
        "package": package,
    })
    report_path = total_dir / "MU_SENSITIVITY_REPORT.md"
    write_report(
        report_path,
        run_records=run_records,
        per_run_path=per_run_path,
        summary_path=summary_path,
        package=package,
    )
    print("mu_sensitivity_total_dir=" + str(total_dir), flush=True)
    print("mu_sensitivity_archive=" + str(package["archive"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
