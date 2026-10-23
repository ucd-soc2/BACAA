from __future__ import annotations

import argparse
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from no_preflight_experiment_common import load_runner, run_no_preflight_batch, write_json


COLD_START_THRESHOLD_SECONDS = 55.0
BATCH_PREFIX = "main_transformer_direct_x3_cold55_no_preflight_gate"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")


def default_batch() -> str:
    return f"{BATCH_PREFIX}_{utc_stamp()}"


def build_runner_argv(batch: str) -> list[str]:
    return [
        "--batch", batch,
        "--methods", "transformer_direct",
        "--repetitions", "3",
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
        "--pod-capacity-per-window", "12.5",
        "--allow-workload-slice-overrides",
    ]


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
    cmd = ["tar", "--dereference", "-czf", str(archive), *paths]
    try:
        subprocess.run(cmd, cwd=str(runner.base.ROOT), check=True, timeout=900)
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
    print(f"transformer_direct_archive={archive}", flush=True)
    print(f"transformer_direct_archive_sha256={sha}", flush=True)
    return manifest


def run(batch: str) -> int:
    runner = load_runner()
    runner_argv = build_runner_argv(batch)
    batch_dir = batch_dir_for(runner, batch)
    manifest = {
        "selected_methods": ["transformer_direct"],
        "valid_runs_required": 3,
        "checkpoint": "checkpoints_transformer_h2",
        "yaml": "stage2/yamls/qwen-transformer.yaml",
        "decision_mode_expected": "direct",
        "queue_mode_expected": "none",
        "no_proposed": True,
        "cold_start_threshold_seconds": COLD_START_THRESHOLD_SECONDS,
        "user_restart_supervisor": "outer supervisor restarts this same batch if the launcher exits nonzero or required artifacts are incomplete",
    }
    rc = 1
    try:
        result = run_no_preflight_batch(
            runner=runner,
            runner_argv=runner_argv,
            cold_start_threshold_seconds=COLD_START_THRESHOLD_SECONDS,
            summary_manifest_key="transformer_direct_no_preflight_audit",
            summary_manifest=manifest,
            manifest_filename="TRANSFORMER_DIRECT_NO_PREFLIGHT_MANIFEST.json",
            manifest_payload=manifest,
        )
        print("transformer_direct_batch_dir=" + result["batch_dir"], flush=True)
        rc = 0
        return rc
    except BaseException as exc:
        print(f"ERROR: {exc}", flush=True)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        rc = 1
        return rc
    finally:
        package_batch(runner, batch, batch_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default=default_batch())
    args = parser.parse_args(argv)
    return run(args.batch)


if __name__ == "__main__":
    raise SystemExit(main())
