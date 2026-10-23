#!/usr/bin/env python3
"""Run the frozen Stage-2 eight-method comparison with resumable repetitions."""

from __future__ import annotations

import argparse
import ast
import csv
import fcntl
import hashlib
import json
import os
import shutil
import statistics
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_e2e_sanity_check as base
import yaml


EXPECTED_QWEN_IMAGE = (
    "ko.local/qwen-local@sha256:"
    "b48add5d8f2db57bcb85e1c58992eb6519d5d17aed7143b67de108b88160d76d"
)
IMAGE_PRECHECK_NODES = ["llm-node-1", "llm-node2", "llm-node3"]
IMAGE_PRECHECK_LABEL = "qwen-image-precheck"


METHOD_ORDER = [
    "kpa", "hpa",
    "ets_direct", "bilstm_direct", "transformer_direct",
    "ets_kpa", "bilstm_kpa", "proposed",
]
QUEUE_ABLATION_METHODS = [
    "proposed_no_queue",
    "proposed_current_queue",
    "proposed_ready_queue",
]
ORACLE_METHODS = [
    "proposed_oracle",
]
METHOD_CHOICES = list(METHOD_ORDER) + QUEUE_ABLATION_METHODS + ORACLE_METHODS
METHOD_YAMLS = {
    "kpa": "stage2/yamls/qwen-kpa.yaml",
    "hpa": "stage2/yamls/qwen-hpa.yaml",
    "ets_direct": "stage2/yamls/qwen-ets.yaml",
    "bilstm_direct": "stage2/yamls/qwen-bilstm.yaml",
    "transformer_direct": "stage2/yamls/qwen-transformer.yaml",
    "ets_kpa": "stage2/yamls/qwen-ets-kpa.yaml",
    "bilstm_kpa": "stage2/yamls/qwen-bilstm-kpa.yaml",
    "proposed": "stage2/yamls/qwen-transformer.yaml",
    "proposed_no_queue": "stage2/yamls/qwen-transformer.yaml",
    "proposed_current_queue": "stage2/yamls/qwen-transformer.yaml",
    "proposed_ready_queue": "stage2/yamls/qwen-transformer.yaml",
    "proposed_oracle": "stage2/yamls/qwen-transformer.yaml",
}

REPETITION_SEEDS = {
    1: 42,
    2: 43,
    3: 44,
}
REPETITION_ORDER_INDEXES = {
    1: [0, 1, 2, 3, 4, 5, 6],  # A B C D E F G
    2: [3, 6, 0, 5, 1, 4, 2],  # D G A F B E C
    3: [5, 2, 4, 1, 6, 0, 3],  # F C E B G A D
}

FROZEN_DECISION_WINDOW_SECONDS = 60.0
FROZEN_POD_CAPACITY = 12.5
FROZEN_WORKLOAD = "data/test_workload_burst.csv"
# User-selected 3h test slice: window_id 87309..87488.
# Runtime --start-row is pandas iloc/0-based, so use 11708.
FROZEN_START_ROW = 11708
FROZEN_DURATION_MINUTES = 180.0
FROZEN_CHECKPOINT = "checkpoints_transformer_h2"
FROZEN_CARBON_CSV = "stage2/data/carbon_intensity_ie_2025-09-20_24h.csv"
FROZEN_CI_TRACE_START = "2025-09-20 00:00:00+00:00"
FROZEN_CI_WINDOW_SECONDS = 480.0
LEGACY_CI1440_WINDOW_SECONDS = 1440.0
LEGACY_CI1440_CARBON_LOOKUP_WINDOW_SECONDS = 20.0
FROZEN_FORMAL_START_DELAY_SECONDS = 30.0
LOCK_PATH = base.BASE / ".stage2_baselines_x3.lock"

base.METHOD_YAMLS.clear()
base.METHOD_YAMLS.update(METHOD_YAMLS)


def resolve_carbon_lookup_window_seconds(
    ci_window_seconds: float,
    explicit: float | None,
) -> float:
    if explicit is not None:
        return explicit
    if abs(ci_window_seconds - LEGACY_CI1440_WINDOW_SECONDS) <= 1e-9:
        return LEGACY_CI1440_CARBON_LOOKUP_WINDOW_SECONDS
    return FROZEN_DECISION_WINDOW_SECONDS


def resolve_ci_time_scale(
    ci_window_seconds: float,
    explicit: float | None,
) -> float:
    if explicit is not None:
        return explicit
    return ci_window_seconds / FROZEN_DECISION_WINDOW_SECONDS


def cleanup_image_precheck_pods() -> None:
    base.run_capture(
        [
            "kubectl", "delete", "pod", "-n", "default",
            "-l", f"experiment={IMAGE_PRECHECK_LABEL}",
            "--ignore-not-found=true", "--force", "--grace-period=0", "--wait=false",
        ],
        check=False,
    )


def validate_method_yaml_images(methods: list[str], expected_image: str) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    for method in dict.fromkeys(methods):
        yaml_path = base.BASE / METHOD_YAMLS[method]
        doc = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        containers = doc["spec"]["template"]["spec"]["containers"]
        user_container = next(
            (container for container in containers if container.get("name") == "user-container"),
            containers[0],
        )
        image = str(user_container.get("image") or "")
        pull_policy = str(user_container.get("imagePullPolicy") or "")
        check = {
            "method": method,
            "yaml": str(yaml_path),
            "image": image,
            "image_pull_policy": pull_policy,
        }
        checks.append(check)
        if image != expected_image:
            raise RuntimeError(
                f"{method} YAML image is not the expected immutable image: {image!r}"
            )
        if pull_policy != "Never":
            raise RuntimeError(
                f"{method} YAML imagePullPolicy must remain Never for the offline cluster"
            )
    return checks


def validate_frozen_yaml_contract(methods: list[str]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    required_args = {
        "--parallel": "2",
        "-t": "2",
        "--threads-batch": "2",
        "-c": "4096",
        "--cache-ram": "0",
    }
    required_flags = {"--no-cache-prompt", "--ignore-eos"}
    for method in dict.fromkeys(methods):
        yaml_path = base.BASE / METHOD_YAMLS[method]
        doc = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        template = doc["spec"]["template"]
        spec = template["spec"]
        annotations = template["metadata"]["annotations"]
        containers = spec["containers"]
        container = next(
            (item for item in containers if item.get("name") == "user-container"),
            containers[0],
        )
        args = [str(value) for value in container.get("args") or []]
        observed: dict[str, str] = {}
        for key, expected in required_args.items():
            if key not in args:
                raise RuntimeError(f"{method} YAML missing argument {key}")
            index = args.index(key)
            if index + 1 >= len(args):
                raise RuntimeError(f"{method} YAML argument {key} has no value")
            observed[key] = args[index + 1]
            if observed[key] != expected:
                raise RuntimeError(
                    f"{method} YAML {key}={observed[key]!r}, expected {expected!r}"
                )
        missing_flags = sorted(required_flags - set(args))
        if missing_flags:
            raise RuntimeError(f"{method} YAML missing flags {missing_flags}")
        if int(spec.get("containerConcurrency") or 0) != 2:
            raise RuntimeError(f"{method} YAML containerConcurrency must be 2")
        resources = container.get("resources") or {}
        requests = resources.get("requests") or {}
        limits = resources.get("limits") or {}
        if str(requests.get("cpu")) != "2" or str(limits.get("cpu")) != "2":
            raise RuntimeError(f"{method} YAML CPU request/limit must both be 2")
        if str(annotations.get("autoscaling.knative.dev/min-scale")) != "2":
            raise RuntimeError(f"{method} YAML min-scale must be 2")
        if str(annotations.get("autoscaling.knative.dev/max-scale")) != "4":
            raise RuntimeError(f"{method} YAML max-scale must be 4")
        checks.append({
            "method": method,
            "yaml": str(yaml_path),
            "container_concurrency": 2,
            "cpu_request": "2",
            "cpu_limit": "2",
            "args": observed,
            "flags": sorted(required_flags),
            "min_scale": 2,
            "max_scale": 4,
        })
    return checks


def collect_image_precheck_nodes(nodes: list[str]) -> list[dict[str, Any]]:
    doc = base.kubectl_json(["get", "nodes"])
    wanted = set(nodes)
    observations: list[dict[str, Any]] = []
    for item in doc.get("items", []):
        name = item.get("metadata", {}).get("name")
        if name not in wanted:
            continue
        conditions = {
            condition.get("type"): condition
            for condition in item.get("status", {}).get("conditions", [])
        }
        ready = conditions.get("Ready") or {}
        observations.append({
            "node": name,
            "ready": ready.get("status") == "True",
            "ready_status": ready.get("status"),
            "reason": ready.get("reason"),
            "message": ready.get("message"),
        })
    return sorted(observations, key=lambda row: str(row.get("node")))


def selected_worker_nodes_for_run(args: argparse.Namespace | None = None) -> list[str]:
    nodes = list(IMAGE_PRECHECK_NODES)
    if args is None or not getattr(args, "allow_smoke_overrides", False):
        return nodes
    observations = collect_image_precheck_nodes(nodes)
    ready_nodes = [str(row["node"]) for row in observations if row.get("ready")]
    if not ready_nodes:
        raise RuntimeError(f"smoke override found no Ready worker nodes: {observations}")
    skipped = sorted(set(nodes) - set(ready_nodes))
    if skipped:
        print(
            f"SMOKE OVERRIDE: skipping non-Ready precheck nodes {skipped}; "
            f"using {ready_nodes}",
            flush=True,
        )
    return ready_nodes


def parse_runtime_image_digest(image_id: object) -> str:
    text = str(image_id or "")
    marker = "sha256:"
    if marker not in text:
        return ""
    digest = text.rsplit(marker, 1)[1].split("@", 1)[-1].split(":", 1)[0]
    digest = digest.strip()
    if len(digest) == 64 and all(char in "0123456789abcdef" for char in digest):
        return digest
    return ""


def collect_image_precheck_pods() -> list[dict[str, Any]]:
    doc = base.kubectl_json([
        "get", "pods", "-n", "default",
        "-l", f"experiment={IMAGE_PRECHECK_LABEL}",
    ])
    observations: list[dict[str, Any]] = []
    for item in doc.get("items", []):
        statuses = item.get("status", {}).get("containerStatuses") or []
        status = statuses[0] if statuses else {}
        spec_containers = item.get("spec", {}).get("containers") or [{}]
        image_id = status.get("imageID")
        observations.append({
            "pod": item.get("metadata", {}).get("name"),
            "node": item.get("spec", {}).get("nodeName"),
            "phase": item.get("status", {}).get("phase"),
            "ready": bool(status.get("ready")),
            "spec_image": spec_containers[0].get("image"),
            "runtime_image": status.get("image"),
            "image_id": image_id,
            "runtime_digest": parse_runtime_image_digest(image_id),
            "state": status.get("state"),
        })
    return sorted(observations, key=lambda row: str(row.get("node")))


def run_image_precheck(batch_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    expected_image = args.expected_qwen_image
    nodes = selected_worker_nodes_for_run(args)
    report_path = batch_dir / "image_precheck.json"
    manifest_path = batch_dir / "image_precheck_pods.json"
    items = []
    for node in nodes:
        pod_name = "qwen-image-precheck-" + node.lower().replace("_", "-")
        items.append({
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": pod_name,
                "namespace": "default",
                "labels": {"experiment": IMAGE_PRECHECK_LABEL},
            },
            "spec": {
                "restartPolicy": "Never",
                "terminationGracePeriodSeconds": 0,
                "nodeName": node,
                "containers": [{
                    "name": "audit",
                    "image": expected_image,
                    "imagePullPolicy": "Never",
                    "command": ["/bin/sh", "-c", "sleep 300"],
                }],
            },
        })
    manifest_path.write_text(
        json.dumps({"apiVersion": "v1", "kind": "List", "items": items}, indent=2),
        encoding="utf-8",
    )
    report: dict[str, Any] = {
        "started_at_utc": base.utc_now(),
        "expected_image": expected_image,
        "expected_nodes": nodes,
        "image_pull_policy": "Never",
        "passed": False,
        "node_observations": [],
        "observations": [],
    }
    cleanup_image_precheck_pods()
    try:
        node_observations = collect_image_precheck_nodes(nodes)
        report["node_observations"] = node_observations
        observed_nodes = {str(row.get("node")) for row in node_observations}
        if observed_nodes != set(nodes):
            raise RuntimeError(
                f"image precheck nodes missing from cluster: expected={nodes}, observed={sorted(observed_nodes)}"
            )
        not_ready = [row for row in node_observations if not row.get("ready")]
        if not_ready:
            raise RuntimeError(f"image precheck nodes are not Ready: {not_ready}")
        base.run_capture(["kubectl", "apply", "--validate=false", "-f", str(manifest_path)])
        deadline = time.monotonic() + args.image_precheck_timeout_seconds
        observations: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            observations = collect_image_precheck_pods()
            if len(observations) == len(nodes) and all(row.get("ready") for row in observations):
                break
            fatal_waits = []
            for row in observations:
                waiting = (row.get("state") or {}).get("waiting") or {}
                if waiting.get("reason") in {"ErrImageNeverPull", "ImagePullBackOff", "InvalidImageName"}:
                    fatal_waits.append(row)
            if fatal_waits:
                raise RuntimeError(f"image precheck Pods cannot use the pinned image: {fatal_waits}")
            time.sleep(2.0)
        else:
            raise RuntimeError(f"image precheck Pods did not become Ready: {observations}")
        report["observations"] = observations
        observed_nodes = {str(row.get("node")) for row in observations}
        if observed_nodes != set(nodes):
            raise RuntimeError(
                f"image precheck node mismatch: expected={nodes}, observed={sorted(observed_nodes)}"
            )
        if not all(row.get("ready") for row in observations):
            raise RuntimeError("image precheck Pods did not all become Ready")
        spec_images = {str(row.get("spec_image")) for row in observations}
        if spec_images != {expected_image}:
            raise RuntimeError(
                f"image precheck Pod spec image mismatch: {sorted(spec_images)}"
            )
        expected_digest = expected_image.split("@", 1)[1]
        expected_digest_hex = expected_digest.removeprefix("sha256:")
        node_digests = {
            str(row.get("node")): str(row.get("runtime_digest") or "")
            for row in observations
        }
        missing_digest = [
            row for row in observations if not str(row.get("runtime_digest") or "")
        ]
        if missing_digest:
            raise RuntimeError(f"worker image digest missing: {missing_digest}")
        mismatched_digest = [
            row for row in observations
            if row.get("runtime_digest") != expected_digest_hex
        ]
        if mismatched_digest:
            raise RuntimeError(
                f"worker image digests do not match {expected_digest}: {mismatched_digest}"
            )
        report["node_image_digests"] = node_digests
        report["node_image_ids"] = {
            str(row.get("node")): str(row.get("image_id") or "")
            for row in observations
        }
        report["passed"] = True
        report["finished_at_utc"] = base.utc_now()
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return report
    except Exception as exc:
        report["error"] = str(exc)
        try:
            report["observations"] = collect_image_precheck_pods()
        except Exception as collect_exc:  # noqa: BLE001
            report["collection_error"] = str(collect_exc)
        report["failed_at_utc"] = base.utc_now()
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        raise RuntimeError(f"worker image precheck failed: {exc}") from exc
    finally:
        cleanup_image_precheck_pods()


def duration_label(minutes: float) -> str:
    value = float(minutes)
    if value.is_integer():
        return f"{int(value)}min"
    return (f"{value:g}min").replace(".", "p")


def legacy_metrics_args(revision: str, args: argparse.Namespace) -> list[object]:
    return base.legacy_metrics_args(revision, args)


def seed_for_repetition(repetition: int, args: argparse.Namespace) -> int:
    if repetition in REPETITION_SEEDS:
        return REPETITION_SEEDS[repetition]
    return int(args.seed) + repetition - 1


def proposed_queue_mode_for_method(method: str, args: argparse.Namespace) -> str:
    if method == "proposed_no_queue":
        return "none"
    if method == "proposed_current_queue":
        return "current"
    if method == "proposed_ready_queue":
        return "ready"
    return args.proposed_queue_mode


def method_command(
    method: str,
    run_id: str,
    revision: str,
    workload: Path,
    args: argparse.Namespace,
    repetition: int,
) -> list[object]:
    run_seed = seed_for_repetition(repetition, args)
    common = [
        "--start-row", str(args.start_row),
        "--duration-minutes", str(args.duration_minutes),
        "--window-size-seconds", str(FROZEN_DECISION_WINDOW_SECONDS),
        "--seed", str(run_seed),
        "--timeout-seconds", str(args.request_timeout_seconds),
        "--start-delay-seconds", str(args.formal_start_delay_seconds),
        "--run-id", run_id,
        "--output-dir", str(base.RESULTS),
        "--save-request-log",
    ]
    if method == "kpa":
        return [
            base.PYTHON, "run_kpa_baseline.py",
            "--csv", str(workload),
            "--skip-prepare-kpa-service",
            "--min-scale", str(args.initial_pods),
            "--max-scale", "4",
            *common,
            *legacy_metrics_args(revision, args),
        ]
    if method == "hpa":
        return [
            base.PYTHON, "run_hpa_baseline.py",
            "--csv", str(workload),
            "--skip-prepare-hpa-service",
            "--min-scale", str(args.initial_pods),
            "--max-scale", "4",
            *common,
            *legacy_metrics_args(revision, args),
        ]
    if method in {"ets_direct", "ets_kpa"}:
        script = (
            "code/run_ets_kpa_baseline_tplus2.py"
            if method == "ets_kpa"
            else "code/run_ets_baseline_tplus2.py"
        )
        ets_target = "1" if method == "ets_kpa" else "100"
        kpa_guard_args = []
        if method == "ets_kpa":
            kpa_guard_args = [
                "--scale-up-ready-latency-seconds",
                str(args.ready_latency_p95_seconds),
                "--scale-up-slack-seconds",
                str(args.scale_up_slack_seconds),
                "--scale-down-min-ready-pods",
                str(args.kpa_scale_down_min_ready_pods),
                "--scale-down-guard-seconds",
                str(args.kpa_scale_down_guard_seconds),
                "--scale-down-post-check-seconds",
                str(args.kpa_scale_down_post_check_seconds),
                "--scale-down-post-check-poll-seconds",
                str(args.kpa_scale_down_post_check_poll_seconds),
            ]
        return [
            base.PYTHON, script,
            "--train-csv", "data/train_workload_burst.csv",
            "--test-csv", str(workload),
            "--ets-config-json", "results/ets_best_config.json",
            "--pod-capacity-per-window", str(args.pod_capacity_per_window),
            "--min-pods", "1",
            "--max-pods", "4",
            "--autoscale-target", "revision",
            "--revision-name", revision,
            "--ets-target", ets_target,
            "--first-patch-lead-seconds", str(args.first_patch_lead_seconds),
            *kpa_guard_args,
            *common,
            *legacy_metrics_args(revision, args),
        ]
    if method in {"bilstm_direct", "bilstm_kpa"}:
        script = (
            "code/run_bilstm_kpa_baseline_tplus2.py"
            if method == "bilstm_kpa"
            else "code/run_bilstm_baseline_tplus2.py"
        )
        bilstm_target = "1" if method == "bilstm_kpa" else "100"
        kpa_guard_args = []
        if method == "bilstm_kpa":
            kpa_guard_args = [
                "--scale-up-ready-latency-seconds",
                str(args.ready_latency_p95_seconds),
                "--scale-up-slack-seconds",
                str(args.scale_up_slack_seconds),
                "--scale-down-min-ready-pods",
                str(args.kpa_scale_down_min_ready_pods),
                "--scale-down-guard-seconds",
                str(args.kpa_scale_down_guard_seconds),
                "--scale-down-post-check-seconds",
                str(args.kpa_scale_down_post_check_seconds),
                "--scale-down-post-check-poll-seconds",
                str(args.kpa_scale_down_post_check_poll_seconds),
                "--ready-guard-min-ready-pods",
                str(args.kpa_ready_guard_min_ready_pods),
                "--ready-guard-recovery-pods",
                str(args.kpa_ready_guard_recovery_pods),
                "--ready-guard-stable-seconds",
                str(args.kpa_ready_guard_stable_seconds),
                "--ready-guard-poll-seconds",
                str(args.kpa_ready_guard_poll_seconds),
            ]
            if not args.kpa_ready_guard_enabled:
                kpa_guard_args.append("--disable-ready-guard")
        return [
            base.PYTHON, script,
            "--model-path", "code/bilstm_autoscaler_model.pt",
            "--train-csv", "data/train_workload_burst.csv",
            "--test-csv", str(workload),
            "--pod-capacity-per-window", str(args.pod_capacity_per_window),
            "--min-pods", "1",
            "--max-pods", "4",
            "--autoscale-target", "revision",
            "--revision-name", revision,
            "--bilstm-target", bilstm_target,
            "--first-patch-lead-seconds", str(args.first_patch_lead_seconds),
            *kpa_guard_args,
            *common,
            *legacy_metrics_args(revision, args),
        ]
    if method in {"transformer_direct", "proposed", *QUEUE_ABLATION_METHODS, *ORACLE_METHODS}:
        decision_mode = "direct" if method == "transformer_direct" else "proposed"
        queue_mode = "none" if method == "transformer_direct" else proposed_queue_mode_for_method(method, args)
        oracle_enabled = (
            method in ORACLE_METHODS
            or (method == "proposed" and getattr(args, "proposed_oracle_predictor", False))
        )
        cmd = [
            base.PYTHON, "stage2/controller_stage2.py", FROZEN_CHECKPOINT,
            "--live",
            "--decision-mode", decision_mode,
            "--baseline-name", method,
            "--config-mu", str(args.pod_capacity_per_window),
            "--ci-csv", args.carbon_intensity_csv,
            "--trace", str(workload),
            "--start-row", str(args.start_row),
            "--duration-minutes", str(args.duration_minutes),
            "--seed", str(run_seed),
            "--start-delay", str(args.formal_start_delay_seconds),
            "--first-patch-lead", str(args.first_patch_lead_seconds),
            "--bootstrap-target-pods", str(args.initial_pods),
            "--run-id", run_id,
            "--output-dir", str(base.RESULTS),
            "--save-request-log",
            "--prometheus-url", args.prometheus_url,
            "--queue-mode", queue_mode,
            "--ready-latency-p95-seconds", str(args.ready_latency_p95_seconds),
            "--scale-up-slack-seconds", str(args.scale_up_slack_seconds),
            "--queue-sample-period-seconds", str(args.queue_sample_period_seconds),
            "--queue-max-sample-age-seconds", str(args.queue_max_sample_age_seconds),
            "--queue-query-timeout-seconds", str(args.queue_query_timeout_seconds),
            "--queue-samples-output", str(base.RESULTS / f"{run_id}_queue_samples.csv"),
            "--window-decisions-output", str(base.RESULTS / f"{run_id}_window_decisions.csv"),
            "--scale-events-output", str(base.RESULTS / f"{run_id}_scale_events.csv"),
        ]
        if oracle_enabled:
            cmd.extend([
                "--oracle-predictor",
                "--oracle-prediction-trace", str(workload),
                "--oracle-confidence", str(args.oracle_confidence),
                "--oracle-interval-pad", str(args.oracle_interval_pad),
            ])
        if args.ci_trace_start:
            cmd.extend([
                "--ci-trace-start", args.ci_trace_start,
                "--ci-window-seconds", str(args.ci_window_seconds),
                "--ci-time-scale", str(args.ci_time_scale),
            ])
        if args.controller_k_ci is not None:
            cmd.extend(["--config-k-ci", str(args.controller_k_ci)])
        if args.controller_k_g is not None:
            cmd.extend(["--config-k-g", str(args.controller_k_g)])
        if args.controller_alpha_base is not None:
            cmd.extend(["--config-alpha-base", str(args.controller_alpha_base)])
        if args.controller_alpha_min is not None:
            cmd.extend(["--config-alpha-min", str(args.controller_alpha_min)])
        if args.controller_alpha_max is not None:
            cmd.extend(["--config-alpha-max", str(args.controller_alpha_max)])
        if decision_mode == "proposed":
            if getattr(args, "controller_scale_down_confirm_windows", None) is not None:
                cmd.extend(["--scale-down-confirm-windows",
                            str(args.controller_scale_down_confirm_windows)])
            if getattr(args, "controller_scale_down_cooldown_seconds", 0.0):
                cmd.extend(["--scale-down-cooldown-seconds",
                            str(args.controller_scale_down_cooldown_seconds)])
            if getattr(args, "controller_disable_downtrend", False):
                cmd.append("--disable-downtrend")
        if queue_mode == "none":
            cmd.append("--disable-queue")
        if decision_mode == "proposed" and getattr(args, "controller_allow_missing_queue", False):
            cmd.append("--allow-missing-queue")
        return cmd
    raise ValueError(method)


base.method_command = method_command


def validate_log(method: str, log_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    capacity_for_validation = (
        float(args.pod_capacity_per_window)
        if getattr(args, "allow_mu_sensitivity_overrides", False)
        else FROZEN_POD_CAPACITY
    )
    expected_capacity = f"{capacity_for_validation:g}"
    expected_window = f"{FROZEN_DECISION_WINDOW_SECONDS:g}"
    expected_window_alt = f"{FROZEN_DECISION_WINDOW_SECONDS:.1f}"
    expected_windows = int(
        (float(args.duration_minutes) * 60.0) // FROZEN_DECISION_WINDOW_SECONDS
    )
    if (
        f"window_size_seconds={expected_window}" not in text
        and f"window_size_seconds={expected_window_alt}" not in text
    ):
        raise RuntimeError(
            f"{method} log did not confirm window_size_seconds={expected_window}"
        )
    if f"windows={expected_windows}" not in text:
        raise RuntimeError(f"{method} log did not confirm windows={expected_windows}")
    if method == "proposed" or method in QUEUE_ABLATION_METHODS or method in ORACLE_METHODS:
        needle = "decision_mode=proposed"
        ok = needle in text and (
            f"mu={expected_capacity}" in text
            or f'"mu": {expected_capacity}' in text
        )
        oracle_expected = (
            method in ORACLE_METHODS
            or (method == "proposed" and getattr(args, "proposed_oracle_predictor", False))
        )
        if ok and oracle_expected and "prediction_source=oracle" not in text:
            raise RuntimeError(f"{method} log did not confirm prediction_source=oracle")
    elif method == "transformer_direct":
        needle = "decision_mode=direct"
        ok = needle in text and (
            f"mu={expected_capacity}" in text
            or f'"mu": {expected_capacity}' in text
        )
    elif method in {"ets_direct", "ets_kpa", "bilstm_direct", "bilstm_kpa"}:
        needle = f"pod_capacity_per_window={expected_capacity}"
        ok = needle in text or (
            "pod_capacity_per_window" in text and expected_capacity in text
        )
    else:
        needle = "internal Knative autoscaler"
        ok = True
    if not ok:
        raise RuntimeError(f"{method} log did not contain {needle}")
    return {
        "needle": needle,
        "mu_or_capacity_logged": ok,
        "window_size_seconds": FROZEN_DECISION_WINDOW_SECONDS,
        "windows": expected_windows,
    }


base.validate_log = validate_log


def run_prometheus_metrics(
    method: str,
    run_id: str,
    start_epoch: float,
    end_epoch: float,
    args: argparse.Namespace,
    log_path: Path,
) -> dict[str, Any]:
    if args.prometheus_query_delay_seconds > 0:
        time.sleep(args.prometheus_query_delay_seconds)
    cmd = [
        base.PYTHON, "prometheus_energy_metrics.py",
        "--method", method,
        "--run-id", run_id,
        "--output-dir", str(base.RESULTS),
        "--prometheus-url", args.prometheus_url,
        "--start-epoch", f"{start_epoch:.3f}",
        "--end-epoch", f"{end_epoch:.3f}",
        "--step-seconds", str(args.prometheus_step_seconds),
        "--cpu-rate-window-seconds", str(args.prometheus_cpu_rate_window_seconds),
        "--window-size-seconds", str(FROZEN_DECISION_WINDOW_SECONDS),
        "--carbon-lookup-window-seconds", str(args.carbon_lookup_window_seconds),
        "--pod-capacity-per-window", str(args.pod_capacity_per_window),
        "--carbon-intensity-csv", args.carbon_intensity_csv,
        "--carbon-start-index", str(args.start_row),
        "--cpu-joules-per-core-second", str(args.cpu_joules_per_core_second),
        "--memory-joules-per-gb-second", str(args.memory_joules_per_gb_second),
        "--cold-start-model-joules", str(args.cold_start_model_joules),
        "--cold-start-window-seconds", str(args.cold_start_window_seconds),
    ]
    if args.ci_trace_start:
        cmd.extend([
            "--ci-trace-start", args.ci_trace_start,
            "--ci-window-seconds", str(args.ci_window_seconds),
            "--ci-time-scale", str(args.ci_time_scale),
        ])
    base.run_logged(cmd, log_path)
    summary_path = base.RESULTS / f"{run_id}_prometheus_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("resource_sample_rows") or 0) <= 0:
        raise RuntimeError(f"Prometheus resource samples missing for {run_id}")
    if float(summary.get("memory_gb_seconds") or 0.0) <= 0:
        raise RuntimeError(f"Prometheus memory samples missing for {run_id}")
    return summary


base.run_prometheus_metrics = run_prometheus_metrics


def prometheus_query_range(
    prometheus_url: str,
    query: str,
    start_epoch: float,
    end_epoch: float,
    step_seconds: float,
) -> dict[str, Any]:
    endpoint = prometheus_url.rstrip("/") + "/api/v1/query_range"
    params = urllib.parse.urlencode(
        {
            "query": query,
            "start": f"{start_epoch:.3f}",
            "end": f"{end_epoch:.3f}",
            "step": f"{step_seconds:g}",
        }
    )
    req = urllib.request.Request(
        endpoint + "?" + params,
        headers={"Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("status") != "success":
        raise RuntimeError(f"Prometheus query_range failed: {payload!r}")
    return payload


def run_extra_prometheus_monitoring(
    method: str,
    run_id: str,
    start_epoch: float,
    end_epoch: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    window = f"{int(args.prometheus_cpu_rate_window_seconds)}s"
    queries = {
        "kourier_cpu_cores": (
            'sum(rate(container_cpu_usage_seconds_total{namespace="kourier-system"}['
            f"{window}"
            "]))"
        ),
        "kourier_memory_working_set_bytes": (
            'sum(container_memory_working_set_bytes{namespace="kourier-system"})'
        ),
        "kourier_request_rate": (
            'sum(rate({__name__=~"kourier.*request.*|envoy_http_downstream_rq_total",'
            'namespace=~"kourier-system|knative-serving"}['
            f"{window}"
            "]))"
        ),
        "kourier_p95_latency_ms": (
            'histogram_quantile(0.95, sum(rate({__name__=~"kourier.*duration.*_bucket|'
            'envoy_http_downstream_rq_time_bucket",'
            'namespace=~"kourier-system|knative-serving"}['
            f"{window}"
            "])) by (le))"
        ),
        "queue_proxy_request_rate_by_pod": (
            'sum by (pod) (rate({__name__=~".*request.*",'
            'namespace="default",container="queue-proxy",pod=~"qwen.*"}['
            f"{window}"
            "]))"
        ),
    }
    payload: dict[str, Any] = {
        "method": method,
        "run_id": run_id,
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "step_seconds": args.prometheus_step_seconds,
        "queries": {},
    }
    for name, query in queries.items():
        try:
            result = prometheus_query_range(
                args.prometheus_url,
                query,
                start_epoch,
                end_epoch,
                args.prometheus_step_seconds,
            )
            series = (result.get("data") or {}).get("result") or []
            payload["queries"][name] = {
                "status": "ok",
                "query": query,
                "series_count": len(series),
                "result": series,
            }
        except Exception as exc:  # noqa: BLE001
            payload["queries"][name] = {
                "status": "failed",
                "query": query,
                "error": str(exc),
                "series_count": 0,
                "result": [],
            }
    output_path = base.RESULTS / f"{run_id}_extra_prometheus_monitoring.json"
    write_json(output_path, payload)
    return {
        "path": str(output_path),
        "queries": {
            name: {
                "status": data.get("status"),
                "series_count": data.get("series_count"),
            }
            for name, data in payload["queries"].items()
        },
    }


def run_controller_mu_dry_run(workload: Path, batch_dir: Path, args: argparse.Namespace) -> None:
    log_path = batch_dir / "controller_mu_dry_run.log"
    cmd = [
        base.PYTHON, "stage2/controller_stage2.py", FROZEN_CHECKPOINT,
        "--dry-run",
        "--config-mu", str(args.pod_capacity_per_window),
        "--ci-csv", args.carbon_intensity_csv,
        "--trace", str(workload),
        "--duration-minutes", "0",
        "--start-row", str(args.start_row),
    ]
    base.run_logged(cmd, log_path)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    expected_capacity = f"{args.pod_capacity_per_window:g}"
    if (
        f"mu={expected_capacity}" not in text
        and f'"mu": {expected_capacity}' not in text
    ):
        raise RuntimeError(
            f"controller dry-run did not print mu={expected_capacity}"
        )


def parse_int(value: object) -> int:
    try:
        return int(float(str(value or 0)))
    except (TypeError, ValueError):
        return 0


def parse_float(value: object) -> float:
    try:
        return float(str(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def percentile(values: list[float], q: float) -> float:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * (float(q) / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(clean) - 1)
    frac = pos - lo
    return clean[lo] * (1.0 - frac) + clean[hi] * frac


def mean_or_zero(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def stddev_or_zero(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def json_compact(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def group_timeout_periods(timeout_by_window: dict[int, int]) -> list[dict[str, int]]:
    periods: list[dict[str, int]] = []
    current: dict[str, int] | None = None
    previous_window: int | None = None
    for window_index in sorted(timeout_by_window):
        timeout_requests = int(timeout_by_window[window_index])
        if current is None or previous_window is None or window_index != previous_window + 1:
            if current is not None:
                periods.append(current)
            current = {
                "start_window_index": int(window_index),
                "end_window_index": int(window_index),
                "timeout_requests": timeout_requests,
            }
        else:
            current["end_window_index"] = int(window_index)
            current["timeout_requests"] += timeout_requests
        previous_window = window_index
    if current is not None:
        periods.append(current)
    return periods


def derive_required_request_metrics(run_id: str) -> dict[str, Any]:
    request_path = base.RESULTS / f"{run_id}_replay_request_log.csv"
    if not request_path.exists():
        raise FileNotFoundError(f"request log missing: {request_path}")
    with request_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    successful_latencies: list[float] = []
    timeout_count = 0
    status_counts: dict[str, int] = {}
    timeout_by_window: dict[int, int] = {}
    node_latency_rows: dict[str, list[float]] = {}
    for row in rows:
        status = parse_int(row.get("status"))
        status_key = str(status)
        status_counts[status_key] = status_counts.get(status_key, 0) + 1
        error = str(row.get("error") or "").lower()
        elapsed = parse_float(row.get("elapsed"))
        if status == 200:
            successful_latencies.append(elapsed)
            node_key = (
                row.get("node_name")
                or row.get("node")
                or row.get("pod_node")
                or row.get("pod_name")
                or row.get("pod")
            )
            if node_key:
                node_latency_rows.setdefault(str(node_key), []).append(elapsed)
        if status == 0 or "timeout" in error or "timed out" in error:
            timeout_count += 1
            window_index = parse_int(row.get("window_index"))
            timeout_by_window[window_index] = timeout_by_window.get(window_index, 0) + 1
    total = len(rows)
    successful = len(successful_latencies)
    failed = total - successful
    over_5 = sum(value > 5.0 for value in successful_latencies)
    over_10 = sum(value > 10.0 for value in successful_latencies)
    timeout_periods = group_timeout_periods(timeout_by_window)
    node_latency_summary = {
        node: {
            "successful_requests": len(values),
            "p50_latency_seconds": percentile(values, 50),
            "p95_latency_seconds": percentile(values, 95),
            "p99_latency_seconds": percentile(values, 99),
            "mean_latency_seconds": mean_or_zero(values),
        }
        for node, values in sorted(node_latency_rows.items())
    }
    return {
        "request_log": str(request_path),
        "total_requests": total,
        "successful_requests": successful,
        "failed_requests": failed,
        "timeout_requests": timeout_count,
        "success_rate_pct": successful / max(total, 1) * 100.0,
        "failure_rate_pct": failed / max(total, 1) * 100.0,
        "timeout_rate_pct": timeout_count / max(total, 1) * 100.0,
        "status_counts": status_counts,
        "timeout_window_count": len(timeout_by_window),
        "timeout_windows_json": json_compact(
            [
                {"window_index": window, "timeout_requests": count}
                for window, count in sorted(timeout_by_window.items())
            ]
        ),
        "timeout_periods_json": json_compact(timeout_periods),
        "node_latency_available": bool(node_latency_summary),
        "node_latency_summary_json": json_compact(node_latency_summary),
        "node_latency_note": (
            "computed_from_request_log_node_or_pod_column"
            if node_latency_summary
            else "request_log_has_no_node_or_pod_column"
        ),
        "over_5s_success_count": over_5,
        "over_5s_success_rate_pct": over_5 / max(successful, 1) * 100.0,
        "over_10s_success_count": over_10,
        "over_10s_success_rate_pct": over_10 / max(successful, 1) * 100.0,
        "over_5s_with_failures_count": over_5 + failed,
        "over_5s_with_failures_rate_pct": (over_5 + failed) / max(total, 1) * 100.0,
        "over_10s_with_failures_count": over_10 + failed,
        "over_10s_with_failures_rate_pct": (over_10 + failed) / max(total, 1) * 100.0,
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def derive_controller_diagnostics(run_id: str) -> dict[str, Any]:
    window_rows = read_csv_rows(base.RESULTS / f"{run_id}_window_decisions.csv")
    queue_rows = read_csv_rows(base.RESULTS / f"{run_id}_queue_samples.csv")
    scale_rows = read_csv_rows(base.RESULTS / f"{run_id}_scale_events.csv")

    ready_values = [
        parse_float(row.get("ready_pods_before"))
        for row in window_rows
        if row.get("ready_pods_before") not in (None, "")
    ]
    target_values = [
        parse_float(row.get("target_pods_after"))
        for row in window_rows
        if row.get("target_pods_after") not in (None, "")
    ]
    queue_source_rows = queue_rows or window_rows
    queue_values = [
        parse_float(row.get("queue_depth"))
        for row in queue_source_rows
        if row.get("queue_depth") not in (None, "")
    ]
    pending_values = [
        parse_float(row.get("pending_requests"))
        for row in queue_source_rows
        if row.get("pending_requests") not in (None, "")
    ]

    scale_up_events = sum(1 for row in scale_rows if str(row.get("direction")) == "up")
    scale_down_events = sum(1 for row in scale_rows if str(row.get("direction")) == "down")
    ready_latencies: list[float] = []
    sorted_windows = sorted(
        window_rows,
        key=lambda row: parse_float(row.get("logged_at_epoch")),
    )
    for event in scale_rows:
        if str(event.get("direction")) != "up":
            continue
        event_time = parse_float(event.get("event_time_epoch"))
        target = parse_int(event.get("to_target_pods"))
        for window in sorted_windows:
            logged_at = parse_float(window.get("logged_at_epoch"))
            ready = parse_int(window.get("ready_pods_before"))
            if logged_at >= event_time and ready >= target:
                ready_latencies.append(max(0.0, logged_at - event_time))
                break

    return {
        "controller_window_count": len(window_rows),
        "queue_sample_count": len(queue_rows),
        "scale_events_captured": len(scale_rows),
        "max_ready_pods_observed": max(ready_values) if ready_values else 0.0,
        "mean_ready_pods_observed": mean_or_zero(ready_values),
        "max_target_pods_observed": max(target_values) if target_values else 0.0,
        "mean_target_pods_observed": mean_or_zero(target_values),
        "peak_queue_depth": max(queue_values) if queue_values else 0.0,
        "mean_queue_depth": mean_or_zero(queue_values),
        "peak_pending_requests": max(pending_values) if pending_values else 0.0,
        "mean_pending_requests": mean_or_zero(pending_values),
        "scale_up_events": scale_up_events,
        "scale_down_events": scale_down_events,
        "scale_to_ready_observed_count": len(ready_latencies),
        "scale_to_ready_latency_mean_seconds": mean_or_zero(ready_latencies),
        "scale_to_ready_latency_p50_seconds": percentile(ready_latencies, 50),
        "scale_to_ready_latency_p95_seconds": percentile(ready_latencies, 95),
        "scale_to_ready_latency_max_seconds": max(ready_latencies) if ready_latencies else 0.0,
    }


def derive_node_energy_efficiency(run_id: str, successful_requests: int) -> dict[str, Any]:
    rows = read_csv_rows(base.RESULTS / f"{run_id}_prometheus_resource_samples.csv")
    by_node: dict[str, dict[str, float]] = {}
    for row in rows:
        node = str(row.get("node_name") or row.get("node") or "")
        if not node:
            continue
        entry = by_node.setdefault(
            node,
            {
                "estimated_energy_j": 0.0,
                "estimated_carbon_gco2e": 0.0,
                "cpu_core_seconds": 0.0,
                "memory_gb_seconds": 0.0,
                "sample_rows": 0.0,
            },
        )
        entry["estimated_energy_j"] += parse_float(row.get("estimated_energy_j"))
        entry["estimated_carbon_gco2e"] += parse_float(row.get("estimated_carbon_gco2e"))
        entry["cpu_core_seconds"] += parse_float(row.get("cpu_core_seconds"))
        entry["memory_gb_seconds"] += parse_float(row.get("memory_gb_seconds"))
        entry["sample_rows"] += 1.0

    summary = {}
    for node, values in sorted(by_node.items()):
        energy = values["estimated_energy_j"]
        carbon = values["estimated_carbon_gco2e"]
        summary[node] = {
            **values,
            "energy_per_global_success_j": energy / max(successful_requests, 1),
            "carbon_per_global_success_gco2e": carbon / max(successful_requests, 1),
        }
    return {
        "node_energy_efficiency_available": bool(summary),
        "node_energy_efficiency_json": json_compact(summary),
        "node_energy_efficiency_note": (
            "estimated_from_prometheus_resource_samples_by_node"
            if summary
            else "prometheus_resource_samples_missing_or_node_name_empty"
        ),
    }


def summarize_formal_pod_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    running = [parse_float(sample.get("running")) for sample in samples]
    ready = [parse_float(sample.get("ready")) for sample in samples]
    total = [parse_float(sample.get("total")) for sample in samples]
    return {
        "formal_pod_sample_count": len(samples),
        "max_running_pods_observed": max(running) if running else 0.0,
        "mean_running_pods_observed": mean_or_zero(running),
        "max_ready_pods_sampled": max(ready) if ready else 0.0,
        "mean_ready_pods_sampled": mean_or_zero(ready),
        "max_total_pods_sampled": max(total) if total else 0.0,
        "mean_total_pods_sampled": mean_or_zero(total),
    }


def build_run_metrics(
    method: str,
    repetition: int,
    attempt: int,
    run_id: str,
    prom: dict[str, Any],
    required: dict[str, Any],
) -> dict[str, Any]:
    return {
        "method": method,
        "repetition": repetition,
        "attempt": attempt,
        "run_id": run_id,
        "p50_latency_seconds": parse_float(prom.get("p50_latency_seconds")),
        "p95_latency_seconds": parse_float(prom.get("p95_latency_seconds")),
        "p99_latency_seconds": parse_float(prom.get("p99_latency_seconds")),
        "mean_latency_seconds": parse_float(prom.get("mean_latency_seconds")),
        "success_rate_pct": parse_float(required.get("success_rate_pct")),
        "failure_rate_pct": parse_float(required.get("failure_rate_pct")),
        "timeout_rate_pct": parse_float(required.get("timeout_rate_pct")),
        "over_5s_success_rate_pct": parse_float(required.get("over_5s_success_rate_pct")),
        "over_10s_success_rate_pct": parse_float(required.get("over_10s_success_rate_pct")),
        "over_5s_with_failures_rate_pct": parse_float(
            required.get("over_5s_with_failures_rate_pct")
        ),
        "over_10s_with_failures_rate_pct": parse_float(
            required.get("over_10s_with_failures_rate_pct")
        ),
        "total_requests": parse_int(required.get("total_requests")),
        "successful_requests": parse_int(required.get("successful_requests")),
        "failed_requests": parse_int(required.get("failed_requests")),
        "timeout_requests": parse_int(required.get("timeout_requests")),
        "timeout_window_count": parse_int(required.get("timeout_window_count")),
        "timeout_windows_json": required.get("timeout_windows_json") or "[]",
        "timeout_periods_json": required.get("timeout_periods_json") or "[]",
        "node_latency_available": bool(required.get("node_latency_available")),
        "node_latency_summary_json": required.get("node_latency_summary_json") or "{}",
        "node_latency_note": required.get("node_latency_note") or "",
        "estimated_total_energy_j": parse_float(prom.get("estimated_total_energy_j")),
        "estimated_total_energy_kwh": parse_float(prom.get("estimated_total_energy_kwh")),
        "estimated_total_carbon_gco2e": parse_float(
            prom.get("estimated_total_carbon_gco2e")
        ),
        "energy_per_successful_request_j": parse_float(
            prom.get("energy_per_successful_request_j")
        ),
        "carbon_per_successful_request_gco2e": parse_float(
            prom.get("carbon_per_successful_request_gco2e")
        ),
        "cold_start_count": parse_int(prom.get("cold_start_count")),
        "cold_start_measured_count": parse_int(prom.get("cold_start_measured_count")),
        "cold_start_fallback_count": parse_int(prom.get("cold_start_fallback_count")),
        "cold_start_latency_mean_seconds": parse_float(prom.get("cold_start_latency_mean_seconds")),
        "cold_start_latency_p95_seconds": parse_float(prom.get("cold_start_latency_p95_seconds")),
        "cold_start_latency_max_seconds": parse_float(prom.get("cold_start_latency_max_seconds")),
    }


RUN_ARTIFACT_SUFFIXES = {
    "request_log": "_replay_request_log.csv",
    "window_decisions": "_window_decisions.csv",
    "queue_samples": "_queue_samples.csv",
    "scale_events": "_scale_events.csv",
    "extra_prometheus_monitoring": "_extra_prometheus_monitoring.json",
}


def copy_run_artifacts(run_id: str, method_dir: Path) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for name, suffix in RUN_ARTIFACT_SUFFIXES.items():
        source = base.RESULTS / f"{run_id}{suffix}"
        destination = method_dir / source.name
        if not source.exists():
            artifacts[name] = {
                "source": str(source),
                "missing": True,
            }
            continue
        shutil.copy2(source, destination)
        artifacts[name] = {
            "source": str(source),
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "missing": False,
        }
    return artifacts


def validate_observability_artifacts(
    method: str,
    artifacts: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    request_log = artifacts.get("request_log") or {}
    if request_log.get("missing"):
        raise FileNotFoundError(f"{method} request log was not archived: {request_log}")

    controller_methods = {"transformer_direct", "proposed", *QUEUE_ABLATION_METHODS, *ORACLE_METHODS}
    if method in controller_methods:
        window_decisions = artifacts.get("window_decisions") or {}
        if window_decisions.get("missing"):
            raise FileNotFoundError(
                f"{method} window decisions were not archived: {window_decisions}"
            )

    queue_mode = proposed_queue_mode_for_method(method, args)
    needs_queue_samples = (
        method in {"proposed", "proposed_current_queue", "proposed_ready_queue", *ORACLE_METHODS}
        and queue_mode != "none"
    )
    if needs_queue_samples:
        queue_samples = artifacts.get("queue_samples") or {}
        if queue_samples.get("missing"):
            raise FileNotFoundError(
                f"{method} queue samples were not archived: {queue_samples}"
            )


def uses_hpa_method(methods: list[str]) -> bool:
    return "hpa" in set(methods)


def metrics_api_available_condition(api_service: dict[str, Any]) -> dict[str, Any]:
    for condition in api_service.get("status", {}).get("conditions", []):
        if condition.get("type") == "Available":
            return condition
    return {}


def validate_hpa_metrics_preflight(label: str) -> dict[str, Any]:
    """Fail fast when CPU HPA cannot read Kubernetes resource metrics."""
    report: dict[str, Any] = {
        "label": label,
        "checked_at_utc": base.utc_now(),
        "metrics_api": {},
        "metrics_server_endpoints": [],
        "top_nodes": "",
    }
    api_service = base.kubectl_json(["get", "apiservice", "v1beta1.metrics.k8s.io"])
    condition = metrics_api_available_condition(api_service)
    report["metrics_api"] = {
        "available": condition.get("status") == "True",
        "status": condition.get("status"),
        "reason": condition.get("reason"),
        "message": condition.get("message"),
        "last_transition_time": condition.get("lastTransitionTime"),
    }
    if condition.get("status") != "True":
        raise RuntimeError(
            f"{label}: HPA CPU metrics API is not Available: {report['metrics_api']}"
        )

    endpoint_addresses = base.run_capture(
        [
            "kubectl", "get", "endpointslice", "-n", "kube-system",
            "-l", "kubernetes.io/service-name=metrics-server",
            "-o", "jsonpath={.items[*].endpoints[*].addresses[*]}",
        ],
        check=False,
    ).split()
    if not endpoint_addresses:
        endpoint_addresses = base.run_capture(
            [
                "kubectl", "get", "endpoints", "metrics-server", "-n", "kube-system",
                "-o", "jsonpath={.subsets[*].addresses[*].ip}",
            ],
            check=False,
        ).split()
    report["metrics_server_endpoints"] = endpoint_addresses
    if not endpoint_addresses:
        raise RuntimeError(f"{label}: metrics-server has no Ready endpoint")

    top_nodes = base.run_capture(["kubectl", "top", "nodes"])
    report["top_nodes"] = "\n".join(top_nodes.strip().splitlines()[:8])
    if "<unknown>" in top_nodes.lower():
        raise RuntimeError(f"{label}: kubectl top nodes returned unknown metrics")
    return report


def collect_hpa_diagnostics(revision: str, method_dir: Path, label: str) -> dict[str, Any]:
    diagnostics_dir = method_dir / "hpa_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "label": label,
        "revision": revision,
        "collected_at_utc": base.utc_now(),
        "files": {},
        "hpa_status": {},
    }
    commands = {
        "hpa_yaml": ["kubectl", "get", "hpa", revision, "-n", base.NAMESPACE, "-o", "yaml"],
        "hpa_json": ["kubectl", "get", "hpa", revision, "-n", base.NAMESPACE, "-o", "json"],
        "hpa_describe": ["kubectl", "describe", "hpa", revision, "-n", base.NAMESPACE],
        "events": ["kubectl", "get", "events", "-n", base.NAMESPACE, "--sort-by=.lastTimestamp"],
    }
    for name, cmd in commands.items():
        output = base.run_capture(cmd, check=False)
        suffix = "json" if name == "hpa_json" else "txt"
        path = diagnostics_dir / f"{label}_{name}.{suffix}"
        path.write_text(output, encoding="utf-8", errors="replace")
        report["files"][name] = str(path)
        if name == "hpa_json":
            try:
                payload = json.loads(output)
            except json.JSONDecodeError:
                report["hpa_status"]["json_error"] = output.strip()[:500]
            else:
                status = payload.get("status") or {}
                report["hpa_status"] = {
                    "current_replicas": status.get("currentReplicas"),
                    "desired_replicas": status.get("desiredReplicas"),
                    "current_metrics": status.get("currentMetrics"),
                    "conditions": status.get("conditions"),
                }
    return report


def collect_hpa_failure_diagnostics(
    revision: str,
    method_dir: Path,
    label: str,
) -> dict[str, Any]:
    try:
        report = collect_hpa_diagnostics(revision, method_dir, label)
        report["status"] = "collected"
        return report
    except Exception as exc:  # noqa: BLE001
        return {
            "label": label,
            "revision": revision,
            "status": "failed",
            "error": compact_error(exc),
            "collected_at_utc": base.utc_now(),
        }


def workload_capacity_pressure(workload: Path, args: argparse.Namespace) -> dict[str, Any]:
    window_count = max(
        1,
        int(round(float(args.duration_minutes) * 60.0 / FROZEN_DECISION_WINDOW_SECONDS)),
    )
    capacity = float(args.initial_pods) * float(args.pod_capacity_per_window)
    request_columns = ("requests_per_60s", "requests_per_20s", "requests", "request_count")
    above_capacity: list[dict[str, Any]] = []
    max_requests = 0.0
    with workload.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for offset, row in enumerate(rows[int(args.start_row): int(args.start_row) + window_count]):
        request_value = None
        request_column = ""
        for column in request_columns:
            if column in row and row[column] not in {"", None}:
                request_value = float(row[column])
                request_column = column
                break
        if request_value is None:
            continue
        max_requests = max(max_requests, request_value)
        if request_value > capacity:
            above_capacity.append({
                "window_index": offset,
                "window_id": row.get("window_id"),
                "requests": request_value,
                "request_column": request_column,
            })
    return {
        "initial_pods": int(args.initial_pods),
        "pod_capacity_per_window": float(args.pod_capacity_per_window),
        "initial_capacity_per_window": capacity,
        "window_count": window_count,
        "max_requests_per_window": max_requests,
        "above_initial_capacity_window_count": len(above_capacity),
        "first_above_initial_capacity_windows": above_capacity[:10],
    }


def validate_hpa_scale_activity(
    workload: Path,
    args: argparse.Namespace,
    ready_counts: list[int],
    total_counts: list[int],
    prom: dict[str, Any],
) -> dict[str, Any]:
    pressure = workload_capacity_pressure(workload, args)
    observed_scale_up = parse_int(prom.get("observed_scale_up_pod_count"))
    new_pod_count = parse_int(prom.get("new_pod_name_count_after_start"))
    report = {
        "passed": True,
        "ready_counts_unique": sorted(set(ready_counts)),
        "total_counts_unique": sorted(set(total_counts)),
        "max_ready_pods": max(ready_counts) if ready_counts else 0,
        "max_total_pods": max(total_counts) if total_counts else 0,
        "observed_scale_up_pod_count": observed_scale_up,
        "new_pod_name_count_after_start": new_pod_count,
        "capacity_pressure": pressure,
    }
    scale_needed = pressure["above_initial_capacity_window_count"] > 0
    no_scale_up = (
        report["max_ready_pods"] <= int(args.initial_pods)
        and report["max_total_pods"] <= int(args.initial_pods)
        and observed_scale_up <= 0
        and new_pod_count <= 0
    )
    if scale_needed and no_scale_up:
        report["passed"] = False
        report["error"] = (
            "HPA control-path failure: workload exceeded initial-pod capacity in "
            f"{pressure['above_initial_capacity_window_count']} windows, but HPA never "
            f"scaled beyond {args.initial_pods} pods"
        )
    return report


def run_method(
    method: str,
    repetition: int,
    attempt: int,
    args: argparse.Namespace,
    method_dir: Path,
    workload: Path,
) -> dict[str, Any]:
    method_dir.mkdir(parents=True, exist_ok=True)
    run_id = (
        f"stage2_{method}_rep{repetition:02d}_attempt{attempt:02d}_"
        f"s{args.start_row}_{duration_label(args.duration_minutes)}_{args.batch}"
    )
    result: dict[str, Any] = {
        "method": method,
        "repetition": repetition,
        "attempt": attempt,
        "run_id": run_id,
        "seed": seed_for_repetition(repetition, args),
        "started_at_utc": base.utc_now(),
        "yaml": METHOD_YAMLS[method],
        "checks": {},
    }

    expected_nodes = base.initial_placement_nodes(args)
    result["checks"]["initial_placement_contract"] = {
        "required_nodes_before_warmup_and_formal_replay": expected_nodes,
        "runtime_scale_placement": "scheduler-managed, audited only",
        "recovery": (
            "cleanup and recreate qwen before warmup when the initial Ready "
            "pod placement does not match"
        ),
    }
    revision, initial, recovery = base.prepare_initial_revision_with_recovery(
        method,
        args,
        method_dir,
        method,
    )
    result["revision"] = revision
    result["checks"]["initial_placement_recovery"] = recovery
    result["checks"]["initial_ready_pods"] = initial
    result["checks"]["initial_scale"] = args.initial_pods
    base.cooldown(args.pre_warmup_settle_seconds, f"{method}_pre_warmup_settle")

    result["checks"]["warmup"] = base.light_warmup_and_verify(args, method_dir, method)
    base.cooldown(args.drain_seconds, f"{method}_drain")

    result["checks"]["before_formal_ready_pods"] = base.wait_exact_ready(
        args.initial_pods,
        args.wait_ready_seconds,
        f"{method}_before_formal",
        expected_nodes=expected_nodes,
    )
    result["checks"]["latest_revision_before_release"] = base.latest_revision()
    base.release_revision_for_formal(revision, min_scale=args.initial_pods, max_scale=4)
    result["checks"]["formal_release_bounds"] = {"min_scale": args.initial_pods, "max_scale": 4}
    result["checks"]["latest_revision_after_release"] = base.assert_latest_revision_unchanged(
        revision,
        f"{method}_after_release",
    )
    result["checks"]["formal_start_ready_pods"] = base.wait_exact_ready(
        args.initial_pods,
        args.wait_ready_seconds,
        f"{method}_formal_start",
        expected_nodes=expected_nodes,
    )
    if method == "hpa" and args.collect_hpa_diagnostics_on_success:
        result["checks"]["hpa_after_release"] = collect_hpa_diagnostics(
            revision,
            method_dir,
            "after_release",
        )

    formal_log = base.LOGS / f"{run_id}.log"
    cmd = method_command(method, run_id, revision, workload, args, repetition)
    try:
        start_epoch, end_epoch, samples = base.run_with_pod_sampling(cmd, formal_log, args)
    except Exception:
        if method == "hpa":
            result["checks"]["hpa_failure_diagnostics"] = collect_hpa_failure_diagnostics(
                revision,
                method_dir,
                "formal_failure",
            )
        raise
    if method == "hpa" and args.collect_hpa_diagnostics_on_success:
        result["checks"]["hpa_after_formal"] = collect_hpa_diagnostics(
            revision,
            method_dir,
            "after_formal",
        )
    result["checks"]["formal_pod_samples"] = samples
    placement_audit = base.summarize_formal_placement(samples, expected_nodes)
    result["checks"]["formal_placement_audit"] = placement_audit
    write_json(method_dir / "formal_placement_audit.json", placement_audit)
    running_counts = [int(sample["running"]) for sample in samples]
    ready_counts = [int(sample["ready"]) for sample in samples]
    total_counts = [int(sample["total"]) for sample in samples]
    terminating_counts = [int(sample["terminating"]) for sample in samples]
    result["checks"]["formal_pod_count_validation"] = {
        "running_counts": running_counts,
        "ready_counts": ready_counts,
        "total_counts": total_counts,
        "terminating_counts": terminating_counts,
        "max_total_pods_including_pending_or_terminating": max(total_counts) if total_counts else 0,
        "rule": "formal active pod count uses non-deleting Running pods; total pod count is diagnostic",
    }
    if not running_counts or min(running_counts) < 1 or max(running_counts) > 4:
        raise RuntimeError(f"{method} running pod count outside [1,4]: {running_counts}")
    if ready_counts and max(ready_counts) > 4:
        raise RuntimeError(f"{method} ready pod count exceeded 4: {ready_counts}")
    if total_counts and max(total_counts) > 4:
        print(f"WARNING: {method} transient total pod count exceeded 4: {total_counts}", flush=True)

    result["checks"]["log_check"] = validate_log(method, formal_log, args)
    prom = run_prometheus_metrics(
        method,
        run_id,
        start_epoch,
        end_epoch,
        args,
        base.LOGS / f"{run_id}_prometheus_metrics.log",
    )
    result["checks"]["prometheus_summary"] = prom
    if args.collect_extra_prometheus_monitoring:
        result["checks"]["extra_prometheus_monitoring"] = run_extra_prometheus_monitoring(
            method,
            run_id,
            start_epoch,
            end_epoch,
            args,
        )
    else:
        result["checks"]["extra_prometheus_monitoring"] = {
            "skipped": True,
            "reason": "extra_prometheus_monitoring_disabled",
        }
    if int(prom.get("successful_requests") or 0) <= 0:
        raise RuntimeError(f"{method} workload did not produce successful requests through Knative")
    required = derive_required_request_metrics(run_id)
    result["checks"]["required_request_metrics"] = required
    controller_diagnostics = derive_controller_diagnostics(run_id)
    result["checks"]["controller_diagnostics"] = controller_diagnostics
    pod_sample_metrics = summarize_formal_pod_samples(samples)
    result["checks"]["formal_pod_sample_metrics"] = pod_sample_metrics
    node_energy_efficiency = derive_node_energy_efficiency(
        run_id,
        parse_int(required.get("successful_requests")),
    )
    result["checks"]["node_energy_efficiency"] = node_energy_efficiency
    if method == "hpa":
        hpa_scale_validation = validate_hpa_scale_activity(
            workload,
            args,
            ready_counts,
            total_counts,
            prom,
        )
        result["checks"]["hpa_scale_activity_validation"] = hpa_scale_validation
        if not hpa_scale_validation["passed"]:
            result["checks"]["hpa_failure_diagnostics"] = collect_hpa_failure_diagnostics(
                revision,
                method_dir,
                "scale_validation_failure",
            )
            result["finished_at_utc"] = base.utc_now()
            result["passed"] = False
            result["failure_type"] = "hpa_control_path_failure"
            (method_dir / "method_summary.json").write_text(
                json.dumps(result, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            raise RuntimeError(hpa_scale_validation["error"])
    run_metrics = build_run_metrics(
        method,
        repetition,
        attempt,
        run_id,
        prom,
        required,
    )
    run_metrics.update(controller_diagnostics)
    run_metrics.update(pod_sample_metrics)
    run_metrics.update(node_energy_efficiency)
    run_metrics["pod_capacity_per_window"] = float(args.pod_capacity_per_window)
    run_metrics["fixed_four_pod_capacity_per_window"] = 4.0 * float(args.pod_capacity_per_window)
    result["run_metrics"] = run_metrics
    result["artifacts"] = copy_run_artifacts(run_id, method_dir)
    validate_observability_artifacts(method, result["artifacts"], args)

    result["finished_at_utc"] = base.utc_now()
    result["passed"] = True
    (method_dir / "method_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--methods", nargs="+", choices=METHOD_CHOICES, default=METHOD_ORDER)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-attempts-per-run", type=int, default=2)
    parser.add_argument("--workload", default=str(base.BASE / FROZEN_WORKLOAD))
    parser.add_argument("--start-row", type=int, default=FROZEN_START_ROW)
    parser.add_argument("--duration-minutes", type=float, default=FROZEN_DURATION_MINUTES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-pods", type=int, default=2)
    parser.add_argument(
        "--initial-placement-nodes",
        default=",".join(base.INITIAL_PLACEMENT_NODES),
        help=(
            "Comma-separated exact node multiset required for the initial Ready pods "
            "before warmup and before formal replay."
        ),
    )
    parser.add_argument("--expected-qwen-image", default=EXPECTED_QWEN_IMAGE)
    parser.add_argument("--image-precheck-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--image-precheck-only", action="store_true")
    parser.add_argument(
        "--allow-smoke-overrides",
        action="store_true",
        help="Permit shorter duration/start-row values for pre-formal smoke tests.",
    )
    parser.add_argument(
        "--allow-workload-slice-overrides",
        action="store_true",
        help="Permit a non-default workload start row and duration while keeping formal checks enabled.",
    )
    parser.add_argument(
        "--allow-oracle-control-run",
        action="store_true",
        help="Permit a one-repetition proposed_oracle control run on the frozen slice.",
    )
    parser.add_argument(
        "--allow-mu-sensitivity-overrides",
        action="store_true",
        help="Permit non-default pod capacity only for explicit mu sensitivity runs.",
    )
    parser.add_argument("--pod-capacity-per-window", type=float, default=FROZEN_POD_CAPACITY)
    parser.add_argument("--request-timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--formal-start-delay-seconds",
        type=float,
        default=FROZEN_FORMAL_START_DELAY_SECONDS,
    )
    parser.add_argument("--first-patch-lead-seconds", type=float, default=1.0)
    parser.add_argument("--wait-ready-seconds", type=float, default=300.0)
    parser.add_argument("--cleanup-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--initial-placement-recovery-attempts", type=int, default=4)
    parser.add_argument("--initial-placement-recovery-delay-seconds", type=float, default=10.0)
    parser.add_argument("--cooldown-seconds", type=float, default=120.0)
    parser.add_argument("--drain-seconds", type=float, default=30.0)
    parser.add_argument("--pod-sample-seconds", type=float, default=60.0)
    parser.add_argument("--legacy-metrics-poll-seconds", type=float, default=60.0)
    parser.add_argument("--kpa-scale-down-min-ready-pods", type=int, default=1)
    parser.add_argument("--kpa-scale-down-guard-seconds", type=float, default=120.0)
    parser.add_argument("--kpa-scale-down-post-check-seconds", type=float, default=30.0)
    parser.add_argument("--kpa-scale-down-post-check-poll-seconds", type=float, default=10.0)
    parser.add_argument(
        "--kpa-disable-ready-guard",
        dest="kpa_ready_guard_enabled",
        action="store_false",
        help="Disable the unified Ready availability guard for bilstm_kpa.",
    )
    parser.set_defaults(kpa_ready_guard_enabled=True)
    parser.add_argument("--kpa-ready-guard-min-ready-pods", type=int, default=1)
    parser.add_argument("--kpa-ready-guard-recovery-pods", type=int, default=2)
    parser.add_argument("--kpa-ready-guard-stable-seconds", type=float, default=120.0)
    parser.add_argument("--kpa-ready-guard-poll-seconds", type=float, default=10.0)
    parser.add_argument("--api-health-checks", type=int, default=1)
    parser.add_argument("--api-health-interval-seconds", type=float, default=0.0)
    parser.add_argument("--api-health-timeout-seconds", type=float, default=15.0)
    parser.add_argument(
        "--run-attempt-pre-health-checks",
        action="store_true",
        help="Run the Kubernetes health gate before every method attempt.",
    )
    parser.add_argument(
        "--run-post-health-checks",
        action="store_true",
        help="Run post-attempt Kubernetes health checks as non-blocking diagnostics.",
    )
    parser.add_argument(
        "--node-notready-wait-seconds",
        type=float,
        default=0.0,
        help=(
            "Seconds to keep retrying cluster health/cleanup after worker "
            "NodeNotReady, API, or webhook health failures; negative waits "
            "indefinitely."
        ),
    )
    parser.add_argument(
        "--node-notready-wait-poll-seconds",
        type=float,
        default=30.0,
        help="Poll interval for --node-notready-wait-seconds.",
    )
    parser.add_argument(
        "--api-outage-grace-seconds",
        type=float,
        default=base.API_OUTAGE_GRACE_SECONDS,
        help=(
            "Grace period for transient Kubernetes API outages during formal "
            "pod sampling before classifying the attempt as infra failure."
        ),
    )
    parser.add_argument("--warmup-duration-seconds", type=float, default=60.0)
    parser.add_argument("--warmup-qps", type=float, default=1.5)
    parser.add_argument("--warmup-concurrency", type=int, default=2)
    parser.add_argument("--warmup-attempts", type=int, default=2)
    parser.add_argument("--warmup-min-success", type=int, default=30)
    parser.add_argument("--warmup-request-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--warmup-max-tokens", type=int, default=100)
    parser.add_argument("--warmup-cpu-query-delay-seconds", type=float, default=10.0)
    parser.add_argument("--warmup-min-cpu-delta", type=float, default=0.001)
    parser.add_argument("--warmup-targeted-requests-per-pod", type=int, default=8)
    parser.add_argument("--warmup-targeted-min-success-per-pod", type=int, default=None)
    parser.add_argument("--warmup-targeted-concurrency", type=int, default=2)
    parser.add_argument("--warmup-targeted-min-cpu-delta", type=float, default=0.05)
    parser.add_argument("--pre-warmup-settle-seconds", type=float, default=20.0)
    parser.add_argument("--prometheus-url", default=base.PROMETHEUS_URL)
    parser.add_argument("--prometheus-step-seconds", type=float, default=10.0)
    parser.add_argument("--prometheus-cpu-rate-window-seconds", type=float, default=30.0)
    parser.add_argument("--prometheus-query-delay-seconds", type=float, default=20.0)
    parser.add_argument(
        "--collect-extra-prometheus-monitoring",
        action="store_true",
        help="Collect optional Kourier/queue-proxy query_range diagnostics after each run.",
    )
    parser.add_argument(
        "--collect-hpa-diagnostics-on-success",
        action="store_true",
        help="Collect verbose HPA YAML/describe/event diagnostics on successful HPA runs.",
    )
    parser.add_argument("--carbon-intensity-csv", default=FROZEN_CARBON_CSV)
    parser.add_argument("--ci-trace-start", default=FROZEN_CI_TRACE_START)
    parser.add_argument("--ci-window-seconds", type=float, default=FROZEN_CI_WINDOW_SECONDS)
    parser.add_argument(
        "--ci-time-scale",
        type=float,
        default=None,
        help=(
            "Synthetic CI seconds per formal elapsed second for Prometheus "
            "accounting. Defaults to --ci-window-seconds / 60; use 8 for 3h->24h."
        ),
    )
    parser.add_argument(
        "--carbon-lookup-window-seconds",
        type=float,
        default=None,
        help=(
            "Legacy elapsed-time window for Prometheus CI lookup when "
            "--ci-time-scale is not used. Defaults to the legacy 20s lookup "
            "for ci1440, otherwise the 60s decision window."
        ),
    )
    parser.add_argument("--controller-k-ci", type=float, default=None)
    parser.add_argument("--controller-k-g", type=float, default=None)
    parser.add_argument("--controller-alpha-base", type=float, default=None)
    parser.add_argument("--controller-alpha-min", type=float, default=None)
    parser.add_argument("--controller-alpha-max", type=float, default=None)
    parser.add_argument("--controller-scale-down-confirm-windows", type=int, default=None,
                        help="consecutive scale-in confirmation windows (proposed only)")
    parser.add_argument("--controller-scale-down-cooldown-seconds", type=float, default=0.0,
                        help="block scale-down for N seconds after a scale-up (proposed only)")
    parser.add_argument("--controller-disable-downtrend", action="store_true",
                        help="disable the downtrend quantile cap (proposed only)")
    parser.add_argument("--controller-allow-missing-queue", action="store_true")
    parser.add_argument("--proposed-queue-mode", choices=["none", "current", "ready"], default="ready")
    parser.add_argument("--proposed-oracle-predictor", action="store_true",
                        help="use the oracle predictor while keeping the method label as proposed")
    parser.add_argument("--oracle-confidence", type=float, default=1.0)
    parser.add_argument("--oracle-interval-pad", type=float, default=0.0)
    parser.add_argument("--ready-latency-p95-seconds", type=float, default=40.0)
    parser.add_argument("--scale-up-slack-seconds", type=float, default=1.0)
    parser.add_argument("--queue-sample-period-seconds", type=float, default=2.0)
    parser.add_argument("--queue-max-sample-age-seconds", type=float, default=10.0)
    parser.add_argument("--queue-query-timeout-seconds", type=float, default=1.5)
    parser.add_argument("--cpu-joules-per-core-second", type=float, default=10.8)
    parser.add_argument("--memory-joules-per-gb-second", type=float, default=0.3725)
    parser.add_argument("--cold-start-model-joules", type=float, default=10.38)
    parser.add_argument("--cold-start-window-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.repetitions <= 0:
        raise SystemExit("--repetitions must be positive")
    if args.max_attempts_per_run <= 0:
        raise SystemExit("--max-attempts-per-run must be positive")
    if args.initial_placement_recovery_attempts <= 0:
        raise SystemExit("--initial-placement-recovery-attempts must be positive")
    if args.initial_placement_recovery_delay_seconds < 0:
        raise SystemExit("--initial-placement-recovery-delay-seconds must be non-negative")
    if args.oracle_confidence < 0.0 or args.oracle_confidence > 1.0:
        raise SystemExit("--oracle-confidence must be in [0, 1]")
    if args.oracle_interval_pad < 0.0:
        raise SystemExit("--oracle-interval-pad must be non-negative")
    if args.proposed_oracle_predictor and args.methods != ["proposed"]:
        raise SystemExit(
            "--proposed-oracle-predictor is only supported with exactly --methods proposed; "
            "use --methods proposed_oracle for a separately labelled oracle control run"
        )
    if (
        abs(args.pod_capacity_per_window - FROZEN_POD_CAPACITY) > 1e-9
        and not args.allow_mu_sensitivity_overrides
    ):
        raise SystemExit(
            f"Stage-2 protocol requires --pod-capacity-per-window={FROZEN_POD_CAPACITY}"
        )
    allow_custom_workload_slice = (
        args.allow_smoke_overrides or args.allow_workload_slice_overrides
    )
    oracle_control_run = (
        args.allow_oracle_control_run
        and args.repetitions == 1
        and (
            args.methods == ["proposed_oracle"]
            or (args.methods == ["proposed"] and args.proposed_oracle_predictor)
        )
    )
    if not args.allow_smoke_overrides:
        if (
            abs(
                args.formal_start_delay_seconds
                - FROZEN_FORMAL_START_DELAY_SECONDS
            )
            > 1e-9
        ):
            raise SystemExit(
                "Stage-2 protocol requires "
                f"--formal-start-delay-seconds={FROZEN_FORMAL_START_DELAY_SECONDS}"
            )
    if not allow_custom_workload_slice:
        expected_workload = (base.BASE / FROZEN_WORKLOAD).resolve()
        if Path(args.workload).resolve() != expected_workload:
            raise SystemExit(f"Stage-2 protocol requires --workload={expected_workload}")
        if args.start_row != FROZEN_START_ROW:
            raise SystemExit(f"Stage-2 protocol requires --start-row={FROZEN_START_ROW}")
        if abs(args.duration_minutes - FROZEN_DURATION_MINUTES) > 1e-9:
            raise SystemExit(
                f"Stage-2 protocol requires --duration-minutes={FROZEN_DURATION_MINUTES:g}"
            )
        if args.repetitions != 3 and not oracle_control_run:
            raise SystemExit("Stage-2 formal protocol requires --repetitions=3")
    if args.ci_window_seconds <= 0:
        raise SystemExit("--ci-window-seconds must be positive")
    args.ci_time_scale = resolve_ci_time_scale(args.ci_window_seconds, args.ci_time_scale)
    if args.ci_time_scale <= 0:
        raise SystemExit("--ci-time-scale must be positive")
    args.carbon_lookup_window_seconds = resolve_carbon_lookup_window_seconds(
        args.ci_window_seconds,
        args.carbon_lookup_window_seconds,
    )
    if args.carbon_lookup_window_seconds <= 0:
        raise SystemExit("--carbon-lookup-window-seconds must be positive")
    if args.image_precheck_timeout_seconds <= 0:
        raise SystemExit("--image-precheck-timeout-seconds must be positive")
    if args.ready_latency_p95_seconds < 0:
        raise SystemExit("--ready-latency-p95-seconds must be non-negative")
    if args.scale_up_slack_seconds < 0:
        raise SystemExit("--scale-up-slack-seconds must be non-negative")
    if args.kpa_ready_guard_min_ready_pods < 0:
        raise SystemExit("--kpa-ready-guard-min-ready-pods must be non-negative")
    if args.kpa_ready_guard_recovery_pods < args.kpa_ready_guard_min_ready_pods:
        raise SystemExit(
            "--kpa-ready-guard-recovery-pods must be greater than or equal to "
            "--kpa-ready-guard-min-ready-pods"
        )
    if args.kpa_ready_guard_recovery_pods > 4:
        raise SystemExit("--kpa-ready-guard-recovery-pods must not exceed 4")
    if args.kpa_ready_guard_stable_seconds < 0:
        raise SystemExit("--kpa-ready-guard-stable-seconds must be non-negative")
    if args.kpa_ready_guard_poll_seconds <= 0:
        raise SystemExit("--kpa-ready-guard-poll-seconds must be positive")
    if args.queue_sample_period_seconds <= 0:
        raise SystemExit("--queue-sample-period-seconds must be positive")
    if args.queue_max_sample_age_seconds <= 0:
        raise SystemExit("--queue-max-sample-age-seconds must be positive")
    if args.queue_query_timeout_seconds <= 0:
        raise SystemExit("--queue-query-timeout-seconds must be positive")
    if args.api_health_checks <= 0:
        raise SystemExit("--api-health-checks must be positive")
    if args.api_health_interval_seconds < 0:
        raise SystemExit("--api-health-interval-seconds must be non-negative")
    if args.api_health_timeout_seconds <= 0:
        raise SystemExit("--api-health-timeout-seconds must be positive")
    if args.node_notready_wait_poll_seconds <= 0:
        raise SystemExit("--node-notready-wait-poll-seconds must be positive")
    if args.api_outage_grace_seconds < 0:
        raise SystemExit("--api-outage-grace-seconds must be non-negative")
    if args.warmup_targeted_requests_per_pod < 0:
        raise SystemExit("--warmup-targeted-requests-per-pod must be non-negative")
    if args.warmup_targeted_concurrency <= 0:
        raise SystemExit("--warmup-targeted-concurrency must be positive")
    if args.warmup_targeted_min_cpu_delta < 0:
        raise SystemExit("--warmup-targeted-min-cpu-delta must be non-negative")
    if (
        args.warmup_targeted_min_success_per_pod is not None
        and args.warmup_targeted_min_success_per_pod < 0
    ):
        raise SystemExit("--warmup-targeted-min-success-per-pod must be non-negative")
    if (
        args.warmup_targeted_requests_per_pod > 0
        and args.warmup_targeted_min_success_per_pod is not None
        and args.warmup_targeted_min_success_per_pod > args.warmup_targeted_requests_per_pod
    ):
        raise SystemExit(
            "--warmup-targeted-min-success-per-pod must not exceed "
            "--warmup-targeted-requests-per-pod"
        )
    prefix = "ko.local/qwen-local@sha256:"
    digest = args.expected_qwen_image.removeprefix(prefix)
    if (
        not args.expected_qwen_image.startswith(prefix)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise SystemExit("--expected-qwen-image must be an immutable ko.local/qwen-local@sha256:<64 hex> reference")
    return args


AGGREGATE_METRICS = [
    "p50_latency_seconds",
    "p95_latency_seconds",
    "p99_latency_seconds",
    "mean_latency_seconds",
    "success_rate_pct",
    "failure_rate_pct",
    "timeout_rate_pct",
    "over_5s_success_rate_pct",
    "over_10s_success_rate_pct",
    "over_5s_with_failures_rate_pct",
    "over_10s_with_failures_rate_pct",
    "total_requests",
    "successful_requests",
    "failed_requests",
    "timeout_requests",
    "timeout_window_count",
    "estimated_total_energy_j",
    "estimated_total_energy_kwh",
    "estimated_total_carbon_gco2e",
    "energy_per_successful_request_j",
    "carbon_per_successful_request_gco2e",
    "cold_start_count",
    "cold_start_latency_mean_seconds",
    "cold_start_latency_p95_seconds",
    "cold_start_latency_max_seconds",
    "max_ready_pods_observed",
    "mean_ready_pods_observed",
    "max_running_pods_observed",
    "mean_running_pods_observed",
    "max_ready_pods_sampled",
    "mean_ready_pods_sampled",
    "max_total_pods_sampled",
    "mean_total_pods_sampled",
    "peak_queue_depth",
    "mean_queue_depth",
    "peak_pending_requests",
    "mean_pending_requests",
    "scale_up_events",
    "scale_down_events",
    "scale_to_ready_observed_count",
    "scale_to_ready_latency_mean_seconds",
    "scale_to_ready_latency_p50_seconds",
    "scale_to_ready_latency_p95_seconds",
    "scale_to_ready_latency_max_seconds",
    "pod_capacity_per_window",
    "fixed_four_pod_capacity_per_window",
]


def write_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_llm_request_config(workload_replayer_path: Path) -> dict[str, Any]:
    tree = ast.parse(workload_replayer_path.read_text(encoding="utf-8-sig"))
    payload: dict[str, Any] | None = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "PAYLOAD" for target in node.targets):
            continue
        payload = ast.literal_eval(node.value)
        break
    if not isinstance(payload, dict):
        raise RuntimeError(f"PAYLOAD dict not found in {workload_replayer_path}")
    messages = payload.get("messages") or []
    prompt = ""
    if messages and isinstance(messages[0], dict):
        prompt = str(messages[0].get("content") or "")
    return {
        "model": payload.get("model"),
        "max_tokens": payload.get("max_tokens"),
        "temperature": payload.get("temperature"),
        "prompt": prompt,
        "source": str(workload_replayer_path),
    }


def frozen_input_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths: dict[str, Path] = {
        "workload": Path(args.workload),
        "carbon_csv": base.BASE / args.carbon_intensity_csv,
        "train_workload": base.BASE / "data/train_workload_burst.csv",
        "ets_config": base.BASE / "results/ets_best_config.json",
        "bilstm_model": base.BASE / "code/bilstm_autoscaler_model.pt",
        "transformer_model": base.BASE / FROZEN_CHECKPOINT / "best_model.pt",
        "stage2_runner": Path(__file__).resolve(),
        "stage2_controller": base.BASE / "stage2/controller_stage2.py",
        "frozen_protocol": base.BASE / "stage2/FROZEN_PROTOCOL.md",
        "sanity_runner": base.BASE / "run_e2e_sanity_check.py",
        "kpa_runner": base.BASE / "run_kpa_baseline.py",
        "hpa_runner": base.BASE / "run_hpa_baseline.py",
        "ets_direct_runner": base.BASE / "code/run_ets_baseline_tplus2.py",
        "ets_kpa_runner": base.BASE / "code/run_ets_kpa_baseline_tplus2.py",
        "bilstm_direct_runner": base.BASE / "code/run_bilstm_baseline_tplus2.py",
        "bilstm_kpa_runner": base.BASE / "code/run_bilstm_kpa_baseline_tplus2.py",
        "baseline_metrics": base.BASE / "baseline_metrics.py",
        "workload_replayer": base.BASE / "workload_replayer.py",
        "prometheus_energy": base.BASE / "prometheus_energy_metrics.py",
        "inference": base.BASE / "inference.py",
        "decision": base.BASE / "decision.py",
    }
    for method, yaml_name in METHOD_YAMLS.items():
        paths[f"yaml_{method}"] = base.BASE / yaml_name
    return paths


def validate_frozen_inputs(args: argparse.Namespace) -> dict[str, Any]:
    paths = frozen_input_paths(args)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"frozen Stage-2 inputs missing: {missing}")

    with paths["workload"].open("r", newline="", encoding="utf-8") as handle:
        workload_rows = list(csv.DictReader(handle))
    decision_windows = int(args.duration_minutes * 60 / FROZEN_DECISION_WINDOW_SECONDS)
    selected = workload_rows[args.start_row : args.start_row + decision_windows]
    if len(selected) != decision_windows:
        raise RuntimeError("frozen workload slice does not contain the required decision windows")
    allow_custom_workload_slice = (
        args.allow_smoke_overrides or args.allow_workload_slice_overrides
    )
    if not allow_custom_workload_slice:
        if len(selected) != decision_windows:
            raise RuntimeError(
                f"formal workload must contain {decision_windows} decision windows, got {len(selected)}"
            )
        first_window = parse_int(selected[0].get("window_id"))
        last_window = parse_int(selected[-1].get("window_id"))
        if (first_window, last_window) != (87309, 87488):
            raise RuntimeError(
                f"formal workload IDs mismatch: {(first_window, last_window)}"
            )

    with paths["carbon_csv"].open("r", newline="", encoding="utf-8-sig") as handle:
        carbon_rows = list(csv.DictReader(handle))
    if len(carbon_rows) != 288:
        raise RuntimeError(f"frozen carbon subset must have 288 rows, got {len(carbon_rows)}")
    if any(any(value in (None, "") for value in row.values()) for row in carbon_rows):
        raise RuntimeError("frozen carbon subset contains missing values")
    llm_request = extract_llm_request_config(paths["workload_replayer"])
    if parse_int(llm_request.get("max_tokens")) != 100:
        raise RuntimeError(
            "LLM request payload must cap output at 100 tokens; "
            f"observed max_tokens={llm_request.get('max_tokens')!r}"
        )
    selected_requests = [
        parse_float(
            row.get("requests_per_60s")
            or row.get("requests_per_20s")
            or row.get("requests")
            or row.get("request_count")
        )
        for row in selected
    ]

    return {
        "checked_at_utc": base.utc_now(),
        "paths": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
        "workload": {
            "start_row": args.start_row,
            "decision_windows": decision_windows,
            "first_window_id": parse_int(selected[0].get("window_id")),
            "last_window_id": parse_int(selected[-1].get("window_id")),
            "total_selected_requests": sum(selected_requests),
            "mean_requests_per_window": mean_or_zero(selected_requests),
            "min_requests_per_window": min(selected_requests) if selected_requests else 0.0,
            "max_requests_per_window": max(selected_requests) if selected_requests else 0.0,
            "normal_and_burst_note": (
                "The frozen Azure-derived trace slice preserves low/normal windows "
                "and burst windows; see the selected request statistics above and "
                "per-window replay schedules for the exact mix."
            ),
        },
        "carbon": {
            "rows": len(carbon_rows),
            "first_timestamp": carbon_rows[0].get("Datetime (UTC)"),
            "last_timestamp": carbon_rows[-1].get("Datetime (UTC)"),
            "ci_trace_start": args.ci_trace_start,
            "ci_window_seconds": args.ci_window_seconds,
            "ci_time_scale": args.ci_time_scale,
            "carbon_lookup_window_seconds": args.carbon_lookup_window_seconds,
        },
        "llm_request": llm_request,
    }


def assert_frozen_inputs_unchanged(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> None:
    previous_paths = previous.get("paths") or {}
    current_paths = current.get("paths") or {}
    mismatches: list[dict[str, Any]] = []
    for name in sorted(set(previous_paths) | set(current_paths)):
        before = (previous_paths.get(name) or {}).get("sha256")
        after = (current_paths.get(name) or {}).get("sha256")
        if before != after:
            mismatches.append({"name": name, "before": before, "after": after})
    if mismatches:
        raise RuntimeError(f"frozen Stage-2 inputs changed since batch start: {mismatches}")


def validate_formal_resume_args(
    summary: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    if args.allow_smoke_overrides or not summary.get("args"):
        return
    ignored = {
        "max_attempts_per_run",
        "image_precheck_only",
        "api_health_checks",
        "api_health_interval_seconds",
        "api_health_timeout_seconds",
        "run_attempt_pre_health_checks",
        "run_post_health_checks",
        "pod_sample_seconds",
        "legacy_metrics_poll_seconds",
        "prometheus_step_seconds",
        "collect_extra_prometheus_monitoring",
        "collect_hpa_diagnostics_on_success",
    }
    original = summary["args"]
    current = vars(args)
    differences = {
        key: {"original": original.get(key), "current": current.get(key)}
        for key in sorted(set(original) | set(current))
        if key not in ignored and original.get(key) != current.get(key)
    }
    if differences:
        raise RuntimeError(f"formal resume arguments changed: {differences}")


def write_config_snapshot(
    batch_dir: Path,
    args: argparse.Namespace,
    frozen_inputs: dict[str, Any],
) -> dict[str, Any]:
    snapshot_dir = batch_dir / "config_snapshot"
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    files_dir = snapshot_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    excluded = {"workload", "train_workload", "bilstm_model", "transformer_model"}
    copied: dict[str, Any] = {}
    paths = frozen_input_paths(args)
    for name, source in paths.items():
        if name in excluded:
            continue
        try:
            relative = source.resolve().relative_to(base.BASE.resolve())
        except ValueError:
            relative = Path(source.name)
        destination = files_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied[name] = {
            "source": str(source),
            "snapshot": str(destination),
            "sha256": sha256_file(destination),
            "bytes": destination.stat().st_size,
        }

    write_json(snapshot_dir / "effective_args.json", vars(args))
    write_json(snapshot_dir / "frozen_inputs.json", frozen_inputs)
    manifest = {
        "created_at_utc": base.utc_now(),
        "copied_files": copied,
        "effective_args": str(snapshot_dir / "effective_args.json"),
        "frozen_inputs": str(snapshot_dir / "frozen_inputs.json"),
    }
    write_json(manifest_path, manifest)
    return manifest


def validate_cluster_health(label: str, args: argparse.Namespace | None = None) -> dict[str, Any]:
    nodes = selected_worker_nodes_for_run(args)
    node_rows = collect_image_precheck_nodes(nodes)
    if {row["node"] for row in node_rows} != set(nodes):
        raise RuntimeError(f"{label}: worker node set mismatch: {node_rows}")
    not_ready = [row for row in node_rows if not row.get("ready")]
    if not_ready:
        raise RuntimeError(f"{label}: worker nodes not Ready: {not_ready}")
    health_checks = max(1, int(getattr(args, "api_health_checks", 1) or 1))
    interval_s = max(0.0, float(getattr(args, "api_health_interval_seconds", 0.0) or 0.0))
    timeout_s = max(1.0, float(getattr(args, "api_health_timeout_seconds", 15.0) or 15.0))
    readyz_samples = []
    readyz = ""
    for check_index in range(health_checks):
        readyz = base.run_capture(
            ["kubectl", "get", "--raw=/readyz"],
            timeout_s=timeout_s,
        )
        readyz_samples.append({
            "check": check_index + 1,
            "readyz": readyz.strip(),
            "checked_at_utc": base.utc_now(),
        })
        if "ok" not in readyz.lower():
            raise base.KubeApiUnavailable(
                f"{label}: Kubernetes readyz failed: {readyz!r}"
            )
        if check_index + 1 < health_checks and interval_s > 0:
            time.sleep(interval_s)
    webhook = base.run_capture([
        "kubectl", "get", "endpointslice", "-n", "knative-serving",
        "-l", "kubernetes.io/service-name=webhook",
        "-o", "jsonpath={.items[*].endpoints[*].addresses[*]}",
    ])
    if not webhook.strip():
        raise RuntimeError(f"{label}: Knative webhook has no endpoint")
    return {
        "label": label,
        "checked_at_utc": base.utc_now(),
        "worker_nodes": node_rows,
        "readyz": readyz.strip(),
        "readyz_samples": readyz_samples,
        "webhook_endpoints": webhook.split(),
    }


def node_notready_wait_seconds(args: argparse.Namespace | None) -> float:
    return float(getattr(args, "node_notready_wait_seconds", 0.0) or 0.0)


def node_notready_wait_poll_seconds(args: argparse.Namespace | None) -> float:
    return max(
        1.0,
        float(getattr(args, "node_notready_wait_poll_seconds", 30.0) or 30.0),
    )


def cluster_wait_remaining(wait_s: float, elapsed_s: float) -> str:
    if wait_s < 0:
        return "unbounded"
    return f"{max(0.0, wait_s - elapsed_s):.1f}s"


def compact_error(exc: BaseException) -> str:
    return " ".join(str(exc).split())[-600:]


def wait_for_cluster_health(
    label: str,
    args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    wait_s = node_notready_wait_seconds(args)
    if wait_s == 0:
        return validate_cluster_health(label, args)

    started = time.time()
    attempts = 0
    last_error = ""
    while True:
        attempts += 1
        try:
            result = validate_cluster_health(label, args)
            elapsed_s = time.time() - started
            if attempts > 1:
                result["health_wait"] = {
                    "status": "recovered",
                    "attempts": attempts,
                    "elapsed_seconds": elapsed_s,
                    "last_error": last_error,
                    "wait_seconds": wait_s,
                }
                print(
                    "cluster_health_wait "
                    f"label={label} status=recovered attempts={attempts} "
                    f"elapsed_seconds={elapsed_s:.1f}",
                    flush=True,
                )
            return result
        except Exception as exc:  # noqa: BLE001
            elapsed_s = time.time() - started
            if wait_s > 0 and elapsed_s >= wait_s:
                print(
                    "cluster_health_wait "
                    f"label={label} status=exhausted attempts={attempts} "
                    f"elapsed_seconds={elapsed_s:.1f} error={compact_error(exc)}",
                    flush=True,
                )
                raise
            last_error = compact_error(exc)
            print(
                "cluster_health_wait "
                f"label={label} status=waiting attempts={attempts} "
                f"elapsed_seconds={elapsed_s:.1f} "
                f"remaining={cluster_wait_remaining(wait_s, elapsed_s)} "
                f"error={last_error}",
                flush=True,
            )
            time.sleep(node_notready_wait_poll_seconds(args))


def is_recoverable_cleanup_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    recoverable_patterns = (
        "node not ready",
        "nodestatusunknown",
        "worker nodes not ready",
        "kubernetes readyz failed",
        "etcd failed",
        "context deadline exceeded",
        "serviceunavailable",
        "the server is currently unable to handle the request",
        "no endpoints available for service \"webhook\"",
        "failed calling webhook",
        "connection refused",
        "unable to connect to the server",
        "timed out",
    )
    return isinstance(exc, base.KubeApiUnavailable) or any(
        pattern in text for pattern in recoverable_patterns
    )


def cleanup_qwen_with_wait(
    args: argparse.Namespace,
    label: str,
) -> dict[str, Any]:
    wait_s = node_notready_wait_seconds(args)
    if wait_s == 0:
        base.cleanup_qwen(args.cleanup_timeout_seconds)
        return {
            "label": label,
            "status": "completed",
            "checked_at_utc": base.utc_now(),
            "wait_seconds": wait_s,
        }

    started = time.time()
    attempts = 0
    last_error = ""
    while True:
        attempts += 1
        try:
            base.cleanup_qwen(args.cleanup_timeout_seconds)
            elapsed_s = time.time() - started
            if attempts > 1:
                print(
                    "cleanup_wait "
                    f"label={label} status=recovered attempts={attempts} "
                    f"elapsed_seconds={elapsed_s:.1f}",
                    flush=True,
                )
            return {
                "label": label,
                "status": "completed",
                "checked_at_utc": base.utc_now(),
                "attempts": attempts,
                "elapsed_seconds": elapsed_s,
                "last_error": last_error,
                "wait_seconds": wait_s,
            }
        except Exception as exc:  # noqa: BLE001
            if not is_recoverable_cleanup_error(exc):
                raise
            elapsed_s = time.time() - started
            if wait_s > 0 and elapsed_s >= wait_s:
                print(
                    "cleanup_wait "
                    f"label={label} status=exhausted attempts={attempts} "
                    f"elapsed_seconds={elapsed_s:.1f} error={compact_error(exc)}",
                    flush=True,
                )
                raise
            last_error = compact_error(exc)
            print(
                "cleanup_wait "
                f"label={label} status=waiting attempts={attempts} "
                f"elapsed_seconds={elapsed_s:.1f} "
                f"remaining={cluster_wait_remaining(wait_s, elapsed_s)} "
                f"error={last_error}",
                flush=True,
            )
            time.sleep(node_notready_wait_poll_seconds(args))


def collect_valid_run_metrics(batch_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(batch_dir.glob("rep*/**/valid_run_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("run_metrics")
        if payload.get("passed") is True and isinstance(metrics, dict):
            rows.append(metrics)
    return sorted(
        rows,
        key=lambda row: (
            int(row["repetition"]),
            METHOD_CHOICES.index(row["method"]) if row["method"] in METHOD_CHOICES else 999,
        ),
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_teacher_requirements_summary(
    batch_dir: Path,
    rows: list[dict[str, Any]],
    expected_methods: list[str],
    expected_repetitions: int,
    per_run_path: Path,
    aggregate_path: Path,
    aggregate_wide_path: Path,
) -> dict[str, str]:
    frozen_inputs_path = batch_dir / "config_snapshot" / "frozen_inputs.json"
    effective_args_path = batch_dir / "config_snapshot" / "effective_args.json"
    frozen_inputs: dict[str, Any] = {}
    if frozen_inputs_path.exists():
        frozen_inputs = json.loads(frozen_inputs_path.read_text(encoding="utf-8"))

    payload = {
        "generated_at_utc": base.utc_now(),
        "expected_methods": expected_methods,
        "expected_repetitions": expected_repetitions,
        "valid_run_count": len(rows),
        "dataset": frozen_inputs.get("workload", {}),
        "carbon_trace": frozen_inputs.get("carbon", {}),
        "llm_request": frozen_inputs.get("llm_request", {}),
        "transformer_and_experiment_config": {
            "effective_args_json": str(effective_args_path),
            "frozen_inputs_json": str(frozen_inputs_path),
            "transformer_model": (
                frozen_inputs.get("paths", {})
                .get("transformer_model", {})
                .get("path", "")
            ),
        },
        "primary_metric_artifacts": {
            "per_run_metrics_csv": str(per_run_path),
            "method_aggregate_metrics_csv": str(aggregate_path),
            "method_aggregate_metrics_wide_csv": str(aggregate_wide_path),
        },
        "diagnostic_artifacts_per_run": {
            "request_latency_success_failure_timeout": "*_replay_request_log.csv",
            "workload_ready_target_queue_scale": "*_window_decisions.csv",
            "queue_backlog_samples": "*_queue_samples.csv",
            "scale_start_events": "*_scale_events.csv",
            "prometheus_energy_carbon": "*_prometheus_summary.json and *_prometheus_resource_samples.csv",
            "extra_prometheus_diagnostics": "*_extra_prometheus_monitoring.json",
            "formal_pod_samples": "valid_run_summary.json -> checks.formal_pod_samples",
        },
        "notes": [
            "P50/P95/P99, success, failure and timeout metrics are derived from request logs.",
            "Energy and carbon are derived from Prometheus CPU/memory samples and carbon intensity.",
            "Per-success energy/carbon are computed after the run from totals and successful requests.",
            "Node latency is computed only when request logs include a node or pod column.",
            "Scale-to-Ready latency is matched post-run from scale_events.csv and window_decisions.csv.",
        ],
    }
    json_path = batch_dir / "TEACHER_REQUIREMENTS_SUMMARY.json"
    write_json(json_path, payload)

    md_path = batch_dir / "TEACHER_REQUIREMENTS_SUMMARY.md"
    lines = [
        "# Teacher Requirements Summary",
        "",
        f"- Generated: `{payload['generated_at_utc']}`",
        f"- Valid runs: `{len(rows)}/{len(expected_methods) * expected_repetitions}`",
        f"- Methods: `{', '.join(expected_methods)}`",
        "",
        "## Primary Tables",
        "",
        f"- Per-run metrics: `{per_run_path}`",
        f"- Aggregate long table: `{aggregate_path}`",
        f"- Aggregate wide table: `{aggregate_wide_path}`",
        "",
        "## Diagnostics",
        "",
    ]
    for label, pattern in payload["diagnostic_artifacts_per_run"].items():
        lines.append(f"- `{label}`: `{pattern}`")
    lines.extend([
        "",
        "## Notes",
        "",
    ])
    lines.extend(f"- {note}" for note in payload["notes"])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "teacher_requirements_summary_json": str(json_path),
        "teacher_requirements_summary_md": str(md_path),
    }


def write_aggregate_artifacts(
    batch_dir: Path,
    expected_repetitions: int,
    expected_methods: list[str],
) -> dict[str, Any]:
    rows = collect_valid_run_metrics(batch_dir)
    per_run_path = batch_dir / "per_run_metrics.csv"
    per_run_fields = [
        "method", "repetition", "attempt", "run_id",
        "p50_latency_seconds", "p95_latency_seconds", "p99_latency_seconds",
        "mean_latency_seconds",
        "success_rate_pct", "failure_rate_pct", "timeout_rate_pct",
        "over_5s_success_rate_pct", "over_10s_success_rate_pct",
        "over_5s_with_failures_rate_pct", "over_10s_with_failures_rate_pct",
        "total_requests", "successful_requests", "failed_requests", "timeout_requests",
        "timeout_window_count", "timeout_windows_json", "timeout_periods_json",
        "estimated_total_energy_j", "estimated_total_energy_kwh",
        "estimated_total_carbon_gco2e",
        "energy_per_successful_request_j",
        "carbon_per_successful_request_gco2e",
        "cold_start_count", "cold_start_measured_count", "cold_start_fallback_count",
        "cold_start_latency_mean_seconds", "cold_start_latency_p95_seconds",
        "cold_start_latency_max_seconds",
        "formal_pod_sample_count",
        "max_running_pods_observed", "mean_running_pods_observed",
        "max_ready_pods_observed", "mean_ready_pods_observed",
        "max_ready_pods_sampled", "mean_ready_pods_sampled",
        "max_total_pods_sampled", "mean_total_pods_sampled",
        "controller_window_count", "queue_sample_count", "scale_events_captured",
        "peak_queue_depth", "mean_queue_depth",
        "peak_pending_requests", "mean_pending_requests",
        "scale_up_events", "scale_down_events",
        "scale_to_ready_observed_count",
        "scale_to_ready_latency_mean_seconds",
        "scale_to_ready_latency_p50_seconds",
        "scale_to_ready_latency_p95_seconds",
        "scale_to_ready_latency_max_seconds",
        "pod_capacity_per_window", "fixed_four_pod_capacity_per_window",
        "node_latency_available", "node_latency_note", "node_latency_summary_json",
        "node_energy_efficiency_available", "node_energy_efficiency_note",
        "node_energy_efficiency_json",
    ]
    write_csv(per_run_path, rows, per_run_fields)

    aggregate_rows: list[dict[str, Any]] = []
    aggregate_wide_rows: list[dict[str, Any]] = []
    for method in expected_methods:
        method_rows = [row for row in rows if row["method"] == method]
        wide_row: dict[str, Any] = {
            "method": method,
            "valid_runs": len(method_rows),
        }
        for metric in AGGREGATE_METRICS:
            values = [parse_float(row.get(metric)) for row in method_rows]
            average = statistics.fmean(values) if values else ""
            stddev = stddev_or_zero(values) if values else ""
            aggregate_rows.append({
                "method": method,
                "metric": metric,
                "valid_runs": len(values),
                "min": min(values) if values else "",
                "max": max(values) if values else "",
                "average": average,
                "stddev": stddev,
            })
            wide_row[f"{metric}_mean"] = average
            wide_row[f"{metric}_stddev"] = stddev
        aggregate_wide_rows.append(wide_row)
    aggregate_path = batch_dir / "method_aggregate_metrics.csv"
    write_csv(
        aggregate_path,
        aggregate_rows,
        ["method", "metric", "valid_runs", "min", "max", "average", "stddev"],
    )
    aggregate_wide_path = batch_dir / "method_aggregate_metrics_wide.csv"
    wide_fields = ["method", "valid_runs"]
    for metric in AGGREGATE_METRICS:
        wide_fields.extend([f"{metric}_mean", f"{metric}_stddev"])
    write_csv(aggregate_wide_path, aggregate_wide_rows, wide_fields)
    teacher_summary_paths = write_teacher_requirements_summary(
        batch_dir,
        rows,
        expected_methods,
        expected_repetitions,
        per_run_path,
        aggregate_path,
        aggregate_wide_path,
    )

    counts = {
        method: sum(row["method"] == method for row in rows)
        for method in expected_methods
    }
    complete = all(value == expected_repetitions for value in counts.values())
    report_path = batch_dir / "FINAL_REPORT.md"
    lines = [
        "# Stage 2: all baselines x3",
        "",
        f"- Generated: `{base.utc_now()}`",
        f"- Valid runs: `{len(rows)}/{len(expected_methods) * expected_repetitions}`",
        f"- Complete: `{complete}`",
        "",
        "## Valid run counts",
        "",
        "| Method | Valid runs |",
        "|---|---:|",
    ]
    lines.extend(f"| {method} | {counts[method]} |" for method in expected_methods)
    lines.extend([
        "",
        "## Aggregate metrics",
        "",
        "See `method_aggregate_metrics.csv` for min/max/average/stddev of every required metric.",
        "See `method_aggregate_metrics_wide.csv` for one row per method with mean/stddev pairs.",
        "See `per_run_metrics.csv` for all valid formal runs.",
        "",
        "Latency threshold rates are reported both over successful requests and with failures",
        "counted as threshold violations. Energy/carbon values are Prometheus-based estimates.",
        "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "valid_run_count": len(rows),
        "expected_run_count": len(expected_methods) * expected_repetitions,
        "counts": counts,
        "complete": complete,
        "per_run_metrics": str(per_run_path),
        "aggregate_metrics": str(aggregate_path),
        "aggregate_metrics_wide": str(aggregate_wide_path),
        "report": str(report_path),
        **teacher_summary_paths,
    }


def acquire_lock():
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle = LOCK_PATH.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"another Stage-2 runner holds {LOCK_PATH}")
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} started_at_utc={base.utc_now()}\n")
    handle.flush()
    return handle


def order_for_repetition(repetition: int, selected_methods: list[str]) -> list[str]:
    indexes = REPETITION_ORDER_INDEXES.get(repetition)
    if not indexes:
        return list(selected_methods)
    ordered: list[str] = []
    used_indexes: set[int] = set()
    for index in indexes:
        if index < len(selected_methods):
            ordered.append(selected_methods[index])
            used_indexes.add(index)
    for index, method in enumerate(selected_methods):
        if index not in used_indexes:
            ordered.append(method)
    return ordered


def configure_child_pythonpath() -> None:
    """Make project-root modules importable by scripts executed from code/."""
    project_root = str(base.BASE.resolve())
    existing = [
        entry
        for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if entry
    ]
    if project_root not in existing:
        existing.insert(0, project_root)
    os.environ["PYTHONPATH"] = os.pathsep.join(existing)


def next_attempt_number(method_slot: Path) -> int:
    attempts: list[int] = []
    for path in method_slot.glob("attempt*"):
        suffix = path.name.removeprefix("attempt")
        if suffix.isdigit():
            attempts.append(int(suffix))
    return max(attempts, default=0) + 1


def main() -> int:
    args = parse_args()
    if not args.allow_smoke_overrides and args.initial_pods != 2:
        raise SystemExit("formal protocol requires --initial-pods 2")
    base.normalize_initial_placement_nodes(args)
    configure_child_pythonpath()
    base.LOGS.mkdir(parents=True, exist_ok=True)
    base.RESULTS.mkdir(parents=True, exist_ok=True)
    workload = Path(args.workload)
    if not workload.exists():
        raise FileNotFoundError(workload)
    label = duration_label(args.duration_minutes)
    batch_dir = base.RESULTS / f"stage2_all_baselines_x3_s{args.start_row}_{label}_{args.batch}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    summary_path = batch_dir / "stage2_state.json"
    lock_handle = acquire_lock()
    summary: dict[str, Any]
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        resumed_at = base.utc_now()
        summary["resumed_at_utc"] = resumed_at
        summary["resume_count"] = int(summary.get("resume_count") or 0) + 1
        summary.setdefault("resume_history", []).append(
            {
                "resumed_at_utc": resumed_at,
                "args": vars(args),
            }
        )
        summary.pop("error", None)
        summary.pop("failed_at_utc", None)
    else:
        summary = {
            "batch": args.batch,
            "started_at_utc": base.utc_now(),
            "args": vars(args),
            "workload": str(workload),
            "selected_methods": args.methods,
            "repetitions": args.repetitions,
            "formal_repetition_seeds": {
                str(rep): seed_for_repetition(rep, args)
                for rep in range(1, args.repetitions + 1)
            },
            "formal_repetition_order_indexes": REPETITION_ORDER_INDEXES,
            "passed": False,
            "complete": False,
            "resume_count": 0,
            "events": [],
        }
    summary["formal_repetition_seeds"] = {
        str(rep): seed_for_repetition(rep, args)
        for rep in range(1, args.repetitions + 1)
    }
    summary["formal_repetition_order_indexes"] = REPETITION_ORDER_INDEXES
    write_json(summary_path, summary)
    try:
        validate_formal_resume_args(summary, args)
        current_frozen_inputs = validate_frozen_inputs(args)
        if summary.get("frozen_inputs"):
            assert_frozen_inputs_unchanged(
                summary["frozen_inputs"],
                current_frozen_inputs,
            )
        else:
            summary["frozen_inputs"] = current_frozen_inputs
        summary["config_snapshot"] = write_config_snapshot(
            batch_dir,
            args,
            summary["frozen_inputs"],
        )
        summary["yaml_contract"] = validate_frozen_yaml_contract(args.methods)
        summary["yaml_image_precheck"] = validate_method_yaml_images(
            args.methods,
            args.expected_qwen_image,
        )
        summary["image_precheck"] = run_image_precheck(batch_dir, args)
        if args.image_precheck_only:
            summary["passed"] = True
            summary["complete"] = False
            summary["finished_at_utc"] = base.utc_now()
            write_json(summary_path, summary)
            print("stage2_state=" + str(summary_path), flush=True)
            return 0
        run_controller_mu_dry_run(workload, batch_dir, args)
        if uses_hpa_method(args.methods):
            summary["hpa_metrics_preflight"] = validate_hpa_metrics_preflight(
                "batch_start_hpa_metrics"
            )
            write_json(summary_path, summary)
        summary["initial_health"] = wait_for_cluster_health("initial", args)
        summary["initial_cleanup"] = cleanup_qwen_with_wait(args, "initial")
        base.cooldown(args.cooldown_seconds, "initial")

        for repetition in range(1, args.repetitions + 1):
            repetition_order = order_for_repetition(repetition, args.methods)
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
                start_attempt = next_attempt_number(method_slot)
                stop_attempt = start_attempt + args.max_attempts_per_run
                for attempt in range(start_attempt, stop_attempt):
                    attempt_dir = method_slot / f"attempt{attempt:02d}"
                    attempt_dir.mkdir(parents=True, exist_ok=True)
                    print(
                        "========== Stage-2 "
                        f"repetition={repetition} method={method} attempt={attempt} ==========",
                        flush=True,
                    )
                    event: dict[str, Any] = {
                        "repetition": repetition,
                        "method": method,
                        "attempt": attempt,
                        "seed": seed_for_repetition(repetition, args),
                        "repetition_order": repetition_order,
                        "started_at_utc": base.utc_now(),
                    }
                    try:
                        if args.run_attempt_pre_health_checks:
                            event["pre_health"] = wait_for_cluster_health(
                                f"rep{repetition:02d}_{method}_attempt{attempt:02d}_pre",
                                args,
                            )
                        else:
                            event["pre_health"] = {
                                "skipped": True,
                                "reason": "attempt_pre_health_check_disabled",
                            }
                        if method == "hpa":
                            event["hpa_metrics_preflight"] = {
                                "skipped": True,
                                "reason": "batch_start_hpa_metrics_preflight_only",
                            }
                        method_result = run_method(
                            method,
                            repetition,
                            attempt,
                            args,
                            attempt_dir,
                            workload,
                        )
                        write_json(valid_summary_path, method_result)
                        event["passed"] = True
                        event["run_id"] = method_result["run_id"]
                        success = True
                    except base.KubeApiUnavailable as exc:
                        last_error = str(exc)
                        event["passed"] = False
                        event["failure_type"] = "infra_failure"
                        event["error"] = last_error
                        write_json(
                            attempt_dir / "failed_attempt.json",
                            {
                                **event,
                                "failed_at_utc": base.utc_now(),
                            },
                        )
                        print(
                            f"INFRA FAILURE repetition={repetition} "
                            f"method={method} attempt={attempt}: {exc}",
                            flush=True,
                        )
                    except Exception as exc:
                        last_error = str(exc)
                        event["passed"] = False
                        event["failure_type"] = "method_failure"
                        event["error"] = last_error
                        write_json(
                            attempt_dir / "failed_attempt.json",
                            {
                                **event,
                                "failed_at_utc": base.utc_now(),
                            },
                        )
                        print(
                            f"ATTEMPT FAILED repetition={repetition} "
                            f"method={method} attempt={attempt}: {exc}",
                            flush=True,
                        )
                    finally:
                        event["finished_at_utc"] = base.utc_now()
                        summary.setdefault("events", []).append(event)
                        write_json(summary_path, summary)
                        try:
                            event["cleanup"] = cleanup_qwen_with_wait(
                                args,
                                f"rep{repetition:02d}_{method}_attempt{attempt:02d}",
                            )
                            base.cooldown(
                                args.cooldown_seconds,
                                f"rep{repetition:02d}_{method}_attempt{attempt:02d}",
                            )
                        except base.KubeApiUnavailable as cleanup_exc:
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
                        if args.run_post_health_checks:
                            try:
                                event["post_health"] = wait_for_cluster_health(
                                    f"rep{repetition:02d}_{method}_attempt{attempt:02d}_post",
                                    args,
                                )
                            except Exception as health_exc:  # noqa: BLE001
                                event["post_health"] = {
                                    "status": "failed_non_blocking",
                                    "error": str(health_exc),
                                }
                                event["post_health_warning"] = compact_error(health_exc)
                                print(
                                    "post-attempt health check warning: "
                                    f"{health_exc}",
                                    flush=True,
                                )
                        else:
                            event["post_health"] = {
                                "skipped": True,
                                "reason": "post_attempt_health_check_disabled",
                            }
                        write_json(summary_path, summary)
                    if event.get("failure_type") == "infra_failure":
                        raise RuntimeError(
                            "infrastructure failure; stopping batch before retrying "
                            f"repetition={repetition} method={method}: {last_error}"
                        )
                    if success:
                        aggregate = write_aggregate_artifacts(
                            batch_dir,
                            args.repetitions,
                            args.methods,
                        )
                        summary["aggregate"] = aggregate
                        write_json(summary_path, summary)
                        break
                if not success:
                    raise RuntimeError(
                        f"no valid run for repetition={repetition} method={method} "
                        f"after attempts {start_attempt}..{stop_attempt - 1}: {last_error}"
                    )

        aggregate = write_aggregate_artifacts(
            batch_dir,
            args.repetitions,
            args.methods,
        )
        summary["aggregate"] = aggregate
        if not aggregate["complete"]:
            raise RuntimeError(f"Stage-2 run matrix incomplete: {aggregate['counts']}")
        summary["passed"] = True
        summary["complete"] = True
        summary["finished_at_utc"] = base.utc_now()
        try:
            summary["final_health"] = wait_for_cluster_health("final", args)
        except Exception as health_exc:  # noqa: BLE001
            summary["final_health"] = {
                "status": "failed_non_blocking",
                "error": str(health_exc),
            }
            print(f"final health check warning: {health_exc}", flush=True)
        write_json(summary_path, summary)
        print("stage2_state=" + str(summary_path), flush=True)
        return 0
    except BaseException as exc:
        summary["passed"] = False
        summary["complete"] = False
        summary["failed_at_utc"] = base.utc_now()
        summary["error"] = str(exc)
        write_json(summary_path, summary)
        print(f"ERROR: {exc}", flush=True)
        if not args.image_precheck_only:
            try:
                cleanup_qwen_with_wait(args, "error")
            except Exception as cleanup_exc:  # noqa: BLE001
                print(f"cleanup after error also failed: {cleanup_exc}", flush=True)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return 1
    finally:
        try:
            lock_handle.seek(0)
            lock_handle.truncate()
            lock_handle.flush()
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
