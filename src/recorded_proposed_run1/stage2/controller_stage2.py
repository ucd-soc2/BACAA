"""Recorded controller for Proposed run 1.

Supports live workload replay, offline trace input, and dry-run decisions.
"""

import argparse
import asyncio
import csv
import gzip
import json
import os
import sys
import time
import subprocess
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from collections import deque
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import QPSPredictor
from decision import DecisionLayer, CILoader
from workload_replayer import (
    DEFAULT_HOST,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TEST_CSV,
    DEFAULT_URL,
    ReplayConfig,
    ReplayRuntimeState,
    make_run_id,
    replay_workload,
    sleep_until,
)


# ============================================
# Controller configuration.

DECISION_WINDOW_S = 60
PREDICTION_HORIZON = 2
# 2026-06-26 initial scale-to-Ready calibration set a conservative 85s lead.
# Later active-Revision phase breakdown showed target Ready max 50.629s, so
# current Stage-2 runs use a 60s default lead.
SCALE_UP_LEAD_S = 40.0
CI_CSV = "carbon_intensity_ie_2025_5min.csv"

DECISION_CONFIG = {
    "mu":             12.5,      # Stage-2 calibrated single-pod capacity (requests/60s)
    "N_min":          1,
    "N_max":          4,
    "W":              3,
    "hist_beta":      0.9,
    "alpha_base":     0.50,       # carbon-aware scale-in baseline weight
    "k_ci":           0.20,       # high CI shifts weight toward current lower demand
    "k_g":            0.10,       # large drops still damp abrupt multi-pod release
    "alpha_min":      0.40,
    "alpha_max":      0.75,
    "scale_in_confirm_windows": 1,
    "downtrend_enabled": False,     # report has no downtrend cap; it accelerates scale-in -> disable
    "queue_alpha":    1.0,      # queue correction weight
    "queue_drain_s":  DECISION_WINDOW_S,  # drain corrected backlog within one decision window
    "scale_up_safety_margin": 1.0,
    "scale_down_horizon": 1,     # use the near horizon for scale-down decisions
    "downtrend_recent_windows": 3,
    "downtrend_previous_windows": 3,
    "downtrend_ratio": 0.75,
    "downtrend_min_drop": 30.0,  # requests/window
    "downtrend_margin": 1.15,
    "downtrend_queue_threshold": 0.0,
}


def apply_decision_config_overrides(args):
    overrides = {
        "mu": args.config_mu,
        "k_ci": args.config_k_ci,
        "k_g": args.config_k_g,
        "alpha_base": args.config_alpha_base,
        "alpha_min": args.config_alpha_min,
        "alpha_max": args.config_alpha_max,
    }
    applied = {}
    for key, value in overrides.items():
        if value is not None:
            DECISION_CONFIG[key] = float(value)
            applied[key] = float(value)
    return applied

SERVICE_NAME = "qwen"
NAMESPACE = "default"
KUBECTL_TIMEOUT_SECONDS = 30.0
KUBECTL_RETRY_ATTEMPTS = 4
KUBECTL_RETRY_DELAY_SECONDS = 3.0
KUBECTL_API_OUTAGE_GRACE_SECONDS = 180.0
API_UNAVAILABLE_PATTERNS = (
    "the connection to the server 127.0.0.1:6443 was refused",
    "connection refused",
    "unable to connect to the server",
    "dial tcp 127.0.0.1:6443",
    "net/http: tls handshake timeout",
    "kubectl timed out after",
    "i/o timeout",
    "context deadline exceeded",
    "apiserver not ready",
    "[-]etcd failed",
    "[-]etcd-readiness failed",
    "serviceunavailable",
    "the server is currently unable to handle the request",
)


# ============================================
# Controller helpers.

class KubeApiUnavailable(RuntimeError):
    pass


def _is_kube_api_unavailable(text):
    lowered = str(text or "").lower()
    return any(pattern in lowered for pattern in API_UNAVAILABLE_PATTERNS)


def _kubectl_completed(args, timeout=15):
    cmd = ["sudo", "-S", "k3s", "kubectl"] + list(args)
    last = subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
    started = time.time()
    attempt = 0
    while True:
        attempt += 1
        try:
            last = subprocess.run(
                cmd,
                input="llm123456\n",
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            last = subprocess.CompletedProcess(
                cmd,
                124,
                stdout="",
                stderr=f"kubectl timed out after {timeout}s",
            )
        except subprocess.SubprocessError as exc:
            last = subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr=f"kubectl failed: {exc}",
            )

        output = f"{last.stdout or ''}{last.stderr or ''}"
        if last.returncode == 0:
            if attempt > 1:
                print(
                    "  [kubectl api grace] recovered "
                    f"attempt={attempt} elapsed={time.time() - started:.1f}s"
                )
            return last
        if not _is_kube_api_unavailable(output):
            return last
        elapsed = time.time() - started
        remaining_grace = KUBECTL_API_OUTAGE_GRACE_SECONDS - elapsed
        if attempt >= KUBECTL_RETRY_ATTEMPTS and remaining_grace <= 0:
            return last
        sleep_s = min(KUBECTL_RETRY_DELAY_SECONDS, max(0.0, remaining_grace))
        print(
            "  [kubectl api grace] API unavailable "
            f"attempt={attempt} min_attempts={KUBECTL_RETRY_ATTEMPTS} "
            f"elapsed={elapsed:.1f}s grace={KUBECTL_API_OUTAGE_GRACE_SECONDS:.1f}s "
            f"sleep={sleep_s:.1f}s output={output.strip()[-300:]}"
        )
        if sleep_s > 0:
            time.sleep(sleep_s)
    return last


def _get_latest_revision():
    r = _kubectl_completed([
        "get", "ksvc", SERVICE_NAME,
        "-n", NAMESPACE, "-o", "jsonpath={.status.latestReadyRevisionName}"
    ], timeout=15)
    if r.returncode != 0:
        output = f"{r.stdout or ''}{r.stderr or ''}"
        if _is_kube_api_unavailable(output):
            raise KubeApiUnavailable(output.strip())
        print(f"  [kubectl ERROR] get revision: {r.stderr.strip()}")
        return None
    rev = r.stdout.strip()
    if not rev:
        print(f"  [kubectl ERROR] no latestReadyRevisionName")
        return None
    return rev


def kubectl_scale(replicas: int):
    revision = _get_latest_revision()
    if not revision:
        return

    patch = (
        '{"metadata":{"annotations":{'
        f'"autoscaling.knative.dev/min-scale":"{replicas}",'
        f'"autoscaling.knative.dev/max-scale":"{replicas}"'
        '}}}'
    )
    r = _kubectl_completed([
        "patch", "revision", revision,
        "-n", NAMESPACE, "--type", "merge", "-p", patch
    ], timeout=15)
    if r.returncode != 0:
        output = f"{r.stdout or ''}{r.stderr or ''}"
        if _is_kube_api_unavailable(output):
            raise KubeApiUnavailable(output.strip())
        print(f"  [kubectl ERROR] {r.stderr.strip()}")
    else:
        print(f"  [kubectl] patched {revision} min=max={replicas}")


def _run_kubectl(args, timeout=15):
    return _kubectl_completed(args, timeout=timeout)


def _parse_queue_depth(metrics_text: str) -> float:
    """Parse queue depth from queue-proxy/Prometheus metrics text.

    Different Knative builds expose this metric with slightly different names,
    so keep the parser permissive and only use samples whose metric name
    contains both "queue" and "depth".
    """
    total = 0.0
    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        metric_head = line.split("{", 1)[0].strip()
        if not metric_head:
            continue
        name_parts = metric_head.split(None, 1)
        if not name_parts:
            continue
        name = name_parts[0]
        if "queue" not in name.lower() or "depth" not in name.lower():
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        try:
            total += float(parts[1])
        except ValueError:
            pass
    return max(0.0, total)


def _parse_in_flight_requests(metrics_text: str) -> float:
    """Parse queue-proxy in-flight/concurrent request gauges from metrics text."""
    total = 0.0
    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        metric_head = line.split("{", 1)[0].strip()
        if not metric_head:
            continue
        name_parts = metric_head.split(None, 1)
        if not name_parts:
            continue
        name = name_parts[0].lower()
        if not (
            "inflight" in name
            or "in_flight" in name
            or ("concurrent" in name and "request" in name)
        ):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        try:
            total += float(parts[1])
        except ValueError:
            pass
    return max(0.0, total)


def _has_request_metric(metrics_text: str) -> bool:
    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        metric_head = line.split("{", 1)[0].strip()
        if not metric_head:
            continue
        name = metric_head.split(None, 1)[0].lower()
        if "request" in name:
            return True
    return False


def _read_pod_metrics_text(pod_name: str, timeout_s: float) -> str | None:
    try:
        raw = subprocess.run([
            "sudo", "-S", "k3s", "kubectl", "get", "--raw",
            f"/api/v1/namespaces/{NAMESPACE}/pods/{pod_name}:9090/proxy/metrics",
        ], input=b"llm123456\n", text=False, capture_output=True, timeout=max(timeout_s, 1.0))
    except subprocess.TimeoutExpired:
        return None
    except subprocess.SubprocessError:
        return None
    if raw.returncode != 0:
        return None
    try:
        return gzip.decompress(raw.stdout).decode()
    except (gzip.BadGzipFile, OSError):
        return raw.stdout.decode(errors="replace")


def _pod_is_ready(pod: dict) -> bool:
    return any(
        condition.get("type") == "Ready" and condition.get("status") == "True"
        for condition in pod.get("status", {}).get("conditions", [])
    )


def collect_queue_kubectl_sample(revision, expected_pods: int, timeout_s: float) -> dict:
    """Fallback queue sample from queue-proxy pod /metrics via kubectl get --raw."""
    sample_time = time.time()
    r = _run_kubectl([
        "get", "pod", "-n", NAMESPACE,
        "-l", f"serving.knative.dev/revision={revision}",
        "--field-selector=status.phase=Running",
        "-o", "json",
    ], timeout=max(timeout_s, 1.0))
    if r.returncode != 0:
        return {
            "sample_time": sample_time,
            "sample_age": 0.0,
            "revision": revision,
            "expected_pods": int(expected_pods),
            "scraped_pods": 0,
            "queue_depth": 0.0,
            "queue_depth_source": "none",
            "backlog_metric_pods": 0,
            "request_metric_pods": 0,
            "in_flight_requests": 0.0,
            "collection_status": "failed",
            "queue_proxy_request_rate_by_pod": {},
            "error": f"kubectl_list_pods_failed:{r.stderr.strip()}",
        }
    try:
        pods = json.loads(r.stdout).get("items", [])
    except json.JSONDecodeError:
        pods = []
    ready_pods = [pod for pod in pods if _pod_is_ready(pod)]

    queue_depth_by_pod: dict[str, float] = {}
    in_flight_by_pod: dict[str, float] = {}
    request_metric_pods = 0
    metrics_read_pods = 0
    for pod in ready_pods:
        pod_name = pod.get("metadata", {}).get("name")
        if not pod_name:
            continue
        metrics_text = _read_pod_metrics_text(pod_name, timeout_s)
        if metrics_text is None:
            continue
        metrics_read_pods += 1
        queue_depth = _parse_queue_depth(metrics_text)
        in_flight = _parse_in_flight_requests(metrics_text)
        if queue_depth > 0.0:
            queue_depth_by_pod[pod_name] = queue_depth
        if in_flight > 0.0:
            in_flight_by_pod[pod_name] = in_flight
        if _has_request_metric(metrics_text):
            request_metric_pods += 1

    if queue_depth_by_pod:
        backlog_values = queue_depth_by_pod
        queue_depth = sum(queue_depth_by_pod.values())
        in_flight_total = sum(in_flight_by_pod.values())
        queue_depth_source = "kubectl_queue_depth"
    elif in_flight_by_pod:
        backlog_values = in_flight_by_pod
        queue_depth = sum(in_flight_by_pod.values())
        in_flight_total = queue_depth
        queue_depth_source = "kubectl_in_flight"
    else:
        backlog_values = {}
        queue_depth = 0.0
        in_flight_total = 0.0
        queue_depth_source = "none"

    scraped_pods = len(backlog_values)
    if expected_pods > 0 and scraped_pods >= expected_pods:
        status = "ok"
        error = ""
    elif expected_pods <= 0 and scraped_pods > 0:
        status = "ok"
        error = ""
    elif scraped_pods > 0:
        status = "partial"
        error = f"kubectl_backlog_metric_scraped_{scraped_pods}_of_{expected_pods}"
    elif metrics_read_pods > 0:
        status = "no_backlog_metric"
        error = "kubectl_queue_depth_or_inflight_metric_missing"
    else:
        status = "failed"
        error = "kubectl_queue_proxy_metrics_unavailable"

    return {
        "sample_time": sample_time,
        "sample_age": 0.0,
        "revision": revision,
        "expected_pods": int(expected_pods),
        "scraped_pods": int(scraped_pods),
        "queue_depth": max(0.0, queue_depth),
        "queue_depth_source": queue_depth_source,
        "backlog_metric_pods": int(scraped_pods),
        "request_metric_pods": int(request_metric_pods),
        "in_flight_requests": max(0.0, in_flight_total),
        "collection_status": status,
        "queue_proxy_request_rate_by_pod": {},
        "error": error,
    }


_QUEUE_CACHE = {"value": 0.0, "ts": 0.0}
_QUEUE_TTL_S = 60.0   # refresh at most every 60s; safe under LEAD_S=15 budget


def get_queue_depth() -> float:
    """Return total pod-side queue depth for the latest Knative revision.

    Cached with a 60-s TTL because each refresh enumerates Pods and calls
    kubectl-proxy on every one (1-3 s per refresh). With SCALE_UP_LEAD_S=10
    the controller has ~10 s of decision budget per window; caching keeps
    the per-decision cost low after the first query, leaving room for
    prediction + downtrend correction + kubectl patch.

    The value is best-effort. If metrics are unavailable, return 0 so the
    controller falls back to the original decision behavior.
    """
    if time.time() - _QUEUE_CACHE["ts"] < _QUEUE_TTL_S:
        return _QUEUE_CACHE["value"]
    revision = _get_latest_revision()
    if not revision:
        return 0.0

    r = _run_kubectl([
        "get", "pod", "-n", NAMESPACE,
        "-l", f"serving.knative.dev/revision={revision}",
        "--field-selector=status.phase=Running",
        "-o", "json",
    ])
    if r.returncode != 0:
        print(f"  [queue] list pods failed: {r.stderr.strip()}")
        return 0.0

    try:
        pods = json.loads(r.stdout).get("items", [])
    except json.JSONDecodeError:
        return 0.0

    total_depth = 0.0
    for pod in pods:
        pod_name = pod.get("metadata", {}).get("name")
        if not pod_name:
            continue
        try:
            raw = subprocess.run([
                "sudo", "-S", "k3s", "kubectl", "get", "--raw",
                f"/api/v1/namespaces/{NAMESPACE}/pods/{pod_name}:9090/proxy/metrics",
            ], input=b"llm123456\n", text=False, capture_output=True, timeout=8)
        except subprocess.TimeoutExpired:
            print(f"  [queue] metrics read timed out for pod {pod_name}; using partial queue depth")
            continue
        except subprocess.SubprocessError as exc:
            print(f"  [queue] metrics read failed for pod {pod_name}: {exc}; using partial queue depth")
            continue
        if raw.returncode != 0:
            continue
        try:
            metrics_text = gzip.decompress(raw.stdout).decode()
        except (gzip.BadGzipFile, OSError):
            metrics_text = raw.stdout.decode(errors='replace')
        total_depth += _parse_queue_depth(metrics_text)

    result = max(0.0, total_depth)
    _QUEUE_CACHE["value"] = result
    _QUEUE_CACHE["ts"] = time.time()
    return result


def apply_queue_correction_legacy(q50: float, q90: float, queue_depth: float):
    """Shift q50/q90 by backlog converted to requests per decision window.

    DecisionLayer internally computes q_plan from q50/q90/conf. Adding the
    same correction to both quantiles preserves uncertainty width while making
    N_req queue-aware.
    """
    alpha = float(DECISION_CONFIG.get("queue_alpha", 1.0))
    drain_s = max(float(DECISION_CONFIG.get("queue_drain_s", 30)), 1.0)
    queue_correction = alpha * queue_depth * DECISION_WINDOW_S / drain_s
    return q50 + queue_correction, q90 + queue_correction, queue_correction


QUEUE_SAMPLE_FIELDS = [
    "sample_time",
    "sample_age",
    "revision",
    "expected_pods",
    "scraped_pods",
    "queue_depth",
    "queue_depth_source",
    "pending_requests",
    "planned_requests",
    "due_requests",
    "started_requests",
    "completed_requests",
    "backlog_metric_pods",
    "request_metric_pods",
    "in_flight_requests",
    "collection_status",
    "queue_proxy_request_rate_by_pod",
    "error",
]


WINDOW_DECISION_FIELDS = [
    "logged_at_epoch",
    "window_index",
    "csv_window_id",
    "expected_requests",
    "generated_requests",
    "planned_requests",
    "due_requests",
    "started_requests",
    "completed_requests",
    "decision_available",
    "planning_mode",
    "selected_horizon",
    "target_window",
    "history_mean",
    "q50_raw",
    "q90_raw",
    "q50",
    "q90",
    "confidence",
    "prediction_source",
    "q_plan",
    "N_req",
    "D_smooth",
    "D_fuse",
    "N_target",
    "N_actual",
    "scale_direction",
    "scale_action",
    "scale_in_counter",
    "ready_pods_before",
    "target_pods_before",
    "target_pods_after",
    "pods_before",
    "up_gate_horizon",
    "up_gate_N_req",
    "up_gate_q_plan",
    "up_gate_raw_N_req",
    "up_gate_raw_q_plan",
    "queue_mode",
    "queue_status",
    "queue_usable",
    "queue_sample_age",
    "queue_depth",
    "queue_depth_source",
    "queue_correction",
    "queue_backlog_ready",
    "queue_lead_arrivals",
    "queue_lead_capacity",
    "pending_requests",
    "in_flight_requests",
    "scraped_pods",
    "expected_pods",
    "ci_now",
    "rho_ci",
    "alpha",
    "r_gap",
    "downtrend_active",
    "downtrend_reason",
    "downtrend_previous_mean",
    "downtrend_recent_mean",
    "downtrend_cap",
]


SCALE_EVENT_FIELDS = [
    "event_time_epoch",
    "event_time_utc",
    "window_index",
    "reason",
    "direction",
    "from_target_pods",
    "to_target_pods",
    "ready_pods_before",
]


class WindowDecisionLogger:
    """Append one row per replay window for post-run mismatch analysis."""

    def __init__(self, output_path=None):
        self.output_path = Path(output_path) if output_path else None

    def append(
        self,
        window_index: int,
        window_actual: dict,
        decision: dict | None,
        target_pods_after: int,
    ) -> None:
        if self.output_path is None:
            return
        row = {
            "logged_at_epoch": time.time(),
            "window_index": window_index,
            "csv_window_id": window_actual.get("csv_window_id"),
            "expected_requests": window_actual.get("expected_requests"),
            "generated_requests": window_actual.get("generated_requests"),
            "planned_requests": window_actual.get("planned_requests"),
            "due_requests": window_actual.get("due_requests"),
            "started_requests": window_actual.get("started_requests"),
            "completed_requests": window_actual.get("completed_requests"),
            "decision_available": decision is not None,
            "target_pods_after": target_pods_after,
        }
        if decision is not None:
            result = decision.get("result") or {}
            downtrend = decision.get("downtrend") or {}
            history_arr = decision.get("history_arr")
            history_mean = (
                float(np.mean(history_arr))
                if history_arr is not None and len(history_arr) > 0
                else None
            )
            row.update({
                "planning_mode": decision.get("planning_mode"),
                "selected_horizon": decision.get("selected_horizon"),
                "target_window": decision.get("target_window"),
                "history_mean": history_mean,
                "q50_raw": decision.get("q50_raw"),
                "q90_raw": decision.get("q90_raw"),
                "q50": decision.get("q50"),
                "q90": decision.get("q90"),
                "confidence": decision.get("conf"),
                "prediction_source": decision.get("prediction_source"),
                "q_plan": result.get("q_plan"),
                "N_req": result.get("N_req"),
                "D_smooth": result.get("D_smooth"),
                "D_fuse": result.get("D_fuse"),
                "N_target": result.get("N_target"),
                "N_actual": decision.get("N_actual"),
                "scale_direction": result.get("scale_direction"),
                "scale_action": decision.get("scale_action"),
                "scale_in_counter": result.get("scale_in_counter"),
                "ready_pods_before": decision.get("ready_pods_before"),
                "target_pods_before": decision.get("target_pods_before"),
                "pods_before": decision.get("pods_before"),
                "up_gate_horizon": decision.get("up_gate_horizon"),
                "up_gate_N_req": decision.get("up_gate_N_req"),
                "up_gate_q_plan": decision.get("up_gate_q_plan"),
                "up_gate_raw_N_req": decision.get("up_gate_raw_N_req"),
                "up_gate_raw_q_plan": decision.get("up_gate_raw_q_plan"),
                "queue_mode": decision.get("queue_mode"),
                "queue_status": decision.get("queue_status"),
                "queue_usable": decision.get("queue_usable"),
                "queue_sample_age": decision.get("queue_sample_age"),
                "queue_depth": decision.get("queue_depth"),
                "queue_depth_source": decision.get("queue_depth_source"),
                "queue_correction": decision.get("queue_correction"),
                "queue_backlog_ready": decision.get("queue_backlog_ready"),
                "queue_lead_arrivals": decision.get("queue_lead_arrivals"),
                "queue_lead_capacity": decision.get("queue_lead_capacity"),
                "pending_requests": decision.get("pending_requests"),
                "in_flight_requests": decision.get("in_flight_requests"),
                "scraped_pods": decision.get("scraped_pods"),
                "expected_pods": decision.get("expected_pods"),
                "ci_now": result.get("ci_now"),
                "rho_ci": result.get("rho_ci"),
                "alpha": result.get("alpha"),
                "r_gap": result.get("r_gap"),
                "downtrend_active": downtrend.get("active"),
                "downtrend_reason": downtrend.get("reason"),
                "downtrend_previous_mean": downtrend.get("previous_mean"),
                "downtrend_recent_mean": downtrend.get("recent_mean"),
                "downtrend_cap": downtrend.get("cap"),
            })
        else:
            row.update({
                "planning_mode": "bootstrap",
                "scale_action": "bootstrap-hold",
            })
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = (
            not self.output_path.exists()
            or self.output_path.stat().st_size == 0
        )
        with self.output_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=WINDOW_DECISION_FIELDS,
                extrasaction="ignore",
            )
            if write_header:
                writer.writeheader()
            writer.writerow(row)


class ScaleEventLogger:
    """Append one row for each controller-issued scale action."""

    def __init__(self, output_path=None):
        self.output_path = Path(output_path) if output_path else None

    def append(
        self,
        *,
        event_time_epoch: float,
        window_index: int | None,
        reason: str,
        direction: str,
        from_target_pods: int,
        to_target_pods: int,
        ready_pods_before: int | None,
    ) -> None:
        if self.output_path is None:
            return
        row = {
            "event_time_epoch": event_time_epoch,
            "event_time_utc": datetime.fromtimestamp(event_time_epoch, timezone.utc).isoformat(),
            "window_index": window_index,
            "reason": reason,
            "direction": direction,
            "from_target_pods": from_target_pods,
            "to_target_pods": to_target_pods,
            "ready_pods_before": ready_pods_before,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = (
            not self.output_path.exists()
            or self.output_path.stat().st_size == 0
        )
        with self.output_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SCALE_EVENT_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


class QueueMetricsCache:
    """Small in-process cache fed by a background Prometheus sampler."""

    def __init__(self, output_path=None):
        self.latest = None
        self.output_path = Path(output_path) if output_path else None
        self._writer_ready = False

    def update(self, sample: dict) -> None:
        sample = dict(sample)
        sample.setdefault("sample_time", time.time())
        sample["sample_age"] = max(0.0, time.time() - float(sample["sample_time"]))
        self.latest = sample
        self._append_sample(sample)

    def current(self, max_age_s: float) -> dict:
        if not self.latest:
            return {
                "sample_time": None,
                "sample_age": None,
                "revision": None,
                "expected_pods": 0,
                "scraped_pods": 0,
                "queue_depth": 0.0,
                "queue_depth_source": "none",
                "backlog_metric_pods": 0,
                "request_metric_pods": 0,
                "in_flight_requests": 0.0,
                "collection_status": "failed",
                "queue_proxy_request_rate_by_pod": {},
                "error": "no_queue_sample",
                "usable": False,
            }
        sample = dict(self.latest)
        sample["sample_age"] = max(0.0, time.time() - float(sample["sample_time"]))
        sample["usable"] = (
            sample.get("collection_status") == "ok"
            and sample["sample_age"] <= max_age_s
        )
        return sample

    def _append_sample(self, sample: dict) -> None:
        if self.output_path is None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        row = {field: sample.get(field) for field in QUEUE_SAMPLE_FIELDS}
        row["queue_proxy_request_rate_by_pod"] = json.dumps(
            row.get("queue_proxy_request_rate_by_pod") or {},
            sort_keys=True,
        )
        write_header = not self.output_path.exists() or self.output_path.stat().st_size == 0
        with self.output_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=QUEUE_SAMPLE_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def _prometheus_query_vector(prometheus_url: str, query: str, timeout_s: float) -> list:
    endpoint = prometheus_url.rstrip("/") + "/api/v1/query"
    params = urllib.parse.urlencode({"query": query})
    req = urllib.request.Request(
        endpoint + "?" + params,
        headers={"Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("status") != "success":
        raise RuntimeError(f"Prometheus query failed: {payload!r}")
    data = payload.get("data") or {}
    return data.get("result") or []


def _vector_total(result: list) -> float:
    total = 0.0
    for item in result:
        value = (item.get("value") or [None, "0"])[1]
        try:
            total += float(value)
        except (TypeError, ValueError):
            continue
    return max(0.0, total)


def _vector_pod_values(result: list) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in result:
        metric = item.get("metric") or {}
        pod = metric.get("pod") or metric.get("pod_name")
        if not pod:
            continue
        value = (item.get("value") or [None, "0"])[1]
        try:
            values[str(pod)] = float(value)
        except (TypeError, ValueError):
            values[str(pod)] = 0.0
    return values


def _promql_regex_literal(value: str) -> str:
    # PromQL string literals reject Python re.escape() output such as "\-".
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def collect_queue_prometheus_sample(
    prometheus_url: str,
    revision,
    expected_pods: int,
    query_timeout_s: float,
) -> dict:
    sample_time = time.time()
    revision_filter = _promql_regex_literal(revision or "")
    revision_clause = f', revision=~"{revision_filter}"' if revision_filter else ""
    pod_clause = f'namespace="{NAMESPACE}", pod=~"{SERVICE_NAME}.*"'
    queue_query = (
        "sum by (pod) "
        f'({{__name__=~".*(queue.*depth|depth.*queue).*", {pod_clause}{revision_clause}}})'
    )
    in_flight_query = (
        "sum by (pod) "
        f'({{__name__=~".*(inflight|in_flight|concurrent.*request|request.*concurrent).*", '
        f'{pod_clause}{revision_clause}}})'
    )
    request_query = (
        "sum by (pod) "
        f'(rate({{__name__=~".*request.*", container="queue-proxy", {pod_clause}}}'
        f'[{int(DECISION_WINDOW_S)}s]))'
    )
    try:
        queue_vector = _prometheus_query_vector(prometheus_url, queue_query, query_timeout_s)
        in_flight_vector = _prometheus_query_vector(prometheus_url, in_flight_query, query_timeout_s)
        request_vector = _prometheus_query_vector(prometheus_url, request_query, query_timeout_s)
        queue_values = _vector_pod_values(queue_vector)
        in_flight_values = _vector_pod_values(in_flight_vector)
        request_values = _vector_pod_values(request_vector)

        if queue_values:
            backlog_values = queue_values
            queue_depth = _vector_total(queue_vector)
            queue_depth_source = "queue_depth"
        elif in_flight_values:
            backlog_values = in_flight_values
            queue_depth = _vector_total(in_flight_vector)
            queue_depth_source = "in_flight"
        else:
            backlog_values = {}
            queue_depth = 0.0
            queue_depth_source = "none"

        scraped_pods = len(backlog_values)
        if expected_pods > 0 and scraped_pods >= expected_pods:
            status = "ok"
            error = ""
        elif expected_pods <= 0 and scraped_pods > 0:
            status = "ok"
            error = ""
        elif scraped_pods > 0:
            status = "partial"
            error = f"backlog_metric_scraped_{scraped_pods}_of_{expected_pods}"
        elif request_values:
            status = "no_backlog_metric"
            error = "queue_depth_or_inflight_metric_missing"
        else:
            status = "failed"
            error = "no_qwen_queue_proxy_metrics"
        prometheus_sample = {
            "sample_time": sample_time,
            "sample_age": 0.0,
            "revision": revision,
            "expected_pods": int(expected_pods),
            "scraped_pods": int(scraped_pods),
            "queue_depth": queue_depth,
            "queue_depth_source": queue_depth_source,
            "backlog_metric_pods": int(scraped_pods),
            "request_metric_pods": int(len(request_values)),
            "in_flight_requests": _vector_total(in_flight_vector),
            "collection_status": status,
            "queue_proxy_request_rate_by_pod": request_values,
            "error": error,
        }
        if status == "ok":
            return prometheus_sample
        kubectl_sample = collect_queue_kubectl_sample(revision, expected_pods, query_timeout_s)
        if kubectl_sample.get("collection_status") != "failed":
            prior = (
                f"prometheus_status={status};"
                f"prometheus_error={error or 'none'};"
                f"prometheus_request_pods={len(request_values)}"
            )
            fallback_error = kubectl_sample.get("error") or ""
            kubectl_sample["error"] = prior if not fallback_error else f"{fallback_error};{prior}"
            return kubectl_sample
        prometheus_sample["error"] = (
            f"{error};kubectl_fallback={kubectl_sample.get('error')}"
            if error
            else f"kubectl_fallback={kubectl_sample.get('error')}"
        )
        return prometheus_sample
    except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        return {
            "sample_time": sample_time,
            "sample_age": 0.0,
            "revision": revision,
            "expected_pods": int(expected_pods),
            "scraped_pods": 0,
            "queue_depth": 0.0,
            "queue_depth_source": "none",
            "backlog_metric_pods": 0,
            "request_metric_pods": 0,
            "in_flight_requests": 0.0,
            "collection_status": "failed",
            "queue_proxy_request_rate_by_pod": {},
            "error": str(exc),
        }


async def queue_prometheus_sampler(args, queue_cache: QueueMetricsCache, expected_pods_getter) -> None:
    period_s = max(float(args.queue_sample_period_seconds), 1.0)
    query_timeout_s = max(float(args.queue_query_timeout_seconds), 0.5)
    while True:
        revision = await asyncio.to_thread(_get_latest_revision)
        expected_pods = max(int(expected_pods_getter()), 0)
        sample = await asyncio.to_thread(
            collect_queue_prometheus_sample,
            args.prometheus_url,
            revision,
            expected_pods,
            query_timeout_s,
        )
        queue_cache.update(sample)
        if sample.get("collection_status") != "ok":
            print(
                "  [queue-sampler] "
                f"status={sample.get('collection_status')} "
                f"scraped={sample.get('scraped_pods')}/{sample.get('expected_pods')} "
                f"source={sample.get('queue_depth_source')} "
                f"request_pods={sample.get('request_metric_pods')} "
                f"error={sample.get('error')}",
                flush=True,
            )
        await asyncio.sleep(period_s)


def collect_queue_replayer_sample(runtime_state: ReplayRuntimeState, expected_pods: int) -> dict:
    """Sample backlog from workload_replayer's own request lifecycle counters."""
    sample_time = time.time()
    snapshot = runtime_state.snapshot()
    queue_depth = float(snapshot["active_requests"])
    expected_pods = max(int(expected_pods), 0)
    return {
        "sample_time": sample_time,
        "sample_age": 0.0,
        "revision": "workload_replayer",
        "expected_pods": expected_pods,
        "scraped_pods": expected_pods,
        "queue_depth": queue_depth,
        "queue_depth_source": "replayer_pending_inflight",
        "pending_requests": float(snapshot["pending_requests"]),
        "planned_requests": int(snapshot["planned_requests"]),
        "due_requests": int(snapshot["due_requests"]),
        "started_requests": int(snapshot["started_requests"]),
        "completed_requests": int(snapshot["completed_requests"]),
        "backlog_metric_pods": 0,
        "request_metric_pods": 0,
        "in_flight_requests": float(snapshot["in_flight_requests"]),
        "collection_status": "ok",
        "queue_proxy_request_rate_by_pod": {},
        "error": "",
    }


async def queue_replayer_sampler(
    args,
    queue_cache: QueueMetricsCache,
    expected_pods_getter,
    runtime_state: ReplayRuntimeState,
) -> None:
    period_s = max(float(args.queue_sample_period_seconds), 1.0)
    while True:
        expected_pods = max(int(expected_pods_getter()), 0)
        queue_cache.update(collect_queue_replayer_sample(runtime_state, expected_pods))
        await asyncio.sleep(period_s)


def _blend_prediction(q50: float, q90: float, confidence: float) -> float:
    c = float(np.clip(confidence, 0.0, 1.0))
    return c * float(q50) + (1.0 - c) * float(q90)


def _lead_arrival_rate(pred: dict, history_arr) -> float:
    recent = np.asarray(history_arr[-3:], dtype=np.float32).reshape(-1)
    history_rate = float(np.mean(recent) / DECISION_WINDOW_S) if len(recent) else 0.0
    try:
        q50_lead, q90_lead, conf_lead = get_planning_prediction(pred, 1)
        prediction_rate = max(
            _blend_prediction(q50_lead, q90_lead, conf_lead),
            0.0,
        ) / DECISION_WINDOW_S
    except KeyError:
        prediction_rate = 0.0
    return max(history_rate, prediction_rate, 0.0)


def build_queue_context(
    pred: dict,
    history_arr,
    current_pods: int,
    queue_sample: dict,
    queue_mode: str,
    ready_latency_s: float,
) -> dict:
    status = queue_sample.get("collection_status", "failed")
    sample_age = queue_sample.get("sample_age")
    queue_depth = float(queue_sample.get("queue_depth") or 0.0)
    context = {
        "queue_mode": queue_mode,
        "queue_status": status,
        "queue_sample_age": sample_age,
        "queue_depth": queue_depth,
        "queue_depth_source": queue_sample.get("queue_depth_source") or "none",
        "pending_requests": float(queue_sample.get("pending_requests") or 0.0),
        "in_flight_requests": float(queue_sample.get("in_flight_requests") or 0.0),
        "scraped_pods": int(queue_sample.get("scraped_pods") or 0),
        "expected_pods": int(queue_sample.get("expected_pods") or 0),
        "queue_correction": 0.0,
        "queue_backlog_ready": 0.0,
        "queue_lead_arrivals": 0.0,
        "queue_lead_capacity": 0.0,
        "queue_usable": bool(queue_sample.get("usable")),
    }
    if queue_mode == "none" or not context["queue_usable"]:
        return context

    alpha = float(DECISION_CONFIG.get("queue_alpha", 1.0))
    drain_s = max(float(DECISION_CONFIG.get("queue_drain_s", DECISION_WINDOW_S)), 1.0)
    if queue_mode == "current":
        backlog_for_planning = queue_depth
    else:
        lead_s = max(float(ready_latency_s), 0.0)
        lead_rate = _lead_arrival_rate(pred, history_arr)
        lead_arrivals = lead_rate * lead_s
        mu_per_second = max(float(DECISION_CONFIG["mu"]), 1e-9) / DECISION_WINDOW_S
        lead_capacity = max(int(current_pods), 0) * mu_per_second * lead_s
        backlog_for_planning = max(0.0, queue_depth + lead_arrivals - lead_capacity)
        context["queue_lead_arrivals"] = lead_arrivals
        context["queue_lead_capacity"] = lead_capacity
    context["queue_backlog_ready"] = backlog_for_planning
    context["queue_correction"] = alpha * backlog_for_planning * DECISION_WINDOW_S / drain_s
    return context


def apply_queue_correction(q50: float, q90: float, queue_context: dict):
    """Shift q50/q90 by the valid queue backlog correction, if available."""
    queue_correction = float(queue_context.get("queue_correction") or 0.0)
    return q50 + queue_correction, q90 + queue_correction, queue_correction


def get_planning_prediction(pred: dict, horizon: int = PREDICTION_HORIZON):
    """Return q50/q90/confidence for the controller planning horizon."""
    q50_key = f"q50_horizon_{horizon}"
    q90_key = f"q90_horizon_{horizon}"
    conf_key = f"confidence_horizon_{horizon}"
    try:
        return pred[q50_key], pred[q90_key], pred[conf_key]
    except KeyError as exc:
        available = ", ".join(sorted(k for k in pred if "horizon" in k))
        raise KeyError(
            f"Predictor output does not contain horizon {horizon}; "
            f"available horizon fields: {available}"
        ) from exc


def estimate_required_pods(q50: float, q90: float, confidence: float):
    """Mirror DecisionLayer's Q_plan -> N_req mapping without mutating history."""
    c = float(np.clip(confidence, 0.0, 1.0))
    q_plan = c * float(q50) + (1.0 - c) * float(q90)
    return estimate_required_pods_from_q_plan(q_plan)


def estimate_required_pods_from_q_plan(q_plan: float):
    """Map a request-count plan to the pod count used by DecisionLayer."""
    mu = max(float(DECISION_CONFIG["mu"]), 1e-9)
    raw = int(np.ceil(max(q_plan, 0.0) / mu))
    n_req = int(np.clip(raw, int(DECISION_CONFIG["N_min"]), int(DECISION_CONFIG["N_max"])))
    return n_req, q_plan


def compute_scale_up_q_plan(q50: float, q90: float, confidence: float):
    """Scale-up demand based on confidence-weighted prediction."""
    c = float(np.clip(confidence, 0.0, 1.0))
    q_plan = c * float(q50) + (1.0 - c) * float(q90)
    margin = max(float(DECISION_CONFIG.get("scale_up_safety_margin", 1.0)), 1.0)
    return q_plan * margin


def _downtrend_state(history_arr, queue_depth: float):
    recent_n = int(DECISION_CONFIG.get("downtrend_recent_windows", 3))
    previous_n = int(DECISION_CONFIG.get("downtrend_previous_windows", 3))
    min_len = recent_n + previous_n
    state = {
        "active": False,
        "reason": "not_enough_history",
        "recent_mean": None,
        "previous_mean": None,
        "cap": None,
    }

    if len(history_arr) < min_len:
        return state

    queue_threshold = float(DECISION_CONFIG.get("downtrend_queue_threshold", 0.0))
    if float(queue_depth) > queue_threshold:
        state["reason"] = "queue_not_empty"
        return state

    arr = np.asarray(history_arr, dtype=np.float32).reshape(-1)
    previous = arr[-min_len:-recent_n]
    recent = arr[-recent_n:]
    previous_mean = float(np.mean(previous))
    recent_mean = float(np.mean(recent))
    state["recent_mean"] = recent_mean
    state["previous_mean"] = previous_mean

    ratio = float(DECISION_CONFIG.get("downtrend_ratio", 0.75))
    min_drop = float(DECISION_CONFIG.get("downtrend_min_drop", 30.0))
    if previous_mean <= 0.0:
        state["reason"] = "previous_mean_zero"
        return state
    if recent_mean > previous_mean * ratio:
        state["reason"] = "drop_ratio_too_small"
        return state
    if previous_mean - recent_mean < min_drop:
        state["reason"] = "drop_abs_too_small"
        return state

    margin = float(DECISION_CONFIG.get("downtrend_margin", 1.15))
    last_value = float(recent[-1])
    state["active"] = True
    state["reason"] = "active"
    state["cap"] = max(last_value * margin, recent_mean * margin)
    return state


def apply_downtrend_correction(q50: float, q90: float, history_arr, queue_depth: float):
    """Cap scale-down-only quantiles when recent demand has clearly fallen."""
    if not bool(DECISION_CONFIG.get("downtrend_enabled", True)):
        # report design has no downtrend cap; keep quantiles untouched.
        state = {
            "active": False, "reason": "disabled",
            "recent_mean": None, "previous_mean": None, "cap": None,
            "q50_before": float(q50), "q90_before": float(q90),
            "q50_after": float(q50), "q90_after": float(q90),
        }
        return q50, q90, state
    state = _downtrend_state(history_arr, queue_depth)
    if not state["active"]:
        state["q50_before"] = float(q50)
        state["q90_before"] = float(q90)
        state["q50_after"] = float(q50)
        state["q90_after"] = float(q90)
        return q50, q90, state

    cap = float(state["cap"])
    q50_new = min(float(q50), cap)
    q90_new = min(float(q90), cap)
    q90_new = max(q90_new, q50_new)
    state["q50_before"] = float(q50)
    state["q90_before"] = float(q90)
    state["q50_after"] = q50_new
    state["q90_after"] = q90_new
    return q50_new, q90_new, state


def _prediction_candidate(pred: dict, horizon: int, queue_context: dict):
    q50_raw, q90_raw, conf = get_planning_prediction(pred, horizon)
    q50, q90, queue_correction = apply_queue_correction(q50_raw, q90_raw, queue_context)
    n_req, q_plan = estimate_required_pods(q50, q90, conf)
    q_plan_up = compute_scale_up_q_plan(q50, q90, conf)
    n_req_up, _ = estimate_required_pods_from_q_plan(q_plan_up)
    result = {
        "horizon": horizon,
        "q50_raw": q50_raw,
        "q90_raw": q90_raw,
        "q50": q50,
        "q90": q90,
        "conf": conf,
        "prediction_source": pred.get("prediction_source", "model"),
        "queue_correction": queue_correction,
        "N_req": n_req,
        "q_plan": q_plan,
        "N_req_up": n_req_up,
        "q_plan_up": q_plan_up,
    }
    result.update(queue_context)
    return result


def prepare_prediction_for_decision(
    pred: dict,
    history_arr,
    current_pods: int,
    queue_sample: dict,
    queue_mode: str,
    ready_latency_s: float,
):
    """Use raw near/far horizons for scale-up, and corrected near horizon for scale-down."""
    scale_down_horizon = int(DECISION_CONFIG.get("scale_down_horizon", 1))
    horizons = sorted({1, PREDICTION_HORIZON, scale_down_horizon})
    queue_context = build_queue_context(
        pred=pred,
        history_arr=history_arr,
        current_pods=current_pods,
        queue_sample=queue_sample,
        queue_mode=queue_mode,
        ready_latency_s=ready_latency_s,
    )
    candidates = {
        horizon: _prediction_candidate(pred, horizon, queue_context)
        for horizon in horizons
    }

    up_candidate = max(candidates.values(), key=lambda item: (item["N_req_up"], item["q_plan_up"]))
    selected = dict(up_candidate)
    selected["mode"] = "scale-up-raw"
    selected["up_gate_horizon"] = up_candidate["horizon"]
    selected["up_gate_N_req"] = up_candidate["N_req_up"]
    selected["up_gate_q_plan"] = up_candidate["q_plan_up"]
    selected["up_gate_raw_N_req"] = up_candidate["N_req"]
    selected["up_gate_raw_q_plan"] = up_candidate["q_plan"]
    selected["q_plan_for_decision"] = up_candidate["q_plan_up"]
    selected["downtrend"] = {
        "active": False,
        "reason": "skipped_scale_up_gate",
        "recent_mean": None,
        "previous_mean": None,
        "cap": None,
        "q50_before": selected["q50"],
        "q90_before": selected["q90"],
        "q50_after": selected["q50"],
        "q90_after": selected["q90"],
    }

    if up_candidate["N_req_up"] > current_pods:
        return selected

    selected = dict(candidates[scale_down_horizon])
    selected["mode"] = "scale-down-near-horizon"
    selected["up_gate_horizon"] = up_candidate["horizon"]
    selected["up_gate_N_req"] = up_candidate["N_req_up"]
    selected["up_gate_q_plan"] = up_candidate["q_plan_up"]
    selected["up_gate_raw_N_req"] = up_candidate["N_req"]
    selected["up_gate_raw_q_plan"] = up_candidate["q_plan"]
    selected["q_plan_for_decision"] = None
    q50_corr, q90_corr, downtrend = apply_downtrend_correction(
        selected["q50"],
        selected["q90"],
        history_arr,
        selected["queue_backlog_ready"],
    )
    selected["q50"] = q50_corr
    selected["q90"] = q90_corr
    selected["downtrend"] = downtrend
    return selected


def prepare_direct_transformer_prediction(pred: dict):
    """Strict Transformer direct baseline: Q_base -> ceil(Q_base / mu)."""
    q50_raw, q90_raw, conf = get_planning_prediction(pred, PREDICTION_HORIZON)
    n_req, q_plan = estimate_required_pods(q50_raw, q90_raw, conf)
    downtrend = {
        "active": False,
        "reason": "strict_transformer_direct",
        "recent_mean": None,
        "previous_mean": None,
        "cap": None,
        "q50_before": float(q50_raw),
        "q90_before": float(q90_raw),
        "q50_after": float(q50_raw),
        "q90_after": float(q90_raw),
    }
    return {
        "horizon": PREDICTION_HORIZON,
        "q50_raw": q50_raw,
        "q90_raw": q90_raw,
        "q50": q50_raw,
        "q90": q90_raw,
        "conf": conf,
        "prediction_source": pred.get("prediction_source", "model"),
        "queue_depth": 0.0,
        "queue_depth_source": "none",
        "queue_correction": 0.0,
        "queue_mode": "none",
        "queue_status": "disabled",
        "queue_sample_age": None,
        "queue_backlog_ready": 0.0,
        "queue_lead_arrivals": 0.0,
        "queue_lead_capacity": 0.0,
        "queue_usable": False,
        "pending_requests": 0.0,
        "in_flight_requests": 0.0,
        "scraped_pods": 0,
        "expected_pods": 0,
        "N_req": n_req,
        "q_plan": q_plan,
        "N_req_up": n_req,
        "q_plan_up": q_plan,
        "mode": "transformer-direct",
        "up_gate_horizon": PREDICTION_HORIZON,
        "up_gate_N_req": n_req,
        "up_gate_q_plan": q_plan,
        "up_gate_raw_N_req": n_req,
        "up_gate_raw_q_plan": q_plan,
        "q_plan_for_decision": q_plan,
        "downtrend": downtrend,
    }


def direct_transformer_result(prepared: dict, current_pods: int, current_time, ci_loader):
    """Return a DecisionLayer-shaped result without smoothing/carbon control."""
    n_req = int(prepared["N_req"])
    n_actual = int(np.clip(n_req, int(DECISION_CONFIG["N_min"]), int(DECISION_CONFIG["N_max"])))
    if n_actual > current_pods:
        direction = "up"
    elif n_actual < current_pods:
        direction = "down"
    else:
        direction = "hold"

    ci_now = ci_loader.lookup(current_time)
    rho_ci = ci_loader.normalized(ci_now)
    r_gap = float((current_pods - n_req) / max(current_pods, 1)) if n_req < current_pods else 0.0
    return {
        "N_actual": n_actual,
        "N_req": n_req,
        "D_smooth": float(n_req),
        "D_fuse": n_req,
        "N_target": n_req,
        "alpha": 0.0,
        "r_gap": round(r_gap, 4),
        "rho_ci": round(rho_ci, 4),
        "ci_now": round(ci_now, 2),
        "q_plan": round(float(prepared["q_plan"]), 3),
        "q50_input": round(float(prepared["q50"]), 3),
        "q90_input": round(float(prepared["q90"]), 3),
        "confidence": round(float(prepared["conf"]), 4),
        "scale_direction": direction,
        "scale_in_counter": 0,
    }


def scale_up_schedule(window_start: float, selected_horizon: int, args):
    """Return when to issue scale-up so capacity is Ready by the target window."""
    horizon = max(int(selected_horizon), 1)
    ready_latency_s = max(float(args.ready_latency_p95_seconds), 0.0)
    slack_s = max(float(getattr(args, "scale_up_slack_seconds", 0.0)), 0.0)
    target_window_start = window_start + (horizon - 1) * DECISION_WINDOW_S
    apply_at = target_window_start - ready_latency_s - slack_s
    return apply_at, target_window_start, ready_latency_s, slack_s


def _ci_lookup_time(args, window_index: int = 0):
    """Return the datetime to use for CI trace lookup.

    If --ci-trace-start is set, map window_index linearly into the CI trace
    (synthetic time). Otherwise fall back to real-time.

    This solves the 2026-experiment-with-2025-CI-CSV problem: real-time
    fell outside the CI range and lookup() clamped to the last row,
    freezing CI as a constant.
    """
    from datetime import timedelta
    start_str = getattr(args, "ci_trace_start", None)
    if not start_str:
        return datetime.now(timezone.utc)
    scale = getattr(args, "ci_time_scale", None)
    if scale is not None:
        elapsed_s = window_index * DECISION_WINDOW_S * scale
    else:
        win_s = getattr(args, "ci_window_seconds", None) or DECISION_WINDOW_S
        elapsed_s = window_index * win_s
    start = datetime.fromisoformat(start_str)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    return start + timedelta(seconds=elapsed_s)


def make_prediction_decision(predictor, layer, args, history_arr,
                              current_pods: int, window_index: int = 0,
                              queue_cache=None):
    if getattr(args, "oracle_predictor", False):
        pred = build_oracle_prediction(args, window_index)
    else:
        pred = predictor.predict(history_arr)
    now = _ci_lookup_time(args, window_index=window_index)
    if getattr(args, "decision_mode", "proposed") == "direct":
        prepared = prepare_direct_transformer_prediction(pred)
        r = direct_transformer_result(prepared, current_pods, now, layer.ci_loader)
        return prepared, r, now

    queue_mode = "none" if args.disable_queue else getattr(args, "queue_mode", "ready")
    if queue_mode == "none":
        queue_sample = {
            "collection_status": "disabled",
            "queue_depth": 0.0,
            "queue_depth_source": "none",
            "sample_age": None,
            "usable": False,
        }
    elif queue_cache is not None:
        queue_sample = queue_cache.current(float(args.queue_max_sample_age_seconds))
    else:
        queue_sample = {
            "collection_status": "failed",
            "queue_depth": 0.0,
            "queue_depth_source": "none",
            "sample_age": None,
            "usable": False,
            "error": "queue_sampler_not_started",
        }
    prepared = prepare_prediction_for_decision(
        pred=pred,
        history_arr=history_arr,
        current_pods=current_pods,
        queue_sample=queue_sample,
        queue_mode=queue_mode,
        ready_latency_s=float(args.ready_latency_p95_seconds),
    )
    r = layer.step(
        q50=prepared["q50"],
        q90=prepared["q90"],
        confidence=prepared["conf"],
        current_pods=current_pods,
        current_time=now,
        q_plan=prepared["q_plan_for_decision"],
    )
    return prepared, r, now


def get_current_pods() -> int:
    revision = _get_latest_revision()
    if not revision:
        return -1

    r = _run_kubectl([
        "get", "pod",
        "-n", NAMESPACE, "-l", f"serving.knative.dev/revision={revision}",
        "--field-selector=status.phase=Running",
        "-o", "json"
    ], timeout=15)
    if r.returncode != 0:
        output = f"{r.stdout or ''}{r.stderr or ''}"
        if _is_kube_api_unavailable(output):
            raise KubeApiUnavailable(output.strip())
        return -1
    try:
        pods = json.loads(r.stdout).get("items", [])
    except json.JSONDecodeError as exc:
        print(f"  [pods] kubectl get pod returned invalid JSON: {exc}")
        return -1
    ready = sum(
        1 for p in pods
        if any(
            c.get("type") == "Ready" and c.get("status") == "True"
            for c in p.get("status", {}).get("conditions", [])
        )
    )
    return ready


# ============================================
# Workload trace input.

def _trace_request_value(row: dict) -> float:
    for column in ("requests_per_60s", "requests_per_20s", "requests", "request_count"):
        value = row.get(column)
        if value not in (None, ""):
            return float(value)
    raise KeyError("trace row has no request count column")


class TraceQPSReader:
    def __init__(self, csv_path: str, start_row: int = 0):
        self._rows = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._rows.append({
                    "ts": int(row["timestamp"]),
                    "window_id": row["window_id"],
                    "qps": _trace_request_value(row),
                })
        self._start_row = int(start_row) % len(self._rows)
        self._idx = self._start_row
        print(f"[TraceQPS] Loaded {len(self._rows)} windows, start_row={start_row}")
        print(f"  Range: window {self._rows[0]['window_id']} to {self._rows[-1]['window_id']}")
        print(f"  Start: window {self._rows[self._start_row]['window_id']}")
        print(f"  QPS range: {min(r['qps'] for r in self._rows):.0f} to "
              f"{max(r['qps'] for r in self._rows):.0f} req/{DECISION_WINDOW_S:.0f}s")

    def get_qps(self, elapsed_seconds: float) -> float:
        window_offset = int(elapsed_seconds / DECISION_WINDOW_S)
        idx = (self._start_row + window_offset) % len(self._rows)
        self._idx = idx
        return self._rows[idx]["qps"]

    @property
    def current_window_id(self) -> int:
        return self._rows[self._idx]["window_id"]


def _load_trace_rows(csv_path: str) -> list:
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(_trace_request_value(row))
    return rows


def _oracle_trace_values(args) -> list:
    trace_path = str(getattr(args, "oracle_prediction_trace", None) or args.trace)
    cached_path = getattr(args, "_oracle_prediction_trace_path", None)
    cached_values = getattr(args, "_oracle_prediction_trace_values", None)
    if cached_path == trace_path and cached_values is not None:
        return cached_values
    values = _load_trace_rows(trace_path)
    if not values:
        raise RuntimeError(f"oracle prediction trace is empty: {trace_path}")
    args._oracle_prediction_trace_path = trace_path
    args._oracle_prediction_trace_values = values
    print(f"[OraclePredictor] Loaded {len(values)} windows from {trace_path}")
    return values


def build_oracle_prediction(args, window_index: int) -> dict:
    """Return actual[t+1] and actual[t+2] using the replay trace as an oracle."""
    values = _oracle_trace_values(args)
    start_row = int(getattr(args, "start_row", 0))
    pad = max(float(getattr(args, "oracle_interval_pad", 0.0)), 0.0)
    confidence = float(np.clip(getattr(args, "oracle_confidence", 1.0), 0.0, 1.0))

    q50_values = []
    source_indexes = {}
    for horizon in range(1, PREDICTION_HORIZON + 1):
        trace_index = (start_row + int(window_index) + horizon - 1) % len(values)
        q50 = max(float(values[trace_index]), 0.0)
        q50_values.append(q50)
        source_indexes[f"horizon_{horizon}"] = trace_index

    q50_arr = np.asarray(q50_values, dtype=np.float32)
    q10_arr = np.maximum(q50_arr - pad, 0.0).astype(np.float32)
    q90_arr = (q50_arr + pad).astype(np.float32)
    conf_arr = np.full(PREDICTION_HORIZON, confidence, dtype=np.float32)

    pred = {
        "q10": q10_arr,
        "q50": q50_arr,
        "q90": q90_arr,
        "confidence": conf_arr,
        "q10_next": float(q10_arr[0]),
        "q50_next": float(q50_arr[0]),
        "q90_next": float(q90_arr[0]),
        "confidence_next": float(conf_arr[0]),
        "prediction_source": "oracle",
        "oracle_source_indexes": source_indexes,
    }
    for offset, horizon in enumerate(range(1, PREDICTION_HORIZON + 1)):
        pred[f"q10_horizon_{horizon}"] = float(q10_arr[offset])
        pred[f"q50_horizon_{horizon}"] = float(q50_arr[offset])
        pred[f"q90_horizon_{horizon}"] = float(q90_arr[offset])
        pred[f"confidence_horizon_{horizon}"] = float(conf_arr[offset])
    return pred


# ============================================
# Live control with workload replay.

async def run_controller_live(predictor, layer, args):
    """Run live controller synchronized with workload_replayer."""
    lookback = predictor.lookback
    qps_history: deque = deque(maxlen=lookback)
    bootstrap_target_pods = int(
        getattr(args, "bootstrap_target_pods", None) or DECISION_CONFIG["N_min"]
    )

    # Initialize forecasting history from the trace.
    trace_rows = _load_trace_rows(args.trace)
    start_row = getattr(args, "start_row", 0)
    start_window = int(start_row)
    for i in range(lookback):
        qps_history.append(trace_rows[(start_window - lookback + i) % len(trace_rows)])

    # Read the initial replica state.
    if args.dry_run:
        current_pods = bootstrap_target_pods
    else:
        current_pods = get_current_pods()
        if current_pods < 0:
            current_pods = bootstrap_target_pods

    print(f"\nInitial state: current_pods={current_pods}")
    print("=" * 60)
    print(f"Starting live control loop (every {DECISION_WINDOW_S}s)...")
    print("=" * 60)

    # Queue completed replay-window observations.
    queue: asyncio.Queue = asyncio.Queue()

    async def observe_window(schedule):
        await queue.put(dict(schedule))

    # Configure workload replay.
    run_id = args.run_id or make_run_id(args.baseline_name or "transformer")
    replay_config = ReplayConfig(
        csv=args.trace,
        url=getattr(args, "url", DEFAULT_URL),
        host=getattr(args, "host", DEFAULT_HOST),
        window_size_seconds=DECISION_WINDOW_S,
        duration_minutes=args.duration_minutes,
        start_row=start_row,
        start_delay_seconds=args.start_delay,
        baseline_name=args.baseline_name or "transformer",
        output_dir=args.output_dir or str(DEFAULT_OUTPUT_DIR),
        run_id=run_id,
        seed=args.seed,
        save_request_log=args.save_request_log,
    )

    # Initialize replay and controller state.
    windows = int((args.duration_minutes * 60) / DECISION_WINDOW_S)
    controller_start = time.monotonic() + args.start_delay
    _target_pods = current_pods
    pending_scale_down = None
    last_scale_up_monotonic = float("-inf")   # for scale-down cooldown
    queue_cache = None
    replay_runtime = ReplayRuntimeState()
    output_dir = Path(args.output_dir or str(DEFAULT_OUTPUT_DIR))
    window_decisions_output = args.window_decisions_output
    if not window_decisions_output:
        window_decisions_output = str(output_dir / f"{run_id}_window_decisions.csv")
    decision_logger = WindowDecisionLogger(window_decisions_output)
    scale_events_output = args.scale_events_output
    if not scale_events_output:
        scale_events_output = str(output_dir / f"{run_id}_scale_events.csv")
    scale_event_logger = ScaleEventLogger(scale_events_output)

    if not args.disable_queue and args.decision_mode == "proposed":
        queue_samples_output = args.queue_samples_output
        if not queue_samples_output:
            queue_samples_output = str(output_dir / f"{run_id}_queue_samples.csv")
        queue_cache = QueueMetricsCache(queue_samples_output)

    def apply_scale(
        target_pods: int,
        reason: str,
        *,
        ready_pods_before: int | None = None,
        window_index: int | None = None,
    ):
        nonlocal _target_pods
        if target_pods == _target_pods:
            return
        from_target_pods = _target_pods
        event_time_epoch = time.time()
        if not args.dry_run:
            kubectl_scale(target_pods)
        print(f"  [scale:{reason}] target_pods {_target_pods} -> {target_pods}")
        _target_pods = target_pods
        direction = "up" if target_pods > from_target_pods else "down"
        scale_event_logger.append(
            event_time_epoch=event_time_epoch,
            window_index=window_index,
            reason=reason,
            direction=direction,
            from_target_pods=from_target_pods,
            to_target_pods=target_pods,
            ready_pods_before=ready_pods_before,
        )

    async def read_ready_pods() -> int:
        if args.dry_run:
            return _target_pods
        ready = await asyncio.to_thread(get_current_pods)
        if ready < 0:
            print(f"  [pods] failed to read Ready pods; fallback target_pods={_target_pods}")
            return _target_pods
        return ready

    async def controller_loop():
        nonlocal _target_pods, pending_scale_down, last_scale_up_monotonic
        for window_index in range(windows):
            window_start = controller_start + window_index * DECISION_WINDOW_S
            await sleep_until(window_start)

            if pending_scale_down is not None:
                ready_for_down = await read_ready_pods()
                if _target_pods > ready_for_down:
                    print(
                        f"  [scale:down@win{window_index}:0s] skipped; "
                        f"scale-up pending target={_target_pods}, ready={ready_for_down}"
                    )
                else:
                    apply_scale(
                        pending_scale_down,
                        f"down@win{window_index}:0s",
                        ready_pods_before=ready_for_down,
                        window_index=window_index,
                    )
                pending_scale_down = None

            decision = None
            ready_pods_before_decision = await read_ready_pods()
            target_pods_before_decision = _target_pods
            pods_before_decision = ready_pods_before_decision

            if window_index == 0:
                scale_action = "bootstrap-hold"
            else:
                # Keep scale-up on raw near/far horizons; only scale-down uses the corrected near horizon.
                history_arr = np.array(list(qps_history), dtype=np.float32)
                prepared, r, now = make_prediction_decision(
                    predictor=predictor,
                    layer=layer,
                    args=args,
                    history_arr=history_arr,
                    current_pods=ready_pods_before_decision,
                    window_index=window_index,
                    queue_cache=queue_cache,
                )

                N_actual = r["N_actual"]
                scale_action = "hold"

                if N_actual > _target_pods:
                    scale_time, target_start, ready_latency_s, slack_s = scale_up_schedule(
                        window_start,
                        prepared["horizon"],
                        args,
                    )
                    now_monotonic = time.monotonic()
                    if scale_time > now_monotonic:
                        await sleep_until(scale_time)
                        delay_s = scale_time - window_start
                        timing = f"+{delay_s:.1f}s"
                    else:
                        timing = "immediate"
                    apply_scale(
                        N_actual,
                        f"up@win{window_index}:{timing}",
                        ready_pods_before=ready_pods_before_decision,
                        window_index=window_index,
                    )
                    last_scale_up_monotonic = time.monotonic()
                    target_offset_s = target_start - window_start
                    scale_action = (
                        f"up@{timing}; ready_by=+{target_offset_s:.1f}s; "
                        f"L_p95={ready_latency_s:.1f}s; slack={slack_s:.1f}s"
                    )
                elif N_actual < ready_pods_before_decision:
                    cooldown_s = float(getattr(args, "scale_down_cooldown_seconds", 0.0) or 0.0)
                    since_up = time.monotonic() - last_scale_up_monotonic
                    if cooldown_s > 0.0 and since_up < cooldown_s:
                        # Keep recently-added (possibly still cold-starting) pods warm.
                        scale_action = f"down_blocked_cooldown({since_up:.0f}/{cooldown_s:.0f}s)"
                    elif _target_pods > ready_pods_before_decision:
                        scale_action = "down_blocked_pending_up"
                    else:
                        pending_scale_down = N_actual
                        scale_action = "down@next_window_start"

                decision = {
                    "history_arr": history_arr,
                    "q50_raw": prepared["q50_raw"],
                    "q90_raw": prepared["q90_raw"],
                    "q50": prepared["q50"],
                    "q90": prepared["q90"],
                    "conf": prepared["conf"],
                    "prediction_source": prepared.get("prediction_source", "model"),
                    "queue_depth": prepared["queue_depth"],
                    "queue_depth_source": prepared["queue_depth_source"],
                    "queue_correction": prepared["queue_correction"],
                    "queue_mode": prepared["queue_mode"],
                    "queue_status": prepared["queue_status"],
                    "queue_sample_age": prepared["queue_sample_age"],
                    "queue_backlog_ready": prepared["queue_backlog_ready"],
                    "queue_lead_arrivals": prepared["queue_lead_arrivals"],
                    "queue_lead_capacity": prepared["queue_lead_capacity"],
                    "queue_usable": prepared["queue_usable"],
                    "pending_requests": prepared["pending_requests"],
                    "in_flight_requests": prepared["in_flight_requests"],
                    "scraped_pods": prepared["scraped_pods"],
                    "expected_pods": prepared["expected_pods"],
                    "now": now,
                    "result": r,
                    "N_actual": N_actual,
                    "pods_before": pods_before_decision,
                    "ready_pods_before": ready_pods_before_decision,
                    "target_pods_before": target_pods_before_decision,
                    "scale_action": scale_action,
                    "target_window": window_index + prepared["horizon"] - 1,
                    "selected_horizon": prepared["horizon"],
                    "planning_mode": prepared["mode"],
                    "up_gate_horizon": prepared["up_gate_horizon"],
                    "up_gate_N_req": prepared["up_gate_N_req"],
                    "up_gate_q_plan": prepared["up_gate_q_plan"],
                    "up_gate_raw_N_req": prepared["up_gate_raw_N_req"],
                    "up_gate_raw_q_plan": prepared["up_gate_raw_q_plan"],
                    "downtrend": prepared["downtrend"],
                }

            # Update demand history from the completed replay window.
            window_actual = await queue.get()

            if window_actual["window_index"] != window_index:
                print(
                    f"  Warning: window mismatch: expected={window_index}, "
                    f"got={window_actual['window_index']}"
                )

            actual_qps = float(window_actual["generated_requests"])
            qps_history.append(actual_qps)
            decision_logger.append(
                window_index=window_index,
                window_actual=window_actual,
                decision=decision,
                target_pods_after=_target_pods,
            )

            # Report the completed decision window.
            print(f"\n[Step {window_index}] win={window_actual['csv_window_id']}")
            print(f"  QPS_actual={actual_qps:.1f}  "
                  f"QPS_expected={window_actual['expected_requests']:.1f}")
            if decision is None:
                print("  Bootstrap: no prediction decision yet; keeping initial pods")
            else:
                r = decision["result"]
                print(f"  Planning: mode={decision['planning_mode']}  "
                      f"horizon={decision['selected_horizon']}  target_win={decision['target_window']}  "
                      f"hist_mean={np.mean(decision['history_arr']):.1f}")
                print(f"  Predict: q50={decision['q50_raw']:.1f}->{decision['q50']:.1f}  "
                      f"q90={decision['q90_raw']:.1f}->{decision['q90']:.1f}  "
                      f"C={decision['conf']:.3f}  "
                      f"source={decision['prediction_source']}")
                print(f"  Scale-up gate: horizon={decision['up_gate_horizon']}  "
                      f"N_req={decision['up_gate_N_req']}  "
                      f"Q_plan={decision['up_gate_q_plan']:.1f}  "
                      f"raw_N_req={decision['up_gate_raw_N_req']}  "
                      f"raw_Q_plan={decision['up_gate_raw_q_plan']:.1f}")
                dt = decision["downtrend"]
                if dt["active"]:
                    print(f"  Downtrend: active  prev3={dt['previous_mean']:.1f}  "
                          f"recent3={dt['recent_mean']:.1f}  cap={dt['cap']:.1f}")
                else:
                    print(f"  Downtrend: skipped ({dt['reason']})")
                age = decision["queue_sample_age"]
                age_text = "NA" if age is None else f"{age:.1f}s"
                print(f"  Queue: mode={decision['queue_mode']} status={decision['queue_status']} "
                      f"usable={decision['queue_usable']} age={age_text} "
                      f"source={decision['queue_depth_source']} "
                      f"scraped={decision['scraped_pods']}/{decision['expected_pods']}")
                print(f"  Queue plan: B_now={decision['queue_depth']:.1f} "
                      f"pending={decision['pending_requests']:.1f} "
                      f"in_flight={decision['in_flight_requests']:.1f} "
                      f"B_ready={decision['queue_backlog_ready']:.1f} "
                      f"A={decision['queue_lead_arrivals']:.1f} "
                      f"C={decision['queue_lead_capacity']:.1f} "
                      f"correction={decision['queue_correction']:.1f} req/window")
                print(f"  Q_plan={r['q_plan']:.1f}  N_req={r['N_req']}  "
                      f"N_ready={decision['ready_pods_before']}  "
                      f"N_target={decision['target_pods_before']}  "
                      f"N_act={decision['N_actual']}  "
                      f"[{r['scale_direction']}; {decision['scale_action']}]")
                print(f"  CI={r['ci_now']:.0f}  rho={r['rho_ci']:.3f}  "
                      f"alpha={r['alpha']:.3f}  cnt={r['scale_in_counter']}  "
                      f"target_pods_now={_target_pods}")

    # Run workload replay and control concurrently.
    tasks = [
        asyncio.create_task(controller_loop()),
        asyncio.create_task(
            replay_workload(
                replay_config,
                window_observer=observe_window,
                runtime_state=replay_runtime,
            )
        ),
    ]
    sampler_task = None
    if queue_cache is not None:
        queue_source = getattr(args, "queue_source", "auto")
        if queue_source == "auto":
            queue_source = "replayer"
        print(f"  [queue-sampler] source={queue_source}", flush=True)
        if queue_source == "replayer":
            sampler_task = asyncio.create_task(
                queue_replayer_sampler(args, queue_cache, lambda: _target_pods, replay_runtime)
            )
        else:
            sampler_task = asyncio.create_task(
                queue_prometheus_sampler(args, queue_cache, lambda: _target_pods)
            )
        tasks.append(sampler_task)
    try:
        await asyncio.gather(*tasks[:2])
    finally:
        if sampler_task is not None:
            sampler_task.cancel()
            await asyncio.gather(sampler_task, return_exceptions=True)


# ============================================
# Offline trace-based controller.

def run_controller_offline(predictor, layer, args):
    """Run controller from trace CSV without workload_replayer."""
    lookback = predictor.lookback
    qps_history: deque = deque(maxlen=lookback)
    bootstrap_target_pods = int(
        getattr(args, "bootstrap_target_pods", None) or DECISION_CONFIG["N_min"]
    )

    start_row = getattr(args, "start_row", 0)
    qps_source = TraceQPSReader(args.trace, start_row=start_row)
    print(f"\nQPS source: workload trace (start_row={start_row})")

    trace_rows = _load_trace_rows(args.trace)
    start_window = int(start_row)
    for i in range(lookback):
        qps_history.append(trace_rows[(start_window - lookback + i) % len(trace_rows)])

    if args.dry_run:
        current_pods = bootstrap_target_pods
        print("DRY RUN: no kubectl commands will be executed")
    else:
        current_pods = get_current_pods()
        if current_pods < 0:
            current_pods = bootstrap_target_pods

    print(f"\nInitial state: current_pods={current_pods}")
    print("=" * 60)
    print(f"Starting control loop (every {DECISION_WINDOW_S}s)...")
    print("=" * 60)

    step = 0
    t_start = time.time()
    max_windows = int(args.duration_minutes * 60 / DECISION_WINDOW_S)
    target_pods_current = current_pods
    pending_scale_down = None
    output_dir = Path(args.output_dir or str(DEFAULT_OUTPUT_DIR))
    run_id = args.run_id or make_run_id(args.baseline_name or "transformer")
    scale_events_output = args.scale_events_output
    if not scale_events_output:
        scale_events_output = str(output_dir / f"{run_id}_scale_events.csv")
    scale_event_logger = ScaleEventLogger(scale_events_output)

    def apply_scale(
        new_target_pods: int,
        reason: str,
        *,
        ready_pods_before: int | None = None,
        window_index: int | None = None,
    ):
        nonlocal target_pods_current
        if new_target_pods == target_pods_current:
            return
        from_target_pods = target_pods_current
        event_time_epoch = time.time()
        if not args.dry_run:
            kubectl_scale(new_target_pods)
        print(f"  [scale:{reason}] target_pods {target_pods_current} -> {new_target_pods}")
        target_pods_current = new_target_pods
        direction = "up" if new_target_pods > from_target_pods else "down"
        scale_event_logger.append(
            event_time_epoch=event_time_epoch,
            window_index=window_index,
            reason=reason,
            direction=direction,
            from_target_pods=from_target_pods,
            to_target_pods=new_target_pods,
            ready_pods_before=ready_pods_before,
        )

    def read_ready_pods() -> int:
        if args.dry_run:
            return target_pods_current
        ready = get_current_pods()
        if ready < 0:
            print(f"  [pods] failed to read Ready pods; fallback target_pods={target_pods_current}")
            return target_pods_current
        return ready

    while step < max_windows:
        loop_start = time.time()
        window_index = step
        step += 1
        elapsed = loop_start - t_start

        if pending_scale_down is not None:
            ready_for_down = read_ready_pods()
            if target_pods_current > ready_for_down:
                print(
                    f"  [scale:down@win{window_index}:0s] skipped; "
                    f"scale-up pending target={target_pods_current}, ready={ready_for_down}"
                )
            else:
                apply_scale(
                    pending_scale_down,
                    f"down@win{window_index}:0s",
                    ready_pods_before=ready_for_down,
                    window_index=window_index,
                )
            pending_scale_down = None

        decision = None
        ready_pods_before_decision = read_ready_pods()
        target_pods_before_decision = target_pods_current
        pods_before_decision = ready_pods_before_decision

        if window_index != 0:
            history_arr = np.array(list(qps_history), dtype=np.float32)
            prepared, r, now = make_prediction_decision(
                predictor=predictor,
                layer=layer,
                args=args,
                history_arr=history_arr,
                current_pods=ready_pods_before_decision,
                window_index=window_index,
            )

            N_actual = r["N_actual"]
            scale_action = "hold"

            if N_actual > target_pods_current:
                scale_at, target_start, ready_latency_s, slack_s = scale_up_schedule(
                    loop_start,
                    prepared["horizon"],
                    args,
                )
                sleep_until_up = max(0, scale_at - time.time())
                if sleep_until_up > 0:
                    time.sleep(sleep_until_up)
                    delay_s = scale_at - loop_start
                    timing = f"+{delay_s:.1f}s"
                else:
                    timing = "immediate"
                apply_scale(
                    N_actual,
                    f"up@win{window_index}:{timing}",
                    ready_pods_before=ready_pods_before_decision,
                    window_index=window_index,
                )
                target_offset_s = target_start - loop_start
                scale_action = (
                    f"up@{timing}; ready_by=+{target_offset_s:.1f}s; "
                    f"L_p95={ready_latency_s:.1f}s; slack={slack_s:.1f}s"
                )
            elif N_actual < ready_pods_before_decision:
                if target_pods_current > ready_pods_before_decision:
                    scale_action = "down_blocked_pending_up"
                else:
                    pending_scale_down = N_actual
                    scale_action = "down@next_window_start"

            decision = {
                "history_arr": history_arr,
                "q50_raw": prepared["q50_raw"],
                "q90_raw": prepared["q90_raw"],
                "q50": prepared["q50"],
                "q90": prepared["q90"],
                "conf": prepared["conf"],
                "prediction_source": prepared.get("prediction_source", "model"),
                "queue_depth": prepared["queue_depth"],
                "queue_depth_source": prepared["queue_depth_source"],
                "queue_correction": prepared["queue_correction"],
                "queue_mode": prepared["queue_mode"],
                "queue_status": prepared["queue_status"],
                "queue_sample_age": prepared["queue_sample_age"],
                "queue_backlog_ready": prepared["queue_backlog_ready"],
                "queue_lead_arrivals": prepared["queue_lead_arrivals"],
                "queue_lead_capacity": prepared["queue_lead_capacity"],
                "queue_usable": prepared["queue_usable"],
                "pending_requests": prepared["pending_requests"],
                "in_flight_requests": prepared["in_flight_requests"],
                "scraped_pods": prepared["scraped_pods"],
                "expected_pods": prepared["expected_pods"],
                "now": now,
                "result": r,
                "N_actual": N_actual,
                "pods_before": pods_before_decision,
                "ready_pods_before": ready_pods_before_decision,
                "target_pods_before": target_pods_before_decision,
                "scale_action": scale_action,
                "target_window": window_index + prepared["horizon"] - 1,
                "selected_horizon": prepared["horizon"],
                "planning_mode": prepared["mode"],
                "up_gate_horizon": prepared["up_gate_horizon"],
                "up_gate_N_req": prepared["up_gate_N_req"],
                "up_gate_q_plan": prepared["up_gate_q_plan"],
                "up_gate_raw_N_req": prepared["up_gate_raw_N_req"],
                "up_gate_raw_q_plan": prepared["up_gate_raw_q_plan"],
                "downtrend": prepared["downtrend"],
            }

        qps_now = qps_source.get_qps(elapsed)
        qps_history.append(qps_now)

        winfo = f"win={qps_source.current_window_id}"
        print(f"\n[Step {step}] {winfo}")
        print(f"  QPS_now={qps_now:.1f}")
        if decision is None:
            print("  Bootstrap: no prediction decision yet; keeping initial pods")
        else:
            r = decision["result"]
            print(f"  Planning: mode={decision['planning_mode']}  "
                  f"horizon={decision['selected_horizon']}  target_win={decision['target_window']}  "
                  f"hist_mean={np.mean(decision['history_arr']):.1f}")
            print(f"  Predict: q50={decision['q50_raw']:.1f}->{decision['q50']:.1f}  "
                  f"q90={decision['q90_raw']:.1f}->{decision['q90']:.1f}  "
                  f"C={decision['conf']:.3f}  "
                  f"source={decision['prediction_source']}")
            print(f"  Scale-up gate: horizon={decision['up_gate_horizon']}  "
                  f"N_req={decision['up_gate_N_req']}  "
                  f"Q_plan={decision['up_gate_q_plan']:.1f}  "
                  f"raw_N_req={decision['up_gate_raw_N_req']}  "
                  f"raw_Q_plan={decision['up_gate_raw_q_plan']:.1f}")
            dt = decision["downtrend"]
            if dt["active"]:
                print(f"  Downtrend: active  prev3={dt['previous_mean']:.1f}  "
                      f"recent3={dt['recent_mean']:.1f}  cap={dt['cap']:.1f}")
            else:
                print(f"  Downtrend: skipped ({dt['reason']})")
            age = decision["queue_sample_age"]
            age_text = "NA" if age is None else f"{age:.1f}s"
            print(f"  Queue: mode={decision['queue_mode']} status={decision['queue_status']} "
                  f"usable={decision['queue_usable']} age={age_text} "
                  f"source={decision['queue_depth_source']} "
                  f"scraped={decision['scraped_pods']}/{decision['expected_pods']}")
            print(f"  Queue plan: B_now={decision['queue_depth']:.1f} "
                  f"pending={decision['pending_requests']:.1f} "
                  f"in_flight={decision['in_flight_requests']:.1f} "
                  f"B_ready={decision['queue_backlog_ready']:.1f} "
                  f"A={decision['queue_lead_arrivals']:.1f} "
                  f"C={decision['queue_lead_capacity']:.1f} "
                  f"correction={decision['queue_correction']:.1f} req/window")
            print(f"  Q_plan={r['q_plan']:.1f}  N_req={r['N_req']}  "
                  f"N_ready={decision['ready_pods_before']}  "
                  f"N_target={decision['target_pods_before']}  "
                  f"N_act={decision['N_actual']}  "
                  f"[{r['scale_direction']}; {decision['scale_action']}]")
            print(f"  CI={r['ci_now']:.0f}  rho={r['rho_ci']:.3f}  "
                  f"alpha={r['alpha']:.3f}  cnt={r['scale_in_counter']}  "
                  f"target_pods_now={target_pods_current}")

        elapsed_step = time.time() - loop_start
        sleep_time = max(0, DECISION_WINDOW_S - elapsed_step)
        if sleep_time < DECISION_WINDOW_S * 0.5:
            print(f"  Warning: Step took {elapsed_step:.1f}s, only sleeping {sleep_time:.1f}s")
        time.sleep(sleep_time)


# ============================================
# Command-line interface.

def main():
    parser = argparse.ArgumentParser(
        description="ETS Controller with Transformer + Carbon-Aware Decision Layer"
    )
    parser.add_argument("checkpoint_dir", help="model checkpoint directory")
    parser.add_argument("--trace", default=DEFAULT_TEST_CSV,
                        help="workload trace CSV path")
    parser.add_argument("--live", action="store_true",
                        help="run live with workload_replayer")
    parser.add_argument("--dry-run", action="store_true",
                        help="run without kubectl changes")
    parser.add_argument("--start-row", type=int, default=0,
                        help="start row in the trace CSV")
    parser.add_argument("--mock-qps", type=float, default=0.0,
                        help="mock QPS for dry-run mode")
    parser.add_argument("--duration-minutes", type=float, default=5.0,
                        help="experiment duration in minutes")
    parser.add_argument("--start-delay", type=float, default=5.0,
                        help="start delay in seconds")
    parser.add_argument("--first-patch-lead", type=float, default=5.0,
                        help=argparse.SUPPRESS)
    parser.add_argument("--bootstrap-target-pods", type=int, default=None,
                        help="fallback target pod count for bootstrap/dry-run initialization")
    parser.add_argument("--baseline-name", default=None,
                        help="baseline name")
    parser.add_argument("--run-id", default=None,
                        help="experiment run ID")
    parser.add_argument("--output-dir", default=None,
                        help="result output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed")
    parser.add_argument("--save-request-log", action="store_true",
                        help="save request log")
    parser.add_argument("--url", default=DEFAULT_URL,
                        help="API URL")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help="HTTP Host header")
    parser.add_argument("--prometheus", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--prometheus-url", default="http://127.0.0.1:30090",
                        help="Prometheus base URL for background queue sampling")
    parser.add_argument("--disable-queue", action="store_true",
                        help="disable queue-aware correction and use original decision behavior")
    parser.add_argument("--queue-mode", choices=["none", "current", "ready"], default="ready",
                        help=(
                            "queue ablation mode: none disables queue correction; current uses "
                            "B_now; ready uses B_ready=max(0,B_now+A-C)"
                        ))
    parser.add_argument("--queue-source", choices=["auto", "prometheus", "replayer"], default="auto",
                        help=(
                            "queue sample source. auto uses workload_replayer pending/in-flight "
                            "state in live mode; prometheus keeps the queue-proxy scrape path"
                        ))
    parser.add_argument("--allow-missing-queue", action="store_true",
                        help="log missing/partial queue samples and continue without queue correction")
    parser.add_argument("--ready-latency-p95-seconds", type=float, default=SCALE_UP_LEAD_S,
                        help="P95 latency L until added capacity is Ready")
    parser.add_argument("--scale-up-slack-seconds", type=float, default=1.0,
                        help="extra lead time subtracted from scale-up apply time")
    parser.add_argument("--queue-sample-period-seconds", type=float, default=2.0,
                        help="background queue Prometheus sampling period")
    parser.add_argument("--queue-max-sample-age-seconds", type=float, default=10.0,
                        help="maximum age of an ok queue sample allowed in decisions")
    parser.add_argument("--queue-query-timeout-seconds", type=float, default=1.5,
                        help="timeout for each Prometheus queue query")
    parser.add_argument("--queue-samples-output", default=None,
                        help="CSV path for background queue samples")
    parser.add_argument("--window-decisions-output", default=None,
                        help="CSV path for one-row-per-window controller decisions")
    parser.add_argument("--scale-events-output", default=None,
                        help="CSV path for controller scale events")
    parser.add_argument("--oracle-predictor", action="store_true",
                        help="replace model forecasts with actual future trace windows")
    parser.add_argument("--oracle-prediction-trace", default=None,
                        help="trace CSV used for oracle h1/h2; defaults to --trace")
    parser.add_argument("--oracle-confidence", type=float, default=1.0,
                        help="confidence assigned to oracle horizons")
    parser.add_argument("--oracle-interval-pad", type=float, default=0.0,
                        help="optional +/- request padding around oracle q50 for q10/q90")
    parser.add_argument("--decision-mode", choices=["proposed", "direct"], default="proposed",
                        help=(
                            "proposed uses the SLA/carbon-aware decision layer; direct maps "
                            "Transformer Q_base directly to a pod target for baseline runs"
                        ))
    parser.add_argument("--ci-trace-start", default=None,
                        help=(
                            "Override CI lookup time so window_index maps linearly into the "
                            "CI trace (e.g. '2025-06-28 00:00:00'). If unset, controller uses "
                            "datetime.now(UTC), which can fall outside the CI CSV range and "
                            "clamp to the last row, freezing rho_CI."
                        ))
    parser.add_argument("--ci-csv", default=CI_CSV,
                        help="Carbon-intensity CSV used by the decision layer.")
    parser.add_argument("--ci-window-seconds", type=float, default=None,
                        help=(
                            "Synthetic seconds-per-window when --ci-trace-start is set. "
                            "Default = DECISION_WINDOW_S (=60). Ignored when "
                            "--ci-time-scale is set."
                        ))
    parser.add_argument("--ci-time-scale", type=float, default=None,
                        help=(
                            "Synthetic CI seconds per real decision second when "
                            "--ci-trace-start is set. Use 8 to map a 3-hour formal "
                            "run onto a 24-hour CI trace."
                        ))
    parser.add_argument("--config-mu", type=float, default=None,
                        help="Override DECISION_CONFIG['mu'] for a frozen serving contract.")
    parser.add_argument("--config-k-ci", type=float, default=None,
                        help="Override DECISION_CONFIG['k_ci'] for sensitivity runs.")
    parser.add_argument("--config-k-g", type=float, default=None,
                        help="Override DECISION_CONFIG['k_g'] for sensitivity runs.")
    parser.add_argument("--config-alpha-base", type=float, default=None,
                        help="Override DECISION_CONFIG['alpha_base'] for sensitivity runs.")
    parser.add_argument("--config-alpha-min", type=float, default=None,
                        help="Override DECISION_CONFIG['alpha_min'] for sensitivity runs.")
    parser.add_argument("--config-alpha-max", type=float, default=None,
                        help="Override DECISION_CONFIG['alpha_max'] for sensitivity runs.")
    parser.add_argument("--scale-down-confirm-windows", type=int, default=None,
                        help="Consecutive scale-in confirmation windows (report 'cautious in'). "
                             "1 disables the gate; 2-3 restores cautious scale-down.")
    parser.add_argument("--scale-down-cooldown-seconds", type=float, default=0.0,
                        help="Block scale-down for this many seconds after a scale-up, so "
                             "freshly-added (cold-starting) pods stay warm. 0 disables.")
    parser.add_argument("--disable-downtrend", dest="disable_downtrend", action="store_true",
                        help="Disable the downtrend quantile cap (report design has none).")
    args = parser.parse_args()
    if args.scale_down_confirm_windows is not None:
        DECISION_CONFIG["scale_in_confirm_windows"] = max(int(args.scale_down_confirm_windows), 1)
    if getattr(args, "disable_downtrend", False):
        DECISION_CONFIG["downtrend_enabled"] = False
    if args.scale_down_cooldown_seconds < 0:
        raise SystemExit("--scale-down-cooldown-seconds must be non-negative")
    if args.bootstrap_target_pods is not None and args.bootstrap_target_pods < 0:
        raise SystemExit("--bootstrap-target-pods must be non-negative")
    if args.disable_queue:
        args.queue_mode = "none"
    if args.queue_sample_period_seconds <= 0:
        raise SystemExit("--queue-sample-period-seconds must be positive")
    if args.queue_max_sample_age_seconds <= 0:
        raise SystemExit("--queue-max-sample-age-seconds must be positive")
    if args.queue_source == "replayer" and not args.live:
        raise SystemExit("--queue-source replayer requires --live")
    if args.ready_latency_p95_seconds < 0:
        raise SystemExit("--ready-latency-p95-seconds must be non-negative")
    if args.scale_up_slack_seconds < 0:
        raise SystemExit("--scale-up-slack-seconds must be non-negative")
    if args.oracle_confidence < 0.0 or args.oracle_confidence > 1.0:
        raise SystemExit("--oracle-confidence must be in [0, 1]")
    if args.oracle_interval_pad < 0.0:
        raise SystemExit("--oracle-interval-pad must be non-negative")
    if args.oracle_predictor:
        oracle_trace = Path(args.oracle_prediction_trace or args.trace)
        if not oracle_trace.exists():
            raise SystemExit(f"--oracle-prediction-trace does not exist: {oracle_trace}")
    if args.ci_time_scale is not None and args.ci_time_scale <= 0:
        raise SystemExit("--ci-time-scale must be positive")
    applied_overrides = apply_decision_config_overrides(args)

    # Load the forecasting model.
    print("=" * 60)
    print("Loading Transformer model...")
    predictor = QPSPredictor(args.checkpoint_dir)
    lookback = predictor.lookback
    print(f"  lookback = {lookback}")

    # Initialize the decision layer.
    print("\nLoading decision layer...")
    ci = CILoader(args.ci_csv)
    layer = DecisionLayer(DECISION_CONFIG, ci)
    print(f"  decision_mode={args.decision_mode}")
    prediction_source = "oracle" if args.oracle_predictor else "model"
    print(f"  prediction_source={prediction_source}")
    if args.oracle_predictor:
        print(
            "  oracle_predictor=True "
            f"trace={args.oracle_prediction_trace or args.trace} "
            f"confidence={args.oracle_confidence:g} "
            f"interval_pad={args.oracle_interval_pad:g}"
        )
    print(
        f"  queue_mode={args.queue_mode}, ready_latency_p95={args.ready_latency_p95_seconds}s, "
        f"scale_up_slack={args.scale_up_slack_seconds}s, "
        f"queue_source={args.queue_source}, "
        f"queue_sample_period={args.queue_sample_period_seconds}s, "
        f"queue_max_age={args.queue_max_sample_age_seconds}s, "
        f"ci_time_scale={args.ci_time_scale}"
    )
    print("  DECISION_CONFIG=" + json.dumps(DECISION_CONFIG, sort_keys=True))
    if applied_overrides:
        print("  DECISION_CONFIG_OVERRIDES=" + json.dumps(applied_overrides, sort_keys=True))

    # Run the selected controller mode.
    if args.live:
        print(f"\nQPS source: live (workload_replayer via asyncio.Queue)")
        print(f"  duration={args.duration_minutes}min, start_delay={args.start_delay}s, "
              f"horizon={PREDICTION_HORIZON}, scale_up_lead={SCALE_UP_LEAD_S}s")
        asyncio.run(run_controller_live(predictor, layer, args))
    else:
        run_controller_offline(predictor, layer, args)


if __name__ == "__main__":
    main()
