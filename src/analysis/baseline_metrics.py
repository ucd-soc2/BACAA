import asyncio
import csv
import json
import math
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CARBON_INTENSITY_CSV = str(
    Path(__file__).with_name("carbon_intensity_ie_2025_5min.csv")
)


@dataclass
class MetricsConfig:
    baseline_name: str
    run_id: str
    output_dir: str
    namespace: str
    service_name: str
    pod_metrics_selector: str
    pod_metrics_revision_name: str
    window_size_seconds: float
    duration_minutes: float
    timeout_seconds: float
    sla_threshold_seconds: float
    pod_capacity_per_window: float
    power_active_watts: float
    power_idle_watts: float
    cold_start_energy_joules: float
    carbon_intensity_g_per_kwh: float
    carbon_intensity_csv: str
    collect_pod_metrics: bool
    metrics_poll_seconds: float
    collect_busy_pod_metrics: bool
    pod_busy_metrics_port: int
    pod_busy_metrics_path: str
    pod_busy_metrics_timeout_seconds: float


def add_metrics_args(parser, default_pod_capacity_per_window=13.75):
    parser.add_argument("--collect-pod-metrics", action="store_true")
    parser.add_argument("--metrics-poll-seconds", type=float, default=1.0)
    parser.add_argument(
        "--pod-metrics-selector",
        choices=["revision", "service"],
        default="revision",
        help=(
            "Select pods for scaling metrics by latest Ready Revision by default. "
            "Use service only for debugging old behavior that includes all Revisions."
        ),
    )
    parser.add_argument(
        "--pod-metrics-revision-name",
        default=None,
        help=(
            "Revision name used when --pod-metrics-selector=revision. Defaults to "
            "--revision-name when available, otherwise resolves latestReadyRevisionName."
        ),
    )
    parser.add_argument(
        "--sla-threshold-seconds",
        type=float,
        default=2.0,
        help="Latency threshold for the primary P99 SLA objective, e.g. P99 latency < 2s.",
    )
    parser.add_argument(
        "--metrics-pod-capacity-per-window",
        type=float,
        default=default_pod_capacity_per_window,
        help="Used to estimate active versus idle pods in energy/carbon metrics.",
    )
    parser.add_argument(
        "--power-active-watts",
        type=float,
        default=0.0,
        help="Per-active-pod power draw. Set this to enable energy/carbon estimates.",
    )
    parser.add_argument(
        "--power-idle-watts",
        type=float,
        default=0.0,
        help="Per-idle-pod power draw. Set this to enable energy/carbon estimates.",
    )
    parser.add_argument(
        "--cold-start-energy-joules",
        type=float,
        default=0.0,
        help="Extra energy per newly started pod. Set this to include cold-start energy.",
    )
    parser.add_argument(
        "--carbon-intensity-g-per-kwh",
        type=float,
        default=0.0,
        help="Fallback constant carbon intensity if --carbon-intensity-csv is unavailable.",
    )
    parser.add_argument(
        "--carbon-intensity-csv",
        default=DEFAULT_CARBON_INTENSITY_CSV,
        help="CSV containing Datetime (UTC) and carbon intensity columns.",
    )
    parser.add_argument(
        "--collect-busy-pod-metrics",
        action="store_true",
        help=(
            "Also query each running pod's metrics endpoint and use busy/idle "
            "pods directly for active versus idle energy."
        ),
    )
    parser.add_argument("--pod-busy-metrics-port", type=int, default=8080)
    parser.add_argument("--pod-busy-metrics-path", default="/metrics")
    parser.add_argument(
        "--pod-busy-metrics-timeout-seconds",
        type=float,
        default=1.0,
    )


def build_metrics_config(args, baseline_name, run_id, pod_capacity_per_window=None):
    capacity = (
        pod_capacity_per_window
        if pod_capacity_per_window is not None
        else args.metrics_pod_capacity_per_window
    )
    return MetricsConfig(
        baseline_name=baseline_name,
        run_id=run_id,
        output_dir=args.output_dir,
        namespace=args.namespace,
        service_name=args.service_name,
        pod_metrics_selector=getattr(args, "pod_metrics_selector", "revision"),
        pod_metrics_revision_name=(
            getattr(args, "pod_metrics_revision_name", None)
            or getattr(args, "revision_name", None)
            or ""
        ),
        window_size_seconds=args.window_size_seconds,
        duration_minutes=args.duration_minutes,
        timeout_seconds=args.timeout_seconds,
        sla_threshold_seconds=args.sla_threshold_seconds,
        pod_capacity_per_window=capacity,
        power_active_watts=args.power_active_watts,
        power_idle_watts=args.power_idle_watts,
        cold_start_energy_joules=args.cold_start_energy_joules,
        carbon_intensity_g_per_kwh=args.carbon_intensity_g_per_kwh,
        carbon_intensity_csv=args.carbon_intensity_csv,
        collect_pod_metrics=args.collect_pod_metrics,
        metrics_poll_seconds=args.metrics_poll_seconds,
        collect_busy_pod_metrics=getattr(args, "collect_busy_pod_metrics", False),
        pod_busy_metrics_port=getattr(args, "pod_busy_metrics_port", 8080),
        pod_busy_metrics_path=getattr(args, "pod_busy_metrics_path", "/metrics"),
        pod_busy_metrics_timeout_seconds=getattr(
            args,
            "pod_busy_metrics_timeout_seconds",
            1.0,
        ),
    )


def percentile(values, q):
    if not values:
        return 0.0
    return float(np.percentile(values, q))


def is_timeout_result(result, timeout_seconds):
    status = result.get("status")
    if status in (408, 504):
        return True
    if status != 0:
        return False
    elapsed = float(result.get("elapsed", 0.0))
    error = str(result.get("error", "")).lower()
    return elapsed >= timeout_seconds or "timeout" in error or "timed out" in error


def summarize_request_metrics(replay_result, config):
    results = replay_result.get("results", [])
    window_summaries = replay_result.get("window_summaries", [])
    total = len(results)
    ok_results = [r for r in results if r.get("status") == 200]
    ok = len(ok_results)
    failed = total - ok
    latencies = [float(r.get("elapsed", 0.0)) for r in ok_results]
    duration_seconds = config.duration_minutes * 60.0
    status_counts = {}
    for result in results:
        key = str(result.get("status", 0))
        status_counts[key] = status_counts.get(key, 0) + 1

    timeout_indexes = {
        index
        for index, result in enumerate(results)
        if is_timeout_result(result, config.timeout_seconds)
    }
    server_error_indexes = {
        index
        for index, result in enumerate(results)
        if isinstance(result.get("status"), int) and 500 <= result["status"] <= 599
    }
    timeout_count = len(timeout_indexes)
    server_error_count = len(server_error_indexes)
    error_count = len(timeout_indexes | server_error_indexes)
    p50_latency = percentile(latencies, 50)
    p95_latency = percentile(latencies, 95)
    p99_latency = percentile(latencies, 99)
    p95_sla_met = p95_latency < config.sla_threshold_seconds if latencies else False
    p95_sla_margin = config.sla_threshold_seconds - p95_latency
    p99_sla_met = p99_latency < config.sla_threshold_seconds if latencies else False
    p99_sla_margin = config.sla_threshold_seconds - p99_latency

    request_latency_threshold_violations = sum(
        1 for latency in latencies if latency > config.sla_threshold_seconds
    )
    request_threshold_violations_with_failures = (
        request_latency_threshold_violations + failed
    )
    p95_window_threshold_violations = sum(
        1
        for summary in window_summaries
        if float(summary.get("p95_latency_seconds", 0.0))
        > config.sla_threshold_seconds
    )

    return {
        "total_requests": total,
        "successful_requests": ok,
        "failed_requests": failed,
        "success_rate_pct": ok / max(total, 1) * 100.0,
        "failure_rate_pct": failed / max(total, 1) * 100.0,
        "total_timeouts": timeout_count,
        "timeout_rate_pct": timeout_count / max(total, 1) * 100.0,
        "server_error_count": server_error_count,
        "server_error_rate_pct": server_error_count / max(total, 1) * 100.0,
        "error_count": error_count,
        "error_rate_pct": error_count / max(total, 1) * 100.0,
        "avg_throughput_rps": ok / max(duration_seconds, 1e-9),
        "sent_rps": total / max(duration_seconds, 1e-9),
        "mean_latency_seconds": sum(latencies) / len(latencies)
        if latencies
        else 0.0,
        "p50_latency_seconds": p50_latency,
        "p95_latency_seconds": p95_latency,
        "p99_latency_seconds": p99_latency,
        "max_latency_seconds": max(latencies) if latencies else 0.0,
        "sla_threshold_seconds": config.sla_threshold_seconds,
        "sla_objective": "p99_latency_seconds < sla_threshold_seconds",
        "p95_sla_met": p95_sla_met,
        "p95_sla_margin_seconds": p95_sla_margin,
        "p95_sla_violation_seconds": max(0.0, -p95_sla_margin),
        "p99_sla_met": p99_sla_met,
        "p99_sla_margin_seconds": p99_sla_margin,
        "p99_sla_violation_seconds": max(0.0, -p99_sla_margin),
        "sla_met": p99_sla_met,
        "sla_margin_seconds": p99_sla_margin,
        "sla_violation_seconds": max(0.0, -p99_sla_margin),
        "request_latency_threshold_violation_count": request_latency_threshold_violations,
        "request_latency_threshold_violation_rate_pct": (
            request_latency_threshold_violations / max(ok, 1) * 100.0
        ),
        "request_latency_threshold_violation_count_with_failures": (
            request_threshold_violations_with_failures
        ),
        "request_latency_threshold_violation_rate_pct_with_failures": (
            request_threshold_violations_with_failures / max(total, 1) * 100.0
        ),
        "p95_window_threshold_violation_count": p95_window_threshold_violations,
        # Backward-compatible aliases for earlier result parsing scripts.
        "sla_violation_count": request_threshold_violations_with_failures,
        "sla_violation_rate_pct": request_threshold_violations_with_failures
        / max(total, 1)
        * 100.0,
        "sla_violation_windows": p95_window_threshold_violations,
        "status_counts": status_counts,
    }


def pod_is_ready(item):
    status = item.get("status", {})
    conditions = status.get("conditions", [])
    return status.get("phase") == "Running" and any(
        condition.get("type") == "Ready" and condition.get("status") == "True"
        for condition in conditions
    )


def parse_busy_metric(payload_text):
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict) and "busy" in payload:
        return float(payload.get("busy") or 0)

    for line in payload_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not stripped.startswith("busy"):
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            return float(parts[-1])

    raise ValueError("Could not find busy metric")


def pod_metrics_url(pod_ip, config):
    host = f"[{pod_ip}]" if ":" in pod_ip and not pod_ip.startswith("[") else pod_ip
    path = config.pod_busy_metrics_path
    if not path.startswith("/"):
        path = f"/{path}"
    return f"http://{host}:{config.pod_busy_metrics_port}{path}"


def read_pod_busy_value(pod_ip, config):
    request = urllib.request.Request(
        pod_metrics_url(pod_ip, config),
        headers={"Accept": "application/json,text/plain;q=0.9,*/*;q=0.8"},
    )
    with urllib.request.urlopen(  # nosec B310 - user-configured pod metrics URL.
        request,
        timeout=config.pod_busy_metrics_timeout_seconds,
    ) as response:
        payload_text = response.read().decode("utf-8", errors="replace")
    return parse_busy_metric(payload_text)


def latest_ready_revision_name(kubectl_cmd, config):
    cmd = kubectl_cmd + [
        "get",
        "kservice",
        config.service_name,
        "-n",
        config.namespace,
        "-o",
        "jsonpath={.status.latestReadyRevisionName}",
    ]
    completed = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    revision_name = completed.stdout.strip()
    if not revision_name:
        raise RuntimeError(
            f"Could not resolve latestReadyRevisionName for kservice {config.service_name}"
        )
    return revision_name


def pod_label_selector(kubectl_cmd, config):
    selector_mode = getattr(config, "pod_metrics_selector", "revision")
    if selector_mode == "service":
        return f"serving.knative.dev/service={config.service_name}"
    if selector_mode != "revision":
        raise ValueError(f"unsupported pod_metrics_selector: {selector_mode}")

    revision_name = getattr(config, "pod_metrics_revision_name", "") or ""
    if not revision_name:
        revision_name = latest_ready_revision_name(kubectl_cmd, config)
        config.pod_metrics_revision_name = revision_name
    return f"serving.knative.dev/revision={revision_name}"


def read_pod_sample(kubectl_cmd, config):
    label_selector = pod_label_selector(kubectl_cmd, config)
    cmd = kubectl_cmd + [
        "get",
        "pods",
        "-n",
        config.namespace,
        "-l",
        label_selector,
        "-o",
        "json",
    ]
    completed = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(completed.stdout)
    total = 0
    ready = 0
    running = 0
    pending = 0
    terminating = 0
    running_pod_ips = []
    for item in payload.get("items", []):
        metadata = item.get("metadata", {})
        status = item.get("status", {})
        if metadata.get("deletionTimestamp"):
            terminating += 1
            continue
        total += 1
        phase = status.get("phase")
        if phase == "Running":
            running += 1
            pod_ip = status.get("podIP")
            if pod_ip:
                running_pod_ips.append(pod_ip)
        elif phase == "Pending":
            pending += 1
        if pod_is_ready(item):
            ready += 1

    now = datetime.now(timezone.utc)
    sample = {
        "timestamp": now.isoformat(timespec="seconds"),
        "epoch_seconds": time.time(),
        "service_name": config.service_name,
        "pod_metrics_selector": getattr(config, "pod_metrics_selector", "revision"),
        "pod_label_selector": label_selector,
        "pod_metrics_revision_name": getattr(config, "pod_metrics_revision_name", ""),
        "total_pods": total,
        "ready_pods": ready,
        "running_pods": running,
        "pending_pods": pending,
        "terminating_pods": terminating,
    }
    if config.collect_busy_pod_metrics:
        busy = 0
        errors = []
        for pod_ip in running_pod_ips:
            try:
                busy_value = read_pod_busy_value(pod_ip, config)
                if busy_value > 0:
                    busy += 1
            except (OSError, ValueError, urllib.error.URLError) as exc:
                errors.append(f"{pod_ip}: {exc}")
        sample["busy_pods"] = None if errors else busy
        sample["idle_pods"] = None if errors else max(0, running - busy)
        sample["partial_busy_pods"] = busy if errors else None
        sample["busy_metric_error_count"] = len(errors)
        sample["busy_metric_errors"] = "; ".join(errors)
    return sample


async def sample_pod_metrics(kubectl_cmd, config, stop_event, samples):
    while not stop_event.is_set():
        try:
            sample = await asyncio.to_thread(read_pod_sample, kubectl_cmd, config)
            samples.append(sample)
        except Exception as exc:
            now = datetime.now(timezone.utc)
            samples.append({
                "timestamp": now.isoformat(timespec="seconds"),
                "epoch_seconds": time.time(),
                "service_name": config.service_name,
                "pod_metrics_selector": getattr(
                    config,
                    "pod_metrics_selector",
                    "revision",
                ),
                "pod_label_selector": "",
                "pod_metrics_revision_name": getattr(
                    config,
                    "pod_metrics_revision_name",
                    "",
                ),
                "total_pods": None,
                "ready_pods": None,
                "running_pods": None,
                "pending_pods": None,
                "terminating_pods": None,
                "busy_pods": None,
                "idle_pods": None,
                "partial_busy_pods": None,
                "busy_metric_error_count": None,
                "busy_metric_errors": "",
                "error": str(exc),
            })
        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=config.metrics_poll_seconds,
            )
        except asyncio.TimeoutError:
            pass


def summarize_scaling_metrics(pod_samples):
    pod_values = [
        int(sample["total_pods"])
        for sample in pod_samples
        if sample.get("total_pods") is not None
    ]
    if not pod_values:
        return {}

    diffs = [
        pod_values[index] - pod_values[index - 1]
        for index in range(1, len(pod_values))
    ]
    ready_values = [
        int(sample["ready_pods"])
        for sample in pod_samples
        if sample.get("ready_pods") is not None
    ]
    busy_values = [
        int(sample["busy_pods"])
        for sample in pod_samples
        if sample.get("busy_pods") is not None
    ]
    idle_values = [
        int(sample["idle_pods"])
        for sample in pod_samples
        if sample.get("idle_pods") is not None
    ]
    sample_epochs = [
        float(sample["epoch_seconds"])
        for sample in pod_samples
        if sample.get("total_pods") is not None and sample.get("epoch_seconds")
    ]
    pod_change_count = sum(1 for diff in diffs if diff != 0)
    sample_span_seconds = (
        max(sample_epochs) - min(sample_epochs)
        if len(sample_epochs) > 1
        else 0.0
    )
    summary = {
        "scale_up_count": sum(1 for diff in diffs if diff > 0),
        "scale_down_count": sum(1 for diff in diffs if diff < 0),
        "scale_up_event_count": sum(1 for diff in diffs if diff > 0),
        "scale_down_event_count": sum(1 for diff in diffs if diff < 0),
        "scale_up_pod_count": sum(max(0, diff) for diff in diffs),
        "scale_down_pod_count": sum(max(0, -diff) for diff in diffs),
        "cold_start_count": sum(max(0, diff) for diff in diffs),
        "pod_change_count": pod_change_count,
        "pod_change_rate_per_minute": pod_change_count
        / max(sample_span_seconds / 60.0, 1e-9),
        "pod_std": statistics.pstdev(pod_values) if len(pod_values) > 1 else 0.0,
        "avg_pods": sum(pod_values) / len(pod_values),
        "max_pods": max(pod_values),
        "min_pods": min(pod_values),
        "pod_sample_count": len(pod_values),
        "pod_count_source": "total_pods",
    }
    if ready_values:
        summary.update({
            "avg_ready_pods": sum(ready_values) / len(ready_values),
            "max_ready_pods": max(ready_values),
            "min_ready_pods": min(ready_values),
            "ready_pod_std": (
                statistics.pstdev(ready_values) if len(ready_values) > 1 else 0.0
            ),
        })
    if busy_values and idle_values:
        total_values = [
            int(sample["total_pods"])
            for sample in pod_samples
            if sample.get("total_pods") is not None
            and sample.get("busy_pods") is not None
            and sample.get("idle_pods") is not None
        ]
        utilization_values = [
            busy / max(total, 1)
            for busy, total in zip(busy_values, total_values)
        ]
        idle_ratio_values = [
            idle / max(total, 1)
            for idle, total in zip(idle_values, total_values)
        ]
        summary.update({
            "busy_pod_sample_count": len(busy_values),
            "avg_busy_pods": sum(busy_values) / len(busy_values),
            "avg_idle_pods": sum(idle_values) / len(idle_values),
            "avg_measured_pod_utilization_pct": (
                sum(utilization_values) / len(utilization_values) * 100.0
            ),
            "avg_measured_idle_ratio_pct": (
                sum(idle_ratio_values) / len(idle_ratio_values) * 100.0
            ),
        })
    return summary


def find_carbon_intensity_column(columns):
    for column in columns:
        if "life cycle" in column.lower():
            return column
    for column in columns:
        lowered = column.lower()
        if "carbon intensity" in lowered and "kwh" in lowered:
            return column
    raise ValueError("Could not find carbon intensity column in CSV")


def load_carbon_intensity_series(config):
    csv_path = Path(config.carbon_intensity_csv)
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    time_column = "Datetime (UTC)"
    if time_column not in df.columns:
        time_column = df.columns[0]
    ci_column = find_carbon_intensity_column(df.columns)

    times = pd.to_datetime(df[time_column], utc=True, errors="coerce")
    values = pd.to_numeric(df[ci_column], errors="coerce")
    ci_df = pd.DataFrame({"timestamp": times, "ci": values}).dropna()
    ci_df = ci_df.sort_values("timestamp").reset_index(drop=True)
    if ci_df.empty:
        return None
    return ci_df


def carbon_intensity_at(ci_df, timestamp, fallback):
    if ci_df is None:
        return fallback
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    index = ci_df["timestamp"].searchsorted(ts, side="right") - 1
    index = max(0, min(int(index), len(ci_df) - 1))
    return float(ci_df.iloc[index]["ci"])


def summarize_energy_metrics(request_metrics, scaling_metrics, config):
    return summarize_energy_metrics_from_samples(
        request_metrics,
        scaling_metrics,
        config,
        [],
    )


def summarize_energy_metrics_from_samples(
    request_metrics,
    scaling_metrics,
    config,
    pod_samples,
):
    ci_df = load_carbon_intensity_series(config)
    duration_seconds = config.duration_minutes * 60.0
    avg_pods = scaling_metrics.get("avg_pods", 0.0)
    pod_qps_capacity = config.pod_capacity_per_window / max(
        config.window_size_seconds,
        1e-9,
    )
    estimated_active_pods = min(
        avg_pods,
        request_metrics.get("sent_rps", 0.0) / max(pod_qps_capacity, 1e-9),
    )
    estimated_idle_pods = max(0.0, avg_pods - estimated_active_pods)

    valid_samples = [
        sample
        for sample in pod_samples
        if sample.get("total_pods") is not None and sample.get("epoch_seconds")
    ]
    measured_busy_samples = [
        sample
        for sample in valid_samples
        if sample.get("busy_pods") is not None and sample.get("idle_pods") is not None
    ]
    active_pod_source = (
        "measured_busy_pods" if measured_busy_samples else "estimated_from_sent_rps"
    )
    measured_utilization_values = [
        float(sample.get("busy_pods") or 0)
        / max(float(sample.get("total_pods") or 0), 1.0)
        for sample in measured_busy_samples
    ]
    measured_idle_ratio_values = [
        float(sample.get("idle_pods") or 0)
        / max(float(sample.get("total_pods") or 0), 1.0)
        for sample in measured_busy_samples
    ]
    active_energy_kwh = 0.0
    idle_energy_kwh = 0.0
    active_carbon_g = 0.0
    idle_carbon_g = 0.0
    cold_start_energy_kwh = 0.0
    cold_start_carbon_g = 0.0
    cold_start_count = 0.0
    ci_values = []

    if len(valid_samples) >= 2:
        for index, sample in enumerate(valid_samples):
            if index + 1 < len(valid_samples):
                dt = max(
                    0.0,
                    float(valid_samples[index + 1]["epoch_seconds"])
                    - float(sample["epoch_seconds"]),
                )
            else:
                dt = config.metrics_poll_seconds
            total_pods = float(sample.get("total_pods") or 0)
            if sample.get("busy_pods") is not None and sample.get("idle_pods") is not None:
                active_pods = min(float(sample.get("busy_pods") or 0), total_pods)
                idle_pods = min(float(sample.get("idle_pods") or 0), total_pods)
            else:
                active_pods = min(estimated_active_pods, total_pods)
                idle_pods = max(0.0, total_pods - active_pods)
            sample_time = pd.Timestamp(sample["timestamp"])
            ci = carbon_intensity_at(
                ci_df,
                sample_time,
                config.carbon_intensity_g_per_kwh,
            )
            ci_values.append(ci)
            sample_active_energy = (
                active_pods * config.power_active_watts * dt / 3_600_000
            )
            sample_idle_energy = (
                idle_pods * config.power_idle_watts * dt / 3_600_000
            )
            active_energy_kwh += sample_active_energy
            idle_energy_kwh += sample_idle_energy
            active_carbon_g += sample_active_energy * ci
            idle_carbon_g += sample_idle_energy * ci

            if index > 0:
                previous_pods = float(valid_samples[index - 1].get("total_pods") or 0)
                new_pods = max(0.0, total_pods - previous_pods)
                if new_pods > 0:
                    cold_start_count += new_pods
                    sample_cold_start_energy = (
                        new_pods * config.cold_start_energy_joules / 3_600_000
                    )
                    cold_start_energy_kwh += sample_cold_start_energy
                    cold_start_carbon_g += sample_cold_start_energy * ci
    else:
        active_energy_kwh = (
            estimated_active_pods * config.power_active_watts * duration_seconds
            / 3_600_000
        )
        idle_energy_kwh = (
            estimated_idle_pods * config.power_idle_watts * duration_seconds
            / 3_600_000
        )
        ci = carbon_intensity_at(
            ci_df,
            datetime.now(timezone.utc),
            config.carbon_intensity_g_per_kwh,
        )
        ci_values.append(ci)
        active_carbon_g = active_energy_kwh * ci
        idle_carbon_g = idle_energy_kwh * ci
        cold_start_count = scaling_metrics.get(
            "cold_start_count",
            scaling_metrics.get(
                "scale_up_pod_count",
                scaling_metrics.get("scale_up_count", 0),
            ),
        )
        cold_start_energy_kwh = (
            cold_start_count * config.cold_start_energy_joules / 3_600_000
        )
        cold_start_carbon_g = cold_start_energy_kwh * ci

    avg_ci = sum(ci_values) / len(ci_values) if ci_values else 0.0
    runtime_energy_kwh = active_energy_kwh + idle_energy_kwh
    total_carbon_g = active_carbon_g + idle_carbon_g + cold_start_carbon_g
    total_energy_with_cold_start_kwh = runtime_energy_kwh + cold_start_energy_kwh
    successful = request_metrics.get("successful_requests", 0)
    estimated_utilization_pct = estimated_active_pods / max(avg_pods, 1e-9) * 100
    estimated_idle_ratio_pct = estimated_idle_pods / max(avg_pods, 1e-9) * 100
    measured_utilization_pct = (
        sum(measured_utilization_values) / len(measured_utilization_values) * 100.0
        if measured_utilization_values
        else None
    )
    measured_idle_ratio_pct = (
        sum(measured_idle_ratio_values) / len(measured_idle_ratio_values) * 100.0
        if measured_idle_ratio_values
        else None
    )
    avg_pod_utilization_pct = (
        measured_utilization_pct
        if measured_utilization_pct is not None
        else estimated_utilization_pct
    )
    avg_idle_ratio_pct = (
        measured_idle_ratio_pct
        if measured_idle_ratio_pct is not None
        else estimated_idle_ratio_pct
    )

    return {
        "estimated_active_pods": estimated_active_pods,
        "estimated_idle_pods": estimated_idle_pods,
        "active_pod_source": active_pod_source,
        "measured_busy_pod_sample_count": len(measured_busy_samples),
        "avg_pod_utilization_pct": avg_pod_utilization_pct,
        "avg_idle_ratio_pct": avg_idle_ratio_pct,
        "estimated_pod_utilization_pct": estimated_utilization_pct,
        "estimated_idle_ratio_pct": estimated_idle_ratio_pct,
        "measured_pod_utilization_pct": measured_utilization_pct,
        "measured_idle_ratio_pct": measured_idle_ratio_pct,
        "active_energy_kwh": active_energy_kwh,
        "idle_energy_kwh": idle_energy_kwh,
        "cold_start_energy_kwh": cold_start_energy_kwh,
        "cold_start_count": cold_start_count,
        "runtime_energy_kwh": runtime_energy_kwh,
        "total_energy_kwh": runtime_energy_kwh,
        "total_energy_with_cold_start_kwh": total_energy_with_cold_start_kwh,
        "energy_per_successful_request_kwh": runtime_energy_kwh / max(successful, 1),
        "energy_per_successful_request_with_cold_start_kwh": (
            total_energy_with_cold_start_kwh / max(successful, 1)
        ),
        "active_carbon_g": active_carbon_g,
        "idle_carbon_g": idle_carbon_g,
        "cold_start_carbon_g": cold_start_carbon_g,
        "total_carbon_g": total_carbon_g,
        "carbon_per_successful_request_g": total_carbon_g / max(successful, 1),
        "carbon_per_request_g": total_carbon_g / max(successful, 1),
        "energy_per_request_kwh": runtime_energy_kwh / max(successful, 1),
        "active_carbon_pct": active_carbon_g / max(total_carbon_g, 1e-9) * 100,
        "idle_carbon_pct": idle_carbon_g / max(total_carbon_g, 1e-9) * 100,
        "cold_start_carbon_pct": cold_start_carbon_g
        / max(total_carbon_g, 1e-9)
        * 100,
        "power_active_watts": config.power_active_watts,
        "power_idle_watts": config.power_idle_watts,
        "cold_start_energy_joules": config.cold_start_energy_joules,
        "carbon_intensity_g_per_kwh": avg_ci,
        "carbon_intensity_csv": config.carbon_intensity_csv,
        "carbon_intensity_source": "csv" if ci_df is not None else "constant",
    }


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_metrics_artifacts(config, replay_result, pod_samples):
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    request_metrics = summarize_request_metrics(replay_result, config)
    scaling_metrics = summarize_scaling_metrics(pod_samples)
    energy_metrics = summarize_energy_metrics_from_samples(
        request_metrics,
        scaling_metrics,
        config,
        pod_samples,
    )

    summary = {
        "run_id": config.run_id,
        "baseline_name": config.baseline_name,
        "request_metrics": request_metrics,
        "scaling_metrics": scaling_metrics,
        "energy_carbon_metrics": energy_metrics,
    }

    summary_path = output_dir / f"{config.run_id}_metrics_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    saved_paths = {"metrics_summary": str(summary_path)}
    if pod_samples:
        pod_metrics_path = output_dir / f"{config.run_id}_pod_metrics.csv"
        write_csv(pod_metrics_path, pod_samples)
        saved_paths["pod_metrics"] = str(pod_metrics_path)

    print("\nSaved computed metrics:")
    for name, path in saved_paths.items():
        print(f"  {name}: {path}")

    return summary, saved_paths
