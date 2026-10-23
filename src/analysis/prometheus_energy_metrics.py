#!/usr/bin/env python3
"""Prometheus-based CPU/memory energy and carbon estimator for qwen experiments."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PROMETHEUS_URL = "http://10.43.143.194:9090"
DEFAULT_CARBON_CSV = "carbon_intensity_ie_2025_5min.csv"
JOULES_PER_KWH = 3_600_000.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def parse_status(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return -1


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def prom_get(base_url: str, path: str, params: dict[str, Any]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    url = f"{base_url.rstrip('/')}{path}?{query}"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("status") != "success":
        raise RuntimeError(f"Prometheus query failed: {payload}")
    return payload


def prom_query_range(base_url: str, expr: str, start: float, end: float, step: float) -> list[dict[str, Any]]:
    payload = prom_get(
        base_url,
        "/api/v1/query_range",
        {"query": expr, "start": f"{start:.3f}", "end": f"{end:.3f}", "step": f"{step:g}"},
    )
    return payload.get("data", {}).get("result", [])


def prom_query(base_url: str, expr: str, ts: float | None = None) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"query": expr}
    if ts is not None:
        params["time"] = f"{ts:.3f}"
    payload = prom_get(base_url, "/api/v1/query", params)
    return payload.get("data", {}).get("result", [])


def carbon_column(fieldnames: list[str]) -> str:
    lowered = {name.lower(): name for name in fieldnames}
    for key, name in lowered.items():
        if "carbon intensity" in key and "life cycle" in key and "kwh" in key:
            return name
    for key, name in lowered.items():
        if "carbon intensity" in key and "kwh" in key:
            return name
    raise ValueError("No carbon intensity gCO2e/kWh column found")


def parse_utc_datetime(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_optional_utc_timestamp(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return parse_utc_datetime(text).timestamp()
    except ValueError:
        return None


def load_carbon_values(path: Path) -> tuple[list[float], list[float], str]:
    if not path.exists():
        return [], [], "missing"
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return [], [], "empty"
        column = carbon_column(reader.fieldnames)
        time_column = "Datetime (UTC)" if "Datetime (UTC)" in reader.fieldnames else None
        times: list[float] = []
        values: list[float] = []
        for row in reader:
            values.append(parse_float(row.get(column), 0.0))
            if time_column:
                parsed = parse_optional_utc_timestamp(row.get(time_column))
                if parsed is not None:
                    times.append(parsed)
    if len(times) != len(values):
        times = []
    return times, values, column


def carbon_at_elapsed(
    values: list[float],
    times: list[float],
    start_index: int,
    elapsed_s: float,
    fallback: float,
    ci_trace_start_epoch: float | None,
    real_window_seconds: float,
    ci_window_seconds: float,
    ci_time_scale: float | None = None,
) -> float:
    if not values:
        return fallback
    if ci_trace_start_epoch is not None and times:
        if ci_time_scale is not None:
            target_epoch = ci_trace_start_epoch + max(0.0, elapsed_s) * ci_time_scale
        else:
            window_index = max(0, int(elapsed_s // max(real_window_seconds, 1e-9)))
            target_epoch = ci_trace_start_epoch + window_index * max(ci_window_seconds, 1e-9)
        index = bisect.bisect_right(times, target_epoch) - 1
    else:
        index = start_index + max(0, int(elapsed_s // 300.0))
    if index >= len(values):
        index = len(values) - 1
    if index < 0:
        index = 0
    return values[index]


def read_request_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_window_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_requests(rows: list[dict[str, Any]], duration_s: float) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    ok_latencies: list[float] = []
    timeout_count = 0
    for row in rows:
        status = parse_status(row.get("status"))
        status_key = str(status)
        status_counts[status_key] = status_counts.get(status_key, 0) + 1
        elapsed = parse_float(row.get("elapsed"), 0.0)
        error = str(row.get("error") or "").lower()
        if status == 200:
            ok_latencies.append(elapsed)
        if status == 0 or "timeout" in error:
            timeout_count += 1
    total = len(rows)
    ok = len(ok_latencies)
    return {
        "total_requests": total,
        "successful_requests": ok,
        "failed_requests": total - ok,
        "success_rate": ok / total if total else 0.0,
        "timeout_requests": timeout_count,
        "throughput_success_rps": ok / max(duration_s, 1e-9),
        "p50_latency_seconds": percentile(ok_latencies, 50),
        "p95_latency_seconds": percentile(ok_latencies, 95),
        "p99_latency_seconds": percentile(ok_latencies, 99),
        "mean_latency_seconds": statistics.mean(ok_latencies) if ok_latencies else 0.0,
        "status_counts": status_counts,
    }


def window_request_map(window_rows: list[dict[str, Any]], capacity_per_window: float) -> dict[int, dict[str, float]]:
    mapping: dict[int, dict[str, float]] = {}
    for row in window_rows:
        index = parse_status(row.get("window_index"))
        if index < 0:
            continue
        generated = parse_float(row.get("generated_requests"), parse_float(row.get("sent_requests"), 0.0))
        sent = parse_float(row.get("sent_requests"), generated)
        required = math.ceil(generated / max(capacity_per_window, 1e-9)) if generated > 0 else 0
        mapping[index] = {
            "generated_requests": generated,
            "sent_requests": sent,
            "required_pods": required,
        }
    return mapping


def series_values_by_key(
    series: list[dict[str, Any]],
    label_names: tuple[str, ...],
) -> dict[tuple[int, tuple[str, ...]], float]:
    out: dict[tuple[int, tuple[str, ...]], float] = {}
    for item in series:
        metric = item.get("metric", {})
        labels = tuple(str(metric.get(name, "")) for name in label_names)
        for ts, value in item.get("values", []):
            try:
                out[(int(float(ts)), labels)] = float(value)
            except (TypeError, ValueError):
                continue
    return out


def phase_by_ts_pod(series: list[dict[str, Any]]) -> dict[tuple[int, str], str]:
    out: dict[tuple[int, str], str] = {}
    best: dict[tuple[int, str], tuple[float, str]] = {}
    for item in series:
        metric = item.get("metric", {})
        pod = str(metric.get("pod", ""))
        phase = str(metric.get("phase", ""))
        for ts, value in item.get("values", []):
            try:
                key = (int(float(ts)), pod)
                val = float(value)
            except (TypeError, ValueError):
                continue
            if key not in best or val > best[key][0]:
                best[key] = (val, phase)
    for key, (_, phase) in best.items():
        out[key] = phase
    return out


def latest_node_map(base_url: str, end: float, pod_regex: str) -> dict[str, str]:
    expr = f'kube_pod_info{{namespace="default",pod=~"{pod_regex}"}}'
    result = prom_query(base_url, expr, ts=end)
    mapping: dict[str, str] = {}
    for item in result:
        metric = item.get("metric", {})
        pod = str(metric.get("pod", ""))
        node = str(metric.get("node", ""))
        if pod:
            mapping[pod] = node
    return mapping


def pod_metric_values(
    series: list[dict[str, Any]],
) -> dict[str, list[tuple[int, float]]]:
    out: dict[str, list[tuple[int, float]]] = {}
    for item in series:
        metric = item.get("metric", {})
        pod = str(metric.get("pod", ""))
        if not pod:
            continue
        values = out.setdefault(pod, [])
        for ts, value in item.get("values", []):
            try:
                values.append((int(float(ts)), float(value)))
            except (TypeError, ValueError):
                continue
    for values in out.values():
        values.sort(key=lambda item: item[0])
    return out


def pod_creation_times(args: argparse.Namespace) -> dict[str, float]:
    expr = f'kube_pod_created{{namespace="{args.namespace}",pod=~"{args.pod_regex}"}}'
    out: dict[str, float] = {}
    for pod, values in pod_metric_values(
        prom_query_range(
            args.prometheus_url,
            expr,
            args.start_epoch,
            args.end_epoch,
            args.step_seconds,
        )
    ).items():
        creation_values = [value for _, value in values if value > 0]
        if creation_values:
            out[pod] = min(creation_values)
    return out


def pod_ready_times(args: argparse.Namespace) -> dict[str, float]:
    expr = (
        f'kube_pod_status_ready{{namespace="{args.namespace}",'
        f'pod=~"{args.pod_regex}",condition="true"}}'
    )
    out: dict[str, float] = {}
    for pod, values in pod_metric_values(
        prom_query_range(
            args.prometheus_url,
            expr,
            args.start_epoch,
            args.end_epoch,
            args.step_seconds,
        )
    ).items():
        ready_samples = [ts for ts, value in values if value >= 0.5]
        if ready_samples:
            out[pod] = float(min(ready_samples))
    return out


def pod_ready_times_from_container_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    first_ready_by_container: dict[tuple[str, str], float] = {}
    for row in rows:
        if int(row.get("container_ready") or 0) < 1:
            continue
        pod = str(row["pod_name"])
        container = str(row["container_name"])
        ts = float(row["epoch_seconds"])
        key = (pod, container)
        first_ready_by_container[key] = min(first_ready_by_container.get(key, float("inf")), ts)

    ready_by_pod: dict[str, float] = {}
    containers_by_pod: dict[str, list[float]] = {}
    for (pod, _container), ts in first_ready_by_container.items():
        containers_by_pod.setdefault(pod, []).append(ts)
    for pod, timestamps in containers_by_pod.items():
        ready_by_pod[pod] = max(timestamps)
    return ready_by_pod


def build_cold_start_intervals(
    rows: list[dict[str, Any]],
    first_seen: dict[str, float],
    new_pods: set[str],
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    creation_times = pod_creation_times(args)
    ready_times = pod_ready_times(args)
    inferred_ready_times = pod_ready_times_from_container_rows(rows)

    cold_pods = set(new_pods)
    cold_pods.update(
        pod for pod, created in creation_times.items()
        if args.start_epoch < created <= args.end_epoch
    )

    intervals: dict[str, dict[str, Any]] = {}
    measured_latencies: list[float] = []
    fallback_latencies: list[float] = []
    for pod in sorted(cold_pods):
        first = first_seen.get(pod)
        created = creation_times.get(pod)
        ready = ready_times.get(pod, inferred_ready_times.get(pod))

        if created is not None and ready is not None and ready >= created:
            start = created
            end = ready
            mode = "measured_creation_to_ready"
            measured_latencies.append(max(0.0, end - start))
        elif created is not None:
            start = created
            end = created + args.cold_start_window_seconds
            mode = "fallback_creation_plus_window"
            fallback_latencies.append(max(0.0, end - start))
        elif first is not None and ready is not None and ready >= first:
            start = first
            end = ready
            mode = "fallback_first_seen_to_ready"
            fallback_latencies.append(max(0.0, end - start))
        elif first is not None:
            start = first
            end = first + args.cold_start_window_seconds
            mode = "fallback_first_seen_plus_window"
            fallback_latencies.append(max(0.0, end - start))
        else:
            continue

        start = max(start, args.start_epoch)
        end = min(max(end, start), args.end_epoch)
        intervals[pod] = {
            "start_epoch": start,
            "end_epoch": end,
            "latency_seconds": max(0.0, end - start),
            "mode": mode,
            "created_epoch": created,
            "ready_epoch": ready,
            "first_seen_epoch": first,
        }

    all_latencies = measured_latencies + fallback_latencies
    modes = {entry["mode"] for entry in intervals.values()}
    if not intervals:
        interval_mode = "none"
    elif modes == {"measured_creation_to_ready"}:
        interval_mode = "measured_creation_to_ready"
    elif "measured_creation_to_ready" in modes:
        interval_mode = "mixed_measured_and_fallback"
    else:
        interval_mode = "fallback"

    summary = {
        "cold_start_interval_mode": interval_mode,
        "cold_start_measured_count": len(measured_latencies),
        "cold_start_fallback_count": len(fallback_latencies),
        "cold_start_lifecycle_metric_created_count": len(creation_times),
        "cold_start_lifecycle_metric_ready_count": len(ready_times),
        "cold_start_latency_mean_seconds": (
            statistics.mean(all_latencies) if all_latencies else 0.0
        ),
        "cold_start_latency_p95_seconds": percentile(all_latencies, 95),
        "cold_start_latency_max_seconds": max(all_latencies) if all_latencies else 0.0,
    }
    return intervals, summary


def build_resource_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    pod_regex = args.pod_regex
    cpu_expr = (
        "sum by (pod, container) "
        f'(rate(container_cpu_usage_seconds_total{{namespace="{args.namespace}",pod=~"{pod_regex}",container!="POD",container!=""}}[{int(args.cpu_rate_window_seconds)}s]))'
    )
    mem_expr = (
        "avg by (pod, container) "
        f'(container_memory_working_set_bytes{{namespace="{args.namespace}",pod=~"{pod_regex}",container!="POD",container!=""}})'
    )
    phase_expr = f'max by (pod, phase) (kube_pod_status_phase{{namespace="{args.namespace}",pod=~"{pod_regex}"}})'
    ready_expr = (
        "max by (pod, container) "
        f'(kube_pod_container_status_ready{{namespace="{args.namespace}",pod=~"{pod_regex}",container!="POD",container!=""}})'
    )

    cpu = series_values_by_key(
        prom_query_range(args.prometheus_url, cpu_expr, args.start_epoch, args.end_epoch, args.step_seconds),
        ("pod", "container"),
    )
    mem = series_values_by_key(
        prom_query_range(args.prometheus_url, mem_expr, args.start_epoch, args.end_epoch, args.step_seconds),
        ("pod", "container"),
    )
    ready = series_values_by_key(
        prom_query_range(args.prometheus_url, ready_expr, args.start_epoch, args.end_epoch, args.step_seconds),
        ("pod", "container"),
    )
    phase = phase_by_ts_pod(
        prom_query_range(args.prometheus_url, phase_expr, args.start_epoch, args.end_epoch, args.step_seconds)
    )
    node_map = latest_node_map(args.prometheus_url, args.end_epoch, pod_regex)

    timestamps = sorted({key[0] for key in set(cpu) | set(mem)})
    next_dt: dict[int, float] = {}
    for index, ts in enumerate(timestamps):
        if index + 1 < len(timestamps):
            next_dt[ts] = max(0.0, min(float(timestamps[index + 1] - ts), args.step_seconds * 2))
        else:
            next_dt[ts] = args.step_seconds

    keys = sorted(set(cpu) | set(mem), key=lambda item: (item[0], item[1]))
    rows: list[dict[str, Any]] = []
    for ts, labels in keys:
        pod, container = labels
        if ts < int(args.start_epoch) or ts > int(args.end_epoch):
            continue
        cpu_cores = max(0.0, cpu.get((ts, labels), 0.0))
        memory_bytes = max(0.0, mem.get((ts, labels), 0.0))
        dt = next_dt.get(ts, args.step_seconds)
        memory_gb = memory_bytes / 1_073_741_824.0
        elapsed = max(0.0, float(ts) - args.start_epoch)
        rows.append({
            "timestamp": datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds"),
            "epoch_seconds": float(ts),
            "elapsed_seconds": elapsed,
            "dt_seconds": dt,
            "run_id": args.run_id,
            "namespace": args.namespace,
            "pod_name": pod,
            "container_name": container,
            "node_name": node_map.get(pod, ""),
            "pod_phase": phase.get((ts, pod), ""),
            "container_ready": int(ready.get((ts, labels), 0.0)),
            "cpu_cores": cpu_cores,
            "memory_bytes": memory_bytes,
            "cpu_core_seconds": cpu_cores * dt,
            "memory_gb_seconds": memory_gb * dt,
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_energy_and_carbon(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    carbon_times: list[float],
    carbon_values: list[float],
) -> None:
    for row in rows:
        elapsed = float(row["elapsed_seconds"])
        ci = carbon_at_elapsed(
            carbon_values,
            carbon_times,
            args.carbon_start_index,
            elapsed,
            args.fallback_carbon_intensity_g_per_kwh,
            args._ci_trace_start_epoch,
            args.carbon_lookup_window_seconds,
            args.ci_window_seconds,
            args.ci_time_scale,
        )
        cpu_j = float(row["cpu_core_seconds"]) * args.cpu_joules_per_core_second
        mem_j = float(row["memory_gb_seconds"]) * args.memory_joules_per_gb_second
        energy_j = cpu_j + mem_j
        row["carbon_intensity_g_per_kwh"] = ci
        row["cpu_energy_j"] = cpu_j
        row["memory_energy_j"] = mem_j
        row["estimated_energy_j"] = energy_j
        row["estimated_energy_kwh"] = energy_j / JOULES_PER_KWH
        row["estimated_carbon_gco2e"] = (energy_j / JOULES_PER_KWH) * ci


def aggregate_window_rows(
    rows: list[dict[str, Any]],
    request_by_window: dict[int, dict[str, float]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_window: dict[int, dict[str, Any]] = {}
    pods_by_ts: dict[float, set[str]] = {}
    for row in rows:
        ts = float(row["epoch_seconds"])
        pods_by_ts.setdefault(ts, set()).add(str(row["pod_name"]))

    running_counts = {ts: len(pods) for ts, pods in pods_by_ts.items()}
    sorted_ts = sorted(running_counts)
    scale_up_pod_count = 0
    scale_down_pod_count = 0
    previous_count = None
    for ts in sorted_ts:
        count = running_counts[ts]
        if previous_count is not None:
            diff = count - previous_count
            if diff > 0:
                scale_up_pod_count += diff
            elif diff < 0:
                scale_down_pod_count += -diff
        previous_count = count

    first_seen: dict[str, float] = {}
    for row in rows:
        pod = str(row["pod_name"])
        first_seen[pod] = min(first_seen.get(pod, float("inf")), float(row["epoch_seconds"]))
    initial_cutoff = args.start_epoch + args.step_seconds + 0.001
    new_pods = {pod for pod, first in first_seen.items() if first > initial_cutoff}
    cold_start_intervals, cold_start_interval_summary = build_cold_start_intervals(
        rows,
        first_seen,
        new_pods,
        args,
    )

    total_idle_capacity_energy_j = 0.0
    total_cold_resource_energy_j = 0.0

    energy_by_ts: dict[float, float] = {}
    ci_by_ts: dict[float, float] = {}
    for row in rows:
        ts = float(row["epoch_seconds"])
        energy_by_ts[ts] = energy_by_ts.get(ts, 0.0) + float(row["estimated_energy_j"])
        ci_by_ts[ts] = float(row["carbon_intensity_g_per_kwh"])

    extra_ratio_by_ts: dict[float, float] = {}
    for ts, pods in pods_by_ts.items():
        elapsed = max(0.0, ts - args.start_epoch)
        window_index = int(elapsed // args.window_size_seconds)
        running = len(pods)
        required = int(request_by_window.get(window_index, {}).get("required_pods", 0))
        extra = max(0, running - required)
        extra_ratio_by_ts[ts] = extra / running if running > 0 else 0.0

    for row in rows:
        ts = float(row["epoch_seconds"])
        elapsed = max(0.0, ts - args.start_epoch)
        window_index = int(elapsed // args.window_size_seconds)
        entry = by_window.setdefault(window_index, {
            "run_id": args.run_id,
            "window_index": window_index,
            "sample_count": 0,
            "cpu_core_seconds": 0.0,
            "memory_gb_seconds": 0.0,
            "cpu_energy_j": 0.0,
            "memory_energy_j": 0.0,
            "estimated_energy_j": 0.0,
            "estimated_carbon_gco2e": 0.0,
            "max_running_pods": 0,
            "avg_running_pods_accum": 0.0,
            "carbon_intensity_g_per_kwh": float(row["carbon_intensity_g_per_kwh"]),
            **request_by_window.get(window_index, {}),
        })
        entry["sample_count"] += 1
        entry["cpu_core_seconds"] += float(row["cpu_core_seconds"])
        entry["memory_gb_seconds"] += float(row["memory_gb_seconds"])
        entry["cpu_energy_j"] += float(row["cpu_energy_j"])
        entry["memory_energy_j"] += float(row["memory_energy_j"])
        entry["estimated_energy_j"] += float(row["estimated_energy_j"])
        entry["estimated_carbon_gco2e"] += float(row["estimated_carbon_gco2e"])
        running = running_counts.get(ts, 0)
        entry["max_running_pods"] = max(int(entry["max_running_pods"]), running)
        entry["avg_running_pods_accum"] += running

        idle_piece = float(row["estimated_energy_j"]) * extra_ratio_by_ts.get(ts, 0.0)
        total_idle_capacity_energy_j += idle_piece
        row["idle_capacity_energy_j"] = idle_piece
        row["idle_capacity_carbon_gco2e"] = (
            idle_piece / JOULES_PER_KWH
        ) * float(row["carbon_intensity_g_per_kwh"])

        pod = str(row["pod_name"])
        interval = cold_start_intervals.get(pod)
        cold_piece = 0.0
        overlap_seconds = 0.0
        if interval:
            sample_start = ts
            sample_end = ts + float(row["dt_seconds"])
            overlap_seconds = max(
                0.0,
                min(sample_end, float(interval["end_epoch"]))
                - max(sample_start, float(interval["start_epoch"])),
            )
            if overlap_seconds > 0.0 and float(row["dt_seconds"]) > 0.0:
                cold_piece = (
                    float(row["estimated_energy_j"])
                    * overlap_seconds
                    / float(row["dt_seconds"])
                )
        total_cold_resource_energy_j += cold_piece
        row["cold_start_interval_start_epoch"] = (
            float(interval["start_epoch"]) if interval else ""
        )
        row["cold_start_interval_end_epoch"] = (
            float(interval["end_epoch"]) if interval else ""
        )
        row["cold_start_interval_mode"] = str(interval["mode"]) if interval else ""
        row["cold_start_latency_seconds"] = (
            float(interval["latency_seconds"]) if interval else ""
        )
        row["cold_start_created_epoch"] = (
            float(interval["created_epoch"])
            if interval and interval["created_epoch"] is not None else ""
        )
        row["cold_start_ready_epoch"] = (
            float(interval["ready_epoch"])
            if interval and interval["ready_epoch"] is not None else ""
        )
        row["cold_start_first_seen_epoch"] = (
            float(interval["first_seen_epoch"])
            if interval and interval["first_seen_epoch"] is not None else ""
        )
        row["cold_start_sample_overlap_seconds"] = overlap_seconds
        row["cold_start_resource_energy_j"] = cold_piece
        row["cold_start_resource_carbon_gco2e"] = (
            cold_piece / JOULES_PER_KWH
        ) * float(row["carbon_intensity_g_per_kwh"])

    window_rows = []
    for index in sorted(by_window):
        entry = by_window[index]
        samples = max(int(entry.pop("sample_count")), 1)
        entry["avg_running_pods"] = entry.pop("avg_running_pods_accum") / samples
        entry["estimated_energy_kwh"] = entry["estimated_energy_j"] / JOULES_PER_KWH
        window_rows.append(entry)

    scale_summary = {
        "observed_scale_up_pod_count": scale_up_pod_count,
        "observed_scale_down_pod_count": scale_down_pod_count,
        "new_pod_name_count_after_start": len(new_pods),
        "cold_start_interval_pod_count": len(cold_start_intervals),
        "cold_start_count": max(scale_up_pod_count, len(cold_start_intervals)),
        "idle_capacity_energy_j": total_idle_capacity_energy_j,
        "cold_start_resource_energy_j": total_cold_resource_energy_j,
        **cold_start_interval_summary,
    }
    return window_rows, scale_summary


def sensitivity_rows(
    rows: list[dict[str, Any]],
    cold_count: float,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    scenarios = [
        ("default", 1.0, 1.0, 1.0),
        ("low_cpu_mem", 0.5, 0.5, 1.0),
        ("high_cpu_mem", 2.0, 2.0, 1.0),
        ("high_cpu", 2.0, 1.0, 1.0),
        ("high_memory", 1.0, 2.0, 1.0),
        ("no_cold_model", 1.0, 1.0, 0.0),
        ("high_cold_model", 1.0, 1.0, 3.0),
    ]
    out = []
    for name, cpu_mult, mem_mult, cold_mult in scenarios:
        energy_j = 0.0
        carbon_g = 0.0
        for row in rows:
            e = (
                float(row["cpu_energy_j"]) * cpu_mult
                + float(row["memory_energy_j"]) * mem_mult
            )
            energy_j += e
            carbon_g += (e / JOULES_PER_KWH) * float(row["carbon_intensity_g_per_kwh"])
        cold_j = cold_count * args.cold_start_model_joules * cold_mult
        # Assign model cold-start carbon using the average CI observed in samples.
        avg_ci = (
            sum(float(row["carbon_intensity_g_per_kwh"]) for row in rows) / len(rows)
            if rows else args.fallback_carbon_intensity_g_per_kwh
        )
        carbon_g += (cold_j / JOULES_PER_KWH) * avg_ci
        energy_j += cold_j
        out.append({
            "run_id": args.run_id,
            "scenario": name,
            "cpu_multiplier": cpu_mult,
            "memory_multiplier": mem_mult,
            "cold_multiplier": cold_mult,
            "estimated_energy_j": energy_j,
            "estimated_energy_kwh": energy_j / JOULES_PER_KWH,
            "estimated_carbon_gco2e": carbon_g,
        })
    return out


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    carbon_times, carbon_values, carbon_source = load_carbon_values(Path(args.carbon_intensity_csv))

    resource_rows = build_resource_rows(args)
    add_energy_and_carbon(resource_rows, args, carbon_times, carbon_values)

    request_rows = read_request_log(output_dir / f"{args.run_id}_replay_request_log.csv")
    window_summary_rows = read_window_summary(output_dir / f"{args.run_id}_replay_window_summary.csv")
    request_metrics = summarize_requests(request_rows, args.end_epoch - args.start_epoch)
    request_by_window = window_request_map(window_summary_rows, args.pod_capacity_per_window)
    window_rows, scale_summary = aggregate_window_rows(resource_rows, request_by_window, args)

    total_energy_j = sum(float(row["estimated_energy_j"]) for row in resource_rows)
    total_carbon_g = sum(float(row["estimated_carbon_gco2e"]) for row in resource_rows)
    cpu_core_seconds = sum(float(row["cpu_core_seconds"]) for row in resource_rows)
    memory_gb_seconds = sum(float(row["memory_gb_seconds"]) for row in resource_rows)
    cold_model_j = scale_summary["cold_start_count"] * args.cold_start_model_joules
    avg_ci = (
        sum(float(row["carbon_intensity_g_per_kwh"]) for row in resource_rows) / len(resource_rows)
        if resource_rows else args.fallback_carbon_intensity_g_per_kwh
    )
    cold_model_carbon = (cold_model_j / JOULES_PER_KWH) * avg_ci
    total_with_cold_model_j = total_energy_j + cold_model_j
    total_with_cold_model_carbon = total_carbon_g + cold_model_carbon
    successful = max(int(request_metrics.get("successful_requests") or 0), 1)

    summary = {
        "run_id": args.run_id,
        "method": args.method,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prometheus_url": args.prometheus_url,
        "start_epoch": args.start_epoch,
        "end_epoch": args.end_epoch,
        "step_seconds": args.step_seconds,
        "cpu_rate_window_seconds": args.cpu_rate_window_seconds,
        "carbon_intensity_csv": args.carbon_intensity_csv,
        "carbon_intensity_column": carbon_source,
        "carbon_start_index": args.carbon_start_index,
        "ci_trace_start": args.ci_trace_start,
        "ci_window_seconds": args.ci_window_seconds,
        "ci_time_scale": args.ci_time_scale,
        "window_size_seconds": args.window_size_seconds,
        "carbon_lookup_window_seconds": args.carbon_lookup_window_seconds,
        "ci_mapping_mode": (
            "synthetic_elapsed_time_scale"
            if args.ci_trace_start and args.ci_time_scale is not None
            else "synthetic_window_time"
            if args.ci_trace_start
            else "legacy_start_index_5min"
        ),
        "cpu_joules_per_core_second": args.cpu_joules_per_core_second,
        "memory_joules_per_gb_second": args.memory_joules_per_gb_second,
        "cold_start_model_joules": args.cold_start_model_joules,
        "cold_start_fallback_window_seconds": args.cold_start_window_seconds,
        "resource_sample_rows": len(resource_rows),
        "cpu_core_seconds": cpu_core_seconds,
        "memory_gb_seconds": memory_gb_seconds,
        "estimated_resource_energy_j": total_energy_j,
        "estimated_resource_energy_kwh": total_energy_j / JOULES_PER_KWH,
        "estimated_resource_carbon_gco2e": total_carbon_g,
        "cold_start_model_energy_j": cold_model_j,
        "cold_start_model_carbon_gco2e": cold_model_carbon,
        "estimated_total_energy_j": total_with_cold_model_j,
        "estimated_total_energy_kwh": total_with_cold_model_j / JOULES_PER_KWH,
        "estimated_total_carbon_gco2e": total_with_cold_model_carbon,
        "energy_per_successful_request_j": total_with_cold_model_j / successful,
        "carbon_per_successful_request_gco2e": total_with_cold_model_carbon / successful,
        "idle_capacity_energy_j": scale_summary["idle_capacity_energy_j"],
        "cold_start_resource_energy_j": scale_summary["cold_start_resource_energy_j"],
        **request_metrics,
        **scale_summary,
        "notes": (
            "Estimated from Prometheus CPU core-seconds and memory GB-seconds. "
            "This is not measured wall power."
        ),
    }

    sensitivity = sensitivity_rows(resource_rows, scale_summary["cold_start_count"], args)
    for row in sensitivity:
        ok = max(int(request_metrics.get("successful_requests") or 0), 1)
        row["energy_per_successful_request_j"] = float(row["estimated_energy_j"]) / ok
        row["carbon_per_successful_request_gco2e"] = float(row["estimated_carbon_gco2e"]) / ok

    paths = {
        "resource_samples_csv": output_dir / f"{args.run_id}_prometheus_resource_samples.csv",
        "window_metrics_csv": output_dir / f"{args.run_id}_prometheus_windows.csv",
        "sensitivity_csv": output_dir / f"{args.run_id}_prometheus_sensitivity.csv",
        "summary_json": output_dir / f"{args.run_id}_prometheus_summary.json",
        "summary_csv": output_dir / f"{args.run_id}_prometheus_summary.csv",
    }
    write_csv(paths["resource_samples_csv"], resource_rows)
    write_csv(paths["window_metrics_csv"], window_rows)
    write_csv(paths["sensitivity_csv"], sensitivity)
    paths["summary_json"].write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(paths["summary_csv"], [summary])

    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2), flush=True)
    print(json.dumps({
        "run_id": args.run_id,
        "method": args.method,
        "successful_requests": summary["successful_requests"],
        "p95_latency_seconds": summary["p95_latency_seconds"],
        "p99_latency_seconds": summary["p99_latency_seconds"],
        "estimated_total_energy_j": summary["estimated_total_energy_j"],
        "estimated_total_carbon_gco2e": summary["estimated_total_carbon_gco2e"],
        "energy_per_successful_request_j": summary["energy_per_successful_request_j"],
        "carbon_per_successful_request_gco2e": summary["carbon_per_successful_request_gco2e"],
    }, indent=2), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--pod-regex", default="qwen.*")
    parser.add_argument("--prometheus-url", default=DEFAULT_PROMETHEUS_URL)
    parser.add_argument("--start-epoch", type=float, required=True)
    parser.add_argument("--end-epoch", type=float, required=True)
    parser.add_argument("--step-seconds", type=float, default=5.0)
    parser.add_argument("--cpu-rate-window-seconds", type=float, default=30.0)
    parser.add_argument("--window-size-seconds", type=float, default=20.0)
    parser.add_argument("--pod-capacity-per-window", type=float, default=13.75)
    parser.add_argument("--carbon-intensity-csv", default=DEFAULT_CARBON_CSV)
    parser.add_argument("--carbon-start-index", type=int, default=296)
    parser.add_argument("--ci-trace-start", type=str, default=None)
    parser.add_argument("--ci-window-seconds", type=float, default=20.0)
    parser.add_argument(
        "--ci-time-scale",
        type=float,
        default=None,
        help=(
            "Synthetic CI seconds per real elapsed second. When set with "
            "--ci-trace-start, CI lookup uses trace_start + elapsed_seconds * scale."
        ),
    )
    parser.add_argument(
        "--carbon-lookup-window-seconds",
        type=float,
        default=None,
        help=(
            "Elapsed-time window used only for CI trace lookup; defaults to "
            "--window-size-seconds."
        ),
    )
    parser.add_argument("--fallback-carbon-intensity-g-per-kwh", type=float, default=0.0)
    parser.add_argument("--cpu-joules-per-core-second", type=float, default=10.8)
    parser.add_argument("--memory-joules-per-gb-second", type=float, default=0.3725)
    parser.add_argument("--cold-start-model-joules", type=float, default=10.38)
    parser.add_argument("--cold-start-window-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.end_epoch <= args.start_epoch:
        raise SystemExit("--end-epoch must be greater than --start-epoch")
    if args.ci_window_seconds <= 0:
        raise SystemExit("--ci-window-seconds must be positive")
    if args.ci_time_scale is not None and args.ci_time_scale <= 0:
        raise SystemExit("--ci-time-scale must be positive")
    if args.carbon_lookup_window_seconds is None:
        args.carbon_lookup_window_seconds = args.window_size_seconds
    if args.carbon_lookup_window_seconds <= 0:
        raise SystemExit("--carbon-lookup-window-seconds must be positive")
    args._ci_trace_start_epoch = (
        parse_utc_datetime(args.ci_trace_start).timestamp()
        if args.ci_trace_start else None
    )
    return args


if __name__ == "__main__":
    raise SystemExit(main())
