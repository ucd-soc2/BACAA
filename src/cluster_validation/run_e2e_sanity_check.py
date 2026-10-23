#!/usr/bin/env python3
"""Short end-to-end sanity check for the formal qwen autoscaler protocol."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE = Path.home() / "online_experiment"
RESULTS = BASE / "results"
LOGS = BASE / "logs"
PYTHON = BASE / ".venv" / "bin" / "python"
NAMESPACE = "default"
SERVICE = "qwen"
QWEN_URL = "http://192.168.10.101:31011/v1/chat/completions"
QWEN_HOST = "qwen.default.example.com"
PROMETHEUS_URL = "http://10.43.143.194:9090"
WORKER_NODES = {"llm-node-1", "llm-node2", "llm-node3"}
INITIAL_PLACEMENT_NODES = ["llm-node-1", "llm-node2"]

KUBECTL_TIMEOUT_SECONDS = 30.0
KUBECTL_RETRY_ATTEMPTS = 4
KUBECTL_RETRY_DELAY_SECONDS = 3.0
API_OUTAGE_GRACE_SECONDS = 180.0
API_UNAVAILABLE_PATTERNS = (
    "the connection to the server 127.0.0.1:6443 was refused",
    "command timed out after",
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


def parse_duration_seconds(value: str) -> float | None:
    text = str(value).strip().lower()
    if not text:
        return None
    multiplier = 1.0
    if text.endswith("ms"):
        multiplier = 0.001
        text = text[:-2]
    elif text.endswith("s"):
        text = text[:-1]
    elif text.endswith("m"):
        multiplier = 60.0
        text = text[:-1]
    elif text.endswith("h"):
        multiplier = 3600.0
        text = text[:-1]
    try:
        return float(text) * multiplier
    except ValueError:
        return None


def kubectl_declared_timeout_seconds(cmd_s: list[str]) -> float | None:
    for index, part in enumerate(cmd_s):
        if part == "--timeout" and index + 1 < len(cmd_s):
            return parse_duration_seconds(cmd_s[index + 1])
        if part.startswith("--timeout="):
            return parse_duration_seconds(part.split("=", 1)[1])
    return None

METHOD_YAMLS = {
    "kpa": "experiment-yamls/qwen-kpa.yaml",
    "hpa": "experiment-yamls/qwen-hpa.yaml",
    "ets": "experiment-yamls/qwen-ets.yaml",
    "bilstm": "experiment-yamls/qwen-bilstm.yaml",
    "transformer": "experiment-yamls/qwen-transformer.yaml",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class KubeApiUnavailable(RuntimeError):
    """Raised when kubectl cannot reach a healthy Kubernetes API server."""


def is_kubectl_command(cmd_s: list[str]) -> bool:
    return any(part.endswith("kubectl") for part in cmd_s[:3])


def is_kube_api_unavailable(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in API_UNAVAILABLE_PATTERNS)


def run_capture(
    cmd: list[object],
    *,
    check: bool = True,
    timeout_s: float | None = None,
    attempts: int | None = None,
) -> str:
    cmd_s = [str(part) for part in cmd]
    print("+ " + " ".join(cmd_s), flush=True)
    kubectl_cmd = is_kubectl_command(cmd_s)
    if timeout_s is None and kubectl_cmd:
        declared_timeout_s = kubectl_declared_timeout_seconds(cmd_s)
        if declared_timeout_s is None:
            timeout_s = KUBECTL_TIMEOUT_SECONDS
        else:
            timeout_s = max(KUBECTL_TIMEOUT_SECONDS, declared_timeout_s + 30.0)
    if attempts is None:
        attempts = KUBECTL_RETRY_ATTEMPTS if kubectl_cmd else 1

    output = ""
    returncode = 0
    for attempt in range(1, max(1, attempts) + 1):
        try:
            proc = subprocess.run(
                cmd_s,
                cwd=BASE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                timeout=timeout_s,
            )
            output = proc.stdout or ""
            returncode = proc.returncode
        except subprocess.TimeoutExpired as exc:
            partial_stdout = exc.stdout or ""
            partial_stderr = exc.stderr or ""
            if isinstance(partial_stdout, bytes):
                partial_stdout = partial_stdout.decode(errors="replace")
            if isinstance(partial_stderr, bytes):
                partial_stderr = partial_stderr.decode(errors="replace")
            output = (
                f"command timed out after {timeout_s}s\n"
                f"{partial_stdout}{partial_stderr}"
            )
            returncode = 124

        if returncode == 0:
            return output.strip()

        transient_api = kubectl_cmd and is_kube_api_unavailable(output)
        if transient_api and attempt < max(1, attempts):
            print(
                "kubectl_api_unavailable_retry="
                + json.dumps(
                    {
                        "attempt": attempt,
                        "attempts": attempts,
                        "sleep_seconds": KUBECTL_RETRY_DELAY_SECONDS,
                        "cmd": cmd_s,
                        "output": output.strip()[-500:],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            time.sleep(KUBECTL_RETRY_DELAY_SECONDS)
            continue
        break

    if check and returncode != 0:
        if kubectl_cmd and is_kube_api_unavailable(output):
            raise KubeApiUnavailable(output.strip())
        raise RuntimeError(output)
    return output.strip()


def run_logged(cmd: list[object], log_path: Path, *, check: bool = True) -> int:
    cmd_s = [str(part) for part in cmd]
    print("+ " + " ".join(cmd_s), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write("+ " + " ".join(cmd_s) + "\n")
        handle.flush()
        proc = subprocess.run(
            cmd_s,
            cwd=BASE,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed with {proc.returncode}: {' '.join(cmd_s)}")
    return proc.returncode


def kubectl_json(args: list[str]) -> dict[str, Any]:
    output = run_capture(["kubectl", *args, "-o", "json"])
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        if is_kube_api_unavailable(output):
            raise KubeApiUnavailable(output) from exc
        raise


def resource_exists(kind: str, name: str) -> bool:
    cmd = ["kubectl", "get", kind, name, "-n", NAMESPACE, "-o", "name"]
    print("+ " + " ".join(cmd), flush=True)
    output = ""
    for attempt in range(1, KUBECTL_RETRY_ATTEMPTS + 1):
        try:
            proc = subprocess.run(
                cmd,
                cwd=BASE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=KUBECTL_TIMEOUT_SECONDS,
            )
            output = proc.stdout or ""
            if proc.returncode == 0:
                return bool(output.strip())
            if not is_kube_api_unavailable(output):
                return False
        except subprocess.TimeoutExpired:
            output = f"kubectl timed out after {KUBECTL_TIMEOUT_SECONDS}s"
        if attempt < KUBECTL_RETRY_ATTEMPTS:
            time.sleep(KUBECTL_RETRY_DELAY_SECONDS)
    raise KubeApiUnavailable(output.strip())


def qwen_resource_counts() -> dict[str, int]:
    counts = {"ksvc": int(resource_exists("ksvc", SERVICE))}
    for kind, key in (
        ("revision", "revision"),
        ("deployment", "deployment"),
        ("pod", "pod"),
        ("hpa", "hpa"),
    ):
        doc = kubectl_json([
            "get",
            kind,
            "-n",
            NAMESPACE,
            "-l",
            f"serving.knative.dev/service={SERVICE}",
        ])
        counts[key] = len(doc.get("items", []))
    return counts


def service_pods() -> list[dict[str, Any]]:
    doc = kubectl_json([
        "get",
        "pod",
        "-n",
        NAMESPACE,
        "-l",
        f"serving.knative.dev/service={SERVICE}",
    ])
    rows: list[dict[str, Any]] = []
    for item in doc.get("items", []):
        meta = item.get("metadata", {})
        status = item.get("status", {})
        phase = status.get("phase", "")
        if phase in {"Succeeded", "Failed"}:
            continue
        container_statuses = status.get("containerStatuses", [])
        rows.append({
            "name": meta.get("name", ""),
            "revision": meta.get("labels", {}).get("serving.knative.dev/revision", ""),
            "phase": phase,
            "ready": bool(container_statuses) and all(bool(st.get("ready")) for st in container_statuses),
            "deleting": bool(meta.get("deletionTimestamp")),
            "node": item.get("spec", {}).get("nodeName", ""),
            "pod_ip": status.get("podIP", ""),
        })
    return sorted(rows, key=lambda row: row["name"])


def pod_count_snapshot() -> dict[str, Any]:
    pods = service_pods()
    return {
        "timestamp_utc": utc_now(),
        "total": len(pods),
        "running": sum(1 for pod in pods if pod["phase"] == "Running" and not pod["deleting"]),
        "ready": sum(1 for pod in pods if pod["ready"] and not pod["deleting"]),
        "terminating": sum(1 for pod in pods if pod["deleting"]),
        "nodes": sorted({pod["node"] for pod in pods if pod["node"]}),
        "pods": pods,
    }


def api_outage_grace_seconds(args: argparse.Namespace | None) -> float:
    if args is None:
        return API_OUTAGE_GRACE_SECONDS
    return max(
        0.0,
        float(
            getattr(
                args,
                "api_outage_grace_seconds",
                API_OUTAGE_GRACE_SECONDS,
            )
        ),
    )


def pod_count_snapshot_with_api_grace(
    args: argparse.Namespace | None,
    log_handle: Any | None,
    label: str,
) -> dict[str, Any]:
    started = time.time()
    grace_s = api_outage_grace_seconds(args)
    attempts = 0
    while True:
        attempts += 1
        try:
            snap = pod_count_snapshot()
            if attempts > 1:
                payload = {
                    "label": label,
                    "attempts": attempts,
                    "elapsed_seconds": round(time.time() - started, 3),
                    "status": "recovered",
                }
                line = "kubectl_api_outage_grace=" + json.dumps(payload, sort_keys=True)
                print(line, flush=True)
                if log_handle is not None:
                    log_handle.write(line + "\n")
                    log_handle.flush()
            return snap
        except KubeApiUnavailable as exc:
            elapsed = time.time() - started
            remaining = grace_s - elapsed
            if remaining <= 0:
                payload = {
                    "label": label,
                    "attempts": attempts,
                    "elapsed_seconds": round(elapsed, 3),
                    "grace_seconds": grace_s,
                    "status": "exhausted",
                    "error": str(exc)[-500:],
                }
                line = "kubectl_api_outage_grace=" + json.dumps(payload, sort_keys=True)
                print(line, flush=True)
                if log_handle is not None:
                    log_handle.write(line + "\n")
                    log_handle.flush()
                raise
            payload = {
                "label": label,
                "attempt": attempts,
                "elapsed_seconds": round(elapsed, 3),
                "remaining_seconds": round(remaining, 3),
                "sleep_seconds": min(KUBECTL_RETRY_DELAY_SECONDS, remaining),
                "status": "waiting",
                "error": str(exc)[-500:],
            }
            line = "kubectl_api_outage_grace=" + json.dumps(payload, sort_keys=True)
            print(line, flush=True)
            if log_handle is not None:
                log_handle.write(line + "\n")
                log_handle.flush()
            time.sleep(min(KUBECTL_RETRY_DELAY_SECONDS, remaining))


def parse_node_csv(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def normalize_initial_placement_nodes(args: argparse.Namespace) -> list[str]:
    nodes = parse_node_csv(getattr(args, "initial_placement_nodes", INITIAL_PLACEMENT_NODES))
    if len(nodes) != int(args.initial_pods):
        raise SystemExit(
            "--initial-placement-nodes must list exactly --initial-pods nodes; "
            f"got {nodes} for initial_pods={args.initial_pods}"
        )
    bad_nodes = sorted(set(nodes) - WORKER_NODES)
    if bad_nodes:
        raise SystemExit(f"--initial-placement-nodes includes non-worker nodes: {bad_nodes}")
    setattr(args, "initial_placement_nodes", nodes)
    return nodes


def initial_placement_nodes(args: argparse.Namespace) -> list[str]:
    return parse_node_csv(getattr(args, "initial_placement_nodes", INITIAL_PLACEMENT_NODES))


def active_ready_pods(snap: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        pod
        for pod in snap.get("pods", [])
        if pod["phase"] == "Running" and pod["ready"] and not pod["deleting"]
    ]


def active_running_pods(snap: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        pod
        for pod in snap.get("pods", [])
        if pod["phase"] == "Running" and not pod["deleting"]
    ]


def node_counts(nodes: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in nodes:
        counts[node] = counts.get(node, 0) + 1
    return {node: counts[node] for node in sorted(counts)}


def pod_identity_rows(pods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": pod.get("name", ""),
            "node": pod.get("node", ""),
            "phase": pod.get("phase", ""),
            "ready": bool(pod.get("ready")),
            "deleting": bool(pod.get("deleting")),
            "pod_ip": pod.get("pod_ip", ""),
        }
        for pod in pods
    ]


def validate_ready_placement(
    snap: dict[str, Any],
    expected_nodes: list[str],
    label: str,
) -> dict[str, Any]:
    ready_pods = active_ready_pods(snap)
    observed_nodes = sorted(str(pod.get("node") or "") for pod in ready_pods)
    expected_sorted = sorted(expected_nodes)
    check = {
        "label": label,
        "expected_nodes": list(expected_nodes),
        "expected_node_multiset": expected_sorted,
        "observed_node_multiset": observed_nodes,
        "observed_node_counts": node_counts(observed_nodes),
        "pods": pod_identity_rows(ready_pods),
        "passed": observed_nodes == expected_sorted,
    }
    if not check["passed"]:
        raise RuntimeError(
            f"{label}: expected Ready pod placement {expected_sorted}, "
            f"observed {observed_nodes}; pods={check['pods']}"
        )
    return check


def wait_no_qwen(timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        counts = qwen_resource_counts()
        pods = pod_count_snapshot()
        message = {"resources": counts, "pods": {k: pods[k] for k in ("total", "ready", "terminating")}}
        if message != last:
            print("qwen_absence_check=" + json.dumps(message, sort_keys=True), flush=True)
            last = message
        if all(value == 0 for value in counts.values()) and pods["total"] == 0:
            return
        time.sleep(5)
    raise TimeoutError("qwen resources did not disappear")


def cleanup_qwen(timeout_s: float) -> None:
    run_capture(["kubectl", "delete", "ksvc", SERVICE, "-n", NAMESPACE, "--ignore-not-found=true"], check=True)
    run_capture([
        "kubectl",
        "delete",
        "hpa",
        "-n",
        NAMESPACE,
        "-l",
        f"serving.knative.dev/service={SERVICE}",
        "--ignore-not-found=true",
    ], check=False)
    wait_no_qwen(timeout_s)


def cooldown(seconds: float, label: str) -> None:
    if seconds <= 0:
        return
    print(f"cooldown[{label}] sleeping {seconds:.1f}s", flush=True)
    time.sleep(seconds)


def latest_revision() -> str:
    out = run_capture([
        "kubectl",
        "get",
        "ksvc",
        SERVICE,
        "-n",
        NAMESPACE,
        "-o",
        "jsonpath={.status.latestReadyRevisionName}",
    ])
    if not out:
        raise RuntimeError("latestReadyRevisionName is empty")
    return out


def apply_method_yaml(method: str) -> str:
    yaml_path = METHOD_YAMLS[method]
    run_capture(["kubectl", "apply", "--server-side", "--force-conflicts", "-f", yaml_path])
    run_capture([
        "kubectl",
        "wait",
        "ksvc",
        SERVICE,
        "-n",
        NAMESPACE,
        "--for=condition=Ready",
        "--timeout=300s",
    ])
    revision = latest_revision()
    print(f"method={method} revision={revision}", flush=True)
    return revision


def wait_exact_ready(
    expected: int,
    timeout_s: float,
    label: str,
    expected_nodes: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    required_nodes = parse_node_csv(expected_nodes) if expected_nodes is not None else None
    if required_nodes is not None and len(required_nodes) != expected:
        raise ValueError(
            f"{label}: expected_nodes length {len(required_nodes)} does not match expected={expected}"
        )
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        snap = pod_count_snapshot()
        short = {k: snap[k] for k in ("total", "running", "ready", "terminating", "nodes")}
        if required_nodes is not None:
            short["required_nodes"] = sorted(required_nodes)
        if short != last:
            print(f"pod_check[{label}]=" + json.dumps(short, sort_keys=True), flush=True)
            last = short
        if (
            snap["total"] == expected
            and snap["running"] == expected
            and snap["ready"] == expected
            and snap["terminating"] == 0
        ):
            bad_nodes = sorted(set(snap["nodes"]) - WORKER_NODES)
            if bad_nodes:
                raise RuntimeError(f"pods scheduled outside worker nodes: {bad_nodes}")
            if required_nodes is not None:
                placement = validate_ready_placement(snap, required_nodes, label)
                snap["placement_check"] = placement
                print(
                    f"pod_placement_check[{label}]="
                    + json.dumps(placement, sort_keys=True),
                    flush=True,
                )
            return snap
        time.sleep(5)
    raise TimeoutError(f"did not reach exactly {expected} ready pods for {label}")


def prepare_initial_revision_with_recovery(
    method: str,
    args: argparse.Namespace,
    method_dir: Path,
    label: str,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    expected_nodes = initial_placement_nodes(args)
    max_attempts = int(getattr(args, "initial_placement_recovery_attempts", 1) or 1)
    retry_delay_s = float(
        getattr(args, "initial_placement_recovery_delay_seconds", 0.0) or 0.0
    )
    report: dict[str, Any] = {
        "passed": False,
        "max_attempts": max_attempts,
        "retry_delay_seconds": retry_delay_s,
        "cleanup_timeout_seconds": args.cleanup_timeout_seconds,
        "expected_pods": args.initial_pods,
        "expected_nodes": expected_nodes,
        "attempts": [],
    }
    report_path = method_dir / "initial_placement_recovery.json"

    def persist_report() -> None:
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    for recovery_attempt in range(1, max_attempts + 1):
        attempt_report: dict[str, Any] = {
            "attempt": recovery_attempt,
            "started_at_utc": utc_now(),
        }
        report["attempts"].append(attempt_report)
        revision = apply_method_yaml(method)
        attempt_report["revision"] = revision
        pin_revision_for_start(revision, args.initial_pods)
        try:
            initial = wait_exact_ready(
                args.initial_pods,
                args.wait_ready_seconds,
                f"{label}_initial_recovery{recovery_attempt}",
                expected_nodes=expected_nodes,
            )
        except KubeApiUnavailable:
            raise
        except (RuntimeError, TimeoutError) as exc:
            attempt_report["passed"] = False
            attempt_report["error"] = str(exc)
            attempt_report["failed_at_utc"] = utc_now()
            persist_report()
            if recovery_attempt >= max_attempts:
                report["error"] = str(exc)
                persist_report()
                raise RuntimeError(
                    f"{label}: initial placement recovery exhausted after "
                    f"{max_attempts} attempts: {exc}"
                ) from exc
            print(
                f"initial_placement_recovery[{label}] attempt={recovery_attempt} "
                f"failed: {exc}; cleanup and recreate qwen",
                flush=True,
            )
            try:
                cleanup_qwen(args.cleanup_timeout_seconds)
                attempt_report["cleanup_completed_at_utc"] = utc_now()
            except Exception as cleanup_exc:
                attempt_report["cleanup_error"] = str(cleanup_exc)
                persist_report()
                raise
            persist_report()
            cooldown(retry_delay_s, f"{label}_initial_placement_recovery{recovery_attempt}")
            continue

        attempt_report["passed"] = True
        attempt_report["ready_pods"] = initial
        attempt_report["finished_at_utc"] = utc_now()
        report["passed"] = True
        report["selected_attempt"] = recovery_attempt
        report["selected_revision"] = revision
        persist_report()
        return revision, initial, report

    raise RuntimeError(f"{label}: initial placement recovery did not run")


def summarize_formal_placement(
    samples: list[dict[str, Any]],
    initial_required_nodes: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    ready_multisets: dict[str, int] = {}
    running_multisets: dict[str, int] = {}
    per_node_ready_samples = {node: 0 for node in sorted(WORKER_NODES)}
    unexpected_nodes: set[str] = set()
    sample_rows: list[dict[str, Any]] = []

    for index, sample in enumerate(samples):
        ready_pods = active_ready_pods(sample)
        running_pods = active_running_pods(sample)
        ready_nodes = sorted(str(pod.get("node") or "") for pod in ready_pods)
        running_nodes = sorted(str(pod.get("node") or "") for pod in running_pods)
        ready_key = ",".join(ready_nodes) if ready_nodes else "<none>"
        running_key = ",".join(running_nodes) if running_nodes else "<none>"
        ready_multisets[ready_key] = ready_multisets.get(ready_key, 0) + 1
        running_multisets[running_key] = running_multisets.get(running_key, 0) + 1
        for node in set(ready_nodes):
            if node in per_node_ready_samples:
                per_node_ready_samples[node] += 1
            elif node:
                unexpected_nodes.add(node)
        for pod in sample.get("pods", []):
            node = str(pod.get("node") or "")
            if node and node not in WORKER_NODES:
                unexpected_nodes.add(node)
        sample_rows.append({
            "sample_index": index,
            "timestamp_utc": sample.get("timestamp_utc"),
            "total": sample.get("total"),
            "running": sample.get("running"),
            "ready": sample.get("ready"),
            "terminating": sample.get("terminating"),
            "ready_running_node_multiset": ready_nodes,
            "ready_running_node_counts": node_counts(ready_nodes),
            "running_node_multiset": running_nodes,
            "running_node_counts": node_counts(running_nodes),
            "pods": pod_identity_rows(sample.get("pods", [])),
        })

    return {
        "policy": (
            "initial placement is enforced before warmup and formal replay; "
            "runtime scale placement is scheduler-managed and audited only"
        ),
        "initial_required_nodes": list(initial_required_nodes),
        "sample_count": len(samples),
        "ready_running_node_multiset_counts": {
            key: ready_multisets[key] for key in sorted(ready_multisets)
        },
        "running_node_multiset_counts": {
            key: running_multisets[key] for key in sorted(running_multisets)
        },
        "per_node_ready_running_sample_counts": per_node_ready_samples,
        "unexpected_worker_nodes": sorted(unexpected_nodes),
        "samples": sample_rows,
    }


def prom_get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    url = f"{PROMETHEUS_URL.rstrip('/')}{path}?{query}"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("status") != "success":
        raise RuntimeError(f"Prometheus query failed: {payload}")
    return payload


def prom_query_sum(query: str) -> float:
    payload = prom_get("/api/v1/query", {"query": query})
    total = 0.0
    for item in payload.get("data", {}).get("result", []):
        try:
            total += float(item.get("value", [0, 0])[1])
        except (TypeError, ValueError):
            continue
    return total


def pod_cpu_total(pod_name: str) -> float:
    query = (
        'sum(container_cpu_usage_seconds_total{'
        f'namespace="{NAMESPACE}",pod="{pod_name}",container!="POD",container!=""'
        "})"
    )
    return prom_query_sum(query)


def qwen_warmup_payload(max_tokens: int) -> dict[str, Any]:
    return {
        "model": "qwen",
        "messages": [{
            "role": "user",
            "content": (
                "A SaaS app suddenly receives 10 times more user requests after a product launch. "
                "Explain in simple language why autoscaling is needed, what could happen if scaling "
                "is too slow, and name one metric the system could monitor to decide when to add more "
                "servers. Give one short example. Keep the answer around 75 words."
            ),
        }],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }


def post_qwen(
    max_tokens: int,
    timeout_s: float,
    *,
    url: str = QWEN_URL,
    host: str = QWEN_HOST,
    target_pod: str | None = None,
) -> dict[str, Any]:
    payload = qwen_warmup_payload(max_tokens)
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Host": host, "Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            response.read()
            status: int | str = response.status
            ok = status == 200
    except urllib.error.HTTPError as exc:
        exc.read()
        status = exc.code
        ok = False
    except Exception as exc:  # noqa: BLE001
        status = type(exc).__name__
        ok = False
    row = {"ok": ok, "status": status, "latency_s": time.perf_counter() - start}
    if target_pod is not None:
        row["target_pod"] = target_pod
    return row


def summarize_requests(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    latencies: list[float] = []
    for row in rows:
        statuses[str(row["status"])] = statuses.get(str(row["status"]), 0) + 1
        if row["ok"]:
            latencies.append(float(row["latency_s"]))
    latencies.sort()
    summary: dict[str, Any] = {
        "sent": len(rows),
        "ok": len(latencies),
        "failed": len(rows) - len(latencies),
        "statuses": statuses,
    }
    if latencies:
        summary["p50_latency_s"] = latencies[len(latencies) // 2]
        summary["p95_latency_s"] = latencies[max(0, int(len(latencies) * 0.95) - 1)]
    return summary


def ready_warmup_pods(snap: dict[str, Any], expected: int, label: str) -> list[dict[str, Any]]:
    pods = [
        pod
        for pod in snap["pods"]
        if pod["phase"] == "Running" and pod["ready"] and not pod["deleting"]
    ]
    if len(pods) != expected:
        raise RuntimeError(f"{label}: expected {expected} ready warmup pods, found {len(pods)}")
    missing_ip = [pod["name"] for pod in pods if not pod.get("pod_ip")]
    if missing_ip:
        raise RuntimeError(f"{label}: ready warmup pods missing pod_ip: {missing_ip}")
    return sorted(pods, key=lambda pod: pod["name"])


def targeted_pod_warmup(args: argparse.Namespace, pods: list[dict[str, Any]]) -> dict[str, Any]:
    requests_per_pod = int(getattr(args, "warmup_targeted_requests_per_pod", 0) or 0)
    if requests_per_pod <= 0:
        return {"enabled": False, "passed": True, "reason": "warmup_targeted_requests_per_pod<=0"}

    min_success = getattr(args, "warmup_targeted_min_success_per_pod", None)
    if min_success is None:
        min_success = requests_per_pod
    min_success = int(min_success)
    min_cpu_delta = float(getattr(args, "warmup_targeted_min_cpu_delta", 0.0) or 0.0)
    concurrency = max(1, int(getattr(args, "warmup_targeted_concurrency", 1) or 1))

    cpu_before = {pod["name"]: pod_cpu_total(pod["name"]) for pod in pods}
    per_pod: dict[str, Any] = {}
    failed_request_counts: dict[str, int] = {}

    for pod in pods:
        pod_name = str(pod["name"])
        endpoint = f"http://{pod['pod_ip']}:8080/v1/chat/completions"
        print(
            "warmup_targeted_pod="
            + json.dumps(
                {
                    "pod": pod_name,
                    "pod_ip": pod["pod_ip"],
                    "endpoint": endpoint,
                    "requests": requests_per_pod,
                    "concurrency": min(concurrency, requests_per_pod),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        rows: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(concurrency, requests_per_pod)) as pool:
            futures = [
                pool.submit(
                    post_qwen,
                    args.warmup_max_tokens,
                    args.warmup_request_timeout_seconds,
                    url=endpoint,
                    host=pod_name,
                    target_pod=pod_name,
                )
                for _ in range(requests_per_pod)
            ]
            for future in concurrent.futures.as_completed(futures):
                rows.append(future.result())
        summary = summarize_requests(rows)
        if int(summary["ok"]) < min_success:
            failed_request_counts[pod_name] = int(summary["ok"])
        per_pod[pod_name] = {
            "pod_ip": pod["pod_ip"],
            "endpoint": endpoint,
            "summary": summary,
        }

    if args.warmup_cpu_query_delay_seconds > 0:
        time.sleep(args.warmup_cpu_query_delay_seconds)

    cpu_after = {pod["name"]: pod_cpu_total(pod["name"]) for pod in pods}
    failed_cpu_delta: dict[str, float] = {}
    for pod in pods:
        pod_name = str(pod["name"])
        delta = cpu_after.get(pod_name, 0.0) - cpu_before.get(pod_name, 0.0)
        per_pod[pod_name]["cpu_before"] = cpu_before.get(pod_name, 0.0)
        per_pod[pod_name]["cpu_after"] = cpu_after.get(pod_name, 0.0)
        per_pod[pod_name]["cpu_delta"] = delta
        if delta < min_cpu_delta:
            failed_cpu_delta[pod_name] = delta

    result = {
        "enabled": True,
        "passed": not failed_request_counts and not failed_cpu_delta,
        "requests_per_pod": requests_per_pod,
        "min_success_per_pod": min_success,
        "min_cpu_delta": min_cpu_delta,
        "concurrency": concurrency,
        "per_pod": per_pod,
        "failed_request_counts": failed_request_counts,
        "failed_cpu_delta": failed_cpu_delta,
    }
    print("warmup_targeted_summary=" + json.dumps(result, sort_keys=True), flush=True)
    return result


def light_warmup_and_verify(args: argparse.Namespace, method_dir: Path, method: str) -> dict[str, Any]:
    total_requests = int(round(args.warmup_duration_seconds * args.warmup_qps))
    interval = 1.0 / max(args.warmup_qps, 0.001)
    attempts: list[dict[str, Any]] = []
    last_failed: dict[str, Any] = {}
    expected_nodes = initial_placement_nodes(args)
    for attempt in range(1, args.warmup_attempts + 1):
        snap = wait_exact_ready(
            args.initial_pods,
            args.wait_ready_seconds,
            f"{method}_warmup_attempt{attempt}",
            expected_nodes=expected_nodes,
        )
        pods = ready_warmup_pods(snap, args.initial_pods, f"{method}_warmup_attempt{attempt}")
        cpu_before = {pod["name"]: pod_cpu_total(pod["name"]) for pod in pods}
        targeted = targeted_pod_warmup(args, pods)
        if not targeted["passed"]:
            attempt_result = {
                "attempt": attempt,
                "targeted_pod_warmup": targeted,
                "passed": False,
                "failed_targeted_request_counts": targeted.get("failed_request_counts", {}),
                "failed_targeted_cpu_delta": targeted.get("failed_cpu_delta", {}),
            }
            attempts.append(attempt_result)
            last_failed = {
                "targeted_request_counts": targeted.get("failed_request_counts", {}),
                "targeted_cpu_delta": targeted.get("failed_cpu_delta", {}),
            }
            print(
                f"targeted pod warmup attempt {attempt} failed; retrying if attempts remain: {last_failed}",
                flush=True,
            )
            continue
        after_targeted = wait_exact_ready(
            args.initial_pods,
            args.wait_ready_seconds,
            f"{method}_after_targeted_warmup_attempt{attempt}",
            expected_nodes=expected_nodes,
        )
        after_targeted_names = {pod["name"] for pod in ready_warmup_pods(
            after_targeted,
            args.initial_pods,
            f"{method}_after_targeted_warmup_attempt{attempt}",
        )}
        original_names = {pod["name"] for pod in pods}
        if after_targeted_names != original_names:
            raise RuntimeError(
                f"warmup pod set changed during targeted warmup: "
                f"before={sorted(original_names)} after={sorted(after_targeted_names)}"
            )
        print(
            f"warmup attempt={attempt} sending {total_requests} requests over "
            f"{args.warmup_duration_seconds}s concurrency={args.warmup_concurrency}",
            flush=True,
        )
        start = time.monotonic()
        rows: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.warmup_concurrency) as pool:
            futures = []
            for i in range(total_requests):
                delay = start + i * interval - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                futures.append(pool.submit(post_qwen, args.warmup_max_tokens, args.warmup_request_timeout_seconds))
            for future in concurrent.futures.as_completed(futures):
                rows.append(future.result())
        summary = summarize_requests(rows)
        print("warmup_summary=" + json.dumps(summary, sort_keys=True), flush=True)
        if summary["ok"] < args.warmup_min_success:
            raise RuntimeError(f"warmup successes too low: {summary}")
        if args.warmup_cpu_query_delay_seconds > 0:
            time.sleep(args.warmup_cpu_query_delay_seconds)
        cpu_after = {pod["name"]: pod_cpu_total(pod["name"]) for pod in pods}
        delta = {pod: cpu_after.get(pod, 0.0) - before for pod, before in cpu_before.items()}
        failed = {pod: value for pod, value in delta.items() if value < args.warmup_min_cpu_delta}
        attempt_result = {
            "attempt": attempt,
            "targeted_pod_warmup": targeted,
            "summary": summary,
            "cpu_before": cpu_before,
            "cpu_after": cpu_after,
            "cpu_delta": delta,
            "passed": not failed,
            "failed_cpu_delta": failed,
        }
        attempts.append(attempt_result)
        print("warmup_cpu_delta=" + json.dumps(delta, sort_keys=True), flush=True)
        if not failed:
            result = {"passed": True, "attempts": attempts, "used_attempts": attempt}
            (method_dir / "warmup_cpu_delta.json").write_text(
                json.dumps(result, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            return result
        last_failed = failed
        print(f"warmup CPU delta attempt {attempt} failed; retrying if attempts remain: {failed}", flush=True)
    result = {"passed": False, "attempts": attempts, "failed_cpu_delta": last_failed}
    (method_dir / "warmup_cpu_delta.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    raise RuntimeError(f"warmup CPU delta verification failed after {args.warmup_attempts} attempts: {last_failed}")


def write_sanity_workload(path: Path) -> None:
    rows = [
        {"timestamp": 0, "window_id": 1, "requests_per_20s": 10, "workload_type": "sanity"},
        {"timestamp": 20, "window_id": 2, "requests_per_20s": 50, "workload_type": "sanity"},
        {"timestamp": 40, "window_id": 3, "requests_per_20s": 20, "workload_type": "sanity"},
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def legacy_metrics_args(revision: str, args: argparse.Namespace) -> list[object]:
    return [
        "--collect-pod-metrics",
        "--metrics-poll-seconds", str(args.legacy_metrics_poll_seconds),
        "--pod-metrics-selector", "revision",
        "--pod-metrics-revision-name", revision,
        "--metrics-pod-capacity-per-window", str(args.pod_capacity_per_window),
        "--power-active-watts", "3.46",
        "--power-idle-watts", "0.024",
        "--cold-start-energy-joules", "10.38",
        "--carbon-intensity-csv", args.carbon_intensity_csv,
    ]


def method_command(method: str, run_id: str, revision: str, workload: Path, args: argparse.Namespace) -> list[object]:
    common = [
        "--start-row", "0",
        "--duration-minutes", str(args.duration_minutes),
        "--seed", str(args.seed),
        "--timeout-seconds", str(args.request_timeout_seconds),
        "--start-delay-seconds", str(args.formal_start_delay_seconds),
        "--run-id", run_id,
        "--output-dir", str(RESULTS),
        "--save-request-log",
    ]
    if method == "kpa":
        return [
            PYTHON, "run_kpa_baseline.py",
            "--csv", str(workload),
            "--skip-prepare-kpa-service",
            "--min-scale", str(args.initial_pods),
            "--max-scale", "4",
            *common,
            *legacy_metrics_args(revision, args),
        ]
    if method == "hpa":
        return [
            PYTHON, "run_hpa_baseline.py",
            "--csv", str(workload),
            "--skip-prepare-hpa-service",
            "--min-scale", str(args.initial_pods),
            "--max-scale", "4",
            *common,
            *legacy_metrics_args(revision, args),
        ]
    if method == "ets":
        return [
            PYTHON, "run_ets_baseline.py",
            "--train-csv", "train_workload.csv",
            "--test-csv", str(workload),
            "--pod-capacity-per-window", str(args.pod_capacity_per_window),
            "--min-pods", "1",
            "--max-pods", "4",
            "--autoscale-target", "revision",
            "--revision-name", revision,
            "--first-patch-lead-seconds", str(args.first_patch_lead_seconds),
            *common,
            *legacy_metrics_args(revision, args),
        ]
    if method == "bilstm":
        return [
            PYTHON, "run_bilstm_baseline.py",
            "--model-path", "bilstm_autoscaler_model.pt",
            "--train-csv", "train_workload.csv",
            "--test-csv", str(workload),
            "--pod-capacity-per-window", str(args.pod_capacity_per_window),
            "--min-pods", "1",
            "--max-pods", "4",
            "--autoscale-target", "revision",
            "--revision-name", revision,
            "--first-patch-lead-seconds", str(args.first_patch_lead_seconds),
            *common,
            *legacy_metrics_args(revision, args),
        ]
    if method == "transformer":
        return [
            PYTHON, "controller.py", "checkpoints_transformer_h2",
            "--live",
            "--trace", str(workload),
            "--start-row", "0",
            "--duration-minutes", str(args.duration_minutes),
            "--seed", str(args.seed),
            "--start-delay", str(args.formal_start_delay_seconds),
            "--first-patch-lead", str(args.first_patch_lead_seconds),
            "--run-id", run_id,
            "--output-dir", str(RESULTS),
            "--save-request-log",
        ]
    raise ValueError(method)


def run_with_pod_sampling(cmd: list[object], log_path: Path, args: argparse.Namespace) -> tuple[float, float, list[dict[str, Any]]]:
    cmd_s = [str(part) for part in cmd]
    print("+ " + " ".join(cmd_s), flush=True)
    samples: list[dict[str, Any]] = []
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start_epoch = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write("+ " + " ".join(cmd_s) + "\n")
        handle.flush()
        proc = subprocess.Popen(
            cmd_s,
            cwd=BASE,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        def stop_formal_process() -> None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)

        while proc.poll() is None:
            try:
                snap = pod_count_snapshot_with_api_grace(
                    args,
                    handle,
                    "formal_pod_sample",
                )
            except KubeApiUnavailable as exc:
                handle.write(
                    "\n[INFRA] Kubernetes API remained unavailable after grace during pod sampling:\n"
                    + str(exc)
                    + "\n"
                )
                handle.flush()
                stop_formal_process()
                raise
            except Exception as exc:
                handle.write(
                    "\n[RUNNER] Formal pod sampling failed; terminating formal command:\n"
                    + str(exc)
                    + "\n"
                )
                handle.flush()
                stop_formal_process()
                raise
            samples.append(snap)
            print(
                "formal_pod_sample="
                + json.dumps(
                    {k: snap[k] for k in ("total", "running", "ready", "terminating", "nodes")},
                    sort_keys=True,
                ),
                flush=True,
            )
            time.sleep(args.pod_sample_seconds)
        rc = proc.returncode
    end_epoch = time.time()
    samples.append(
        pod_count_snapshot_with_api_grace(args, None, "formal_final_pod_sample")
    )
    if rc != 0:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if is_kube_api_unavailable(log_text):
            raise KubeApiUnavailable(
                f"formal command lost Kubernetes API access with rc={rc}: "
                f"{' '.join(cmd_s)}\n{log_text[-4000:]}"
            )
        raise RuntimeError(f"formal command failed with {rc}: {' '.join(cmd_s)}")
    return start_epoch, end_epoch, samples


def run_prometheus_metrics(method: str, run_id: str, start_epoch: float, end_epoch: float, args: argparse.Namespace, log_path: Path) -> dict[str, Any]:
    if args.prometheus_query_delay_seconds > 0:
        time.sleep(args.prometheus_query_delay_seconds)
    cmd = [
        PYTHON, "prometheus_energy_metrics.py",
        "--method", method,
        "--run-id", run_id,
        "--output-dir", str(RESULTS),
        "--prometheus-url", args.prometheus_url,
        "--start-epoch", f"{start_epoch:.3f}",
        "--end-epoch", f"{end_epoch:.3f}",
        "--step-seconds", str(args.prometheus_step_seconds),
        "--cpu-rate-window-seconds", str(args.prometheus_cpu_rate_window_seconds),
        "--pod-capacity-per-window", str(args.pod_capacity_per_window),
        "--carbon-intensity-csv", args.carbon_intensity_csv,
        "--carbon-start-index", "0",
        "--cpu-joules-per-core-second", str(args.cpu_joules_per_core_second),
        "--memory-joules-per-gb-second", str(args.memory_joules_per_gb_second),
        "--cold-start-model-joules", str(args.cold_start_model_joules),
        "--cold-start-window-seconds", str(args.cold_start_window_seconds),
    ]
    run_logged(cmd, log_path)
    summary_path = RESULTS / f"{run_id}_prometheus_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("resource_sample_rows") or 0) <= 0:
        raise RuntimeError(f"Prometheus resource samples missing for {run_id}")
    if float(summary.get("memory_gb_seconds") or 0.0) <= 0:
        raise RuntimeError(f"Prometheus memory samples missing for {run_id}")
    return summary


def validate_log(method: str, log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    result = {"mu_or_capacity_logged": True, "needle": ""}
    if method == "transformer":
        result["needle"] = "mu=13.75"
        result["mu_or_capacity_logged"] = "mu=13.75" in text
    elif method in {"ets", "bilstm"}:
        result["needle"] = "pod_capacity_per_window=13.75"
        result["mu_or_capacity_logged"] = "pod_capacity_per_window=13.75" in text
    else:
        result["needle"] = "internal Knative autoscaler"
    if not result["mu_or_capacity_logged"]:
        raise RuntimeError(f"{method} log did not contain {result['needle']}")
    return result


def run_controller_mu_dry_run(workload: Path, batch_dir: Path) -> None:
    log_path = batch_dir / "controller_mu_dry_run.log"
    cmd = [
        PYTHON, "controller.py", "checkpoints_transformer_h2",
        "--dry-run",
        "--trace", str(workload),
        "--duration-minutes", "0",
        "--start-row", "0",
    ]
    run_logged(cmd, log_path)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if "mu=13.75" not in text:
        raise RuntimeError("controller dry-run did not print mu=13.75")


def annotate_revision_bounds(revision: str, min_scale: int, max_scale: int) -> None:
    run_capture([
        "kubectl",
        "annotate",
        "revision",
        revision,
        "-n",
        NAMESPACE,
        f"autoscaling.knative.dev/min-scale={min_scale}",
        f"autoscaling.knative.dev/max-scale={max_scale}",
        "--overwrite",
    ])


def scale_revision_deployment(revision: str, replicas: int) -> None:
    run_capture([
        "kubectl",
        "scale",
        "deployment",
        f"{revision}-deployment",
        "-n",
        NAMESPACE,
        f"--replicas={replicas}",
    ], check=False)


def pin_revision_for_start(revision: str, replicas: int) -> None:
    annotate_revision_bounds(revision, replicas, replicas)
    scale_revision_deployment(revision, replicas)


def release_revision_for_formal(revision: str, min_scale: int = 2, max_scale: int = 4) -> None:
    annotate_revision_bounds(revision, min_scale, max_scale)


def assert_latest_revision_unchanged(expected_revision: str, label: str) -> str:
    observed_revision = latest_revision()
    print(
        f"latest_revision_check[{label}]: expected={expected_revision} observed={observed_revision}",
        flush=True,
    )
    if observed_revision != expected_revision:
        raise RuntimeError(
            f"{label}: latestReadyRevisionName changed after release; "
            f"expected {expected_revision}, observed {observed_revision}"
        )
    return observed_revision


def run_method(method: str, args: argparse.Namespace, batch_dir: Path, workload: Path) -> dict[str, Any]:
    method_dir = batch_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{method}_e2e_sanity_{args.batch}"
    result: dict[str, Any] = {
        "method": method,
        "run_id": run_id,
        "started_at_utc": utc_now(),
        "yaml": METHOD_YAMLS[method],
        "checks": {},
    }

    expected_nodes = initial_placement_nodes(args)
    result["checks"]["initial_placement_contract"] = {
        "required_nodes_before_warmup_and_formal_replay": expected_nodes,
        "runtime_scale_placement": "scheduler-managed, audited only",
        "recovery": (
            "cleanup and recreate qwen before warmup when the initial Ready "
            "pod placement does not match"
        ),
    }
    revision, initial, recovery = prepare_initial_revision_with_recovery(
        method,
        args,
        method_dir,
        method,
    )
    result["revision"] = revision
    result["checks"]["initial_placement_recovery"] = recovery
    result["checks"]["initial_ready_pods"] = initial
    result["checks"]["initial_scale"] = args.initial_pods
    cooldown(args.pre_warmup_settle_seconds, f"{method}_pre_warmup_settle")

    warmup = light_warmup_and_verify(args, method_dir, method)
    result["checks"]["warmup"] = warmup
    cooldown(args.drain_seconds, f"{method}_drain")

    before_formal = wait_exact_ready(
        args.initial_pods,
        args.wait_ready_seconds,
        f"{method}_before_formal",
        expected_nodes=expected_nodes,
    )
    result["checks"]["before_formal_ready_pods"] = before_formal
    result["checks"]["latest_revision_before_release"] = latest_revision()
    release_revision_for_formal(revision, min_scale=args.initial_pods, max_scale=4)
    result["checks"]["formal_release_bounds"] = {"min_scale": args.initial_pods, "max_scale": 4}
    result["checks"]["latest_revision_after_release"] = assert_latest_revision_unchanged(
        revision,
        f"{method}_after_release",
    )
    result["checks"]["formal_start_ready_pods"] = wait_exact_ready(
        args.initial_pods,
        args.wait_ready_seconds,
        f"{method}_formal_start",
        expected_nodes=expected_nodes,
    )

    formal_log = LOGS / f"{run_id}.log"
    cmd = method_command(method, run_id, revision, workload, args)
    start_epoch, end_epoch, samples = run_with_pod_sampling(cmd, formal_log, args)
    result["checks"]["formal_pod_samples"] = samples
    placement_audit = summarize_formal_placement(samples, expected_nodes)
    result["checks"]["formal_placement_audit"] = placement_audit
    (method_dir / "formal_placement_audit.json").write_text(
        json.dumps(placement_audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    running_pod_counts = [int(sample["running"]) for sample in samples]
    ready_pod_counts = [int(sample["ready"]) for sample in samples]
    total_pod_counts = [int(sample["total"]) for sample in samples]
    terminating_pod_counts = [int(sample["terminating"]) for sample in samples]
    result["checks"]["formal_pod_count_validation"] = {
        "running_counts": running_pod_counts,
        "ready_counts": ready_pod_counts,
        "total_counts": total_pod_counts,
        "terminating_counts": terminating_pod_counts,
        "max_total_pods_including_pending_or_terminating": max(total_pod_counts) if total_pod_counts else 0,
        "rule": "formal active pod count uses non-deleting Running pods; total pod count is diagnostic",
    }
    if not running_pod_counts or min(running_pod_counts) < 1 or max(running_pod_counts) > 4:
        raise RuntimeError(f"{method} running pod count outside [1,4]: {running_pod_counts}")
    if ready_pod_counts and max(ready_pod_counts) > 4:
        raise RuntimeError(f"{method} ready pod count exceeded 4: {ready_pod_counts}")
    if total_pod_counts and max(total_pod_counts) > 4:
        print(
            f"WARNING: {method} transient total pod count exceeded 4: {total_pod_counts}",
            flush=True,
        )

    log_check = validate_log(method, formal_log)
    result["checks"]["log_check"] = log_check
    prom = run_prometheus_metrics(
        method,
        run_id,
        start_epoch,
        end_epoch,
        args,
        LOGS / f"{run_id}_prometheus_metrics.log",
    )
    result["checks"]["prometheus_summary"] = prom
    if int(prom.get("successful_requests") or 0) <= 0:
        raise RuntimeError(f"{method} workload did not produce successful requests through Knative")

    result["finished_at_utc"] = utc_now()
    result["passed"] = True
    (method_dir / "method_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--methods", nargs="+", choices=list(METHOD_YAMLS), default=list(METHOD_YAMLS))
    parser.add_argument("--duration-minutes", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-pods", type=int, default=2)
    parser.add_argument(
        "--initial-placement-nodes",
        default=",".join(INITIAL_PLACEMENT_NODES),
        help=(
            "Comma-separated exact node multiset required for the initial Ready pods "
            "before warmup and before formal replay."
        ),
    )
    parser.add_argument("--pod-capacity-per-window", type=float, default=13.75)
    parser.add_argument("--request-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--formal-start-delay-seconds", type=float, default=5.0)
    parser.add_argument("--first-patch-lead-seconds", type=float, default=1.0)
    parser.add_argument("--wait-ready-seconds", type=float, default=300.0)
    parser.add_argument("--cleanup-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--initial-placement-recovery-attempts", type=int, default=4)
    parser.add_argument("--initial-placement-recovery-delay-seconds", type=float, default=10.0)
    parser.add_argument("--cooldown-seconds", type=float, default=300.0)
    parser.add_argument("--drain-seconds", type=float, default=30.0)
    parser.add_argument("--pod-sample-seconds", type=float, default=5.0)
    parser.add_argument("--api-outage-grace-seconds", type=float, default=API_OUTAGE_GRACE_SECONDS)
    parser.add_argument("--legacy-metrics-poll-seconds", type=float, default=1.0)
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
    parser.add_argument("--prometheus-url", default=PROMETHEUS_URL)
    parser.add_argument("--prometheus-step-seconds", type=float, default=5.0)
    parser.add_argument("--prometheus-cpu-rate-window-seconds", type=float, default=30.0)
    parser.add_argument("--prometheus-query-delay-seconds", type=float, default=20.0)
    parser.add_argument("--carbon-intensity-csv", default="carbon_intensity_ie_2025_5min.csv")
    parser.add_argument("--cpu-joules-per-core-second", type=float, default=10.8)
    parser.add_argument("--memory-joules-per-gb-second", type=float, default=0.3725)
    parser.add_argument("--cold-start-model-joules", type=float, default=10.38)
    parser.add_argument("--cold-start-window-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.initial_pods != 2:
        raise SystemExit("formal protocol sanity requires --initial-pods 2")
    normalize_initial_placement_nodes(args)
    if args.initial_placement_recovery_attempts <= 0:
        raise SystemExit("--initial-placement-recovery-attempts must be positive")
    if args.initial_placement_recovery_delay_seconds < 0:
        raise SystemExit("--initial-placement-recovery-delay-seconds must be non-negative")
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
    LOGS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    batch_dir = RESULTS / f"e2e_sanity_{args.batch}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    workload = BASE / "sanity_workload_3w.csv"
    write_sanity_workload(workload)

    summary: dict[str, Any] = {
        "batch": args.batch,
        "started_at_utc": utc_now(),
        "args": vars(args),
        "workload": str(workload),
        "methods": [],
        "passed": False,
    }
    summary_path = batch_dir / "sanity_summary.json"
    try:
        run_controller_mu_dry_run(workload, batch_dir)
        cleanup_qwen(args.cleanup_timeout_seconds)
        cooldown(args.cooldown_seconds, "initial")
        for method in args.methods:
            print(f"========== e2e sanity method={method} ==========", flush=True)
            method_result = run_method(method, args, batch_dir, workload)
            summary["methods"].append(method_result)
            cleanup_qwen(args.cleanup_timeout_seconds)
            cooldown(args.cooldown_seconds, f"after_{method}")
        summary["passed"] = True
        summary["finished_at_utc"] = utc_now()
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print("sanity_summary=" + str(summary_path), flush=True)
        return 0
    except Exception as exc:
        summary["passed"] = False
        summary["failed_at_utc"] = utc_now()
        summary["error"] = str(exc)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        try:
            cleanup_qwen(args.cleanup_timeout_seconds)
        except Exception as cleanup_exc:  # noqa: BLE001
            print(f"cleanup after error also failed: {cleanup_exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
