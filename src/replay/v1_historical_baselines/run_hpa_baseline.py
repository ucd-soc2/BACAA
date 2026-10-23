import argparse
import asyncio
import json
import shlex
import shutil
import subprocess

from baseline_metrics import (
    add_metrics_args,
    build_metrics_config,
    sample_pod_metrics,
    save_metrics_artifacts,
)
from workload_replayer import (
    DEFAULT_HOST,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TEST_CSV,
    DEFAULT_URL,
    ReplayConfig,
    make_run_id,
    replay_workload,
)


DEFAULT_SERVICE_NAME = "qwen"
DEFAULT_NAMESPACE = "default"
AUTOSCALING_CLASS_ANNOTATION = "autoscaling.knative.dev/class"
METRIC_ANNOTATION = "autoscaling.knative.dev/metric"
TARGET_ANNOTATION = "autoscaling.knative.dev/target"
MIN_SCALE_ANNOTATION = "autoscaling.knative.dev/min-scale"
MAX_SCALE_ANNOTATION = "autoscaling.knative.dev/max-scale"
HPA_CLASS_VALUE = "hpa.autoscaling.knative.dev"

# Fixed cluster scale bounds for all formal HPA baseline runs.
# These values are enforced in code, so the user does not need to pass
# --min-scale/--max-scale or --prepare-hpa-service manually.
FIXED_MIN_SCALE = "2"
FIXED_MAX_SCALE = "4"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the HPA baseline: replay the same Poisson workload and let "
            "Knative delegate autoscaling to Kubernetes HPA."
        )
    )
    parser.add_argument("--csv", default=DEFAULT_TEST_CSV)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--window-size-seconds", type=float, default=20.0)
    parser.add_argument("--duration-minutes", type=float, default=5.0)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--connector-limit", type=int, default=0)
    parser.add_argument("--start-delay-seconds", type=float, default=5.0)
    parser.add_argument("--start-at-epoch", type=float, default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--save-request-log", action="store_true")

    parser.add_argument("--service-name", default=DEFAULT_SERVICE_NAME)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument(
        "--prepare-hpa-service",
        action="store_true",
        help=(
            "Deprecated: HPA service preparation is now enabled by default. "
            "This flag is kept only for compatibility with old commands."
        ),
    )
    parser.add_argument(
        "--skip-prepare-hpa-service",
        action="store_true",
        help=(
            "Skip patching the Knative Service before replay. Normally do not use this, "
            "because the script now patches HPA min-scale=2 and max-scale=4 automatically."
        ),
    )
    parser.add_argument("--hpa-metric", default="cpu")
    parser.add_argument(
        "--hpa-target",
        default="80",
        help="Autoscaling target. For Knative HPA cpu metric, this is the target CPU percentage.",
    )
    parser.add_argument("--min-scale", default=FIXED_MIN_SCALE)
    parser.add_argument("--max-scale", default=FIXED_MAX_SCALE)
    parser.add_argument("--wait-ready-seconds", type=int, default=120)
    parser.add_argument("--no-wait-ready", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--kubectl-cmd",
        default=None,
        help='Command used to patch Knative, for example "kubectl" or "sudo k3s kubectl".',
    )
    add_metrics_args(parser)
    return parser.parse_args()


def build_replay_config(args):
    run_id = args.run_id or make_run_id("hpa")
    return ReplayConfig(
        csv=args.csv,
        url=args.url,
        host=args.host,
        window_size_seconds=args.window_size_seconds,
        duration_minutes=args.duration_minutes,
        start_row=args.start_row,
        scale=args.scale,
        seed=args.seed,
        timeout_seconds=args.timeout_seconds,
        connector_limit=args.connector_limit,
        baseline_name="hpa",
        start_delay_seconds=args.start_delay_seconds,
        start_at_epoch=args.start_at_epoch,
        output_dir=args.output_dir,
        run_id=run_id,
        save_request_log=args.save_request_log,
    )


def resolve_kubectl_cmd(args):
    if args.kubectl_cmd:
        return shlex.split(args.kubectl_cmd)
    if shutil.which("kubectl"):
        return ["kubectl"]
    if shutil.which("k3s"):
        return ["sudo", "k3s", "kubectl"]
    return ["kubectl"]


def run_kubectl(args, kubectl_args, check=True):
    cmd = resolve_kubectl_cmd(args) + list(kubectl_args)
    print("+ " + " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    output = proc.stdout or ""
    if check and proc.returncode != 0:
        raise RuntimeError(
            "command failed with "
            f"{proc.returncode}: {' '.join(cmd)}\n{output.strip()}"
        )
    return output


def kubectl_json(args, kubectl_args):
    output = run_kubectl(args, [*kubectl_args, "-o", "json"])
    return json.loads(output)


def metrics_api_available_condition(api_service):
    for condition in api_service.get("status", {}).get("conditions", []):
        if condition.get("type") == "Available":
            return condition
    return {}


def validate_hpa_metrics_preflight(args):
    if str(args.hpa_metric).lower() != "cpu":
        return {"skipped": True, "reason": f"hpa_metric={args.hpa_metric}"}

    api_service = kubectl_json(args, ["get", "apiservice", "v1beta1.metrics.k8s.io"])
    condition = metrics_api_available_condition(api_service)
    report = {
        "checked": True,
        "metrics_api": {
            "available": condition.get("status") == "True",
            "status": condition.get("status"),
            "reason": condition.get("reason"),
            "message": condition.get("message"),
            "last_transition_time": condition.get("lastTransitionTime"),
        },
        "metrics_server_endpoints": [],
        "top_nodes": "",
    }
    if condition.get("status") != "True":
        raise RuntimeError(
            "HPA CPU metrics API is not Available: "
            + json.dumps(report["metrics_api"], sort_keys=True)
        )

    endpoint_addresses = run_kubectl(
        args,
        [
            "get",
            "endpointslice",
            "-n",
            "kube-system",
            "-l",
            "kubernetes.io/service-name=metrics-server",
            "-o",
            "jsonpath={.items[*].endpoints[*].addresses[*]}",
        ],
        check=False,
    ).split()
    if not endpoint_addresses:
        endpoint_addresses = run_kubectl(
            args,
            [
                "get",
                "endpoints",
                "metrics-server",
                "-n",
                "kube-system",
                "-o",
                "jsonpath={.subsets[*].addresses[*].ip}",
            ],
            check=False,
        ).split()
    report["metrics_server_endpoints"] = endpoint_addresses
    if not endpoint_addresses:
        raise RuntimeError("metrics-server has no Ready endpoint")

    top_nodes = run_kubectl(args, ["top", "nodes"])
    report["top_nodes"] = "\n".join(top_nodes.strip().splitlines()[:8])
    if "<unknown>" in top_nodes.lower():
        raise RuntimeError("kubectl top nodes returned unknown CPU metrics")
    return report


def enforce_fixed_scale_bounds(args):
    """Force the HPA baseline to use the cluster's fixed pod bounds.

    This avoids accidentally running HPA without max-scale=4/min-scale=2,
    even if the user forgets command-line flags or passes different values.
    """
    requested_min_scale = str(args.min_scale)
    requested_max_scale = str(args.max_scale)
    args.min_scale = FIXED_MIN_SCALE
    args.max_scale = FIXED_MAX_SCALE

    if (
        requested_min_scale != FIXED_MIN_SCALE
        or requested_max_scale != FIXED_MAX_SCALE
    ):
        print(
            "[HPA Scale Bounds] overriding requested "
            f"min/max=[{requested_min_scale}, {requested_max_scale}] "
            f"with fixed cluster bounds [{FIXED_MIN_SCALE}, {FIXED_MAX_SCALE}]"
        )


def prepare_hpa_service(args):
    annotations = {
        AUTOSCALING_CLASS_ANNOTATION: HPA_CLASS_VALUE,
        METRIC_ANNOTATION: args.hpa_metric,
        TARGET_ANNOTATION: args.hpa_target,
        MIN_SCALE_ANNOTATION: str(args.min_scale),
        MAX_SCALE_ANNOTATION: str(args.max_scale),
    }
    patch_json = {
        "spec": {
            "template": {
                "metadata": {
                    "annotations": annotations,
                }
            }
        }
    }
    cmd = resolve_kubectl_cmd(args) + [
        "patch",
        "kservice",
        args.service_name,
        "-n",
        args.namespace,
        "--type",
        "merge",
        "-p",
        json.dumps(patch_json),
    ]

    print("[HPA Prepare]", " ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)
        if not args.no_wait_ready:
            wait_for_service_ready(args)


def wait_for_service_ready(args):
    cmd = resolve_kubectl_cmd(args) + [
        "wait",
        "kservice",
        args.service_name,
        "-n",
        args.namespace,
        "--for=condition=Ready",
        f"--timeout={args.wait_ready_seconds}s",
    ]
    print("[HPA Wait]", " ".join(cmd))
    subprocess.run(cmd, check=True)


async def main():
    args = parse_args()
    enforce_fixed_scale_bounds(args)
    run_id = args.run_id or make_run_id("hpa")
    args.run_id = run_id

    print("========== HPA baseline ==========")
    print("External controller: disabled")
    print("Autoscaling: Knative HPA")
    print(f"hpa_metric={args.hpa_metric}")
    print(f"hpa_target={args.hpa_target}")
    print(f"scale_bounds=[{args.min_scale}, {args.max_scale}]")

    if not args.dry_run:
        metrics_preflight = validate_hpa_metrics_preflight(args)
        print("[HPA Metrics Preflight] " + json.dumps(metrics_preflight, sort_keys=True))

    if args.skip_prepare_hpa_service:
        print("[HPA Prepare] skipped by --skip-prepare-hpa-service")
    else:
        prepare_hpa_service(args)

    metrics_config = build_metrics_config(args, "hpa", run_id)
    pod_samples = []
    stop_metrics = asyncio.Event()
    metrics_task = None
    if args.collect_pod_metrics:
        metrics_task = asyncio.create_task(
            sample_pod_metrics(
                resolve_kubectl_cmd(args),
                metrics_config,
                stop_metrics,
                pod_samples,
            )
        )

    try:
        replay_result = await replay_workload(build_replay_config(args))
    finally:
        if metrics_task:
            stop_metrics.set()
            await metrics_task

    save_metrics_artifacts(metrics_config, replay_result, pod_samples)


if __name__ == "__main__":
    asyncio.run(main())
