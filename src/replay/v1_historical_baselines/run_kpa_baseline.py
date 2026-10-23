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

KPA_CLASS_VALUE = "kpa.autoscaling.knative.dev"

# Fixed cluster scale bounds for all formal KPA baseline runs.
# These values are enforced in code, so the user does not need to pass
# --min-scale/--max-scale or --prepare-kpa-service manually.
FIXED_MIN_SCALE = "2"
FIXED_MAX_SCALE = "4"

# For LLM serving, concurrency target=1 is usually more sensitive than 2.
# If target=2, KPA may not scale much because Poisson replay smooths concurrency.
DEFAULT_KPA_TARGET = "1"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the KPA baseline: replay the same Poisson workload and let "
            "Knative's built-in KPA reactively autoscale."
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
        "--prepare-kpa-service",
        action="store_true",
        help=(
            "Deprecated: KPA service preparation is now enabled by default. "
            "This flag is kept only for compatibility with old commands."
        ),
    )
    parser.add_argument(
        "--skip-prepare-kpa-service",
        action="store_true",
        help=(
            "Skip patching the Knative Service before replay. Normally do not use this, "
            "because the script now patches KPA min-scale=2 and max-scale=4 automatically."
        ),
    )

    parser.add_argument(
        "--kpa-metric",
        default="concurrency",
        help="Knative KPA metric. Usually use concurrency for Knative KPA.",
    )
    parser.add_argument(
        "--kpa-target",
        default=DEFAULT_KPA_TARGET,
        help=(
            "Autoscaling target for KPA. For LLM serving, concurrency target=1 "
            "usually makes KPA more responsive than target=2."
        ),
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
    run_id = args.run_id or make_run_id("kpa")
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
        baseline_name="kpa",
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


def enforce_fixed_scale_bounds(args):
    """Force the KPA baseline to use the cluster's fixed pod bounds.

    This avoids accidentally running KPA without max-scale=4/min-scale=2,
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
            "[KPA Scale Bounds] overriding requested "
            f"min/max=[{requested_min_scale}, {requested_max_scale}] "
            f"with fixed cluster bounds [{FIXED_MIN_SCALE}, {FIXED_MAX_SCALE}]"
        )


def prepare_kpa_service(args):
    annotations = {
        AUTOSCALING_CLASS_ANNOTATION: KPA_CLASS_VALUE,
        METRIC_ANNOTATION: str(args.kpa_metric),
        TARGET_ANNOTATION: str(args.kpa_target),
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

    print("[KPA Prepare]", " ".join(cmd))

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

    print("[KPA Wait]", " ".join(cmd))
    subprocess.run(cmd, check=True)


async def main():
    args = parse_args()

    enforce_fixed_scale_bounds(args)

    run_id = args.run_id or make_run_id("kpa")
    args.run_id = run_id

    print("========== KPA baseline ==========")
    print("External controller: disabled")
    print("Autoscaling: Knative KPA")
    print(f"kpa_metric={args.kpa_metric}")
    print(f"kpa_target={args.kpa_target}")
    print(f"scale_bounds=[{args.min_scale}, {args.max_scale}]")

    if args.skip_prepare_kpa_service:
        print("[KPA Prepare] skipped by --skip-prepare-kpa-service")
    else:
        prepare_kpa_service(args)

    metrics_config = build_metrics_config(args, "kpa", run_id)

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
