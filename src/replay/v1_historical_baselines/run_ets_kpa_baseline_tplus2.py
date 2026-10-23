import argparse
import asyncio
import json
import math
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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
    request_column,
)


DEFAULT_TRAIN_CSV = str(
    Path.home() / "online_experiment" / "data" / "train_workload.csv"
)
DEFAULT_SERVICE_NAME = "qwen"
DEFAULT_NAMESPACE = "default"
AUTOSCALING_CLASS_ANNOTATION = "autoscaling.knative.dev/class"
METRIC_ANNOTATION = "autoscaling.knative.dev/metric"
TARGET_ANNOTATION = "autoscaling.knative.dev/target"
MIN_SCALE_ANNOTATION = "autoscaling.knative.dev/min-scale"
MAX_SCALE_ANNOTATION = "autoscaling.knative.dev/max-scale"
KPA_CLASS_VALUE = "kpa.autoscaling.knative.dev"
KUBECTL_TIMEOUT_SECONDS = 30.0
KUBECTL_RETRY_ATTEMPTS = 4
KUBECTL_RETRY_DELAY_SECONDS = 3.0
API_UNAVAILABLE_PATTERNS = (
    "the connection to the server 127.0.0.1:6443 was refused",
    "connection refused",
    "unable to connect to the server",
    "dial tcp 127.0.0.1:6443",
    "net/http: tls handshake timeout",
    "i/o timeout",
    "context deadline exceeded",
    "serviceunavailable",
    "the server is currently unable to handle the request",
)

# Best ETS parameters selected from validation tuning
BEST_ETS_TREND = "none"
BEST_ETS_SEASONAL = "none"
BEST_ETS_SEASONAL_PERIODS = None
DEFAULT_POD_CAPACITY_PER_WINDOW = 6.25
BEST_ETS_HISTORY_WINDOW_SIZE = 20

@dataclass
class ETSConfig:
    train_csv: str = DEFAULT_TRAIN_CSV
    service_name: str = DEFAULT_SERVICE_NAME
    namespace: str = DEFAULT_NAMESPACE
    window_size_seconds: float = 20.0
    duration_minutes: float = 5.0
    start_delay_seconds: float = 30.0
    start_at_epoch: Optional[float] = None
    first_patch_lead_seconds: float = 5.0
    pod_capacity_per_window: float = DEFAULT_POD_CAPACITY_PER_WINDOW
    min_pods: int = 1
    max_pods: int = 4
    scale_up_ready_latency_seconds: float = 85.0
    scale_up_slack_seconds: float = 1.0
    scale_down_min_ready_pods: int = 1
    scale_down_guard_seconds: float = 0.0
    scale_down_post_check_seconds: float = 0.0
    scale_down_post_check_poll_seconds: float = 2.0
    history_window_size: int = BEST_ETS_HISTORY_WINDOW_SIZE
    min_history_windows: int = 5
    prediction_horizon: int = 2
    trend: str = BEST_ETS_TREND
    seasonal: str = BEST_ETS_SEASONAL
    seasonal_periods: Optional[int] = BEST_ETS_SEASONAL_PERIODS
    smoothing_level: Optional[float] = None
    smoothing_trend: Optional[float] = None
    smoothing_seasonal: Optional[float] = None
    reset_min_scale_on_exit: bool = False
    dry_run: bool = False
    output_dir: Optional[str] = None
    run_id: Optional[str] = None
    kubectl_cmd: Optional[str] = None
    autoscale_target: str = "revision"
    revision_name: Optional[str] = None
    ets_autoscaler_class: str = KPA_CLASS_VALUE
    ets_metric: str = "concurrency"
    ets_target: str = "1"
    record_scale_latency: bool = False
    scale_latency_poll_seconds: float = 1.0
    scale_latency_timeout_seconds: float = 120.0
    ets_config_json: Optional[str] = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the ETS baseline: start a predictive ETS min-scale controller "
            "and replay the same Poisson workload."
        )
    )

    parser.add_argument("--test-csv", default=DEFAULT_TEST_CSV)
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--window-size-seconds", type=float, default=20.0)
    parser.add_argument("--duration-minutes", type=float, default=5.0)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--connector-limit", type=int, default=0)
    parser.add_argument("--start-delay-seconds", type=float, default=30.0)
    parser.add_argument("--start-at-epoch", type=float, default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--save-request-log", action="store_true")
    parser.add_argument(
        "--ets-config-json",
        default=None,
        help=(
            "Path to ets_best_config.json produced by the offline tuning script. "
            "Only ETS hyperparameters are loaded; train/test paths and run settings "
            "still come from the command line."
        ),
    )

    parser.add_argument("--service-name", default=DEFAULT_SERVICE_NAME)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--first-patch-lead-seconds", type=float, default=5.0)
    parser.add_argument(
        "--pod-capacity-per-window",
        type=float,
        default=DEFAULT_POD_CAPACITY_PER_WINDOW,
    )
    parser.add_argument("--min-pods", type=int, default=1)
    parser.add_argument("--max-pods", type=int, default=4)
    parser.add_argument("--scale-up-ready-latency-seconds", type=float, default=85.0)
    parser.add_argument("--scale-up-slack-seconds", type=float, default=1.0)
    parser.add_argument("--scale-down-min-ready-pods", type=int, default=1)
    parser.add_argument("--scale-down-guard-seconds", type=float, default=0.0)
    parser.add_argument("--scale-down-post-check-seconds", type=float, default=0.0)
    parser.add_argument("--scale-down-post-check-poll-seconds", type=float, default=2.0)
    parser.add_argument("--history-window-size", type=int, default=BEST_ETS_HISTORY_WINDOW_SIZE)
    parser.add_argument("--min-history-windows", type=int, default=5)
    parser.add_argument("--prediction-horizon", type=int, default=2)
    parser.add_argument("--trend", choices=["add", "none"], default=BEST_ETS_TREND)
    parser.add_argument("--seasonal", choices=["add", "none"], default=BEST_ETS_SEASONAL)
    parser.add_argument("--seasonal-periods", type=int, default=BEST_ETS_SEASONAL_PERIODS)
    parser.add_argument("--smoothing-level", type=float, default=None)
    parser.add_argument("--smoothing-trend", type=float, default=None)
    parser.add_argument("--smoothing-seasonal", type=float, default=None)
    parser.add_argument("--reset-min-scale-on-exit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--kubectl-cmd",
        default=None,
        help='Command used to patch Knative, for example "kubectl" or "sudo k3s kubectl".',
    )
    parser.add_argument(
        "--autoscale-target",
        choices=["revision", "kservice"],
        default="revision",
        help=(
            "Patch the current Revision by default. Patching kservice edits "
            "spec.template and creates a new Revision on every change."
        ),
    )
    parser.add_argument(
        "--revision-name",
        default=None,
        help="Revision to patch when --autoscale-target=revision. Defaults to the service's latestReadyRevisionName.",
    )
    parser.add_argument("--prepare-ets-service", action="store_true")
    parser.add_argument("--ets-autoscaler-class", default=KPA_CLASS_VALUE)
    parser.add_argument("--ets-metric", default="concurrency")
    parser.add_argument(
        "--ets-target",
        default="1",
        help=(
            "Reactive KPA concurrency target for the clean ETS Revision. "
            "ETS+KPA uses target=1 to match the pure KPA fallback."
        ),
    )
    parser.add_argument("--wait-ready-seconds", type=int, default=120)
    parser.add_argument("--no-wait-ready", action="store_true")
    parser.add_argument("--record-scale-latency", action="store_true")
    parser.add_argument("--scale-latency-poll-seconds", type=float, default=1.0)
    parser.add_argument("--scale-latency-timeout-seconds", type=float, default=120.0)
    add_metrics_args(parser)
    return parser.parse_args()


def build_replay_config(args, start_at_epoch):
    run_id = args.run_id or make_run_id("ets")
    return ReplayConfig(
        csv=args.test_csv,
        url=args.url,
        host=args.host,
        window_size_seconds=args.window_size_seconds,
        duration_minutes=args.duration_minutes,
        start_row=args.start_row,
        scale=args.scale,
        seed=args.seed,
        timeout_seconds=args.timeout_seconds,
        connector_limit=args.connector_limit,
        baseline_name="ets",
        start_delay_seconds=args.start_delay_seconds,
        start_at_epoch=start_at_epoch,
        output_dir=args.output_dir,
        run_id=run_id,
        save_request_log=args.save_request_log,
    )


def build_ets_config(args, start_at_epoch):
    run_id = args.run_id or make_run_id("ets")
    return ETSConfig(
        train_csv=args.train_csv,
        service_name=args.service_name,
        namespace=args.namespace,
        window_size_seconds=args.window_size_seconds,
        duration_minutes=args.duration_minutes,
        start_delay_seconds=args.start_delay_seconds,
        start_at_epoch=start_at_epoch,
        first_patch_lead_seconds=args.first_patch_lead_seconds,
        pod_capacity_per_window=args.pod_capacity_per_window,
        min_pods=args.min_pods,
        max_pods=args.max_pods,
        scale_up_ready_latency_seconds=args.scale_up_ready_latency_seconds,
        scale_up_slack_seconds=args.scale_up_slack_seconds,
        scale_down_min_ready_pods=args.scale_down_min_ready_pods,
        scale_down_guard_seconds=args.scale_down_guard_seconds,
        scale_down_post_check_seconds=args.scale_down_post_check_seconds,
        scale_down_post_check_poll_seconds=args.scale_down_post_check_poll_seconds,
        history_window_size=args.history_window_size,
        min_history_windows=args.min_history_windows,
        prediction_horizon=args.prediction_horizon,
        trend=args.trend,
        seasonal=args.seasonal,
        seasonal_periods=args.seasonal_periods,
        smoothing_level=args.smoothing_level,
        smoothing_trend=args.smoothing_trend,
        smoothing_seasonal=args.smoothing_seasonal,
        reset_min_scale_on_exit=args.reset_min_scale_on_exit,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
        run_id=run_id,
        kubectl_cmd=args.kubectl_cmd,
        autoscale_target=args.autoscale_target,
        revision_name=args.revision_name,
        ets_autoscaler_class=args.ets_autoscaler_class,
        ets_metric=args.ets_metric,
        ets_target=args.ets_target,
        record_scale_latency=args.record_scale_latency,
        scale_latency_poll_seconds=args.scale_latency_poll_seconds,
        scale_latency_timeout_seconds=args.scale_latency_timeout_seconds,
        ets_config_json=args.ets_config_json,
    )



def apply_ets_config_json(args):
    """Load offline-tuned ETS hyperparameters into argparse args.

    The JSON is expected to be produced by tune_ets_baseline.py.  We deliberately
    do not override train/test CSV paths, duration, start row, URL, or output
    directory, because those are runtime experiment settings on the VM.
    """
    if not args.ets_config_json:
        return args

    config_path = Path(args.ets_config_json).expanduser()
    if not config_path.exists():
        print(f"[ETS Config] {config_path} not found; using CLI/default ETS settings")
        return args

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    best_config = payload.get("best_config", payload)

    allowed_keys = [
        "trend",
        "seasonal",
        "seasonal_periods",
        "history_window_size",
        "min_history_windows",
        "prediction_horizon",
        "smoothing_level",
        "smoothing_trend",
        "smoothing_seasonal",
    ]
    ignored_policy_keys = [
        "pod_capacity_per_window",
        "min_pods",
        "max_pods",
    ]

    print(f"[ETS Config] loading offline-tuned config from {config_path}")
    for key in allowed_keys:
        if key in best_config:
            value = best_config[key]
            setattr(args, key, value)
            print(f"[ETS Config] {key}={value}")
    for key in ignored_policy_keys:
        if key in best_config:
            print(f"[ETS Config] ignoring {key}={best_config[key]} from JSON; using live default/CLI value {getattr(args, key)}")

    if args.seasonal == "none":
        args.seasonal_periods = None

    return args



def experiment_window_count(config):
    windows = int((config.duration_minutes * 60.0) // config.window_size_seconds)
    if windows <= 0:
        raise ValueError("duration_minutes must cover at least one window")
    return windows


def validate_ets_config(config):
    if config.max_pods < config.min_pods:
        raise ValueError("max_pods must be greater than or equal to min_pods")
    if config.pod_capacity_per_window <= 0:
        raise ValueError("pod_capacity_per_window must be positive")
    if config.prediction_horizon <= 0:
        raise ValueError("prediction_horizon must be positive")
    if config.scale_up_ready_latency_seconds < 0:
        raise ValueError("scale_up_ready_latency_seconds must be non-negative")
    if config.scale_up_slack_seconds < 0:
        raise ValueError("scale_up_slack_seconds must be non-negative")
    if config.seasonal != "none" and config.seasonal_periods is None:
        raise ValueError("seasonal_periods is required when seasonal is enabled")

    fixed_params = {
        "smoothing_level": config.smoothing_level,
        "smoothing_trend": config.smoothing_trend,
        "smoothing_seasonal": config.smoothing_seasonal,
    }
    if not any(value is not None for value in fixed_params.values()):
        return

    missing = []
    if config.smoothing_level is None:
        missing.append("--smoothing-level")
    if config.trend != "none" and config.smoothing_trend is None:
        missing.append("--smoothing-trend")
    if config.seasonal != "none" and config.smoothing_seasonal is None:
        missing.append("--smoothing-seasonal")

    if missing:
        raise ValueError(
            "When using fixed ETS smoothing parameters, also provide: "
            + ", ".join(missing)
        )


def resolve_experiment_start(start_at_epoch, start_delay_seconds):
    if start_at_epoch is None:
        delay = start_delay_seconds
    else:
        delay = start_at_epoch - time.time()
        if delay < 0:
            raise ValueError("--start-at-epoch is already in the past")
    return time.monotonic() + delay, delay


def load_train_history(config):
    df = pd.read_csv(config.train_csv)
    col = request_column(df)
    history = df[col].astype(float).clip(lower=0).tolist()
    if not history:
        raise ValueError("train workload history is empty")
    print(f"Loaded train history: {len(history)} windows from {config.train_csv}")
    return history


def required_history_size(config):
    size = max(config.history_window_size, config.min_history_windows)
    if config.seasonal != "none":
        size = max(size, 2 * config.seasonal_periods)
    return size


def fit_kwargs(config):
    kwargs = {}
    optimized = True

    if config.smoothing_level is not None:
        kwargs["smoothing_level"] = config.smoothing_level
        optimized = False
    if config.smoothing_trend is not None:
        kwargs["smoothing_trend"] = config.smoothing_trend
        optimized = False
    if config.smoothing_seasonal is not None:
        kwargs["smoothing_seasonal"] = config.smoothing_seasonal
        optimized = False

    kwargs["optimized"] = optimized
    return kwargs


def fallback_prediction(history, config):
    recent = history[-config.history_window_size :]
    return sum(recent) / len(recent)


def predict_next_requests(history, config):
    if len(history) < required_history_size(config):
        return fallback_prediction(history, config)

    model_history = history[-required_history_size(config) :]
    trend = None if config.trend == "none" else config.trend
    seasonal = None if config.seasonal == "none" else config.seasonal

    try:
        model = ExponentialSmoothing(
            model_history,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=config.seasonal_periods if seasonal else None,
            initialization_method="estimated",
        )
        fitted_model = model.fit(**fit_kwargs(config))
        forecast = fitted_model.forecast(int(config.prediction_horizon))
        return max(0.0, float(forecast[int(config.prediction_horizon) - 1]))
    except Exception as exc:
        print(f"[Warning] ETS failed, fallback to moving average: {exc}")
        return fallback_prediction(history, config)


def requests_to_pods(predicted_requests, config):
    desired_pods = math.ceil(predicted_requests / config.pod_capacity_per_window)
    desired_pods = max(config.min_pods, desired_pods)
    desired_pods = min(config.max_pods, desired_pods)
    return desired_pods


def patch_knative_min_scale(desired_pods, config):
    patch_json = min_scale_patch_json(config, str(desired_pods))
    run_kubectl_patch(patch_json, config)


def remove_knative_min_scale(config):
    patch_json = min_scale_patch_json(config, None)
    run_kubectl_patch(patch_json, config)


def resolve_kubectl_cmd(config):
    if config.kubectl_cmd:
        return shlex.split(config.kubectl_cmd)
    if shutil.which("kubectl"):
        return ["kubectl"]
    if shutil.which("k3s"):
        return ["sudo", "k3s", "kubectl"]
    return ["kubectl"]


class KubeApiUnavailable(RuntimeError):
    pass


def is_kube_api_unavailable(text):
    lowered = str(text or "").lower()
    return any(pattern in lowered for pattern in API_UNAVAILABLE_PATTERNS)


def run_kubectl_command(cmd, *, capture=False, check=True, timeout_s=None):
    timeout_s = KUBECTL_TIMEOUT_SECONDS if timeout_s is None else timeout_s
    stdout = subprocess.PIPE if capture else None
    stderr = subprocess.PIPE if capture else None
    last_proc = None
    last_output = ""
    for attempt in range(1, KUBECTL_RETRY_ATTEMPTS + 1):
        try:
            proc = subprocess.run(
                cmd,
                stdout=stdout,
                stderr=stderr,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout or ""
            err = exc.stderr or ""
            if isinstance(out, bytes):
                out = out.decode(errors="replace")
            if isinstance(err, bytes):
                err = err.decode(errors="replace")
            proc = subprocess.CompletedProcess(
                cmd,
                124,
                stdout=out,
                stderr=f"kubectl timed out after {timeout_s}s\n{err}",
            )
        last_proc = proc
        last_output = f"{proc.stdout or ''}{proc.stderr or ''}"
        if proc.returncode == 0:
            return proc
        if not is_kube_api_unavailable(last_output):
            break
        if attempt < KUBECTL_RETRY_ATTEMPTS:
            print(
                "[ETS kubectl] API unavailable; "
                f"retry={attempt}/{KUBECTL_RETRY_ATTEMPTS} "
                f"sleep={KUBECTL_RETRY_DELAY_SECONDS}s "
                f"output={last_output.strip()[-300:]}"
            )
            time.sleep(KUBECTL_RETRY_DELAY_SECONDS)

    if check:
        if is_kube_api_unavailable(last_output):
            raise KubeApiUnavailable(last_output.strip())
        raise subprocess.CalledProcessError(
            last_proc.returncode if last_proc else 1,
            cmd,
            output=last_proc.stdout if last_proc else "",
            stderr=last_proc.stderr if last_proc else last_output,
        )
    return last_proc


def min_scale_patch_json(config, value):
    if config.autoscale_target == "revision":
        return {
            "metadata": {
                "annotations": {
                    MIN_SCALE_ANNOTATION: value,
                }
            }
        }

    return {
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        MIN_SCALE_ANNOTATION: value,
                    }
                }
            }
        }
    }


def get_latest_ready_revision(config):
    cmd = resolve_kubectl_cmd(config) + [
        "get",
        "kservice",
        config.service_name,
        "-n",
        config.namespace,
        "-o",
        "jsonpath={.status.latestReadyRevisionName}",
    ]
    print("[ETS Resolve Revision]", " ".join(cmd))
    completed = run_kubectl_command(
        cmd,
        capture=True,
    )
    revision_name = completed.stdout.strip()
    if not revision_name:
        raise RuntimeError(
            f"Could not resolve latestReadyRevisionName for kservice {config.service_name}"
        )
    return revision_name


def resolve_patch_resource(config):
    if config.autoscale_target == "kservice":
        return "kservice", config.service_name

    if not config.revision_name:
        config.revision_name = get_latest_ready_revision(config)
    return "revision", config.revision_name


def prepare_ets_service(args):
    annotations = {
        AUTOSCALING_CLASS_ANNOTATION: args.ets_autoscaler_class,
        METRIC_ANNOTATION: args.ets_metric,
        TARGET_ANNOTATION: args.ets_target,
        MIN_SCALE_ANNOTATION: str(args.min_pods),
        MAX_SCALE_ANNOTATION: str(args.max_pods),
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

    print("[ETS Prepare]", " ".join(cmd))
    if not args.dry_run:
        run_kubectl_command(cmd)
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
    print("[ETS Wait]", " ".join(cmd))
    run_kubectl_command(cmd)


def run_kubectl_patch(patch_json, config):
    resource_type, resource_name = resolve_patch_resource(config)
    cmd = resolve_kubectl_cmd(config) + [
        "patch",
        resource_type,
        resource_name,
        "-n",
        config.namespace,
        "--type",
        "merge",
        "-p",
        json.dumps(patch_json),
    ]

    print("[ETS Patch]", " ".join(cmd))
    if not config.dry_run:
        run_kubectl_command(cmd)


def ready_pod_counts(config):
    if not config.revision_name:
        config.revision_name = get_latest_ready_revision(config)
    cmd = resolve_kubectl_cmd(config) + [
        "get",
        "pods",
        "-n",
        config.namespace,
        "-l",
        f"serving.knative.dev/revision={config.revision_name}",
        "-o",
        "json",
    ]
    completed = run_kubectl_command(
        cmd,
        capture=True,
    )
    payload = json.loads(completed.stdout)
    ready = 0
    total = 0
    for item in payload.get("items", []):
        metadata = item.get("metadata", {})
        if metadata.get("deletionTimestamp"):
            continue
        total += 1
        status = item.get("status", {})
        conditions = status.get("conditions", [])
        is_ready = any(
            condition.get("type") == "Ready" and condition.get("status") == "True"
            for condition in conditions
        )
        if status.get("phase") == "Running" and is_ready:
            ready += 1
    return ready, total


async def verify_scale_down_min_ready(config, recovery_pods):
    result = {
        "status": "skipped",
        "min_ready_observed": None,
        "final_ready": None,
        "final_total": None,
        "samples": 0,
        "recovery_pods": None,
    }
    if (
        config.dry_run
        or config.scale_down_post_check_seconds <= 0
        or config.scale_down_min_ready_pods <= 0
    ):
        return result

    deadline = time.time() + config.scale_down_post_check_seconds
    poll_seconds = max(0.1, config.scale_down_post_check_poll_seconds)
    result["status"] = "ok"
    while True:
        ready, total = await asyncio.to_thread(ready_pod_counts, config)
        result["samples"] += 1
        result["final_ready"] = ready
        result["final_total"] = total
        if result["min_ready_observed"] is None:
            result["min_ready_observed"] = ready
        else:
            result["min_ready_observed"] = min(result["min_ready_observed"], ready)

        if ready < config.scale_down_min_ready_pods:
            recovery_target = max(config.scale_down_min_ready_pods, recovery_pods)
            print(
                "[ETS Patch] post-scale-down ready guard violated "
                f"ready={ready} total={total} "
                f"required_ready={config.scale_down_min_ready_pods}; "
                f"recovering_min_scale={recovery_target}"
            )
            await asyncio.to_thread(patch_knative_min_scale, recovery_target, config)
            result["status"] = "recovered"
            result["recovery_pods"] = recovery_target
            return result

        remaining = deadline - time.time()
        if remaining <= 0:
            return result
        await asyncio.sleep(min(poll_seconds, remaining))


async def record_scale_latency(
    config,
    scale_latency_records,
    latest_patch_state,
    event_id,
    window_index,
    predicted_requests,
    desired_pods,
    previous_desired_pods,
    decision_epoch,
    patch_done_epoch,
):
    started_at = datetime.fromtimestamp(decision_epoch)
    initial_ready, initial_total = await asyncio.to_thread(ready_pod_counts, config)
    final_ready = initial_ready
    final_total = initial_total
    status = "timeout"
    superseded = False
    ready_epoch = None

    while True:
        now_epoch = time.time()
        final_ready, final_total = await asyncio.to_thread(ready_pod_counts, config)
        superseded = latest_patch_state["event_id"] != event_id

        if final_ready >= desired_pods:
            status = "ready"
            ready_epoch = now_epoch
            break

        if now_epoch - decision_epoch >= config.scale_latency_timeout_seconds:
            break

        await asyncio.sleep(config.scale_latency_poll_seconds)

    finished_epoch = ready_epoch or time.time()
    scale_latency_records.append({
        "event_id": event_id,
        "window_index": window_index,
        "revision_name": config.revision_name,
        "predicted_requests": predicted_requests,
        "previous_desired_pods": previous_desired_pods,
        "desired_pods": desired_pods,
        "decision_time": started_at.isoformat(timespec="seconds"),
        "decision_epoch": decision_epoch,
        "patch_done_time": datetime.fromtimestamp(patch_done_epoch).isoformat(
            timespec="seconds"
        ),
        "patch_done_epoch": patch_done_epoch,
        "ready_time": datetime.fromtimestamp(finished_epoch).isoformat(
            timespec="seconds"
        ) if status == "ready" else "",
        "ready_epoch": ready_epoch,
        "decision_to_ready_seconds": (
            ready_epoch - decision_epoch if ready_epoch is not None else None
        ),
        "patch_done_to_ready_seconds": (
            ready_epoch - patch_done_epoch if ready_epoch is not None else None
        ),
        "initial_ready_pods": initial_ready,
        "initial_total_pods": initial_total,
        "final_ready_pods": final_ready,
        "final_total_pods": final_total,
        "status": status,
        "superseded_by_later_patch": superseded,
    })


def save_ets_artifacts(config, controller_records, scale_latency_records=None):
    if not config.output_dir:
        return {}

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = config.run_id or make_run_id("ets")

    controller_path = output_dir / f"{run_id}_ets_controller_windows.csv"
    metadata_path = output_dir / f"{run_id}_ets_metadata.json"

    pd.DataFrame(controller_records).to_csv(controller_path, index=False)
    saved_paths = {
        "ets_controller_windows": str(controller_path),
        "ets_metadata": str(metadata_path),
    }
    if scale_latency_records is not None:
        scale_latency_path = output_dir / f"{run_id}_ets_scale_latency.csv"
        pd.DataFrame(scale_latency_records).to_csv(scale_latency_path, index=False)
        saved_paths["ets_scale_latency"] = str(scale_latency_path)

    metadata = {
        "run_id": run_id,
        "kubectl_cmd": config.kubectl_cmd,
        "train_csv": config.train_csv,
        "service_name": config.service_name,
        "namespace": config.namespace,
        "window_size_seconds": config.window_size_seconds,
        "duration_minutes": config.duration_minutes,
        "actual_request_source": "workload_replayer.generated_requests",
        "pod_capacity_per_window": config.pod_capacity_per_window,
        "min_pods": config.min_pods,
        "max_pods": config.max_pods,
        "scale_up_ready_latency_seconds": config.scale_up_ready_latency_seconds,
        "scale_up_slack_seconds": config.scale_up_slack_seconds,
        "scale_down_min_ready_pods": config.scale_down_min_ready_pods,
        "scale_down_guard_seconds": config.scale_down_guard_seconds,
        "scale_down_post_check_seconds": config.scale_down_post_check_seconds,
        "scale_down_post_check_poll_seconds": config.scale_down_post_check_poll_seconds,
        "history_window_size": config.history_window_size,
        "min_history_windows": config.min_history_windows,
        "prediction_horizon": config.prediction_horizon,
        "target_window_formula": "window_index + prediction_horizon - 1",
        "trend": config.trend,
        "seasonal": config.seasonal,
        "seasonal_periods": config.seasonal_periods,
        "smoothing_level": config.smoothing_level,
        "smoothing_trend": config.smoothing_trend,
        "smoothing_seasonal": config.smoothing_seasonal,
        "dry_run": config.dry_run,
        "autoscale_target": config.autoscale_target,
        "revision_name": config.revision_name,
        "ets_autoscaler_class": config.ets_autoscaler_class,
        "ets_metric": config.ets_metric,
        "ets_target": config.ets_target,
        "record_scale_latency": config.record_scale_latency,
        "scale_latency_poll_seconds": config.scale_latency_poll_seconds,
        "scale_latency_timeout_seconds": config.scale_latency_timeout_seconds,
        "ets_config_json": config.ets_config_json,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "saved_paths": saved_paths,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nSaved ETS artifacts:")
    for name, path in saved_paths.items():
        print(f"  {name}: {path}")

    return saved_paths


async def sleep_until(target_time):
    delay = target_time - time.monotonic()
    if delay > 0:
        await asyncio.sleep(delay)


def target_window_index(window_index, config):
    return window_index + max(0, int(config.prediction_horizon) - 1)


def scale_up_schedule(controller_start, window_index, config):
    target_index = target_window_index(window_index, config)
    target_start = controller_start + (target_index * config.window_size_seconds)
    lead_seconds = (
        config.scale_up_ready_latency_seconds + config.scale_up_slack_seconds
    )
    return target_start - lead_seconds, target_start, target_index


def empty_post_scale_down_guard():
    return {
        "status": "",
        "min_ready_observed": None,
        "final_ready": None,
        "final_total": None,
        "samples": 0,
        "recovery_pods": None,
    }


async def run_ets_controller(config, actual_request_queue):
    validate_ets_config(config)
    windows = experiment_window_count(config)
    history = load_train_history(config)
    controller_start, start_delay = resolve_experiment_start(
        config.start_at_epoch,
        config.start_delay_seconds,
    )
    planned_end = datetime.now() + timedelta(
        seconds=start_delay + (windows * config.window_size_seconds)
    )
    last_patched_pods = None
    pending_scale_down = None
    scale_down_candidate_pods = None
    scale_down_candidate_since_epoch = None
    scale_down_candidate_windows = 0
    controller_records = []
    scale_latency_records = []
    scale_latency_tasks = []
    latest_patch_state = {"event_id": 0}

    print("========== ETS predictive controller ==========")
    print(f"service={config.service_name} namespace={config.namespace}")
    print("actual_request_source=workload_replayer.poisson_generated")
    print(f"window_size_seconds={config.window_size_seconds}")
    print(f"duration_minutes={config.duration_minutes}")
    print(f"windows={windows}")
    print(f"planned_end={planned_end}")
    print(f"pod_capacity_per_window={config.pod_capacity_per_window}")
    print(f"kubectl_cmd={' '.join(resolve_kubectl_cmd(config))}")
    resource_type, resource_name = resolve_patch_resource(config)
    print(f"autoscale_target={resource_type}/{resource_name}")
    print(f"ets_autoscaler_class={config.ets_autoscaler_class}")
    print(f"ets_metric={config.ets_metric}")
    print(f"ets_target={config.ets_target}")
    print(f"record_scale_latency={config.record_scale_latency}")
    print(f"pod_range=[{config.min_pods}, {config.max_pods}]")
    print(
        "scale_up_schedule="
        f"ready_latency_seconds={config.scale_up_ready_latency_seconds}, "
        f"slack_seconds={config.scale_up_slack_seconds}"
    )
    print(
        "scale_down_ready_guard="
        f"min_ready={config.scale_down_min_ready_pods}, "
        f"guard_seconds={config.scale_down_guard_seconds}, "
        f"post_check_seconds={config.scale_down_post_check_seconds}, "
        f"post_check_poll_seconds={config.scale_down_post_check_poll_seconds}"
    )
    print(f"prediction_horizon={config.prediction_horizon}")
    print(
        "ets="
        f"trend={config.trend}, seasonal={config.seasonal}, "
        f"seasonal_periods={config.seasonal_periods}, "
        f"smoothing_level={config.smoothing_level}, "
        f"smoothing_trend={config.smoothing_trend}, "
        f"smoothing_seasonal={config.smoothing_seasonal}"
    )

    try:
        for window_index in range(windows):
            window_start = controller_start + (
                window_index * config.window_size_seconds
            )
            if window_index == 0:
                patch_time = window_start - config.first_patch_lead_seconds
            else:
                patch_time = window_start
            await sleep_until(patch_time)

            pending_scale_down_status = ""
            pending_scale_down_pods = None
            pending_scale_down_ready = None
            pending_scale_down_total = None
            pending_scale_down_required_ready = None
            pending_scale_down_elapsed_seconds = None
            pending_scale_down_windows = 0
            pending_post_scale_down_guard = empty_post_scale_down_guard()
            if pending_scale_down is not None:
                pending_scale_down_pods = pending_scale_down["pods"]
                pending_scale_down_windows = pending_scale_down["windows"]
                pending_scale_down_elapsed_seconds = (
                    time.time() - pending_scale_down["since_epoch"]
                )
                if (
                    last_patched_pods is None
                    or pending_scale_down_pods >= last_patched_pods
                ):
                    pending_scale_down_status = "skipped_obsolete"
                else:
                    pending_scale_down_required_ready = max(
                        config.scale_down_min_ready_pods,
                        last_patched_pods,
                    )
                    if config.dry_run:
                        pending_scale_down_ready = pending_scale_down_required_ready
                        pending_scale_down_total = pending_scale_down_required_ready
                    else:
                        pending_scale_down_ready, pending_scale_down_total = (
                            await asyncio.to_thread(ready_pod_counts, config)
                        )
                    guard_ready = (
                        pending_scale_down_ready
                        >= pending_scale_down_required_ready
                    )
                    guard_elapsed = (
                        pending_scale_down_elapsed_seconds
                        >= config.scale_down_guard_seconds
                    )
                    if guard_ready and guard_elapsed:
                        decision_epoch = time.time()
                        await asyncio.to_thread(
                            patch_knative_min_scale,
                            pending_scale_down_pods,
                            config,
                        )
                        patch_done_epoch = time.time()
                        previous_pods_before_pending = last_patched_pods
                        last_patched_pods = pending_scale_down_pods
                        pending_scale_down_status = "applied"
                        pending_post_scale_down_guard = (
                            await verify_scale_down_min_ready(
                                config,
                                previous_pods_before_pending,
                            )
                        )
                        if pending_post_scale_down_guard["status"] == "recovered":
                            last_patched_pods = pending_post_scale_down_guard[
                                "recovery_pods"
                            ]
                        if (
                            config.record_scale_latency
                            and not config.dry_run
                            and config.autoscale_target == "revision"
                        ):
                            latest_patch_state["event_id"] += 1
                            event_id = latest_patch_state["event_id"]
                            scale_latency_tasks.append(
                                asyncio.create_task(
                                    record_scale_latency(
                                        config=config,
                                        scale_latency_records=scale_latency_records,
                                        latest_patch_state=latest_patch_state,
                                        event_id=event_id,
                                        window_index=pending_scale_down["window_index"],
                                        predicted_requests=pending_scale_down[
                                            "predicted_requests"
                                        ],
                                        desired_pods=last_patched_pods,
                                        previous_desired_pods=(
                                            previous_pods_before_pending
                                        ),
                                        decision_epoch=decision_epoch,
                                        patch_done_epoch=patch_done_epoch,
                                    )
                                )
                            )
                    else:
                        pending_scale_down_status = (
                            "skipped_ready_guard"
                            f": ready={pending_scale_down_ready}"
                            f" total={pending_scale_down_total}"
                            f" required_ready={pending_scale_down_required_ready}"
                            f" elapsed_s={pending_scale_down_elapsed_seconds:.1f}"
                            f" required_elapsed_s={config.scale_down_guard_seconds:.1f}"
                            f" windows={pending_scale_down_windows}"
                        )
                print(
                    "[ETS Patch] pending scale-down "
                    f"target={pending_scale_down_pods} "
                    f"status={pending_scale_down_status}"
                )
                pending_scale_down = None

            predicted_requests = await asyncio.to_thread(
                predict_next_requests,
                history,
                config,
            )
            desired_pods = requests_to_pods(predicted_requests, config)
            history_windows_before_prediction = len(history)
            current_target_window = target_window_index(window_index, config)

            print(
                f"\n[ETS Window {window_index}] "
                f"target_window={current_target_window} "
                f"history_windows={len(history)} "
                f"predicted_requests={predicted_requests:.2f} "
                f"desired_pods={desired_pods}"
            )

            patch_target_pods = desired_pods
            previous_desired_pods = last_patched_pods
            patched_min_scale = desired_pods != last_patched_pods
            scale_action = "hold"
            scale_up_timing = ""
            scale_up_target_start_epoch = None
            scale_down_suppressed_reason = ""
            scale_down_guard_ready = None
            scale_down_guard_total = None
            scale_down_guard_required_ready = None
            scale_down_guard_elapsed_seconds = None
            scale_down_guard_windows = 0
            post_scale_down_guard = empty_post_scale_down_guard()

            if (
                previous_desired_pods is not None
                and desired_pods < previous_desired_pods
                and config.scale_down_guard_seconds > 0
            ):
                now_epoch = time.time()
                if scale_down_candidate_pods != desired_pods:
                    scale_down_candidate_pods = desired_pods
                    scale_down_candidate_since_epoch = now_epoch
                    scale_down_candidate_windows = 1
                else:
                    scale_down_candidate_windows += 1

                scale_down_guard_windows = scale_down_candidate_windows
                scale_down_guard_elapsed_seconds = (
                    now_epoch - scale_down_candidate_since_epoch
                    if scale_down_candidate_since_epoch is not None
                    else 0.0
                )
                scale_down_guard_required_ready = max(
                    config.scale_down_min_ready_pods,
                    previous_desired_pods,
                )
                if config.dry_run:
                    scale_down_guard_ready = scale_down_guard_required_ready
                    scale_down_guard_total = scale_down_guard_required_ready
                else:
                    scale_down_guard_ready, scale_down_guard_total = (
                        await asyncio.to_thread(ready_pod_counts, config)
                    )

                guard_ready = scale_down_guard_ready >= scale_down_guard_required_ready
                guard_elapsed = (
                    scale_down_guard_elapsed_seconds
                    >= config.scale_down_guard_seconds
                )
                if not (guard_ready and guard_elapsed):
                    patch_target_pods = previous_desired_pods
                    patched_min_scale = False
                    scale_action = "down_suppressed_ready_guard"
                    scale_down_suppressed_reason = (
                        "ready_guard"
                        f": ready={scale_down_guard_ready}"
                        f" total={scale_down_guard_total}"
                        f" required_ready={scale_down_guard_required_ready}"
                        f" elapsed_s={scale_down_guard_elapsed_seconds:.1f}"
                        f" required_elapsed_s={config.scale_down_guard_seconds:.1f}"
                        f" windows={scale_down_guard_windows}"
                    )
                    print(
                        "[ETS Patch] scale-down suppressed "
                        f"raw_desired_pods={desired_pods} "
                        f"current_min_scale={previous_desired_pods} "
                        f"reason={scale_down_suppressed_reason}"
                    )
            elif previous_desired_pods is None or desired_pods >= previous_desired_pods:
                scale_down_candidate_pods = None
                scale_down_candidate_since_epoch = None
                scale_down_candidate_windows = 0

            if (
                previous_desired_pods is not None
                and desired_pods < previous_desired_pods
                and not scale_down_suppressed_reason
            ):
                pending_scale_down = {
                    "pods": desired_pods,
                    "since_epoch": scale_down_candidate_since_epoch or time.time(),
                    "windows": max(1, scale_down_candidate_windows),
                    "window_index": window_index,
                    "target_window": current_target_window,
                    "predicted_requests": predicted_requests,
                    "previous_desired_pods": previous_desired_pods,
                }
                patch_target_pods = previous_desired_pods
                patched_min_scale = False
                scale_action = "down_pending_next_window"
                scale_down_suppressed_reason = (
                    "pending_next_window"
                    f": target={desired_pods}"
                    f" current_min_scale={previous_desired_pods}"
                    f" target_window={current_target_window}"
                )
                print(
                    "[ETS Patch] scale-down pending "
                    f"raw_desired_pods={desired_pods} "
                    f"current_min_scale={previous_desired_pods} "
                    f"target_window={current_target_window}"
                )

            if patched_min_scale:
                decision_epoch = time.time()
                was_scale_down_patch = (
                    previous_desired_pods is not None
                    and patch_target_pods < previous_desired_pods
                )
                if previous_desired_pods is None:
                    scale_action = "bootstrap_patch"
                elif patch_target_pods > previous_desired_pods:
                    scale_time, target_start, _ = scale_up_schedule(
                        controller_start,
                        window_index,
                        config,
                    )
                    scale_up_target_start_epoch = target_start
                    now_monotonic = time.monotonic()
                    if scale_time > now_monotonic:
                        await sleep_until(scale_time)
                        delay_s = scale_time - window_start
                        scale_up_timing = f"+{delay_s:.1f}s"
                    else:
                        scale_up_timing = "immediate"
                    target_offset_s = target_start - window_start
                    scale_action = (
                        f"up@{scale_up_timing}; "
                        f"ready_by=+{target_offset_s:.1f}s"
                    )
                await asyncio.to_thread(patch_knative_min_scale, patch_target_pods, config)
                patch_done_epoch = time.time()
                last_patched_pods = patch_target_pods
                if was_scale_down_patch:
                    post_scale_down_guard = await verify_scale_down_min_ready(
                        config,
                        previous_desired_pods,
                    )
                    if post_scale_down_guard["status"] == "recovered":
                        last_patched_pods = post_scale_down_guard["recovery_pods"]
                        patch_target_pods = last_patched_pods
                if patch_target_pods >= (previous_desired_pods or patch_target_pods):
                    scale_down_candidate_pods = None
                    scale_down_candidate_since_epoch = None
                    scale_down_candidate_windows = 0
                if (
                    config.record_scale_latency
                    and not config.dry_run
                    and config.autoscale_target == "revision"
                ):
                    latest_patch_state["event_id"] += 1
                    event_id = latest_patch_state["event_id"]
                    scale_latency_tasks.append(
                        asyncio.create_task(
                            record_scale_latency(
                                config=config,
                                scale_latency_records=scale_latency_records,
                                latest_patch_state=latest_patch_state,
                                event_id=event_id,
                                window_index=window_index,
                                predicted_requests=predicted_requests,
                                desired_pods=patch_target_pods,
                                previous_desired_pods=previous_desired_pods,
                                decision_epoch=decision_epoch,
                                patch_done_epoch=patch_done_epoch,
                            )
                        )
                    )
            else:
                if not scale_down_suppressed_reason:
                    print(f"[ETS Patch] skipped, min-scale already {patch_target_pods}")

            window_actual = await actual_request_queue.get()
            if window_actual["window_index"] != window_index:
                print(
                    "[Warning] ETS received window stats out of order: "
                    f"expected={window_index}, got={window_actual['window_index']}"
                )

            actual_requests = float(window_actual["generated_requests"])
            history.append(actual_requests)
            controller_records.append({
                "window_index": window_index,
                "csv_window_id": window_actual["csv_window_id"],
                "timestamp": window_actual["timestamp"],
                "workload_type": window_actual["workload_type"],
                "history_windows_before_prediction": history_windows_before_prediction,
                "prediction_horizon": config.prediction_horizon,
                "target_window": current_target_window,
                "predicted_requests": predicted_requests,
                "raw_desired_pods": desired_pods,
                "desired_pods": desired_pods,
                "applied_desired_pods": patch_target_pods,
                "previous_desired_pods": previous_desired_pods,
                "patched_min_scale": patched_min_scale,
                "scale_action": scale_action,
                "scale_up_timing": scale_up_timing,
                "scale_up_target_start_epoch": scale_up_target_start_epoch,
                "pending_scale_down_status": pending_scale_down_status,
                "pending_scale_down_pods": pending_scale_down_pods,
                "pending_scale_down_ready": pending_scale_down_ready,
                "pending_scale_down_total": pending_scale_down_total,
                "pending_scale_down_required_ready": pending_scale_down_required_ready,
                "pending_scale_down_elapsed_seconds": (
                    pending_scale_down_elapsed_seconds
                ),
                "pending_scale_down_windows": pending_scale_down_windows,
                "scale_down_suppressed_reason": scale_down_suppressed_reason,
                "scale_down_guard_required_ready": scale_down_guard_required_ready,
                "scale_down_guard_ready": scale_down_guard_ready,
                "scale_down_guard_total": scale_down_guard_total,
                "scale_down_guard_elapsed_seconds": scale_down_guard_elapsed_seconds,
                "scale_down_guard_windows": scale_down_guard_windows,
                "post_scale_down_guard_status": (
                    pending_post_scale_down_guard["status"]
                    or post_scale_down_guard["status"]
                ),
                "post_scale_down_min_ready_observed": (
                    pending_post_scale_down_guard["min_ready_observed"]
                    if pending_post_scale_down_guard["status"]
                    else post_scale_down_guard["min_ready_observed"]
                ),
                "post_scale_down_final_ready": (
                    pending_post_scale_down_guard["final_ready"]
                    if pending_post_scale_down_guard["status"]
                    else post_scale_down_guard["final_ready"]
                ),
                "post_scale_down_final_total": (
                    pending_post_scale_down_guard["final_total"]
                    if pending_post_scale_down_guard["status"]
                    else post_scale_down_guard["final_total"]
                ),
                "post_scale_down_samples": (
                    pending_post_scale_down_guard["samples"]
                    if pending_post_scale_down_guard["status"]
                    else post_scale_down_guard["samples"]
                ),
                "post_scale_down_recovery_pods": (
                    pending_post_scale_down_guard["recovery_pods"]
                    if pending_post_scale_down_guard["status"]
                    else post_scale_down_guard["recovery_pods"]
                ),
                "actual_generated_requests": actual_requests,
                "expected_requests": window_actual["expected_requests"],
                "pod_capacity_per_window": config.pod_capacity_per_window,
            })
            print(
                "actual_requests_from_replayer="
                f"{actual_requests:.2f} "
                f"expected_requests={window_actual['expected_requests']:.2f}"
            )

    finally:
        if config.reset_min_scale_on_exit:
            print("\nResetting Knative min-scale annotation...")
            await asyncio.to_thread(remove_knative_min_scale, config)

    if scale_latency_tasks:
        task_results = await asyncio.gather(
            *scale_latency_tasks,
            return_exceptions=True,
        )
        for task_result in task_results:
            if isinstance(task_result, Exception):
                print(f"[Warning] scale latency recorder failed: {task_result}")

    saved_paths = save_ets_artifacts(
        config,
        controller_records,
        scale_latency_records if config.record_scale_latency else None,
    )
    print(f"\nETS controller finished={datetime.now()}")
    return {
        "controller_records": controller_records,
        "saved_paths": saved_paths,
    }


async def main():
    args = parse_args()
    args = apply_ets_config_json(args)

    if args.prepare_ets_service:
        prepare_ets_service(args)

    start_at_epoch = args.start_at_epoch
    if start_at_epoch is None:
        start_at_epoch = time.time() + args.start_delay_seconds

    run_id = args.run_id or make_run_id("ets")
    args.run_id = run_id
    replay_config = build_replay_config(args, start_at_epoch)
    ets_config = build_ets_config(args, start_at_epoch)
    metrics_config = build_metrics_config(
        args,
        "ets",
        run_id,
        pod_capacity_per_window=args.pod_capacity_per_window,
    )
    actual_request_queue = asyncio.Queue()
    pod_samples = []
    stop_metrics = asyncio.Event()
    metrics_task = None

    async def observe_window(schedule):
        await actual_request_queue.put(schedule)

    controller_task = asyncio.create_task(
        run_ets_controller(ets_config, actual_request_queue)
    )
    await asyncio.sleep(0.1)
    if args.collect_pod_metrics:
        metrics_task = asyncio.create_task(
            sample_pod_metrics(
                resolve_kubectl_cmd(ets_config),
                metrics_config,
                stop_metrics,
                pod_samples,
            )
        )
    replay_task = asyncio.create_task(
        replay_workload(replay_config, window_observer=observe_window)
    )

    try:
        _, replay_result = await asyncio.gather(controller_task, replay_task)
    finally:
        if metrics_task:
            stop_metrics.set()
            await metrics_task

    save_metrics_artifacts(metrics_config, replay_result, pod_samples)


if __name__ == "__main__":
    asyncio.run(main())
