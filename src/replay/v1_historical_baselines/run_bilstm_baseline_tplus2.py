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

import numpy as np
import pandas as pd

from train_bilstm_autoscaler import (
    request_column,
    require_torch,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_DIR = Path.home() / "online_experiment"
DEFAULT_DATA_DIR = DEFAULT_BASE_DIR / "data"
DEFAULT_RESULTS_DIR = DEFAULT_BASE_DIR / "results"

DEFAULT_TRAIN_CSV = str(DEFAULT_DATA_DIR / "train_workload_burst.csv")
DEFAULT_TEST_CSV = str(DEFAULT_DATA_DIR / "test_workload_burst.csv")

DEFAULT_MODEL_PATH = str(SCRIPT_DIR / "bilstm_autoscaler_model.pt")
DEFAULT_POD_CAPACITY_PER_WINDOW = 6.25
DEFAULT_URL = "http://192.168.10.101:31011/v1/chat/completions"
DEFAULT_HOST = "qwen.default.example.com"
DEFAULT_OUTPUT_DIR = str(DEFAULT_RESULTS_DIR)
DEFAULT_SERVICE_NAME = "qwen"
DEFAULT_NAMESPACE = "default"

AUTOSCALING_CLASS_ANNOTATION = "autoscaling.knative.dev/class"
METRIC_ANNOTATION = "autoscaling.knative.dev/metric"
TARGET_ANNOTATION = "autoscaling.knative.dev/target"
MIN_SCALE_ANNOTATION = "autoscaling.knative.dev/min-scale"
MAX_SCALE_ANNOTATION = "autoscaling.knative.dev/max-scale"
KPA_CLASS_VALUE = "kpa.autoscaling.knative.dev"


@dataclass
class ModelBundle:
    model: object
    mean: float
    std: float
    window_size: int
    prediction_horizon: int
    model_path: str


@dataclass
class BiLSTMConfig:
    model_path: str = DEFAULT_MODEL_PATH
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
    history_window_size: int = 10
    prediction_horizon: int = 2
    reset_min_scale_on_exit: bool = False
    dry_run: bool = False
    output_dir: Optional[str] = None
    run_id: Optional[str] = None
    kubectl_cmd: Optional[str] = None
    autoscale_target: str = "revision"
    revision_name: Optional[str] = None
    bilstm_autoscaler_class: str = KPA_CLASS_VALUE
    bilstm_metric: str = "concurrency"
    bilstm_target: str = "100"
    record_scale_latency: bool = False
    scale_latency_poll_seconds: float = 1.0
    scale_latency_timeout_seconds: float = 120.0


class TorchPredictor:
    def __init__(self, model):
        self.model = model

    def predict(self, normalized_window):
        torch, _, _, _ = require_torch()
        array = np.asarray(normalized_window, dtype=np.float32).reshape(1, -1, 1)
        tensor = torch.tensor(array, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            return float(self.model(tensor).detach().cpu().numpy()[0])


def create_bilstm_forecaster(input_size=1, hidden_size=64, dropout=0.1):
    torch, nn, _, _ = require_torch()

    class BiLSTMForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=True,
            )
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            forward_last = h_n[-2]
            backward_last = h_n[-1]
            sequence_summary = torch.cat([forward_last, backward_last], dim=1)
            return self.head(sequence_summary).squeeze(-1)

    return BiLSTMForecaster()


def map_requests_to_pods(
    predicted_requests,
    pod_capacity_per_window=DEFAULT_POD_CAPACITY_PER_WINDOW,
    min_pods=1,
    max_pods=4,
):
    if max_pods < min_pods:
        raise ValueError("max_pods must be greater than or equal to min_pods")
    if pod_capacity_per_window <= 0:
        raise ValueError("pod_capacity_per_window must be positive")
    raw_pods = math.ceil(max(0.0, float(predicted_requests)) / pod_capacity_per_window)
    return min(max_pods, max(min_pods, raw_pods))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the Bi-LSTM baseline: start a predictive fixed-scale controller "
            "and replay the same Poisson workload used by the other baselines."
        )
    )

    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
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
    parser.add_argument("--history-window-size", type=int, default=10)
    parser.add_argument("--prediction-horizon", type=int, default=2)
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
        help=(
            "Revision to patch when --autoscale-target=revision. Defaults to "
            "the service's latestReadyRevisionName."
        ),
    )
    parser.add_argument("--prepare-bilstm-service", action="store_true")
    parser.add_argument("--bilstm-autoscaler-class", default=KPA_CLASS_VALUE)
    parser.add_argument("--bilstm-metric", default="concurrency")
    parser.add_argument(
        "--bilstm-target",
        default="100",
        help=(
            "Reactive autoscaler target for the clean Bi-LSTM Revision. A high "
            "concurrency target is retained for compatibility; Bi-LSTM fixes each "
            "window by patching min-scale and max-scale to the same pod count."
        ),
    )
    parser.add_argument("--wait-ready-seconds", type=int, default=120)
    parser.add_argument("--no-wait-ready", action="store_true")
    parser.add_argument("--record-scale-latency", action="store_true")
    parser.add_argument("--scale-latency-poll-seconds", type=float, default=1.0)
    parser.add_argument("--scale-latency-timeout-seconds", type=float, default=120.0)

    from baseline_metrics import add_metrics_args

    add_metrics_args(parser, default_pod_capacity_per_window=DEFAULT_POD_CAPACITY_PER_WINDOW)
    return parser.parse_args()


def make_run_id(baseline_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{baseline_name}_{timestamp}"


def build_replay_config(args, start_at_epoch):
    from workload_replayer import ReplayConfig

    run_id = args.run_id or make_run_id("bilstm")
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
        baseline_name="bilstm",
        start_delay_seconds=args.start_delay_seconds,
        start_at_epoch=start_at_epoch,
        output_dir=args.output_dir,
        run_id=run_id,
        save_request_log=args.save_request_log,
    )


def build_bilstm_config(args, start_at_epoch):
    run_id = args.run_id or make_run_id("bilstm")
    return BiLSTMConfig(
        model_path=args.model_path,
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
        history_window_size=args.history_window_size,
        prediction_horizon=args.prediction_horizon,
        reset_min_scale_on_exit=args.reset_min_scale_on_exit,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
        run_id=run_id,
        kubectl_cmd=args.kubectl_cmd,
        autoscale_target=args.autoscale_target,
        revision_name=args.revision_name,
        bilstm_autoscaler_class=args.bilstm_autoscaler_class,
        bilstm_metric=args.bilstm_metric,
        bilstm_target=args.bilstm_target,
        record_scale_latency=args.record_scale_latency,
        scale_latency_poll_seconds=args.scale_latency_poll_seconds,
        scale_latency_timeout_seconds=args.scale_latency_timeout_seconds,
    )


def experiment_window_count(config):
    windows = int((config.duration_minutes * 60.0) // config.window_size_seconds)
    if windows <= 0:
        raise ValueError("duration_minutes must cover at least one window")
    return windows


def validate_bilstm_config(config):
    if config.max_pods < config.min_pods:
        raise ValueError("max_pods must be greater than or equal to min_pods")
    if config.pod_capacity_per_window <= 0:
        raise ValueError("pod_capacity_per_window must be positive")
    if config.history_window_size <= 0:
        raise ValueError("history_window_size must be positive")
    if config.prediction_horizon <= 0:
        raise ValueError("prediction_horizon must be positive")
    if not Path(config.model_path).exists():
        raise FileNotFoundError(f"model file does not exist: {config.model_path}")


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
    column = request_column(df)
    history = df[column].astype(float).clip(lower=0).tolist()
    if not history:
        raise ValueError("train workload history is empty")
    print(f"Loaded train history: {len(history)} windows from {config.train_csv}")
    return history


def load_model_bundle(model_path):
    torch, _, _, _ = require_torch()
    payload = torch.load(model_path, map_location="cpu")
    model_config = payload.get("model_config", {})
    autoscaler_config = payload.get("autoscaler_config", {})
    normalization = autoscaler_config.get("normalization", {})

    hidden_size = int(model_config.get("hidden_size", 64))
    dropout = float(model_config.get("dropout", 0.1))
    window_size = int(model_config.get("window_size", 10))
    prediction_horizon = int(
        model_config.get(
            "prediction_horizon",
            autoscaler_config.get("prediction_horizon", 1),
        )
    )
    mean = float(normalization["mean"])
    std = float(normalization["std"])

    model = create_bilstm_forecaster(hidden_size=hidden_size, dropout=dropout)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return ModelBundle(
        model=TorchPredictor(model),
        mean=mean,
        std=std,
        window_size=window_size,
        prediction_horizon=prediction_horizon,
        model_path=str(model_path),
    )


def fallback_prediction(history):
    if not history:
        return 0.0
    return max(0.0, sum(float(value) for value in history) / len(history))


def predict_next_requests(history, bundle):
    values = [float(value) for value in history]
    if len(values) < bundle.window_size:
        return fallback_prediction(values)

    window = np.asarray(values[-bundle.window_size:], dtype=float).reshape(
        bundle.window_size,
        1,
    )
    normalized = (window - float(bundle.mean)) / float(bundle.std)
    scaled_prediction = float(bundle.model.predict(normalized))
    predicted = (scaled_prediction * float(bundle.std)) + float(bundle.mean)
    return max(0.0, predicted)


def requests_to_pods(predicted_requests, config):
    return map_requests_to_pods(
        predicted_requests,
        config.pod_capacity_per_window,
        config.min_pods,
        config.max_pods,
    )


def patch_knative_fixed_scale(desired_pods, config):
    patch_json = min_scale_patch_json(config, str(desired_pods))
    run_kubectl_patch(patch_json, config)


def remove_knative_fixed_scale(config):
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


def min_scale_patch_json(config, value):
    annotations = {
        MIN_SCALE_ANNOTATION: value,
        MAX_SCALE_ANNOTATION: value,
    }

    if config.autoscale_target == "revision":
        return {
            "metadata": {
                "annotations": annotations,
            }
        }

    return {
        "spec": {
            "template": {
                "metadata": {
                    "annotations": annotations,
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
    print("[BiLSTM Resolve Revision]", " ".join(cmd))
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


def resolve_patch_resource(config):
    if config.autoscale_target == "kservice":
        return "kservice", config.service_name

    if not config.revision_name:
        config.revision_name = get_latest_ready_revision(config)
    return "revision", config.revision_name


def prepare_bilstm_service(args):
    annotations = {
        AUTOSCALING_CLASS_ANNOTATION: args.bilstm_autoscaler_class,
        METRIC_ANNOTATION: args.bilstm_metric,
        TARGET_ANNOTATION: args.bilstm_target,
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

    print("[BiLSTM Prepare]", " ".join(cmd))
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
    print("[BiLSTM Wait]", " ".join(cmd))
    subprocess.run(cmd, check=True)


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

    print("[BiLSTM Patch]", " ".join(cmd))
    if not config.dry_run:
        subprocess.run(cmd, check=True)


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
    completed = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
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
    if previous_desired_pods is None or desired_pods > previous_desired_pods:
        scale_direction = "up"
    elif desired_pods < previous_desired_pods:
        scale_direction = "down"
    else:
        scale_direction = "same"

    while True:
        now_epoch = time.time()
        final_ready, final_total = await asyncio.to_thread(ready_pod_counts, config)
        superseded = latest_patch_state["event_id"] != event_id

        if scale_direction == "down":
            target_reached = final_ready <= desired_pods
        else:
            target_reached = final_ready >= desired_pods

        if target_reached:
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
        "scale_direction": scale_direction,
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


def save_bilstm_artifacts(config, controller_records, scale_latency_records=None):
    if not config.output_dir:
        return {}

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = config.run_id or make_run_id("bilstm")

    controller_path = output_dir / f"{run_id}_bilstm_controller_windows.csv"
    metadata_path = output_dir / f"{run_id}_bilstm_metadata.json"

    pd.DataFrame(controller_records).to_csv(controller_path, index=False)
    saved_paths = {
        "bilstm_controller_windows": str(controller_path),
        "bilstm_metadata": str(metadata_path),
    }
    if scale_latency_records is not None:
        scale_latency_path = output_dir / f"{run_id}_bilstm_scale_latency.csv"
        pd.DataFrame(scale_latency_records).to_csv(scale_latency_path, index=False)
        saved_paths["bilstm_scale_latency"] = str(scale_latency_path)

    metadata = {
        "run_id": run_id,
        "model_path": config.model_path,
        "kubectl_cmd": config.kubectl_cmd,
        "train_csv": config.train_csv,
        "service_name": config.service_name,
        "namespace": config.namespace,
        "window_size_seconds": config.window_size_seconds,
        "duration_minutes": config.duration_minutes,
        "actual_request_source": "workload_replayer.generated_requests",
        "control_mode": "knative_fixed_scale",
        "forced_replicas": True,
        "pod_capacity_per_window": config.pod_capacity_per_window,
        "min_pods": config.min_pods,
        "max_pods": config.max_pods,
        "history_window_size": config.history_window_size,
        "prediction_horizon": config.prediction_horizon,
        "target_window_formula": "window_index + prediction_horizon - 1",
        "dry_run": config.dry_run,
        "autoscale_target": config.autoscale_target,
        "revision_name": config.revision_name,
        "bilstm_autoscaler_class": config.bilstm_autoscaler_class,
        "bilstm_metric": config.bilstm_metric,
        "bilstm_target": config.bilstm_target,
        "record_scale_latency": config.record_scale_latency,
        "scale_latency_poll_seconds": config.scale_latency_poll_seconds,
        "scale_latency_timeout_seconds": config.scale_latency_timeout_seconds,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "saved_paths": saved_paths,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nSaved Bi-LSTM controller artifacts:")
    for name, path in saved_paths.items():
        print(f"  {name}: {path}")

    return saved_paths


async def sleep_until(target_time):
    delay = target_time - time.monotonic()
    if delay > 0:
        await asyncio.sleep(delay)


def target_window_index(window_index, config):
    return window_index + max(0, int(config.prediction_horizon) - 1)


async def run_bilstm_controller(config, actual_request_queue):
    validate_bilstm_config(config)
    windows = experiment_window_count(config)
    history = load_train_history(config)
    bundle = load_model_bundle(config.model_path)
    if bundle.window_size != config.history_window_size:
        print(
            "[Warning] model window size differs from controller history size: "
            f"model={bundle.window_size}, controller={config.history_window_size}; "
            "using the model window size."
        )
        config.history_window_size = bundle.window_size
    if bundle.prediction_horizon != config.prediction_horizon:
        raise ValueError(
            "Bi-LSTM model prediction_horizon does not match controller: "
            f"model={bundle.prediction_horizon}, controller={config.prediction_horizon}. "
            "Use a t+2 model or pass the matching --prediction-horizon."
        )

    controller_start, start_delay = resolve_experiment_start(
        config.start_at_epoch,
        config.start_delay_seconds,
    )
    planned_end = datetime.now() + timedelta(
        seconds=start_delay + (windows * config.window_size_seconds)
    )
    last_patched_pods = None
    controller_records = []
    scale_latency_records = []
    scale_latency_tasks = []
    latest_patch_state = {"event_id": 0}

    print("========== Bi-LSTM predictive controller ==========")
    print(f"service={config.service_name} namespace={config.namespace}")
    print("actual_request_source=workload_replayer.poisson_generated")
    print("control_mode=knative_fixed_scale forced_replicas=True")
    print(f"model_path={config.model_path}")
    print(f"window_size_seconds={config.window_size_seconds}")
    print(f"duration_minutes={config.duration_minutes}")
    print(f"windows={windows}")
    print(f"planned_end={planned_end}")
    print(f"pod_capacity_per_window={config.pod_capacity_per_window}")
    print(f"kubectl_cmd={' '.join(resolve_kubectl_cmd(config))}")
    resource_type, resource_name = resolve_patch_resource(config)
    print(f"autoscale_target={resource_type}/{resource_name}")
    print(f"bilstm_autoscaler_class={config.bilstm_autoscaler_class}")
    print(f"bilstm_metric={config.bilstm_metric}")
    print(f"bilstm_target={config.bilstm_target}")
    print(f"record_scale_latency={config.record_scale_latency}")
    print(f"pod_range=[{config.min_pods}, {config.max_pods}]")
    print(f"history_window_size={config.history_window_size}")
    print(f"prediction_horizon={config.prediction_horizon}")

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

            predicted_requests = await asyncio.to_thread(
                predict_next_requests,
                history,
                bundle,
            )
            desired_pods = requests_to_pods(predicted_requests, config)
            history_windows_before_prediction = len(history)
            current_target_window = target_window_index(window_index, config)

            print(
                f"\n[BiLSTM Window {window_index}] "
                f"target_window={current_target_window} "
                f"history_windows={len(history)} "
                f"predicted_requests={predicted_requests:.2f} "
                f"desired_pods={desired_pods}"
            )

            patched_fixed_scale = desired_pods != last_patched_pods
            previous_desired_pods = last_patched_pods
            patch_error = ""
            if patched_fixed_scale:
                decision_epoch = time.time()
                try:
                    await asyncio.to_thread(
                        patch_knative_fixed_scale,
                        desired_pods,
                        config,
                    )
                    patch_done_epoch = time.time()
                    last_patched_pods = desired_pods
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
                                    desired_pods=desired_pods,
                                    previous_desired_pods=previous_desired_pods,
                                    decision_epoch=decision_epoch,
                                    patch_done_epoch=patch_done_epoch,
                                )
                            )
                        )
                except Exception as exc:
                    patch_error = str(exc)
                    print(f"[Warning] Bi-LSTM patch failed: {patch_error}")
            else:
                print(f"[BiLSTM Patch] skipped, fixed scale already {desired_pods}")

            window_actual = await actual_request_queue.get()
            if window_actual["window_index"] != window_index:
                print(
                    "[Warning] Bi-LSTM received window stats out of order: "
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
                "model_window_size": bundle.window_size,
                "prediction_horizon": config.prediction_horizon,
                "target_window": current_target_window,
                "predicted_requests": predicted_requests,
                "desired_pods": desired_pods,
                "previous_desired_pods": previous_desired_pods,
                "patched_fixed_scale": patched_fixed_scale,
                "patched_min_scale": patched_fixed_scale,
                "patch_error": patch_error,
                "actual_generated_requests": actual_requests,
                "expected_requests": window_actual["expected_requests"],
                "pod_capacity_per_window": config.pod_capacity_per_window,
                "control_mode": "knative_fixed_scale",
                "forced_replicas": True,
            })
            print(
                "actual_requests_from_replayer="
                f"{actual_requests:.2f} "
                f"expected_requests={window_actual['expected_requests']:.2f}"
            )

    finally:
        if config.reset_min_scale_on_exit:
            print("\nResetting Knative min-scale and max-scale annotations...")
            await asyncio.to_thread(remove_knative_fixed_scale, config)

    if scale_latency_tasks:
        task_results = await asyncio.gather(
            *scale_latency_tasks,
            return_exceptions=True,
        )
        for task_result in task_results:
            if isinstance(task_result, Exception):
                print(f"[Warning] scale latency recorder failed: {task_result}")

    saved_paths = save_bilstm_artifacts(
        config,
        controller_records,
        scale_latency_records if config.record_scale_latency else None,
    )
    print(f"\nBi-LSTM controller finished={datetime.now()}")
    return {
        "controller_records": controller_records,
        "saved_paths": saved_paths,
    }


async def main():
    args = parse_args()

    if args.prepare_bilstm_service:
        prepare_bilstm_service(args)

    start_at_epoch = args.start_at_epoch
    if start_at_epoch is None:
        start_at_epoch = time.time() + args.start_delay_seconds

    run_id = args.run_id or make_run_id("bilstm")
    args.run_id = run_id
    replay_config = build_replay_config(args, start_at_epoch)
    bilstm_config = build_bilstm_config(args, start_at_epoch)

    from baseline_metrics import (
        build_metrics_config,
        sample_pod_metrics,
        save_metrics_artifacts,
    )
    from workload_replayer import replay_workload

    metrics_config = build_metrics_config(
        args,
        "bilstm",
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
        run_bilstm_controller(bilstm_config, actual_request_queue)
    )
    await asyncio.sleep(0.1)
    if args.collect_pod_metrics:
        metrics_task = asyncio.create_task(
            sample_pod_metrics(
                resolve_kubectl_cmd(bilstm_config),
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
