import argparse
import asyncio
import json
import time
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Awaitable, Callable, Optional

import aiohttp
import numpy as np
import pandas as pd


DEFAULT_BASE_DIR = Path.home() / "online_experiment"
DEFAULT_DATA_DIR = DEFAULT_BASE_DIR / "data"
DEFAULT_RESULTS_DIR = DEFAULT_BASE_DIR / "results"
DEFAULT_TEST_CSV = str(DEFAULT_DATA_DIR / "test_workload.csv")
DEFAULT_URL = "http://192.168.10.101:31011/v1/chat/completions"
DEFAULT_HOST = "qwen.default.example.com"
DEFAULT_OUTPUT_DIR = str(DEFAULT_RESULTS_DIR)

PAYLOAD = {
    "model": "qwen",
    "messages": [
        {
            "role": "user",
            "content": "A SaaS app suddenly receives 10 times more user requests after a product launch. Explain in simple language why autoscaling is needed, what could happen if scaling is too slow, and name one metric the system could monitor to decide when to add more servers. Give one short example. Keep the answer around 75 words.",
        }
    ],
    "max_tokens": 100,
    "temperature": 0.0,
}


@dataclass
class ReplayConfig:
    csv: str = DEFAULT_TEST_CSV
    url: str = DEFAULT_URL
    host: str = DEFAULT_HOST
    window_size_seconds: float = 20.0
    duration_minutes: float = 5.0
    start_row: int = 0
    scale: float = 1.0
    seed: Optional[int] = 42
    timeout_seconds: float = 60.0
    connector_limit: int = 0
    baseline_name: str = "workload"
    start_delay_seconds: float = 5.0
    start_at_epoch: Optional[float] = None
    output_dir: Optional[str] = None
    run_id: Optional[str] = None
    save_request_log: bool = False


class ReplayRuntimeState:
    """Track due, pending, and in-flight replay requests inside the event loop."""

    def __init__(self):
        self._planned_send_times = []
        self._started_requests = 0
        self._completed_requests = 0
        self._status_counts = Counter()

    def plan_request(self, absolute_send_time: float) -> None:
        self._planned_send_times.append(float(absolute_send_time))

    def finalize_plan(self) -> None:
        self._planned_send_times.sort()

    def mark_started(self) -> None:
        self._started_requests += 1

    def mark_completed(self, status: int) -> None:
        self._completed_requests += 1
        self._status_counts[int(status)] += 1

    def snapshot(self, now: Optional[float] = None) -> dict:
        now = time.monotonic() if now is None else float(now)
        due_requests = bisect_right(self._planned_send_times, now)
        pending_requests = max(0, due_requests - self._started_requests)
        in_flight_requests = max(0, self._started_requests - self._completed_requests)
        return {
            "planned_requests": len(self._planned_send_times),
            "due_requests": int(due_requests),
            "started_requests": int(self._started_requests),
            "completed_requests": int(self._completed_requests),
            "pending_requests": int(pending_requests),
            "in_flight_requests": int(in_flight_requests),
            "active_requests": int(pending_requests + in_flight_requests),
            "status_counts": dict(sorted(self._status_counts.items())),
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay a selected slice of test_workload.csv with a Poisson "
            "arrival process."
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
    parser.add_argument("--baseline-name", default="workload")
    parser.add_argument("--start-delay-seconds", type=float, default=5.0)
    parser.add_argument("--start-at-epoch", type=float, default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--save-request-log", action="store_true")
    return parser.parse_args()


def config_from_args(args):
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
        baseline_name=args.baseline_name,
        start_delay_seconds=args.start_delay_seconds,
        start_at_epoch=args.start_at_epoch,
        output_dir=args.output_dir,
        run_id=args.run_id,
        save_request_log=args.save_request_log,
    )


def request_column(df):
    for column in ("requests_per_60s", "requests_per_20s", "requests", "request_count"):
        if column in df.columns:
            return column
    raise ValueError("CSV has no request count column")


def window_count(config):
    total_seconds = config.duration_minutes * 60.0
    windows = int(total_seconds // config.window_size_seconds)
    if windows <= 0:
        raise ValueError("duration_minutes must cover at least one window")
    return windows


def make_run_id(baseline_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{baseline_name}_{timestamp}"


def load_workload(config):
    df = pd.read_csv(config.csv)
    col = request_column(df)
    windows = window_count(config)
    end_row = config.start_row + windows
    workload = df.iloc[config.start_row:end_row].copy()

    if len(workload) < windows:
        raise ValueError(
            f"CSV only has {len(workload)} rows from start_row={config.start_row}, "
            f"but {windows} are needed"
        )

    workload["_expected_requests"] = (
        workload[col].astype(float).mul(config.scale).clip(lower=0)
    )
    return workload, col, windows


def resolve_experiment_start(config):
    if config.start_at_epoch is None:
        delay = config.start_delay_seconds
    else:
        delay = config.start_at_epoch - time.time()
        if delay < 0:
            raise ValueError("--start-at-epoch is already in the past")
    return time.monotonic() + delay, delay


def poisson_offsets(expected_requests, window_size_seconds, rng):
    request_count = int(round(expected_requests))
    if request_count <= 0:
        return []

    offsets = rng.uniform(0.0, window_size_seconds, size=request_count)
    offsets.sort()
    return offsets.tolist()


async def sleep_until(target_time):
    delay = target_time - time.monotonic()
    if delay > 0:
        await asyncio.sleep(delay)


async def send_one(session, config):
    start_time = time.monotonic()
    try:
        async with session.post(
            config.url,
            json=PAYLOAD,
            headers={
                "Host": config.host,
                "Content-Type": "application/json",
            },
            timeout=config.timeout_seconds,
        ) as resp:
            await resp.text()
            return {
                "status": resp.status,
                "elapsed": time.monotonic() - start_time,
                "error": "",
            }
    except Exception as exc:
        return {
            "status": 0,
            "elapsed": time.monotonic() - start_time,
            "error": str(exc),
        }


async def scheduled_request(session, config, absolute_send_time, window_index, runtime_state=None):
    await sleep_until(absolute_send_time)
    if runtime_state is not None:
        runtime_state.mark_started()
    result = await send_one(session, config)
    if runtime_state is not None:
        runtime_state.mark_completed(result["status"])
    result["window_index"] = window_index
    return result


async def notify_window_observer(window_observer, schedule):
    if window_observer is None:
        return
    await window_observer(dict(schedule))


def print_window_schedule(window_index, row, generated_requests):
    original_window_id = row.get("window_id", "")
    timestamp = row.get("timestamp", "")
    workload_type = row.get("workload_type", "")
    expected = float(row["_expected_requests"])
    print(
        f"[Schedule] local_window={window_index} "
        f"csv_window_id={original_window_id} timestamp={timestamp} "
        f"type={workload_type} expected={expected:.2f} "
        f"poisson_generated={generated_requests}"
    )


def build_window_summaries(results, schedules, windows):
    by_window = defaultdict(list)
    for result in results:
        try:
            window_index = int(result["window_index"])
        except (KeyError, TypeError, ValueError):
            continue
        by_window[window_index].append(result)

    summaries = []
    for window_index in range(min(windows, len(schedules))):
        window_results = by_window[window_index]
        counter = Counter(result["status"] for result in window_results)
        ok = counter.get(200, 0)
        failed = len(window_results) - ok
        latencies = [r["elapsed"] for r in window_results if r["status"] == 200]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        p50_latency = float(np.percentile(latencies, 50)) if latencies else 0.0
        p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0
        schedule = schedules[window_index]

        summaries.append({
            "window_index": window_index,
            "csv_window_id": schedule["csv_window_id"],
            "timestamp": schedule["timestamp"],
            "workload_type": schedule["workload_type"],
            "expected_requests": schedule["expected_requests"],
            "generated_requests": schedule["generated_requests"],
            "sent_requests": len(window_results),
            "ok_requests": ok,
            "failed_requests": failed,
            "avg_latency_seconds": avg_latency,
            "p50_latency_seconds": p50_latency,
            "p95_latency_seconds": p95_latency,
            "status_counts": json.dumps(dict(sorted(counter.items()))),
        })

    return summaries


def write_dataframe_csv_atomic(frame, path):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def write_json_atomic(payload, path):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def runtime_snapshot_or_none(runtime_state):
    if runtime_state is None:
        return None
    return runtime_state.snapshot()


def collect_finished_task_results(tasks, results, seen_task_ids):
    added = 0
    for task in tasks:
        task_id = id(task)
        if task_id in seen_task_ids or not task.done() or task.cancelled():
            continue
        try:
            result = task.result()
        except Exception as exc:
            result = {
                "status": 0,
                "elapsed": 0.0,
                "error": f"task exception: {exc}",
                "window_index": "",
            }
        results.append(result)
        seen_task_ids.add(task_id)
        added += 1
    return added


def summarize_results(window_summaries):
    print("\n========== Replay summary ==========")
    total_counter = Counter()
    for summary in window_summaries:
        status_counts = json.loads(summary["status_counts"])
        total_counter.update({int(k): v for k, v in status_counts.items()})

        print(
            f"[Window {summary['window_index']:02d}] "
            f"expected={summary['expected_requests']:.2f} "
            f"generated={summary['generated_requests']} "
            f"sent={summary['sent_requests']} "
            f"ok={summary['ok_requests']} "
            f"failed={summary['failed_requests']} "
            f"avg_latency={summary['avg_latency_seconds']:.3f}s "
            f"status_counts={status_counts}"
        )

    print(f"Total status counts: {dict(sorted(total_counter.items()))}")


def save_replay_artifacts(
    config,
    schedules,
    window_summaries,
    results,
    *,
    artifact_complete=True,
    partial_reason=None,
    runtime_snapshot=None,
    quiet=False,
):
    if not config.output_dir:
        return {}

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = config.run_id or make_run_id(config.baseline_name)

    schedule_path = output_dir / f"{run_id}_replay_schedule.csv"
    summary_path = output_dir / f"{run_id}_replay_window_summary.csv"
    metadata_path = output_dir / f"{run_id}_replay_metadata.json"

    write_dataframe_csv_atomic(pd.DataFrame(schedules), schedule_path)
    write_dataframe_csv_atomic(pd.DataFrame(window_summaries), summary_path)

    saved_paths = {
        "schedule": str(schedule_path),
        "window_summary": str(summary_path),
        "metadata": str(metadata_path),
    }

    if config.save_request_log:
        request_log_path = output_dir / f"{run_id}_replay_request_log.csv"
        write_dataframe_csv_atomic(pd.DataFrame(results), request_log_path)
        saved_paths["request_log"] = str(request_log_path)

    metadata = {
        "run_id": run_id,
        "baseline_name": config.baseline_name,
        "csv": config.csv,
        "url": config.url,
        "host": config.host,
        "window_size_seconds": config.window_size_seconds,
        "duration_minutes": config.duration_minutes,
        "start_row": config.start_row,
        "scale": config.scale,
        "seed": config.seed,
        "timeout_seconds": config.timeout_seconds,
        "connector_limit": config.connector_limit,
        "save_request_log": config.save_request_log,
        "artifact_complete": bool(artifact_complete),
        "partial_reason": partial_reason,
        "result_count": len(results),
        "schedule_count": len(schedules),
        "window_summary_count": len(window_summaries),
        "runtime_snapshot": runtime_snapshot,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "saved_paths": saved_paths,
    }
    write_json_atomic(metadata, metadata_path)

    if not quiet:
        artifact_label = "replay artifacts" if artifact_complete else "partial replay artifacts"
        print(f"\nSaved {artifact_label}:")
        for name, path in saved_paths.items():
            print(f"  {name}: {path}")

    return saved_paths


async def replay_workload(
    config,
    window_observer: Optional[Callable[[dict], Awaitable[None]]] = None,
    runtime_state: Optional[ReplayRuntimeState] = None,
):
    workload, request_col, windows = load_workload(config)
    experiment_start, start_delay = resolve_experiment_start(config)
    rng = np.random.default_rng(config.seed)
    schedules = []
    tasks = []
    results = []
    seen_task_ids = set()
    window_summaries = []
    saved_paths = {}
    completed_normally = False
    partial_reason = None

    print("========== Poisson workload replay ==========")
    print(f"baseline_name={config.baseline_name}")
    print(f"start_time={datetime.now()}")
    print(f"csv={config.csv}")
    print(f"url={config.url}")
    print(f"host={config.host}")
    print(f"request_column={request_col}")
    print(f"window_size_seconds={config.window_size_seconds}")
    print(f"duration_minutes={config.duration_minutes}")
    print(f"windows={windows}")
    print(f"scale={config.scale}")
    print(f"seed={config.seed}")
    print(f"run_id={config.run_id}")
    print(f"output_dir={config.output_dir}")

    connector = aiohttp.TCPConnector(limit=config.connector_limit)
    timeout = aiohttp.ClientTimeout(
        total=config.timeout_seconds + (config.duration_minutes * 60.0) + 30
    )

    try:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for local_window_index, (_, row) in enumerate(workload.iterrows()):
                offsets = poisson_offsets(
                    expected_requests=float(row["_expected_requests"]),
                    window_size_seconds=config.window_size_seconds,
                    rng=rng,
                )
                generated_requests = len(offsets)
                schedules.append({
                    "window_index": local_window_index,
                    "csv_window_id": row.get("window_id", ""),
                    "timestamp": row.get("timestamp", ""),
                    "workload_type": row.get("workload_type", ""),
                    "expected_requests": float(row["_expected_requests"]),
                    "generated_requests": generated_requests,
                })
                print_window_schedule(local_window_index, row, generated_requests)

                window_start = experiment_start + (
                    local_window_index * config.window_size_seconds
                )
                for offset in offsets:
                    absolute_send_time = window_start + offset
                    if runtime_state is not None:
                        runtime_state.plan_request(absolute_send_time)
                    tasks.append(
                        asyncio.create_task(
                            scheduled_request(
                                session=session,
                                config=config,
                                absolute_send_time=absolute_send_time,
                                window_index=local_window_index,
                                runtime_state=runtime_state,
                            )
                        )
                    )

            if runtime_state is not None:
                runtime_state.finalize_plan()

            duration_seconds = windows * config.window_size_seconds
            planned_end = datetime.now() + timedelta(
                seconds=start_delay + duration_seconds
            )
            print(
                f"\nReplay will start in {start_delay:.1f} seconds "
                f"and run until about {planned_end}."
            )

            for schedule in schedules:
                window_end = experiment_start + (
                    (schedule["window_index"] + 1) * config.window_size_seconds
                )
                await sleep_until(window_end)
                await notify_window_observer(window_observer, schedule)
                collect_finished_task_results(tasks, results, seen_task_ids)
                window_summaries = build_window_summaries(results, schedules, windows)
                saved_paths = save_replay_artifacts(
                    config=config,
                    schedules=schedules,
                    window_summaries=window_summaries,
                    results=results,
                    artifact_complete=False,
                    partial_reason=(
                        "incremental_flush_after_window="
                        f"{schedule['window_index']}"
                    ),
                    runtime_snapshot=runtime_snapshot_or_none(runtime_state),
                    quiet=True,
                )

            print("All windows have been scheduled. Waiting for in-flight requests...")
            if tasks:
                await asyncio.gather(*tasks)
            collect_finished_task_results(tasks, results, seen_task_ids)
            completed_normally = True
    except BaseException as exc:
        partial_reason = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if not completed_normally:
            pending_tasks = [task for task in tasks if not task.done()]
            for task in pending_tasks:
                task.cancel()
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            collect_finished_task_results(tasks, results, seen_task_ids)

        if completed_normally or (config.output_dir and schedules):
            window_summaries = build_window_summaries(results, schedules, windows)
            if completed_normally:
                summarize_results(window_summaries)
            saved_paths = save_replay_artifacts(
                config=config,
                schedules=schedules,
                window_summaries=window_summaries,
                results=results,
                artifact_complete=completed_normally,
                partial_reason=None if completed_normally else partial_reason,
                runtime_snapshot=runtime_snapshot_or_none(runtime_state),
                quiet=False,
            )
            if completed_normally:
                print(f"finished={datetime.now()}")

    return {
        "results": results,
        "schedules": schedules,
        "window_summaries": window_summaries,
        "windows": windows,
        "saved_paths": saved_paths,
    }


async def main():
    config = config_from_args(parse_args())
    await replay_workload(config)


if __name__ == "__main__":
    asyncio.run(main())
