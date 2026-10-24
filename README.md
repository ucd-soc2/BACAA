# Autoscaling Experiment Artifact

Code and experimental data for forecast-driven autoscaling with readiness-aware replica planning and carbon-aware scale-down on a CPU-based Knative deployment. The artifact includes source versions, forecasting checkpoints, experiment configurations, selected run records, and numerical results for comparison with seven autoscaling baselines.

## Repository layout

```text
src/
  controller/              Replica decisions and forecasting interfaces
    v1_historical_baselines/
    v2_proposed_transformer/
  replay/                  Workload replay and baseline experiment runners
    v1_historical_baselines/
    v2_proposed_transformer/
  launchers/               Proposed, Transformer, sensitivity, and ablation batches
  analysis/                Request metrics, resource-based energy/carbon, batch checks
  model_training/          Retained Transformer training and plotting helpers
  cluster_validation/      Deployment and request-path checks
  recorded_proposed_run1/  Controller and batch-runner snapshot for Proposed run 1
models/                    BiLSTM and Transformer checkpoints and model support files
config/                    Recorded environments, frozen inputs, arguments, and YAMLs
data/
  workload/                Workload trace
  carbon/                  Carbon-intensity trace
  runs/                    Selected run records and source-configuration snapshots
  tables/                  Per-run and aggregate numerical results
LICENSE                    MIT license for repository-owned code
```

The two source families preserve differences between experiment batches. They are not interchangeable duplicates. The separate `recorded_proposed_run1` snapshot preserves the controller used for that selected run. Runtime scripts retain imports and paths from the original experiment layout; this archival layout is not an installed Python package.

## Experiment index

| Experiment | Retained runs | Numerical results |
| --- | --- | --- |
| Primary comparison | Eight methods, three selected runs each in [`01_primary_comparison`](data/runs/01_primary_comparison) | [`primary_all_runs_metrics.csv`](data/tables/01_primary_comparison/primary_all_runs_metrics.csv), [`primary_method_summary.csv`](data/tables/01_primary_comparison/primary_method_summary.csv) |
| Capacity-parameter sensitivity | One run at each of 10.00, 11.25, 13.75, and 15.00 requests/window; three reference runs at 12.50 in [`02_capacity_sensitivity`](data/runs/02_capacity_sensitivity) | [`sensitivity_all_runs_metrics.csv`](data/tables/02_capacity_sensitivity/sensitivity_all_runs_metrics.csv), [`sensitivity_summary.csv`](data/tables/02_capacity_sensitivity/sensitivity_summary.csv) |
| Cluster-capacity calibration | Three trials at each of 46, 48, 50, 52, and 54 requests/minute in [`03_cluster_capacity`](data/runs/03_cluster_capacity) | [`capacity_all_runs_metrics.csv`](data/tables/03_cluster_capacity/capacity_all_runs_metrics.csv), [`capacity_by_load_summary.csv`](data/tables/03_cluster_capacity/capacity_by_load_summary.csv), [`capacity_decision_check.csv`](data/tables/03_cluster_capacity/capacity_decision_check.csv) |
| Decision-layer ablation | Three runs for each of four configurations in [`04_decision_ablation`](data/runs/04_decision_ablation) | [`ablation_all_runs_metrics.csv`](data/tables/04_decision_ablation/ablation_all_runs_metrics.csv), [`ablation_summary.csv`](data/tables/04_decision_ablation/ablation_summary.csv) |

The primary method identifiers are `kpa`, `hpa`, `ets_direct`, `bilstm_direct`, `transformer_direct`, `ets_kpa`, `bilstm_kpa`, and `proposed`. The ablations disable carbon-aware fusion, readiness-aware pressure, or upper-quantile planning, respectively.

[`RUN_SELECTION.csv`](data/runs/RUN_SELECTION.csv) maps the repository's run labels to original run and batch identifiers. Its `run_path` and `reference_run_path` columns are relative to `data/runs/`. There are 58 entries representing 52 distinct runs: the sensitivity reference and full-controller ablation reuse the three primary Proposed runs. Their `run_reference.json` files point to the corresponding records; they are not additional repetitions. Selected runs can come from different batches or attempts, so a repository label such as `run3` does not necessarily mean original repetition 3.

## Quick start: inspect the recorded results

Run the following command from the repository root, which contains `src/`, `models/`, `config/`, and `data/`.

The following check requires only Python 3 and its standard library. It validates the selected workload slice and recalculates carbon means, sample standard deviations, and per-run SLO pass counts from the primary per-run table. It does not launch a cluster experiment.

```sh
python3 - <<'PY'
import csv
import statistics
from collections import defaultdict
from pathlib import Path

root = Path.cwd()
with (root / "data/workload/test_workload_burst.csv").open(encoding="utf-8-sig", newline="") as f:
    trace = list(csv.DictReader(f))[11708:11888]
assert len(trace) == 180
assert sum(int(row["requests_per_60s"]) for row in trace) == 4118
assert [int(trace[i]["window_id"]) for i in (0, -1)] == [87309, 87488]

runs = defaultdict(list)
with (root / "data/tables/01_primary_comparison/primary_all_runs_metrics.csv").open(encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        runs[row["method"]].append(row)
assert len(runs) == 8 and all(len(rows) == 3 for rows in runs.values())

def meets_slo(row):
    total = int(row["total_requests"])
    return (float(row["p95_latency_seconds"]) <= 15
            and int(row["successful_requests"]) / total >= 0.999
            and int(row["timeout_requests"]) / total < 0.001)

print("Workload: 180 windows, 4118 requests")
print(f"{'Method':<20} {'Carbon mean +/- SD (gCO2e)':<29} SLO passes")
for method, rows in runs.items():
    carbon = [float(row["estimated_total_carbon_gco2e"]) for row in rows]
    estimate = f"{statistics.mean(carbon):.2f} +/- {statistics.stdev(carbon):.2f}"
    print(f"{method:<20} {estimate:<29} {sum(map(meets_slo, rows))}/3")
PY
```

For `proposed`, the expected carbon result is **14.05 +/- 0.80 gCO2e**, with **3/3** selected runs meeting the joint SLO. These values summarize the recorded measurements; they do not establish reproducibility on a different deployment.

## Inputs and metric conventions

- **Workload:** [`test_workload_burst.csv`](data/workload/test_workload_burst.csv) contains the retained burst workload derived from Azure Functions traces. The online evaluation uses zero-based rows `[11708, 11888)`: 180 one-minute windows, window IDs 87309–87488, and 4,118 issued requests. The calibrated capacity parameter is 12.5 requests per Pod per 60-second window. Some legacy identifiers use `qps` for this per-window demand; check units before changing parameters.
- **Carbon trace:** [`carbon_intensity_ie_2025-09-20_24h.csv`](data/carbon/carbon_intensity_ie_2025-09-20_24h.csv) contains 288 five-minute observations for Ireland on 2025-09-20, sourced from Electricity Maps. Recorded runs map the 180-minute replay onto this 24-hour trace using `ci_time_scale = 8`.
- **Inference workload:** Frozen configurations retain the fixed English request text, `temperature = 0.0`, and `max_tokens = 100`. This request payload and the Qwen serving configuration are experimental inputs.
- **Joint experimental SLO:** A run passes when successful-request P95 latency is at most 15 seconds, request success is at least 99.9%, and the timeout rate is below 0.1%. Evaluate all three gates per run.
- **Request records:** Successful requests have HTTP status 200. The recorded timeout classification includes status 0 or an error containing `timeout`. Latencies are in seconds. See [`baseline_metrics.py`](src/analysis/baseline_metrics.py) and [`prometheus_energy_metrics.py`](src/analysis/prometheus_energy_metrics.py) for the retained metric implementations.
- **Energy and carbon:** The resource proxy uses 10.8 J/core-second, 0.3725 J/GB-second, and a 10.38 J initialization term per cold start, with 3,600,000 J/kWh. Carbon combines estimated energy with the aligned carbon trace in gCO2e/kWh. These are resource-based operational estimates, not direct wall-power measurements. Per-request carbon fields ending in `_gco2e` must be multiplied by 1,000 for mgCO2e.
- **Aggregation:** Summary files report means and sample standard deviations across selected runs. Latency summaries average the per-run percentiles; they are not percentiles of pooled requests. Single-run sensitivity entries have no estimate of between-run variability even where their stored SD field is zero. Empty fields indicate unavailable values, not zero.

A run directory may contain `run_metrics.csv`, request logs, window decisions, queue samples, scale events, Prometheus resource samples and summaries, runtime logs, and `source_configuration/`. Availability varies by run. Prefer the `data/tables/` files and corresponding run/Prometheus summaries for the final reported measures. Older `metrics_summary.json` files can contain a legacy two-second SLA threshold and a different energy approximation.

## Code, checkpoints, and environment

[`config/environment/python_version.txt`](config/environment/python_version.txt) records Python 3.12.3. [`pip_freeze.txt`](config/environment/pip_freeze.txt) records the original environment, including PyTorch, NumPy, pandas, SciPy, statsmodels, and aiohttp. It includes distribution-specific packages and is an environment inventory, not a portable requirements file. CSV/JSON inspection and the quick-start check above do not require these packages.

| Component | Location and use |
| --- | --- |
| Replica planning and carbon-aware decisions | [`src/controller/`](src/controller); forecasting interfaces and control loops for each retained source family |
| Baseline evaluation | [`src/replay/`](src/replay); KPA, HPA, ETS, BiLSTM, and forecast-assisted KPA runners |
| Experiment variants | [`src/launchers/`](src/launchers); batch parameters, ablation switches, capacity perturbations, and recorded retry policies |
| Resource estimation | [`src/analysis/prometheus_energy_metrics.py`](src/analysis/prometheus_energy_metrics.py); CPU/memory integration and carbon estimation |
| Forecast training | [`src/model_training/`](src/model_training) and [`train_bilstm_autoscaler.py`](src/replay/v1_historical_baselines/train_bilstm_autoscaler.py); retained training helpers |
| Deployment validation | [`src/cluster_validation/run_e2e_sanity_check.py`](src/cluster_validation/run_e2e_sanity_check.py); original testbed checks |

| Checkpoint | SHA-256 | Role |
| --- | --- | --- |
| [`transformer_current_7e601e3f.pt`](models/transformer_current_7e601e3f.pt) | `7e601e3f50a7ea1456fd769b1f9b2a8e226e2a8bffbb89e7f6b095b3aaec7d47` | Current Transformer snapshot used by the Proposed/Transformer experiment family |
| [`transformer_historical_b893c709.pt`](models/transformer_historical_b893c709.pt) | `b893c709ef4bbb245fe9d53883d9a24cfeb0f8cc45c2661714cc6b19fd5705ae` | Earlier Transformer snapshot referenced by historical configurations |
| [`bilstm_autoscaler_model_1169265b.pt`](models/bilstm_autoscaler_model_1169265b.pt) | `1169265b85f9a2111b62f98b61206e1a9affcb5e6c1fc4def38dd46a257f92fc` | BiLSTM baseline checkpoint |

The corresponding Transformer directories contain `model.py`, `inference.py`, `results.json`, and `confidence_calc.npz`. `QPSPredictor` expects the checkpoint as `best_model.pt` inside the selected model directory. For a separate working copy, place the matching checkpoint under that name and keep its model definition, training statistics, and confidence parameters together. Model loading and inference require PyTorch and the recorded numerical dependencies. Do not mix support files from different checkpoint versions.

## Running new cluster experiments

Live reruns require a Linux Kubernetes/Knative testbed, a reachable Prometheus endpoint, the original CPU serving image and GGUF model, access to the image on every serving node, and the service/routing configuration represented in `config/*/yamls/`. The supplied YAMLs use a local `ko.local/qwen-local` image with `imagePullPolicy: Never`; this repository does not distribute that image or the served LLM weights.

To prepare a rerun:

1. Select the matching source family and a run's `effective_args.json`, `frozen_inputs.json`, and YAML snapshot. The Proposed run-1 snapshot is separate from later Proposed runs.
2. Assemble a working experiment directory with the modules expected by the runners. Several launchers import `stage2.run_stage2_baselines_x3` and expect sibling replay, metric, model, and data files from the original `online_experiment` layout. Update local paths, service names, routing hosts, and Prometheus addresses for the target cluster.
3. Install compatible dependencies, supply the serving image/model, and configure the cluster. Use the retained validation code to inspect readiness, routing, and metrics before spending time on a full replay.
4. Preserve the chosen workload slice, forecast horizon, request payload, capacity parameter, carbon-time mapping, and metric coefficients. Inspect the selected launcher's skipped checks, cold-start limits, and retry rules: the `no_preflight_gate` variants record a particular experimental protocol and bypass some checks.
5. Write new measurements to a separate results directory. Compare per-run SLO gates and resource estimates using the same definitions before aggregating results.

The repository does not provide a complete cluster provisioning recipe or a verified one-command live reproduction workflow. The offline check above is the directly runnable entry point for examining the retained results.

## Coverage and known limitations

- **Training provenance:** The complete training/validation traces, trace-construction pipeline and function identifiers, and full parameter sweep are absent. The retained training helpers do not reconstruct the exact checkpoint pipeline by themselves. The current checkpoint's `results.json` names the same workload file for validation and testing, with a validation slice overlapping the online replay slice; this snapshot does not independently establish a disjoint held-out evaluation.
- **Resource sampling:** Recorded configurations use five-second sampling for historical baselines and ten-second sampling for several later Proposed, Transformer, sensitivity, and ablation batches.
- **Raw-data coverage:** Proposed run 1 lacks its raw resource-sample export. The 15 cluster-capacity trials retain summary metrics without the underlying request logs. Some historical baseline run-1 records lack scale-event counts needed for the complete scaling-event comparison. Missing fields should remain missing.
- **Latency reconciliation:** Some run-1 saved latency summaries differ slightly from recomputation using their request logs (up to approximately 0.0271 seconds in the inspected latency statistics). Request counts agree. Both records are retained without changing the measurements to force agreement.
- **Figures and deployment:** A complete figure-generation workflow, exact serving-image build, GGUF file, and cluster provisioning materials are not included. The available data and source support inspection of the recorded evaluation, with the above limits on end-to-end reproduction.
- **Provenance records:** Run IDs, source paths, checksums, and attempt histories describe the original execution environment. They are not all paths to files in this repository. Source comments and messages have been translated to English, and local export/report metadata has been removed; original source-time hashes therefore need not match the packaged source files. Measurement CSVs, model weights, and confidence arrays are preserved.

## License

Repository-owned code is licensed under the [MIT License](LICENSE). Third-party data, models, and dependencies remain subject to their respective original terms; the code license does not relicense those materials.
