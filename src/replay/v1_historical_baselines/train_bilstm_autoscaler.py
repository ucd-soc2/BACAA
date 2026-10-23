import argparse
import json
import math
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product


DEFAULT_BASE_DIR = Path(r"G:\PycharmProjects\WorkloadTransformerProject")
DEFAULT_DATA_DIR = DEFAULT_BASE_DIR / "data" / "final"
DEFAULT_OUTPUT_DIR = DEFAULT_BASE_DIR / "baselines" / "bilstm" / "results"
DEFAULT_TRAIN_CSV = DEFAULT_DATA_DIR / "train_workload.csv"
DEFAULT_VAL_CSV = DEFAULT_DATA_DIR / "val_workload.csv"
DEFAULT_TEST_CSV = DEFAULT_DATA_DIR / "test_workload.csv"
DEFAULT_MODEL_FILENAME = "bilstm_autoscaler_model.pt"


@dataclass
class NormalizationStats:
    mean: float
    std: float

    @classmethod
    def from_values(cls, values):
        array = np.asarray(values, dtype=float)
        if array.size == 0:
            raise ValueError("values must not be empty")
        std = float(array.std())
        return cls(mean=float(array.mean()), std=std if std > 1e-9 else 1.0)

    def normalize(self, values):
        return (np.asarray(values, dtype=float) - self.mean) / self.std

    def denormalize(self, values):
        return (np.asarray(values, dtype=float) * self.std) + self.mean


def require_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required for Bi-LSTM training and inference. "
            "Install torch in the VM environment before running this script."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def request_column(df):
    for column in ("requests_per_60s", "requests_per_20s", "requests", "request_count"):
        if column in df.columns:
            return column
    raise ValueError("CSV has no request count column")


def read_workload_values(csv_path):
    df = pd.read_csv(csv_path)
    column = request_column(df)
    return df, df[column].astype(float).to_numpy()


def build_supervised_samples(values, window_size):
    values = np.asarray(values, dtype=float)
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if len(values) <= window_size:
        raise ValueError("values must contain more rows than window_size")

    x_rows = []
    y_rows = []
    target_indexes = []
    for target_index in range(window_size, len(values)):
        x_rows.append(values[target_index - window_size:target_index])
        y_rows.append(values[target_index])
        target_indexes.append(target_index)

    x = np.asarray(x_rows, dtype=np.float32).reshape(-1, window_size, 1)
    y = np.asarray(y_rows, dtype=np.float32)
    return x, y, np.asarray(target_indexes, dtype=int)


def split_supervised_by_target(target_indexes, train_size, val_size):
    train_end = int(train_size)
    val_end = int(train_size + val_size)
    target_indexes = np.asarray(target_indexes, dtype=int)
    return {
        "train": np.flatnonzero(target_indexes < train_end),
        "val": np.flatnonzero(
            (target_indexes >= train_end) & (target_indexes < val_end)
        ),
        "test": np.flatnonzero(target_indexes >= val_end),
    }


def requests_to_pods(
    predicted_requests,
    pod_capacity_per_window=13.75,
    min_pods=1,
    max_pods=6,
):
    if max_pods < min_pods:
        raise ValueError("max_pods must be greater than or equal to min_pods")
    if pod_capacity_per_window <= 0:
        raise ValueError("pod_capacity_per_window must be positive")
    raw_pods = math.ceil(max(0.0, float(predicted_requests)) / pod_capacity_per_window)
    return min(max_pods, max(min_pods, raw_pods))


def create_bilstm_forecaster(input_size=1, hidden_size=64, dropout=0.1):
    _, nn, _, _ = require_torch()

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
            output, _ = self.lstm(x)
            last_step = output[:, -1, :]
            return self.head(last_step).squeeze(-1)

    return BiLSTMForecaster()


def prepare_dataset(train_csv, val_csv, test_csv, window_size):
    train_df, train_values = read_workload_values(train_csv)
    val_df, val_values = read_workload_values(val_csv)
    test_df, test_values = read_workload_values(test_csv)

    combined_values = np.concatenate([train_values, val_values, test_values])
    x, y, target_indexes = build_supervised_samples(combined_values, window_size)
    split_indexes = split_supervised_by_target(
        target_indexes=target_indexes,
        train_size=len(train_values),
        val_size=len(val_values),
    )
    stats = NormalizationStats.from_values(train_values)
    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_values": train_values,
        "val_values": val_values,
        "test_values": test_values,
        "combined_values": combined_values,
        "x": x,
        "y": y,
        "target_indexes": target_indexes,
        "split_indexes": split_indexes,
        "normalization": stats,
    }


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch, _, _, _ = require_torch()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except RuntimeError:
        pass


def make_loader(torch, DataLoader, TensorDataset, x, y, batch_size, shuffle):
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(
    x_train,
    y_train,
    x_val,
    y_val,
    hidden_size=64,
    dropout=0.1,
    epochs=100,
    batch_size=64,
    learning_rate=0.001,
    patience=10,
    seed=42,
):
    torch, nn, DataLoader, TensorDataset = require_torch()
    set_random_seed(seed)
    model = create_bilstm_forecaster(hidden_size=hidden_size, dropout=dropout)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = make_loader(
        torch,
        DataLoader,
        TensorDataset,
        x_train,
        y_train,
        batch_size=batch_size,
        shuffle=True
    )
    val_x = torch.tensor(x_val, dtype=torch.float32)
    val_y = torch.tensor(y_val, dtype=torch.float32)

    best_state = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            prediction = model(batch_x)
            loss = loss_fn(prediction, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_prediction = model(val_x)
            val_loss = float(loss_fn(val_prediction, val_y).item())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })
        print(
            f"[BiLSTM] epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"[BiLSTM] early stopping after epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict_scaled(model, x):
    torch, _, _, _ = require_torch()
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(x, dtype=torch.float32)
        return model(tensor).detach().cpu().numpy()


def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    mae = float(np.mean(np.abs(error))) if actual.size else 0.0
    rmse = float(np.sqrt(np.mean(error ** 2))) if actual.size else 0.0
    return {"mae": mae, "rmse": rmse}


def pod_metrics(actual_requests, predicted_requests, capacity, min_pods, max_pods):
    actual_pods = [
        requests_to_pods(value, capacity, min_pods, max_pods)
        for value in actual_requests
    ]
    predicted_pods = [
        requests_to_pods(value, capacity, min_pods, max_pods)
        for value in predicted_requests
    ]
    if not actual_pods:
        return {
            "pod_exact_match_rate": 0.0,
            "under_provisioned_windows": 0,
            "over_provisioned_windows": 0,
            "under_provisioned_pod_windows": 0,
            "over_provisioned_pod_windows": 0,
        }

    diffs = [pred - actual for pred, actual in zip(predicted_pods, actual_pods)]
    return {
        "pod_exact_match_rate": sum(diff == 0 for diff in diffs) / len(diffs),
        "under_provisioned_windows": sum(diff < 0 for diff in diffs),
        "over_provisioned_windows": sum(diff > 0 for diff in diffs),
        "under_provisioned_pod_windows": int(sum(max(0, -diff) for diff in diffs)),
        "over_provisioned_pod_windows": int(sum(max(0, diff) for diff in diffs)),
    }


def build_prediction_frame(
    target_indexes,
    actual,
    predicted,
    split_name,
    capacity,
    min_pods,
    max_pods,
):
    rows = []
    for target_index, actual_value, predicted_value in zip(
        target_indexes, actual, predicted
    ):
        rows.append({
            "split": split_name,
            "target_index": int(target_index),
            "actual_requests_per_20s": float(actual_value),
            "predicted_requests_per_20s": float(predicted_value),
            "actual_pods": requests_to_pods(
                actual_value, capacity, min_pods, max_pods
            ),
            "predicted_pods": requests_to_pods(
                predicted_value, capacity, min_pods, max_pods
            ),
        })
    return rows

def plot_actual_vs_predicted(prediction_df, output_dir):
    """
    Plot actual vs predicted requests for TEST set only.
    """

    output_dir = Path(output_dir)

    test_df = prediction_df[
        prediction_df["split"] == "test"
    ].copy()

    if test_df.empty:
        print("No test predictions found.")
        return

    plt.figure(figsize=(14, 6))

    plt.plot(
        test_df["target_index"],
        test_df["actual_requests_per_20s"],
        label="Actual requests"
    )

    plt.plot(
        test_df["target_index"],
        test_df["predicted_requests_per_20s"],
        label="Predicted requests"
    )

    plt.xlabel("Time window index")
    plt.ylabel("Requests per 20s")
    plt.title("Bi-LSTM Actual vs Predicted Requests (Test Set)")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = output_dir / "bilstm_actual_vs_predicted_test.png"

    plt.savefig(output_path, dpi=300)

    plt.close()

    print(f"  plot: {output_path}")

def run_grid_search(args):
    """
    Run grid search for Bi-LSTM hyperparameters using validation RMSE.
    The test set is NOT used during grid search.
    """

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==============================
    # Grid Search Space
    # ==============================
    window_size_list = [5, 10, 15, 20, 25]
    hidden_size_list = [32, 64]
    dropout_list = [0.1]
    learning_rate_list = [0.005]
    batch_size_list = [64]


    results = []
    best_result = None
    best_model = None
    best_dataset = None
    best_history = None

    total_runs = (
        len(window_size_list)
        * len(hidden_size_list)
        * len(dropout_list)
        * len(learning_rate_list)
        * len(batch_size_list)
    )

    run_id = 0

    for window_size, hidden_size, dropout, learning_rate, batch_size in product(
        window_size_list,
        hidden_size_list,
        dropout_list,
        learning_rate_list,
        batch_size_list,
    ):
        run_id += 1

        print("\n" + "=" * 80)
        print(f"[Grid Search] Run {run_id}/{total_runs}")
        print(
            f"window_size={window_size}, "
            f"hidden_size={hidden_size}, "
            f"dropout={dropout}, "
            f"learning_rate={learning_rate}, "
            f"batch_size={batch_size}"
        )
        print("=" * 80)

        try:
            dataset = prepare_dataset(
                train_csv=args.train_csv,
                val_csv=args.val_csv,
                test_csv=args.test_csv,
                window_size=window_size,
            )

            stats = dataset["normalization"]
            x_norm = stats.normalize(dataset["x"]).astype(np.float32)
            y_norm = stats.normalize(dataset["y"]).astype(np.float32)
            splits = dataset["split_indexes"]

            model, history = train_model(
                x_train=x_norm[splits["train"]],
                y_train=y_norm[splits["train"]],
                x_val=x_norm[splits["val"]],
                y_val=y_norm[splits["val"]],
                hidden_size=hidden_size,
                dropout=dropout,
                epochs=args.epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                patience=args.patience,
                seed=args.seed,
            )

            # ==============================
            # Evaluate on validation set
            # ==============================
            val_indexes = splits["val"]
            val_pred_norm = predict_scaled(model, x_norm[val_indexes])
            val_pred = stats.denormalize(val_pred_norm)
            val_actual = dataset["y"][val_indexes]

            val_metrics = regression_metrics(val_actual, val_pred)

            result = {
                "run_id": run_id,
                "window_size": window_size,
                "hidden_size": hidden_size,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "best_val_loss": min(item["val_loss"] for item in history),
                "epochs_trained": len(history),
            }

            results.append(result)

            print(
                f"[Grid Search Result] "
                f"val_mae={result['val_mae']:.4f}, "
                f"val_rmse={result['val_rmse']:.4f}, "
                f"best_val_loss={result['best_val_loss']:.6f}"
            )

            # Choose best model by validation RMSE
            if best_result is None or result["val_rmse"] < best_result["val_rmse"]:
                best_result = result
                best_model = model
                best_dataset = dataset
                best_history = history

                print("[Grid Search] New best model found.")

        except Exception as exc:
            print(f"[Grid Search] Failed for this parameter set: {exc}")

    if best_result is None:
        raise RuntimeError("Grid search failed. No valid model was trained.")

    # ==============================
    # Save grid search results
    # ==============================
    grid_results_path = output_dir / "bilstm_grid_search_results.csv"
    best_params_path = output_dir / "bilstm_best_params.json"

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="val_rmse", ascending=True)
    results_df.to_csv(grid_results_path, index=False)

    best_params_path.write_text(
        json.dumps(best_result, indent=2),
        encoding="utf-8"
    )

    print("\nBest Grid Search Result:")
    print(json.dumps(best_result, indent=2))

    print("\nSaved grid search artifacts:")
    print(f"  grid results: {grid_results_path}")
    print(f"  best params: {best_params_path}")

    return best_model, best_dataset, best_history, best_result

def train_and_save(args):
    # ==============================
    # 1. Grid Search
    # ==============================
    best_model, dataset, history, best_result = run_grid_search(args)

    stats = dataset["normalization"]
    x_norm = stats.normalize(dataset["x"]).astype(np.float32)
    y_norm = stats.normalize(dataset["y"]).astype(np.float32)
    splits = dataset["split_indexes"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==============================
    # 2. Evaluate best model on train/val/test
    # ==============================
    prediction_rows = []
    metrics = {}

    for split_name, indexes in splits.items():
        predicted_norm = predict_scaled(best_model, x_norm[indexes])
        predicted = stats.denormalize(predicted_norm)
        actual = dataset["y"][indexes]

        metrics[split_name] = {
            **regression_metrics(actual, predicted),
            **pod_metrics(
                actual,
                predicted,
                args.pod_capacity_per_window,
                args.min_pods,
                args.max_pods,
            ),
        }

        prediction_rows.extend(
            build_prediction_frame(
                target_indexes=dataset["target_indexes"][indexes],
                actual=actual,
                predicted=predicted,
                split_name=split_name,
                capacity=args.pod_capacity_per_window,
                min_pods=args.min_pods,
                max_pods=args.max_pods,
            )
        )

    # ==============================
    # 3. Save best model
    # ==============================
    torch, _, _, _ = require_torch()

    model_path = output_dir / DEFAULT_MODEL_FILENAME

    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "model_config": {
                "input_size": 1,
                "hidden_size": best_result["hidden_size"],
                "dropout": best_result["dropout"],
                "window_size": best_result["window_size"],
            },
            "autoscaler_config": {
                "normalization": asdict(stats),
                "pod_capacity_per_window": args.pod_capacity_per_window,
                "min_pods": args.min_pods,
                "max_pods": args.max_pods,
                "train_csv": str(args.train_csv),
                "val_csv": str(args.val_csv),
                "test_csv": str(args.test_csv),
            },
            "best_grid_search_result": best_result,
        },
        model_path,
    )

    # ==============================
    # 4. Save predictions, metrics, history
    # ==============================
    predictions_path = output_dir / "bilstm_predictions.csv"
    metrics_path = output_dir / "bilstm_metrics.json"
    history_path = output_dir / "training_history.json"

    prediction_df = pd.DataFrame(prediction_rows)
    prediction_df.to_csv(predictions_path, index=False)

    plot_actual_vs_predicted(prediction_df, output_dir)

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print("\nSaved Bi-LSTM training artifacts:")
    print(f"  model: {model_path}")
    print(f"  predictions: {predictions_path}")
    print(f"  metrics: {metrics_path}")
    print(f"  history: {history_path}")

    print("\nFinal Metrics:")
    print(json.dumps(metrics, indent=2))

    return {
        "model": str(model_path),
        "predictions": str(predictions_path),
        "metrics": str(metrics_path),
        "history": str(history_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Bi-LSTM predictive autoscaler baseline."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--val-csv", default=None)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pod-capacity-per-window", type=float, default=13.75)
    parser.add_argument("--min-pods", type=int, default=1)
    parser.add_argument("--max-pods", type=int, default=6)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    args.train_csv = Path(args.train_csv) if args.train_csv else data_dir / "train_workload.csv"
    args.val_csv = Path(args.val_csv) if args.val_csv else data_dir / "val_workload.csv"
    args.test_csv = Path(args.test_csv) if args.test_csv else data_dir / "test_workload.csv"
    return args


def main():
    args = parse_args()
    if args.max_pods < args.min_pods:
        raise ValueError("max_pods must be greater than or equal to min_pods")
    if args.pod_capacity_per_window <= 0:
        raise ValueError("pod_capacity_per_window must be positive")
    train_and_save(args)


if __name__ == "__main__":
    main()
