"""
Train Quantile Transformer
==========================

Training behavior:
1. Predict the next two windows by default.
2. Normalize targets for training and denormalize them for evaluation and plotting.
3. Select best_model by MAE_q50 by default; total quantile loss is an alternative.
4. Use the fitted confidence calculator when plotting planning values.
"""

import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model import QuantileQPSTransformer, quantile_loss, compute_metrics, ConfidenceCalculator
from dataset import QPSDataset, split_dataset

try:
    from plot import plot_val_prediction_interval
    HAS_PLOT = True
    PLOT_ERROR = None
except Exception as e:
    HAS_PLOT = False
    PLOT_ERROR = e


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_workload(path):
    print(f"Loading workload from {path}...")
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        if "qps" not in df.columns:
            for col in ["QPS", "request_rate", "rate", "value", "requests_per_20s", "requests", "request_count"]:
                if col in df.columns:
                    df = df.rename(columns={col: "qps"})
                    print(f"Renamed column {col} to qps")
                    break
        if "qps" not in df.columns:
            raise ValueError(f"{path} has no qps column. Available columns: {list(df.columns)}")
        return df
    if path.endswith(".npy"):
        return np.load(path)
    raise ValueError(f"Unsupported file format: {path}")


def build_dataset_with_optional_stats(data, args, stats=None):
    return QPSDataset(
        data,
        lookback=args.lookback,
        horizon=args.horizon,
        features=args.features,
        normalize=True,
        stats=stats,
        target_normalize=True,
    )


def build_loaders(args):
    if args.horizon < 1:
        raise ValueError("--horizon must be >= 1.")

    if args.train_workload and args.val_workload and args.test_workload:
        print("Using separate train / val / test workload files.")
        train_data = load_workload(args.train_workload)
        val_data = load_workload(args.val_workload)
        test_data = load_workload(args.test_workload)

        train_set = build_dataset_with_optional_stats(train_data, args, stats=None)
        train_stats = train_set.get_stats()
        val_set = build_dataset_with_optional_stats(val_data, args, stats=train_stats)
        test_set = build_dataset_with_optional_stats(test_data, args, stats=train_stats)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

        print("Dataset size:")
        print(f"  Train: {len(train_set)}")
        print(f"  Val:   {len(val_set)}")
        print(f"  Test:  {len(test_set)}")
        print(f"  Feature dim: {train_stats['feature_dim']}")
        print(f"  target_mean={train_stats['target_mean']:.4f}, target_std={train_stats['target_std']:.4f}")
        return train_loader, val_loader, test_loader, train_stats

    if args.workload:
        data = load_workload(args.workload)
        return split_dataset(
            data,
            train_ratio=0.7,
            val_ratio=0.15,
            lookback=args.lookback,
            horizon=args.horizon,
            features=args.features,
            batch_size=args.batch_size,
            target_normalize=True,
        )

    raise ValueError("Provide --workload or --train_workload/--val_workload/--test_workload.")


def denormalize_y(y_norm, train_stats):
    y_norm = np.asarray(y_norm)
    return y_norm * float(train_stats["target_std"]) + float(train_stats["target_mean"])


def denormalize_pred(pred_norm, train_stats):
    pred_norm = np.asarray(pred_norm)
    return pred_norm * float(train_stats["target_std"]) + float(train_stats["target_mean"])


def clip_nonnegative_pred(pred_raw):
    pred_raw = pred_raw.copy()
    pred_raw[..., 0] = np.maximum(pred_raw[..., 0], 0.0)
    pred_raw[..., 1] = np.maximum(pred_raw[..., 1], 0.0)
    pred_raw[..., 2] = np.maximum(pred_raw[..., 2], 0.0)
    # Enforce quantile ordering after clipping.
    pred_raw[..., 0] = np.minimum(pred_raw[..., 0], pred_raw[..., 1])
    pred_raw[..., 2] = np.maximum(pred_raw[..., 2], pred_raw[..., 1])
    return pred_raw


def train_one_epoch(model, loader, optimizer, device, horizon_weights=None):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = quantile_loss(y, pred, horizon_weights=horizon_weights)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, train_stats, horizon_weights=None):
    model.eval()
    all_preds_norm = []
    all_trues_norm = []
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = quantile_loss(y, pred, horizon_weights=horizon_weights)

        total_loss += float(loss.item())
        n_batches += 1
        all_preds_norm.append(pred.cpu().numpy())
        all_trues_norm.append(y.cpu().numpy())

    preds_norm = np.concatenate(all_preds_norm, axis=0)
    trues_norm = np.concatenate(all_trues_norm, axis=0)

    preds_raw = clip_nonnegative_pred(denormalize_pred(preds_norm, train_stats))
    trues_raw = denormalize_y(trues_norm, train_stats)

    metrics = compute_metrics(trues_raw, preds_raw)
    metrics_h2 = compute_metrics(trues_raw[:, 1:2], preds_raw[:, 1:2, :])
    for k, v in metrics_h2.items(): metrics[f"h2_{k}"] = v
    metrics["loss_norm"] = total_loss / max(n_batches, 1)
    return metrics, preds_raw, trues_raw, preds_norm, trues_norm


def get_score(metrics, mode):
    if mode == "mae":
        return metrics["h2_MAE_q50"]
    if mode == "loss":
        return metrics["loss_norm"]
    raise ValueError("--save_metric must be mae or loss")


def get_horizon_weights(args):
    if args.horizon_weights is None:
        return None
    if len(args.horizon_weights) != args.horizon:
        raise ValueError(
            f"--horizon_weights length must equal --horizon. "
            f"Got horizon={args.horizon}, weights={args.horizon_weights}"
        )
    return [float(w) for w in args.horizon_weights]


def resolve_qps_floor(args, train_stats):
    if args.qps_floor is not None:
        if args.qps_floor <= 0:
            raise ValueError("--qps_floor must be positive")
        return float(args.qps_floor)

    target_mean = float(train_stats.get("target_mean", 0.0))
    return max(target_mean, 1.0)


def auto_plot_validation(args, val_preds_raw, val_trues_raw, confidence=None):
    if not args.auto_plot:
        print("\nAutomatic plotting is disabled.")
        return
    if not HAS_PLOT:
        print("\nSkipping plots because plot.py could not be imported.")
        print(f"Error: {PLOT_ERROR}")
        return

    plot_dir = os.path.join(args.output_dir, "val_plots")
    os.makedirs(plot_dir, exist_ok=True)

    q10 = val_preds_raw[:, :, 0]
    q50 = val_preds_raw[:, :, 1]
    q90 = val_preds_raw[:, :, 2]
    y_true = val_trues_raw

    print("\n" + "=" * 60)
    print("Plotting validation forecasts...")
    print(f"Output directory: {plot_dir}")

    horizon = y_true.shape[1]
    for h in range(horizon):
        save_path = os.path.join(plot_dir, f"val_prediction_horizon_{h + 1}.png")
        plot_val_prediction_interval(
            y_true=y_true,
            q10=q10,
            q50=q50,
            q90=q90,
            horizon_idx=h,
            max_points=args.plot_max_points,
            save_path=save_path,
            c_final=confidence,
            plot_decision=True,
        )
    print("Validation forecast plots saved.")


def main(args):
    set_seed(args.seed)
    train_loader, val_loader, test_loader, train_stats = build_loaders(args)
    args.qps_floor = resolve_qps_floor(args, train_stats)
    print(f"Confidence qps_floor: {args.qps_floor:.4f}")
    horizon_weights = get_horizon_weights(args)
    if horizon_weights is None:
        print("Horizon loss weights: equal weights")
    else:
        print(f"Horizon loss weights: {horizon_weights}")

    if args.features[0] != "qps":
        raise ValueError("The residual q50 model requires qps as the first feature, e.g. --features qps or --features qps response_time")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = QuantileQPSTransformer(
        feature_dim=train_stats["feature_dim"],
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_model * 4,
        dropout=args.dropout,
        lookback=args.lookback,
        horizon=args.horizon,
        target_feature_index=0,
        residual_q50=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.output_dir, exist_ok=True)
    best_score = float("inf")
    best_val_loss_norm = float("inf")
    best_epoch = -1
    history = {"train_loss": [], "val_loss_norm": [], "val_mae": [], "val_metrics": []}

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, horizon_weights=horizon_weights)
        val_metrics, _, _, _, _ = evaluate(
            model, val_loader, device, train_stats, horizon_weights=horizon_weights
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss_norm"].append(val_metrics["loss_norm"])
        history["val_mae"].append(val_metrics["MAE_q50"])
        history["val_metrics"].append(val_metrics)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss_norm: {train_loss:.4f} | "
            f"val_loss_norm: {val_metrics['loss_norm']:.4f} | "
            f"MAE: {val_metrics['MAE_q50']:.2f} | "
            f"RMSE: {val_metrics['RMSE_q50']:.2f} | "
            f"PICP: {val_metrics['PICP_80']:.1f}%"
        )

        score = get_score(val_metrics, args.save_metric)
        if score < best_score:
            best_score = score
            best_val_loss_norm = val_metrics["loss_norm"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": vars(args),
                    "train_stats": train_stats,
                    "val_metrics": val_metrics,
                    "target_normalized": True,
                    "residual_q50": True,
                },
                os.path.join(args.output_dir, "best_model.pt"),
            )

    print("\n" + "=" * 60)
    print("Loading best model for final evaluation...")
    ckpt_path = os.path.join(args.output_dir, "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"Best epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"Best save metric ({args.save_metric}): {best_score:.4f}")

    test_metrics, test_preds_raw, test_trues_raw, test_preds_norm, test_trues_norm = evaluate(
        model, test_loader, device, train_stats, horizon_weights=horizon_weights
    )
    print("\nTest set metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    val_metrics, val_preds_raw, val_trues_raw, val_preds_norm, val_trues_norm = evaluate(
        model, val_loader, device, train_stats, horizon_weights=horizon_weights
    )
    print("\nValidation set metrics with best model:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")

    conf_calc = ConfidenceCalculator(qps_floor=args.qps_floor)
    conf_calc.calibrate(val_preds_raw)
    val_confidence = conf_calc.compute(
        val_preds_raw[:, :, 0],
        val_preds_raw[:, :, 1],
        val_preds_raw[:, :, 2],
    )

    results = {
        "config": vars(args),
        "train_stats": train_stats,
        "best_score": float(best_score),
        "best_val_loss_norm": float(best_val_loss_norm),
        "best_epoch": int(best_epoch),
        "save_metric": args.save_metric,
        "val_metrics": {k: float(v) for k, v in val_metrics.items()},
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "U_ref": float(conf_calc.U_ref),
        "qps_floor": float(conf_calc.qps_floor),
        "target_normalized": True,
        "residual_q50": True,
    }

    with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    np.savez(
        os.path.join(args.output_dir, "confidence_calc.npz"),
        U_ref=conf_calc.U_ref,
        epsilon=conf_calc.epsilon,
        qps_floor=conf_calc.qps_floor,
    )

    np.savez(
        os.path.join(args.output_dir, "history.npz"),
        train_loss=np.asarray(history["train_loss"]),
        val_loss_norm=np.asarray(history["val_loss_norm"]),
        val_mae=np.asarray(history["val_mae"]),
        test_preds=test_preds_raw,
        test_trues=test_trues_raw,
        val_preds=val_preds_raw,
        val_trues=val_trues_raw,
        test_preds_norm=test_preds_norm,
        test_trues_norm=test_trues_norm,
        val_preds_norm=val_preds_norm,
        val_trues_norm=val_trues_norm,
        val_confidence=val_confidence,
    )

    auto_plot_validation(args, val_preds_raw, val_trues_raw, confidence=val_confidence)

    print(f"\nAll results saved to: {args.output_dir}")
    print(f"Validation plots saved to: {os.path.join(args.output_dir, 'val_plots')}")
    print("Use inference.py to load the model for prediction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--workload", type=str, default=None)
    parser.add_argument("--train_workload", type=str, default=None)
    parser.add_argument("--val_workload", type=str, default=None)
    parser.add_argument("--test_workload", type=str, default=None)
    parser.add_argument("--features", type=str, nargs="+", default=["qps"])

    parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--horizon_weights", type=float, nargs="+", default=None)
    parser.add_argument("--qps_floor", type=float, default=None)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_metric", type=str, default="mae", choices=["mae", "loss"])

    parser.add_argument("--auto_plot", action="store_true", default=True)
    parser.add_argument("--plot_max_points", type=int, default=300)
    parser.add_argument("--output_dir", type=str, default="./checkpoints_h2_fixed")

    args = parser.parse_args()
    main(args)
