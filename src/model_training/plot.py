"""Validation forecast plotting.

Plot observed demand, q50 forecasts, q10-q90 intervals, and optional
confidence-aware QPS planning values together to inspect forecast behavior
during bursts and valleys.
"""

import numpy as np
import matplotlib.pyplot as plt


def _as_2d(arr, name):
    """Convert [N] arrays to [N, 1] and validate [N, horizon] inputs."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"{name} must have shape [N] or [N, horizon], got shape={arr.shape}")


def plot_val_prediction_interval(
    y_true,
    q10,
    q50,
    q90,
    horizon_idx=0,
    max_points=300,
    save_path="val_prediction_horizon_1.png",
    c_final=None,
    plot_decision=True,
):
    """Plot validation prediction intervals for one forecast horizon.

    Inputs y_true, q10, q50, and q90 use the original demand scale.
    If c_final is supplied, also plot:
    Q_decision = q50 + (1 - C) * (q90 - q50)
    """
    y_true = _as_2d(y_true, "y_true")
    q10 = _as_2d(q10, "q10")
    q50 = _as_2d(q50, "q50")
    q90 = _as_2d(q90, "q90")

    horizon = y_true.shape[1]
    if horizon_idx < 0 or horizon_idx >= horizon:
        raise ValueError(f"horizon_idx={horizon_idx} is out of range for horizon={horizon}")

    true_line = y_true[:, horizon_idx][:max_points]
    q10_line = q10[:, horizon_idx][:max_points]
    q50_line = q50[:, horizon_idx][:max_points]
    q90_line = q90[:, horizon_idx][:max_points]
    x_axis = np.arange(len(true_line))

    plt.figure(figsize=(14, 6))
    plt.plot(x_axis, true_line, label="True", linewidth=2)
    plt.plot(x_axis, q50_line, label="Pred q50", linewidth=2)

    if plot_decision and c_final is not None:
        if np.isscalar(c_final):
            c_line = np.ones_like(q50_line) * float(c_final)
        else:
            c_final = _as_2d(c_final, "c_final")
            c_line = c_final[:, horizon_idx][:max_points]
        qps_decision = q50_line + (1.0 - c_line) * (q90_line - q50_line)
        plt.plot(x_axis, qps_decision, label="QPS decision", linewidth=2, linestyle="--")

    plt.fill_between(x_axis, q10_line, q90_line, alpha=0.25, label="Prediction Interval q10-q90")
    plt.title(f"Validation Prediction vs True | Horizon {horizon_idx + 1}")
    plt.xlabel("Validation sample index")
    plt.ylabel("Target value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved to: {save_path}")


def plot_all_horizons(y_true, q10, q50, q90, max_points=300):
    """Plot validation prediction intervals for every forecast horizon."""
    y_true = _as_2d(y_true, "y_true")
    for h in range(y_true.shape[1]):
        plot_val_prediction_interval(
            y_true=y_true,
            q10=q10,
            q50=q50,
            q90=q90,
            horizon_idx=h,
            max_points=max_points,
            save_path=f"val_prediction_horizon_{h + 1}.png",
        )
