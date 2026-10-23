"""Model inference interface.

Load a trained Quantile Transformer checkpoint and provide the controller with
q10, q50, q90, confidence, and conservative planning demand estimates for the
next horizon windows.

Key equations:
X_norm = (X - mean_train) / std_train
pred_raw = pred_norm * target_std + target_mean
Q_decision = q50 + (1 - C) * (q90 - q50)
"""

import json
import os
import sys

import numpy as np
import torch

from model import QuantileQPSTransformer, ConfidenceCalculator


class QPSPredictor:
    """Load a checkpoint and expose request-demand quantile forecasts."""

    def __init__(self, checkpoint_dir, device=None):
        """Read results.json, build the model, and load best_model.pt and confidence parameters."""
        self.checkpoint_dir = checkpoint_dir

        results_path = os.path.join(checkpoint_dir, "results.json")
        with open(results_path, "r", encoding="utf-8") as f:
            self.results = json.load(f)

        config = self.results["config"]
        self.lookback = int(config["lookback"])
        self.horizon = int(config["horizon"])
        self.features = config.get("features", ["qps"])

        train_stats = self.results["train_stats"]
        self.mean = np.asarray(train_stats["mean"], dtype=np.float32)
        self.std = np.asarray(train_stats["std"], dtype=np.float32)
        self.feature_dim = int(train_stats["feature_dim"])
        self.target_mean = float(train_stats.get("target_mean", 0.0))
        self.target_std = float(train_stats.get("target_std", 1.0))
        self.target_normalized = bool(self.results.get("target_normalized", train_stats.get("target_normalize", True)))

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model = QuantileQPSTransformer(
            feature_dim=self.feature_dim,
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            d_ff=config["d_model"] * 4,
            dropout=config["dropout"],
            lookback=self.lookback,
            horizon=self.horizon,
            target_feature_index=0,
            residual_q50=bool(self.results.get("residual_q50", True)),
            level_scale_alpha=float(config.get("level_scale_alpha", 0.25)),
            level_scale_min=float(config.get("level_scale_min", 0.3)),
            level_scale_max=float(config.get("level_scale_max", 2.0)),
        ).to(device)

        ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        qps_floor = self.results.get("qps_floor", config.get("qps_floor"))
        if qps_floor is None:
            qps_floor = self.target_mean if self.target_mean > 0 else 20.0
        confidence_config = self.results.get("confidence_config", {})
        self.conf_calc = ConfidenceCalculator(
            qps_floor=float(qps_floor),
            ref_percentile=float(
                confidence_config.get(
                    "ref_percentile",
                    config.get("confidence_ref_percentile", 90.0),
                )
            ),
            temperature=float(
                confidence_config.get(
                    "temperature",
                    config.get("confidence_temperature", 1.0),
                )
            ),
        )
        self.conf_calc.U_ref = float(self.results["U_ref"])

        print(f"[QPSPredictor] Loaded from {checkpoint_dir}")
        print(f"  feature_dim={self.feature_dim}, lookback={self.lookback}, horizon={self.horizon}")
        print(f"  features={self.features}")
        print(f"  target_mean={self.target_mean:.4f}, target_std={self.target_std:.4f}")
        print(f"  U_ref={self.conf_calc.U_ref:.6f}")
        print(f"  qps_floor={self.conf_calc.qps_floor:.2f}")
        print(f"  confidence_ref_percentile={self.conf_calc.ref_percentile:.1f}")
        print(f"  confidence_temperature={self.conf_calc.temperature:.2f}")

    def _denorm_pred(self, pred_norm):
        """Convert model outputs from normalized targets to the original request-demand scale."""
        if self.target_normalized:
            return pred_norm * self.target_std + self.target_mean
        return pred_norm

    @torch.no_grad()
    def predict(self, history):
        """Predict q10, q50, q90, and confidence for the next horizon windows.

        The input history length must match the configured lookback. Clip predictions to
        nonnegative demand and enforce q10 <= q50 <= q90.
        """
        history = np.asarray(history, dtype=np.float32)
        if history.ndim == 1:
            history = history.reshape(-1, 1)

        if history.shape[0] != self.lookback:
            raise ValueError(f"Expected history length {self.lookback}, got {history.shape[0]}")
        if history.shape[1] != self.feature_dim:
            raise ValueError(f"Expected feature dimension {self.feature_dim}, got {history.shape[1]}")

        x_norm = (history - self.mean) / self.std
        x_tensor = torch.from_numpy(x_norm).float().unsqueeze(0).to(self.device)

        pred_norm = self.model(x_tensor)[0].cpu().numpy()
        pred_raw = self._denorm_pred(pred_norm)

        q10 = np.maximum(pred_raw[:, 0], 0.0)
        q50 = np.maximum(pred_raw[:, 1], 0.0)
        q90 = np.maximum(pred_raw[:, 2], 0.0)
        q10 = np.minimum(q10, q50)
        q90 = np.maximum(q90, q50)

        confidence = self.conf_calc.compute(q10, q50, q90)

        result = {
            "q10": q10,
            "q50": q50,
            "q90": q90,
            "confidence": confidence,
            "q10_next": float(q10[0]),
            "q50_next": float(q50[0]),
            "q90_next": float(q90[0]),
            "confidence_next": float(confidence[0]),
        }
        for idx in range(self.horizon):
            suffix = f"horizon_{idx + 1}"
            result[f"q10_{suffix}"] = float(q10[idx])
            result[f"q50_{suffix}"] = float(q50[idx])
            result[f"q90_{suffix}"] = float(q90[idx])
            result[f"confidence_{suffix}"] = float(confidence[idx])
        return result

    def predict_with_decision(self, history):
        """Compute a conservative QPS planning value from the forecast quantiles.

        Equation:
        Q_decision = q50 + (1 - confidence) * (q90 - q50)
        Higher confidence shifts planning toward q50; lower confidence shifts it toward q90.
        """
        result = self.predict(history)
        q50 = result["q50"]
        q90 = result["q90"]
        c_final = result["confidence"]
        qps_decision = q50 + (1.0 - c_final) * (q90 - q50)

        result["qps_decision"] = qps_decision
        result["qps_for_planning"] = qps_decision
        result["qps_decision_next"] = float(qps_decision[0])
        result["qps_for_planning_next"] = float(qps_decision[0])
        for idx in range(self.horizon):
            suffix = f"horizon_{idx + 1}"
            result[f"qps_decision_{suffix}"] = float(qps_decision[idx])
            result[f"qps_for_planning_{suffix}"] = float(qps_decision[idx])
        return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <checkpoint_dir>")
        sys.exit(1)

    predictor = QPSPredictor(sys.argv[1])
    example_low = max(0.0, predictor.target_mean - predictor.target_std)
    example_high = max(example_low + 1.0, predictor.target_mean + predictor.target_std)
    example_history = np.linspace(example_low, example_high, predictor.lookback, dtype=np.float32)
    if predictor.feature_dim > 1:
        example_history = np.stack([example_history] * predictor.feature_dim, axis=1)

    print(f"\nExample input length: {predictor.lookback}")
    result = predictor.predict_with_decision(example_history)
    print(f"\nForecast for the next {predictor.horizon} windows:")
    for idx in range(predictor.horizon):
        print(f"  Horizon {idx + 1}:")
        print(f"    q10:        {result['q10'][idx]:.2f}")
        print(f"    q50:        {result['q50'][idx]:.2f}")
        print(f"    q90:        {result['q90'][idx]:.2f}")
        print(f"    confidence: {result['confidence'][idx]:.3f}")
        print(f"    decision:   {result['qps_decision'][idx]:.2f}")
