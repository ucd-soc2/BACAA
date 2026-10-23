"""模型推理接口模块。

本模块用于加载训练好的 Quantile Transformer checkpoint，并向控制器提供
未来 horizon 个窗口的 q10、q50、q90、confidence 和 conservative planning
需求值。

重要公式：
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


TREND_FEATURE_EPSILON = 1e-6


def _rolling_mean(values, window):
    values = np.asarray(values, dtype=np.float32)
    means = np.empty_like(values, dtype=np.float32)
    for idx in range(values.shape[0]):
        start = max(0, idx - window + 1)
        means[idx] = float(values[start: idx + 1].mean())
    return means


def build_trend_feature_matrix(qps_history, features):
    qps = np.asarray(qps_history, dtype=np.float32).reshape(-1)
    if qps.size == 0:
        return np.empty((0, len(features)), dtype=np.float32)

    diff_1 = np.empty_like(qps, dtype=np.float32)
    diff_1[0] = 0.0
    diff_1[1:] = qps[1:] - qps[:-1]

    diff_2 = np.zeros_like(qps, dtype=np.float32)
    diff_2[2:] = qps[2:] - qps[:-2]

    slope_3 = np.zeros_like(qps, dtype=np.float32)
    slope_3[3:] = (qps[3:] - qps[:-3]) / 3.0

    acceleration = np.empty_like(qps, dtype=np.float32)
    acceleration[0] = 0.0
    acceleration[1:] = diff_1[1:] - diff_1[:-1]

    mean_3 = _rolling_mean(qps, 3)
    mean_12 = _rolling_mean(qps, 12)
    mean_ratio = np.nan_to_num(
        mean_3 / (mean_12 + TREND_FEATURE_EPSILON),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
    max_ratio = np.nan_to_num(
        qps / (mean_12 + TREND_FEATURE_EPSILON),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    feature_values = {
        "qps": qps,
        "diff_1": diff_1,
        "diff_2": diff_2,
        "slope_3": slope_3,
        "acceleration": acceleration,
        "mean_3": mean_3,
        "mean_12": mean_12,
        "mean_ratio": mean_ratio,
        "max_ratio": max_ratio,
    }
    missing = [name for name in features if name not in feature_values]
    if missing:
        raise ValueError(f"Unsupported generated feature(s): {missing}")
    return np.stack([feature_values[name] for name in features], axis=1).astype(np.float32)


class QPSPredictor:
    """加载 checkpoint 并提供请求量分位数预测接口。"""

    def __init__(self, checkpoint_dir, device=None):
        """读取 results.json、构建模型、加载 best_model.pt 和 confidence 参数。"""
        self.checkpoint_dir = checkpoint_dir

        results_path = os.path.join(checkpoint_dir, "results.json")
        with open(results_path, "r", encoding="utf-8") as f:
            self.results = json.load(f)

        config = self.results["config"]
        self.lookback = int(config["lookback"])
        self.horizon = int(config["horizon"])

        train_stats = self.results["train_stats"]
        self.features = config.get("features", train_stats.get("features", ["qps"]))
        self.mean = np.asarray(train_stats["mean"], dtype=np.float32)
        self.std = np.asarray(train_stats["std"], dtype=np.float32)
        self.feature_dim = int(train_stats["feature_dim"])
        self.target_mean = float(train_stats.get("target_mean", 0.0))
        self.target_std = float(train_stats.get("target_std", 1.0))
        self.target_normalized = bool(self.results.get("target_normalized", train_stats.get("target_normalize", True)))
        model_config = self.results.get("model_config", {})
        self.prediction_mode = self.results.get(
            "prediction_mode",
            config.get(
                "prediction_mode",
                model_config.get(
                    "prediction_mode",
                    "residual" if self.results.get("residual_q50", True) else "absolute",
                ),
            ),
        )

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
            prediction_mode=self.prediction_mode,
            hybrid_absolute_weight=float(
                config.get(
                    "hybrid_absolute_weight",
                    model_config.get("hybrid_absolute_weight", 0.7),
                )
            ),
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
        print(f"  prediction_mode={self.prediction_mode}")
        print(f"  target_mean={self.target_mean:.4f}, target_std={self.target_std:.4f}")
        print(f"  U_ref={self.conf_calc.U_ref:.6f}")
        print(f"  qps_floor={self.conf_calc.qps_floor:.2f}")
        print(f"  confidence_ref_percentile={self.conf_calc.ref_percentile:.1f}")
        print(f"  confidence_temperature={self.conf_calc.temperature:.2f}")

    def _denorm_pred(self, pred_norm):
        """把模型输出从归一化 target 空间还原到原始请求量尺度。"""
        if self.target_normalized:
            return pred_norm * self.target_std + self.target_mean
        return pred_norm

    def _prepare_history_features(self, history):
        history = np.asarray(history, dtype=np.float32)
        if history.ndim == 1:
            return build_trend_feature_matrix(history, self.features)
        if history.ndim == 2:
            if history.shape[1] == self.feature_dim:
                return history.astype(np.float32)
            if history.shape[1] == 1:
                return build_trend_feature_matrix(history[:, 0], self.features)
        raise ValueError(
            "History must be a 1-D qps sequence, a single-column qps matrix, "
            "or a feature matrix whose width matches the trained feature_dim."
        )

    @torch.no_grad()
    def predict(self, history):
        """对一个历史窗口序列输出未来 horizon 个窗口的 q10、q50、q90 和 confidence。

        输入 history 的长度必须等于训练配置中的 lookback。模型输出后会裁剪到非负
        请求量，并再次保证 q10 <= q50 <= q90。
        """
        history = self._prepare_history_features(history)

        if history.shape[0] != self.lookback:
            raise ValueError(f"History 长度应为 {self.lookback}，实际 {history.shape[0]}")
        if history.shape[1] != self.feature_dim:
            raise ValueError(f"特征维度应为 {self.feature_dim}，实际 {history.shape[1]}")

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
        """在分位数预测基础上计算 conservative QPS planning value。

        公式为：
        Q_decision = q50 + (1 - confidence) * (q90 - q50)
        confidence 高时更接近 q50，confidence 低时更接近 q90。
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
    print(f"\n示例输入长度: {predictor.lookback}")
    result = predictor.predict_with_decision(example_history)
    print(f"\n预测未来 {predictor.horizon} 个窗口:")
    for idx in range(predictor.horizon):
        print(f"  Horizon {idx + 1}:")
        print(f"    q10:        {result['q10'][idx]:.2f}")
        print(f"    q50:        {result['q50'][idx]:.2f}")
        print(f"    q90:        {result['q90'][idx]:.2f}")
        print(f"    confidence: {result['confidence'][idx]:.3f}")
        print(f"    decision:   {result['qps_decision'][idx]:.2f}")
