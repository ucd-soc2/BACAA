import math
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class QuantileQPSTransformer(nn.Module):
    def __init__(
        self,
        feature_dim: int = 1,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        lookback: int = 400,
        horizon: int = 2,
        target_feature_index: int = 0,
        residual_q50: bool = True,
        level_scale_alpha: float = 0.25,
        level_scale_min: float = 0.3,
        level_scale_max: float = 2.0,
    ):
        super().__init__()
        if horizon < 1:
            raise ValueError("horizon must be >= 1.")
        if target_feature_index < 0 or target_feature_index >= feature_dim:
            raise ValueError(
                f"target_feature_index={target_feature_index} is outside feature_dim={feature_dim}."
            )

        self.feature_dim = feature_dim
        self.lookback = lookback
        self.horizon = horizon
        self.d_model = d_model
        self.target_feature_index = target_feature_index
        self.residual_q50 = residual_q50
        self.level_scale_alpha = float(level_scale_alpha)
        self.level_scale_min = float(level_scale_min)
        self.level_scale_max = float(level_scale_max)

        self.embedding = nn.Linear(feature_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=max(lookback, 1000))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon * 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x)
        h = self.pos_encoding(h)
        h = self.transformer(h)
        h_last = h[:, -1, :]

        out = self.head(h_last).view(-1, self.horizon, 3)
        center_raw = out[:, :, 0]

        if self.residual_q50:
            last_target_norm = x[:, -1, self.target_feature_index].unsqueeze(1)
            q50 = last_target_norm + center_raw
        else:
            q50 = center_raw

        level_scale = torch.clamp(
            torch.exp(self.level_scale_alpha * q50),
            min=self.level_scale_min,
            max=self.level_scale_max,
        )
        lower_dev = F.softplus(out[:, :, 1]) * level_scale + 1e-6
        upper_dev = F.softplus(out[:, :, 2]) * level_scale + 1e-6
        q10 = q50 - lower_dev
        q90 = q50 + upper_dev

        return torch.stack([q10, q50, q90], dim=-1)


def quantile_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    taus: Sequence[float] = (0.1, 0.5, 0.9),
    weights: Sequence[float] = (0.5, 1.4, 0.8),
    horizon_weights: Optional[Sequence[float]] = None,
    peak_under_weight: float = 0.45,
    valley_over_weight: float = 0.35,
    low_width_weight: float = 0.05,
    peak_q90_under_weight: float = 0.0,
    valley_q90_over_weight: float = 0.0,
    peak_gate_threshold: float = 0.4,
    valley_gate_threshold: float = -0.2,
    gate_slope: float = 6.0,
    return_components: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(1)
    if y_pred.dim() != 3 or y_pred.shape[-1] != 3:
        raise ValueError(f"y_pred must have shape (batch, horizon, 3), got {tuple(y_pred.shape)}")
    if len(weights) != len(taus):
        raise ValueError(f"weights length must equal taus length. weights={weights}, taus={taus}")

    horizon = y_true.shape[1]
    if horizon_weights is None:
        h_weights = torch.ones(1, horizon, dtype=y_true.dtype, device=y_true.device)
    else:
        if len(horizon_weights) != horizon:
            raise ValueError(
                f"horizon_weights length must equal horizon. "
                f"horizon={horizon}, horizon_weights length={len(horizon_weights)}"
            )
        h_weights = torch.tensor(horizon_weights, dtype=y_true.dtype, device=y_true.device).view(1, horizon)

    quantile_loss_value = torch.zeros((), dtype=y_true.dtype, device=y_true.device)
    for i, (tau, weight) in enumerate(zip(taus, weights)):
        diff = y_true - y_pred[:, :, i]
        loss = torch.maximum(tau * diff, (tau - 1.0) * diff)
        quantile_loss_value = quantile_loss_value + float(weight) * (loss * h_weights).mean()

    q10 = y_pred[:, :, 0]
    q50 = y_pred[:, :, 1]
    q90 = y_pred[:, :, 2]

    peak_gate = torch.sigmoid(float(gate_slope) * (y_true - float(peak_gate_threshold)))
    valley_gate = torch.sigmoid(float(gate_slope) * (float(valley_gate_threshold) - y_true))

    under_error = F.relu(y_true - q50)
    peak_under_loss = (under_error * h_weights * peak_gate).mean()

    over_error = F.relu(q50 - y_true)
    valley_over_loss = (over_error * h_weights * valley_gate).mean()

    q90_under_error = F.relu(y_true - q90)
    peak_q90_under_loss = (q90_under_error * h_weights * peak_gate).mean()

    q90_over_error = F.relu(q90 - y_true)
    valley_q90_over_loss = (q90_over_error * h_weights * valley_gate).mean()

    width = q90 - q10
    low_width_loss = (width * h_weights * valley_gate).mean()

    total = (
        quantile_loss_value
        + float(peak_under_weight) * peak_under_loss
        + float(valley_over_weight) * valley_over_loss
        + float(low_width_weight) * low_width_loss
        + float(peak_q90_under_weight) * peak_q90_under_loss
        + float(valley_q90_over_weight) * valley_q90_over_loss
    )

    if return_components:
        components = {
            "quantile_loss": quantile_loss_value.detach(),
            "peak_gate_mean": peak_gate.mean().detach(),
            "peak_gate_min": peak_gate.min().detach(),
            "peak_gate_max": peak_gate.max().detach(),
            "valley_gate_mean": valley_gate.mean().detach(),
            "valley_gate_min": valley_gate.min().detach(),
            "valley_gate_max": valley_gate.max().detach(),
            "peak_under_loss": peak_under_loss.detach(),
            "valley_over_loss": valley_over_loss.detach(),
            "peak_q90_under_loss": peak_q90_under_loss.detach(),
            "valley_q90_over_loss": valley_q90_over_loss.detach(),
            "low_width_loss": low_width_loss.detach(),
            "weighted_peak_under_loss": (float(peak_under_weight) * peak_under_loss).detach(),
            "weighted_valley_over_loss": (float(valley_over_weight) * valley_over_loss).detach(),
            "weighted_low_width_loss": (float(low_width_weight) * low_width_loss).detach(),
            "weighted_peak_q90_under_loss": (float(peak_q90_under_weight) * peak_q90_under_loss).detach(),
            "weighted_valley_q90_over_loss": (float(valley_q90_over_weight) * valley_q90_over_loss).detach(),
        }
        return total, components

    return total


def _ensure_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a


def compute_metrics(y_true, y_pred):
    y_true = _ensure_2d(np.asarray(y_true))
    y_pred = np.asarray(y_pred)

    q10 = y_pred[..., 0]
    q50 = y_pred[..., 1]
    q90 = y_pred[..., 2]

    mae = np.mean(np.abs(y_true - q50))
    rmse = np.sqrt(np.mean((y_true - q50) ** 2))
    mape = np.mean(np.abs(y_true - q50) / (np.abs(y_true) + 1e-6)) * 100

    inside = (y_true >= q10) & (y_true <= q90)
    picp = inside.mean() * 100
    mpiw = np.mean(q90 - q10)
    cov_q10 = np.mean(y_true <= q10) * 100
    cov_q90 = np.mean(y_true <= q90) * 100

    pinball_q10 = np.mean(np.maximum(0.1 * (y_true - q10), -0.9 * (y_true - q10)))
    pinball_q50 = np.mean(np.maximum(0.5 * (y_true - q50), -0.5 * (y_true - q50)))
    pinball_q90 = np.mean(np.maximum(0.9 * (y_true - q90), -0.1 * (y_true - q90)))

    return {
        "MAE_q50": float(mae),
        "RMSE_q50": float(rmse),
        "MAPE_q50": float(mape),
        "PICP_80": float(picp),
        "MPIW": float(mpiw),
        "Coverage_q10": float(cov_q10),
        "Coverage_q90": float(cov_q90),
        "Pinball_q10": float(pinball_q10),
        "Pinball_q50": float(pinball_q50),
        "Pinball_q90": float(pinball_q90),
    }


class ConfidenceCalculator:
    def __init__(
        self,
        epsilon: float = 1e-6,
        qps_floor: float = 20.0,
        ref_percentile: float = 90.0,
        temperature: float = 1.0,
    ):
        self.epsilon = epsilon
        self.qps_floor = float(qps_floor)
        self.ref_percentile = float(ref_percentile)
        self.temperature = float(temperature)
        self.U_ref = None

    def _relative_width(self, q10, q50, q90):
        denom = np.abs(q50) + self.qps_floor + self.epsilon
        return (q90 - q10) / denom

    def calibrate(self, val_predictions, verbose: bool = True):
        val_predictions = np.asarray(val_predictions)
        q10 = val_predictions[..., 0]
        q50 = val_predictions[..., 1]
        q90 = val_predictions[..., 2]
        U = self._relative_width(q10, q50, q90)
        self.U_ref = float(np.percentile(U.reshape(-1), self.ref_percentile))
        self.U_ref = max(self.U_ref, self.epsilon)
        if verbose:
            print(
                f"[ConfidenceCalculator] U_ref calibrated = {self.U_ref:.6f}, "
                f"qps_floor={self.qps_floor:.2f}, "
                f"ref_percentile={self.ref_percentile:.1f}, "
                f"temperature={self.temperature:.2f}"
            )

    def compute(self, q10, q50, q90):
        if self.U_ref is None:
            raise RuntimeError("Call calibrate() before compute().")
        q10 = np.asarray(q10)
        q50 = np.asarray(q50)
        q90 = np.asarray(q90)
        U = self._relative_width(q10, q50, q90)
        C = np.exp(-self.temperature * U / (self.U_ref + self.epsilon))
        return np.clip(C, 0.0, 1.0)
