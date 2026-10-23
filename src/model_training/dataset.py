"""Request-demand dataset construction.

Convert a workload time series into supervised learning samples:
X = features from the preceding lookback windows
y = demand in the next horizon windows

Normalization equations:
X_norm = (X - mean_train) / std_train
y_norm = (y - target_mean_train) / target_std_train
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class QPSDataset(Dataset):
    """Construct sliding-window samples for demand forecasting.

    Accept a DataFrame or a one-dimensional series. Compute feature and target
    statistics on training data and reuse them for validation and test normalization.
    """

    def __init__(
        self,
        qps_series,
        lookback=100,
        horizon=2,
        features=None,
        normalize=True,
        stats=None,
        target_normalize=True,
    ):
        """Read the workload, normalize features and targets, and count samples."""
        if horizon < 1:
            raise ValueError("horizon must be >= 1.")

        self.lookback = lookback
        self.horizon = horizon
        self.normalize = normalize
        self.target_normalize = target_normalize

        if isinstance(qps_series, pd.DataFrame):
            if features is None:
                features = ["qps"]
            if "qps" not in qps_series.columns:
                raise ValueError(f"DataFrame must contain a qps target column. Available columns: {list(qps_series.columns)}")
            missing = [c for c in features if c not in qps_series.columns]
            if missing:
                raise ValueError(f"Input is missing feature columns: {missing}. Available columns: {list(qps_series.columns)}")
            data = qps_series[features].values.astype(np.float32)
            qps_raw = qps_series["qps"].values.astype(np.float32)
        else:
            data = np.asarray(qps_series, dtype=np.float32).reshape(-1, 1)
            qps_raw = np.asarray(qps_series, dtype=np.float32)
            if features is None:
                features = ["qps"]

        self.features = list(features)
        self.feature_dim = data.shape[1]
        self.qps_raw = qps_raw.astype(np.float32)

        if normalize:
            if stats is None:
                self.mean = data.mean(axis=0).astype(np.float32)
                self.std = (data.std(axis=0) + 1e-6).astype(np.float32)
            else:
                self.mean = np.asarray(stats["mean"], dtype=np.float32)
                self.std = np.asarray(stats["std"], dtype=np.float32)
            data = (data - self.mean) / self.std
        else:
            self.mean = np.zeros(self.feature_dim, dtype=np.float32)
            self.std = np.ones(self.feature_dim, dtype=np.float32)

        if target_normalize:
            if stats is None or "target_mean" not in stats or "target_std" not in stats:
                self.target_mean = np.float32(qps_raw.mean())
                self.target_std = np.float32(qps_raw.std() + 1e-6)
            else:
                self.target_mean = np.float32(stats["target_mean"])
                self.target_std = np.float32(stats["target_std"])
        else:
            self.target_mean = np.float32(0.0)
            self.target_std = np.float32(1.0)

        self.data = data.astype(np.float32)
        self.qps_norm = ((self.qps_raw - self.target_mean) / self.target_std).astype(np.float32)

        self.n_samples = len(self.data) - self.lookback - self.horizon + 1
        if self.n_samples <= 0:
            raise ValueError(
                f"Sequence too short: need at least {self.lookback + self.horizon} points, got {len(self.data)}."
            )

    def __len__(self):
        """Return the number of available sliding-window samples."""
        return self.n_samples

    def __getitem__(self, idx):
        """Return history X and future targets y for sample idx."""
        x = self.data[idx: idx + self.lookback]
        if self.target_normalize:
            y = self.qps_norm[idx + self.lookback: idx + self.lookback + self.horizon]
        else:
            y = self.qps_raw[idx + self.lookback: idx + self.lookback + self.horizon]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

    def get_stats(self):
        """Export training statistics for validation, testing, and inference."""
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "feature_dim": int(self.feature_dim),
            "features": self.features,
            "target_mean": float(self.target_mean),
            "target_std": float(self.target_std),
            "target_normalize": bool(self.target_normalize),
        }


def split_dataset(
    qps_series,
    train_ratio=0.7,
    val_ratio=0.15,
    lookback=100,
    horizon=2,
    features=None,
    batch_size=64,
    target_normalize=True,
):
    """Split one workload series chronologically and construct DataLoaders.

    Split boundaries:
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    """
    n = len(qps_series)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    if isinstance(qps_series, pd.DataFrame):
        train_data = qps_series.iloc[:train_end].copy()
        val_data = qps_series.iloc[train_end:val_end].copy()
        test_data = qps_series.iloc[val_end:].copy()
    else:
        train_data = qps_series[:train_end]
        val_data = qps_series[train_end:val_end]
        test_data = qps_series[val_end:]

    train_set = QPSDataset(
        train_data,
        lookback=lookback,
        horizon=horizon,
        features=features,
        normalize=True,
        stats=None,
        target_normalize=target_normalize,
    )
    train_stats = train_set.get_stats()

    val_set = QPSDataset(
        val_data,
        lookback=lookback,
        horizon=horizon,
        features=features,
        normalize=True,
        stats=train_stats,
        target_normalize=target_normalize,
    )
    test_set = QPSDataset(
        test_data,
        lookback=lookback,
        horizon=horizon,
        features=features,
        normalize=True,
        stats=train_stats,
        target_normalize=target_normalize,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print("Dataset split:")
    print(f"  Train: {len(train_set)} samples")
    print(f"  Val:   {len(val_set)} samples")
    print(f"  Test:  {len(test_set)} samples")
    print(f"  Feature dim: {train_stats['feature_dim']}")
    print(f"  Lookback: {lookback}")
    print(f"  Horizon:  {horizon}")
    print(f"  target_mean={train_stats['target_mean']:.4f}, target_std={train_stats['target_std']:.4f}")

    return train_loader, val_loader, test_loader, train_stats
