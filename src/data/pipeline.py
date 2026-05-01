"""Data preprocessing pipeline for survival data."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset


class SurvivalDataset(Dataset):
    """PyTorch Dataset for survival data with precomputed risk set masks."""

    def __init__(
        self,
        time: torch.Tensor,
        event: torch.Tensor,
        covariates: torch.Tensor,
    ):
        """
        Args:
            time: normalized observed times, shape [n]
            event: event indicators (0/1), shape [n]
            covariates: standardized covariate matrix, shape [n, p]
        """
        # Sort by time (required for risk set computation)
        order = torch.argsort(time)
        self.time = time[order]
        self.event = event[order]
        self.covariates = covariates[order]
        self.n = len(time)
        self.p = covariates.shape[1]

        # Precompute risk set masks: risk_mask[i, j] = 1 if j is in R(Y_i)
        # R(Y_i) = {j : Y_j >= Y_i}
        t = self.time.unsqueeze(1)  # [n, 1]
        t_col = self.time.unsqueeze(0)  # [1, n]
        self.risk_mask = (t_col >= t).float()  # [n, n]

        self.n_events = int(self.event.sum().item())

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx):
        return {
            "time": self.time[idx],
            "event": self.event[idx],
            "covariates": self.covariates[idx],
        }

    def get_collocation_points(
        self,
        n_points: int,
        method: str = "uniform",
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate collocation points (t, x) for the ODE residual loss.

        Args:
            n_points: number of collocation points N
            method: 'uniform' samples t in [0, t_max]; 'observed_times' samples
                    from observed event times; 'random' draws uniformly
            seed: for reproducibility

        Returns:
            t_col: shape [N, 1]
            x_col: shape [N, p]  (random draw from observed covariates)
        """
        rng = np.random.default_rng(seed)
        t_max = self.time.max().item()

        if method == "uniform":
            t_col = torch.linspace(1e-6, t_max, n_points).unsqueeze(1)
        elif method == "observed_times":
            idx = rng.choice(self.n, size=n_points, replace=True)
            t_col = self.time[idx].unsqueeze(1)
        elif method == "random":
            t_vals = rng.uniform(1e-6, t_max, n_points)
            t_col = torch.tensor(t_vals, dtype=torch.float32).unsqueeze(1)
        else:
            raise ValueError(f"Unknown collocation method: {method}")

        # Pair each t with a randomly selected observed covariate vector
        idx = rng.choice(self.n, size=n_points, replace=True)
        x_col = self.covariates[idx]

        return t_col, x_col


class SurvivalDataPipeline:
    """Fits scalers on training data and produces SurvivalDataset objects."""

    def __init__(self):
        self._time_scaler = MinMaxScaler(feature_range=(0, 1))
        self._cov_scaler = StandardScaler()
        self._fitted = False
        self._covariate_cols: Optional[list] = None

    def fit_transform(self, df: pd.DataFrame) -> SurvivalDataset:
        """Fit scalers and transform training data."""
        self._covariate_cols = [c for c in df.columns if c not in ("time", "event")]
        time = df["time"].values.reshape(-1, 1)
        covariates = df[self._covariate_cols].values

        time_norm = self._time_scaler.fit_transform(time).ravel()
        cov_norm = self._cov_scaler.fit_transform(covariates)
        self._fitted = True

        return SurvivalDataset(
            time=torch.tensor(time_norm, dtype=torch.float32),
            event=torch.tensor(df["event"].values, dtype=torch.float32),
            covariates=torch.tensor(cov_norm, dtype=torch.float32),
        )

    def transform(self, df: pd.DataFrame) -> SurvivalDataset:
        """Transform new data using already-fit scalers."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform before transform.")
        time = df["time"].values.reshape(-1, 1)
        covariates = df[self._covariate_cols].values

        time_norm = self._time_scaler.transform(time).ravel()
        cov_norm = self._cov_scaler.transform(covariates)

        return SurvivalDataset(
            time=torch.tensor(time_norm, dtype=torch.float32),
            event=torch.tensor(df["event"].values, dtype=torch.float32),
            covariates=torch.tensor(cov_norm, dtype=torch.float32),
        )

    def inverse_transform_time(self, t_norm: np.ndarray) -> np.ndarray:
        """Convert normalized time back to original scale."""
        return self._time_scaler.inverse_transform(
            np.asarray(t_norm).reshape(-1, 1)
        ).ravel()

    def get_scalers(self):
        return self._time_scaler, self._cov_scaler
