"""Cox survival data simulation via inverse transform sampling."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import optimize, integrate


class BaselineHazard(ABC):
    """Abstract base for baseline hazard functions α(t)."""

    @abstractmethod
    def hazard(self, t: np.ndarray) -> np.ndarray:
        """Baseline hazard α(t)."""

    def cumulative_hazard(self, t: np.ndarray) -> np.ndarray:
        """Cumulative baseline hazard Λ_0(t) = ∫₀ᵗ α(s)ds."""
        t = np.asarray(t, dtype=float)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)
        result = np.array([integrate.quad(self.hazard, 0, ti)[0] for ti in t])
        return result[0] if scalar else result

    def inverse_cumulative_hazard(self, u: np.ndarray) -> np.ndarray:
        """Λ_0^{-1}(u): used for inverse transform sampling."""
        u = np.atleast_1d(np.asarray(u, dtype=float))
        results = np.empty_like(u)
        for i, ui in enumerate(u):
            if ui <= 0:
                results[i] = 0.0
                continue
            # bracket: find t_max such that Λ_0(t_max) > ui
            t_max = 1.0
            while self.cumulative_hazard(t_max) < ui:
                t_max *= 2
            results[i] = optimize.brentq(
                lambda t: self.cumulative_hazard(t) - ui, 0.0, t_max, xtol=1e-8
            )
        return results


class ExponentialBaseline(BaselineHazard):
    """Constant baseline hazard α(t) = lam."""

    def __init__(self, lam: float = 0.5):
        self.lam = lam

    def hazard(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return np.full_like(t, self.lam)

    def cumulative_hazard(self, t: np.ndarray) -> np.ndarray:
        return self.lam * np.asarray(t, dtype=float)

    def inverse_cumulative_hazard(self, u: np.ndarray) -> np.ndarray:
        return np.asarray(u, dtype=float) / self.lam


class WeibullBaseline(BaselineHazard):
    """Weibull baseline hazard α(t) = k·lam·(lam·t)^{k-1}."""

    def __init__(self, k: float = 1.5, lam: float = 0.5):
        self.k = k
        self.lam = lam

    def hazard(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return self.k * self.lam * (self.lam * t) ** (self.k - 1)

    def cumulative_hazard(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return (self.lam * t) ** self.k

    def inverse_cumulative_hazard(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        return (u ** (1.0 / self.k)) / self.lam


class GompertzBaseline(BaselineHazard):
    """Gompertz baseline hazard α(t) = b·exp(a·t)."""

    def __init__(self, a: float = 0.5, b: float = 0.1):
        self.a = a
        self.b = b

    def hazard(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return self.b * np.exp(self.a * t)

    def cumulative_hazard(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return (self.b / self.a) * (np.exp(self.a * t) - 1)

    def inverse_cumulative_hazard(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        return np.log(1 + u * self.a / self.b) / self.a


class BathtubBaseline(BaselineHazard):
    """Non-monotone bathtub hazard: high early risk, valley, then late rise.

    α(t) = floor + early_amp·exp(-decay·t) + late_amp·(exp(growth·t) - 1)

    The cumulative hazard is closed form; inverse still uses a scalar root find
    on the closed-form cumulative hazard.
    """

    def __init__(
        self,
        floor: float = 0.08,
        early_amp: float = 0.9,
        decay: float = 1.6,
        late_amp: float = 0.02,
        growth: float = 0.35,
    ):
        self.floor = floor
        self.early_amp = early_amp
        self.decay = decay
        self.late_amp = late_amp
        self.growth = growth

    def hazard(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return (
            self.floor
            + self.early_amp * np.exp(-self.decay * t)
            + self.late_amp * (np.exp(self.growth * t) - 1.0)
        )

    def cumulative_hazard(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return (
            self.floor * t
            + self.early_amp * (1.0 - np.exp(-self.decay * t)) / self.decay
            + self.late_amp * (np.exp(self.growth * t) - 1.0) / self.growth
            - self.late_amp * t
        )


class PiecewiseConstantBaseline(BaselineHazard):
    """Piecewise constant hazard defined by cut points and rates."""

    def __init__(self, cutpoints: List[float], rates: List[float]):
        """
        Args:
            cutpoints: interval boundaries (not including 0), length K-1
            rates: hazard value in each interval, length K
        """
        assert len(rates) == len(cutpoints) + 1
        self.cutpoints = np.array([0.0] + list(cutpoints) + [np.inf])
        self.rates = np.array(rates)

    def hazard(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        idx = np.searchsorted(self.cutpoints[1:-1], t, side="right")
        return self.rates[idx]

    def cumulative_hazard(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)
        result = np.zeros_like(t)
        for i, ti in enumerate(t):
            idx = np.searchsorted(self.cutpoints[1:-1], ti, side="right")
            for k in range(idx):
                result[i] += self.rates[k] * (self.cutpoints[k + 1] - self.cutpoints[k])
            result[i] += self.rates[idx] * (ti - self.cutpoints[idx])
        return result[0] if scalar else result


class CoxSimulator:
    """Simulate survival data from the Cox model using inverse transform sampling.

    The hazard for subject i is λ_x(t) = α(t)·exp(x_i^T β).
    """

    def __init__(
        self,
        baseline_hazard: BaselineHazard,
        beta: Union[List[float], np.ndarray],
        n_covariates: Optional[int] = None,
        censoring_rate: float = 0.3,
        censoring_type: str = "random_exponential",
        covariate_dist: Union[str, Callable] = "normal",
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            baseline_hazard: baseline hazard function α(t)
            beta: true covariate coefficients, length p
            n_covariates: number of covariates (inferred from beta if not given)
            censoring_rate: approximate proportion of censored observations
            censoring_type: 'random_exponential' or 'administrative'
            covariate_dist: 'normal', 'uniform', or callable(n, p) -> array
            random_seed: for reproducibility
        """
        self.baseline_hazard = baseline_hazard
        self.beta = np.asarray(beta, dtype=float)
        self.p = len(self.beta) if n_covariates is None else n_covariates
        assert len(self.beta) == self.p, "beta length must match n_covariates"
        self.censoring_rate = censoring_rate
        self.censoring_type = censoring_type
        self.covariate_dist = covariate_dist
        self.rng = np.random.default_rng(random_seed)

    def _sample_covariates(self, n: int) -> np.ndarray:
        if callable(self.covariate_dist):
            return self.covariate_dist(n, self.p)
        elif self.covariate_dist == "normal":
            return self.rng.standard_normal((n, self.p))
        elif self.covariate_dist == "uniform":
            return self.rng.uniform(-1, 1, (n, self.p))
        else:
            raise ValueError(f"Unknown covariate_dist: {self.covariate_dist}")

    def _sample_event_times(self, X: np.ndarray) -> np.ndarray:
        """Sample event times via inverse transform: T = Λ_0^{-1}(-log(U)/exp(x^T β))."""
        n = X.shape[0]
        U = self.rng.uniform(0, 1, n)
        linear_predictors = X @ self.beta
        # quantile of individual cumulative hazard: -log(U) / exp(x^T β)
        quantiles = -np.log(U) / np.exp(linear_predictors)
        return self.baseline_hazard.inverse_cumulative_hazard(quantiles)

    def _sample_censoring_times(self, event_times: np.ndarray) -> np.ndarray:
        """Sample censoring times such that the expected censoring rate is met."""
        if self.censoring_type == "administrative":
            admin_time = np.quantile(event_times, 1 - self.censoring_rate)
            return np.full(len(event_times), admin_time)
        elif self.censoring_type == "random_exponential":
            if self.censoring_rate <= 0:
                return np.full(len(event_times), np.inf)
            # Choose exponential rate so that P(C < T) ≈ censoring_rate
            med_event = np.median(event_times)
            # solve: P(Exp(rate) < median) ≈ censoring_rate
            # approximate: rate ≈ -log(1 - censoring_rate) / med_event
            rate = -np.log(1 - self.censoring_rate + 1e-8) / max(med_event, 1e-8)
            return self.rng.exponential(1.0 / rate, len(event_times))
        else:
            raise ValueError(f"Unknown censoring_type: {self.censoring_type}")

    def simulate(self, n_samples: int) -> pd.DataFrame:
        """Simulate n_samples observations from the Cox model.

        Returns:
            DataFrame with columns: time, event, x_1, ..., x_p
        """
        X = self._sample_covariates(n_samples)
        event_times = self._sample_event_times(X)
        censoring_times = self._sample_censoring_times(event_times)

        observed_times = np.minimum(event_times, censoring_times)
        events = (event_times <= censoring_times).astype(int)

        # Clip very large times to avoid numerical issues
        max_time = np.percentile(observed_times[events == 1], 99) * 2
        observed_times = np.clip(observed_times, 1e-8, max_time)

        actual_censoring = 1 - events.mean()
        if abs(actual_censoring - self.censoring_rate) > 0.15:
            warnings.warn(
                f"Actual censoring rate {actual_censoring:.2f} differs from "
                f"target {self.censoring_rate:.2f}"
            )

        df = pd.DataFrame({"time": observed_times, "event": events})
        for j in range(self.p):
            df[f"x_{j + 1}"] = X[:, j]
        return df
