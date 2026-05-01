"""Surrogate and coefficient neural network components."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


_ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
}


def _build_mlp(
    in_dim: int,
    hidden_dims: List[int],
    out_dim: int,
    activation: str,
    use_layer_norm: bool = False,
) -> nn.Sequential:
    act_cls = _ACTIVATIONS[activation]
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        if use_layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act_cls())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class SurrogateNetwork(nn.Module):
    """Λ_φ(t, x): approximates cumulative hazard. Output is always ≥ 0 (Softplus)."""

    def __init__(
        self,
        n_covariates: int,
        hidden_dims: List[int] = (64, 64, 64),
        activation: str = "tanh",
        use_layer_norm: bool = False,
    ):
        super().__init__()
        in_dim = 1 + n_covariates  # (t, x_1, ..., x_p)
        self.net = _build_mlp(in_dim, list(hidden_dims), 1, activation, use_layer_norm)
        self.output_act = nn.Softplus()

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: shape [batch, 1]
            x: shape [batch, p]
        Returns:
            Λ̂: shape [batch, 1], non-negative
        """
        inp = torch.cat([t, x], dim=1)
        return self.output_act(self.net(inp))


class CoefficientNetwork(nn.Module):
    """γ_θ(t): approximates log-baseline hazard. Output is unconstrained."""

    def __init__(
        self,
        hidden_dims: List[int] = (32, 32),
        activation: str = "tanh",
        use_layer_norm: bool = False,
        time_features: List[str] = ("t",),
        log_time_offset: float = 1e-6,
    ):
        super().__init__()
        self.time_features = list(time_features)
        self.log_time_offset = float(log_time_offset)
        valid = {"t", "log_t", "log_t_shifted"}
        unknown = set(self.time_features) - valid
        if unknown:
            raise ValueError(f"Unknown coefficient time features: {sorted(unknown)}")
        self.net = _build_mlp(
            len(self.time_features), list(hidden_dims), 1, activation, use_layer_norm
        )

    def _transform_time(self, t: torch.Tensor) -> torch.Tensor:
        features = []
        for feature in self.time_features:
            if feature == "t":
                features.append(t)
            elif feature == "log_t":
                features.append(torch.log(torch.clamp(t, min=1e-6)))
            elif feature == "log_t_shifted":
                shifted = torch.clamp(t + self.log_time_offset, min=1e-6)
                features.append(torch.log(shifted))
        return torch.cat(features, dim=1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: shape [batch, 1]
        Returns:
            γ̂: shape [batch, 1]
        """
        return self.net(self._transform_time(t))
