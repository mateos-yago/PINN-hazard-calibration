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
    ):
        super().__init__()
        self.net = _build_mlp(1, list(hidden_dims), 1, activation, use_layer_norm)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: shape [batch, 1]
        Returns:
            γ̂: shape [batch, 1]
        """
        return self.net(t)
