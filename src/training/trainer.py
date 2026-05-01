"""Training loop and experiment management."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
import yaml

from ..data.pipeline import SurvivalDataset
from ..models.pinn import HazardPINN
from .loss import CompositeLoss


@dataclass
class ExperimentConfig:
    experiment_name: str
    rationale: str
    n_epochs: int = 1000
    lr: float = 1e-3
    optimizer_name: str = "adam"
    weight_decay: float = 0.0
    n_collocation_points: int = 200
    collocation_method: str = "uniform"
    checkpoint_every: int = 200
    gradient_clip: Optional[float] = None
    lr_scheduler: Optional[str] = None
    lr_patience: int = 100
    loss_weights: Dict[str, float] = field(default_factory=lambda: {"mle": 1.0, "pl": 0.5, "ode": 1.0, "ic": 1.0})
    model_config: Dict = field(default_factory=dict)
    simulation_config: Dict = field(default_factory=dict)


class Trainer:
    """Runs training loop and saves all experiment artifacts."""

    def __init__(self, model: HazardPINN, loss_fn: CompositeLoss, config: ExperimentConfig):
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self._loss_history: List[Dict] = []
        self._best_loss = float("inf")
        self._best_state: Optional[Dict] = None

    def _build_optimizer(self) -> torch.optim.Optimizer:
        name = self.config.optimizer_name.lower()
        if name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        elif name == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        elif name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {name}")

    def _build_scheduler(self, optimizer):
        if self.config.lr_scheduler is None:
            return None
        if self.config.lr_scheduler == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=self.config.lr_patience, factor=0.5, verbose=False
            )
        raise ValueError(f"Unknown scheduler: {self.config.lr_scheduler}")

    def train(self, dataset: SurvivalDataset) -> Dict[str, List]:
        """Run training loop.

        Returns:
            loss_history: dict with keys 'epoch', 'total', 'mle', 'pl', 'ode', 'ic'
        """
        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer)
        history: Dict[str, List] = {"epoch": [], "total": [], "mle": [], "pl": [], "ode": [], "ic": []}

        t_col, x_col = dataset.get_collocation_points(
            self.config.n_collocation_points,
            method=self.config.collocation_method,
        )

        self.model.train()
        start = time.time()

        for epoch in range(1, self.config.n_epochs + 1):
            optimizer.zero_grad()

            total_loss, components = self.loss_fn(self.model, dataset, t_col, x_col)

            total_loss.backward()

            if self.config.gradient_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

            optimizer.step()

            if scheduler is not None:
                if self.config.lr_scheduler == "reduce_on_plateau":
                    scheduler.step(total_loss.item())

            total_val = total_loss.item()
            history["epoch"].append(epoch)
            history["total"].append(total_val)
            for k in ("mle", "pl", "ode", "ic"):
                history[k].append(components.get(k, 0.0))

            if total_val < self._best_loss:
                self._best_loss = total_val
                self._best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if epoch % self.config.checkpoint_every == 0:
                elapsed = time.time() - start
                print(
                    f"[{epoch:5d}/{self.config.n_epochs}] "
                    f"total={total_val:.4f}  "
                    f"mle={components.get('mle', 0):.4f}  "
                    f"pl={components.get('pl', 0):.4f}  "
                    f"ode={components.get('ode', 0):.4f}  "
                    f"ic={components.get('ic', 0):.4f}  "
                    f"({elapsed:.1f}s)"
                )

        self._loss_history = history
        return history

    def save_experiment(self, results_dir: str, metrics: Optional[Dict] = None) -> str:
        """Save all experiment artifacts to results_dir/{experiment_name}/."""
        exp_dir = os.path.join(results_dir, self.config.experiment_name)
        os.makedirs(exp_dir, exist_ok=True)

        # Model weights
        torch.save(self.model.state_dict(), os.path.join(exp_dir, "weights_final.pt"))
        if self._best_state is not None:
            torch.save(self._best_state, os.path.join(exp_dir, "weights_best.pt"))

        # Loss history
        pd.DataFrame(self._loss_history).to_csv(
            os.path.join(exp_dir, "loss_history.csv"), index=False
        )

        # Config
        with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
            yaml.dump(asdict(self.config), f, default_flow_style=False)

        # Metrics
        if metrics is not None:
            with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

        print(f"Experiment saved to {exp_dir}")
        return exp_dir
