"""GRU-D baseline adapted to the labor landmark protocol.

This is a complete trainable implementation: feature decay, hidden-state
decay, last-observation carry-forward, missingness masks, static context and
labor-stage context are all used in the forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GRUDConfig:
    n_variables: int
    n_static_features: int
    n_stages: int = 3
    n_classes: int = 2
    hidden_size: int = 128
    dropout: float = 0.15


class GRUD(nn.Module):
    """GRU-D with train-fold means equal to zero after normalization."""

    def __init__(self, config: GRUDConfig) -> None:
        super().__init__()
        self.config = config
        v = config.n_variables
        h = config.hidden_size

        # Each feature has its own decay toward its training-fold mean (zero in
        # normalized coordinates). Hidden-state decay may use all feature gaps.
        self.input_decay_weight = nn.Parameter(torch.full((v,), 0.1))
        self.input_decay_bias = nn.Parameter(torch.zeros(v))
        self.hidden_decay = nn.Linear(v, h)
        self.recurrent_cell = nn.GRUCell(2 * v, h)

        self.static_projection = nn.Sequential(
            nn.LayerNorm(2 * config.n_static_features),
            nn.Linear(2 * config.n_static_features, h),
            nn.Tanh(),
        )
        self.stage_embedding = nn.Embedding(config.n_stages, h)
        self.classifier = nn.Sequential(
            nn.LayerNorm(h),
            nn.Dropout(config.dropout),
            nn.Linear(h, config.n_classes),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        values = torch.nan_to_num(batch["dense_values"])
        observed = batch["observed_mask"].to(values.dtype)
        deltas = torch.clamp_min(batch["deltas"], 0.0)
        padding = batch["temporal_padding_mask"].bool()
        if values.ndim != 3 or values.shape != observed.shape or values.shape != deltas.shape:
            raise ValueError("GRU-D expects values, masks and deltas with shape [B,T,V]")

        static = torch.cat(
            [torch.nan_to_num(batch["static"]), batch["static_missing"].to(values.dtype)],
            dim=-1,
        )
        hidden = self.static_projection(static)
        last_observed = torch.zeros(
            values.shape[0], values.shape[2], device=values.device, dtype=values.dtype
        )

        for step in range(values.shape[1]):
            gamma_x = torch.exp(-torch.relu(
                deltas[:, step] * self.input_decay_weight + self.input_decay_bias
            ))
            gamma_h = torch.exp(-torch.relu(self.hidden_decay(deltas[:, step])))
            decayed_hidden = hidden * gamma_h

            # Missing values decay from the last observation toward normalized
            # training mean zero; masks remain explicit recurrent inputs.
            completed = (
                observed[:, step] * values[:, step]
                + (1.0 - observed[:, step]) * gamma_x * last_observed
            )
            candidate = self.recurrent_cell(
                torch.cat([completed, observed[:, step]], dim=-1), decayed_hidden
            )
            valid_step = (~padding[:, step]).to(values.dtype).unsqueeze(-1)
            hidden = valid_step * candidate + (1.0 - valid_step) * hidden

            updated_last = (
                observed[:, step] * values[:, step]
                + (1.0 - observed[:, step]) * last_observed
            )
            last_observed = valid_step * updated_last + (1.0 - valid_step) * last_observed

        representation = hidden + self.stage_embedding(batch["stage_id"])
        return {
            "logits": self.classifier(representation),
            "representation": representation,
        }
