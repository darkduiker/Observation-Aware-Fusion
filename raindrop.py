"""Device-agnostic Raindrop adaptation for the labor landmark task.

The implementation retains the defining graph-guided sensor propagation and
temporal self-attention structure while removing fixed public-dataset shapes,
global CUDA transfers and PyTorch-Geometric coupling from the supplied code.
It is an adapted implementation, not a numerical reproduction claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from opal_fusion.models.components import ContinuousTimeEncoding


@dataclass(frozen=True)
class RaindropConfig:
    n_variables: int
    n_static_features: int
    n_classes: int = 2
    n_stages: int = 3
    sensor_dimension: int = 8
    model_dimension: int = 128
    time_dimension: int = 32
    graph_layers: int = 2
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.15


class SensorGraphPropagation(nn.Module):
    """Learned directed sensor graph with residual message passing."""

    def __init__(self, n_variables: int, dimension: int, dropout: float) -> None:
        super().__init__()
        self.adjacency_logits = nn.Parameter(torch.zeros(n_variables, n_variables))
        self.update = nn.Sequential(
            nn.Linear(2 * dimension, 2 * dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dimension, dimension),
        )
        self.norm = nn.LayerNorm(dimension)

    def forward(
        self, nodes: torch.Tensor, observed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        adjacency = torch.softmax(self.adjacency_logits, dim=-1)
        messages = torch.einsum("vw,btwd->btvd", adjacency, nodes)
        updated = self.norm(nodes + self.update(torch.cat([nodes, messages], dim=-1)))
        return updated * observed.unsqueeze(-1), adjacency


class Raindrop(nn.Module):
    """Graph propagation across variables followed by temporal attention."""

    def __init__(self, config: RaindropConfig) -> None:
        super().__init__()
        self.config = config
        v = config.n_variables
        d_sensor = config.sensor_dimension
        d_model = config.model_dimension

        self.sensor_embedding = nn.Embedding(v, d_sensor)
        self.value_projection = nn.Sequential(
            nn.Linear(1, d_sensor), nn.GELU(), nn.Linear(d_sensor, d_sensor)
        )
        self.graph_layers = nn.ModuleList([
            SensorGraphPropagation(v, d_sensor, config.dropout)
            for _ in range(config.graph_layers)
        ])
        self.sensor_pool = nn.Sequential(
            nn.Linear(v * d_sensor, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.mask_projection = nn.Linear(v, d_model)
        self.time_encoding = ContinuousTimeEncoding(config.time_dimension)
        self.time_projection = nn.Linear(config.time_dimension, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.n_heads,
            dim_feedforward=4 * d_model,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, config.n_layers)
        self.static_encoder = nn.Sequential(
            nn.LayerNorm(2 * config.n_static_features),
            nn.Linear(2 * config.n_static_features, d_model),
            nn.GELU(),
        )
        self.stage_embedding = nn.Embedding(config.n_stages, d_model)
        self.outcome_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d_model, config.n_classes),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        values = torch.nan_to_num(batch["dense_values"])
        observed = batch["observed_mask"].to(values.dtype)
        time_hours = batch["time_hours"]
        padding = batch["temporal_padding_mask"].bool()
        if values.ndim != 3 or values.shape != observed.shape:
            raise ValueError("Raindrop expects dense values and masks with shape [B,T,V]")
        if values.shape[-1] != self.config.n_variables:
            raise ValueError(
                f"Expected {self.config.n_variables} variables, got {values.shape[-1]}"
            )

        sensor_ids = torch.arange(self.config.n_variables, device=values.device)
        sensor_token = self.sensor_embedding(sensor_ids).view(1, 1, self.config.n_variables, -1)
        nodes = self.value_projection(values.unsqueeze(-1)) + sensor_token
        nodes = nodes * observed.unsqueeze(-1)
        adjacency = None
        for graph_layer in self.graph_layers:
            nodes, adjacency = graph_layer(nodes, observed)

        temporal_tokens = self.sensor_pool(nodes.flatten(start_dim=2))
        temporal_tokens = (
            temporal_tokens
            + self.mask_projection(observed)
            + self.time_projection(self.time_encoding(time_hours))
        )
        encoded = self.temporal_encoder(
            temporal_tokens, src_key_padding_mask=padding
        )
        keep = (~padding).to(encoded.dtype).unsqueeze(-1)
        pooled = (encoded * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)

        static = torch.cat([
            torch.nan_to_num(batch["static"]),
            batch["static_missing"].to(values.dtype),
        ], dim=-1)
        representation = (
            pooled
            + self.static_encoder(static)
            + self.stage_embedding(batch["stage_id"])
        )
        return {
            "logits": self.outcome_head(representation),
            "representation": representation,
            "adjacency": adjacency,
        }
