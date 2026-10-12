"""Leakage-safe tabular representations for landmark baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TabularMatrix:
    features: np.ndarray
    target: np.ndarray
    metadata: pd.DataFrame
    feature_names: tuple[str, ...]


def _feature_names(n_variables: int, n_static: int, representation: str) -> tuple[str, ...]:
    names = [f"static_{index}" for index in range(n_static)]
    names += [f"static_missing_{index}" for index in range(n_static)]
    names += ["stage_lt4", "stage_4to6", "stage_6to10"]
    names += [f"latest_value_{index}" for index in range(n_variables)]
    names += [f"available_{index}" for index in range(n_variables)]
    names += [f"latest_recency_{index}" for index in range(n_variables)]
    if representation == "current_delta":
        names += [f"raw_delta_{index}" for index in range(n_variables)]
        names += [f"delta_available_{index}" for index in range(n_variables)]
        names += [f"log_count_{index}" for index in range(n_variables)]
    return tuple(names)


def tabularize_samples(
    samples: list[dict],
    n_variables: int,
    representation: str = "current_delta",
) -> TabularMatrix:
    """Convert pre-query event histories into current-state or delta features.

    Event values are already normalized with training-fold statistics. Missing
    latest values and unavailable deltas are filled with zero and accompanied
    by explicit indicators, so zero is never silently interpreted as observed.
    """
    if representation not in {"current", "current_delta"}:
        raise ValueError(f"Unknown tabular representation: {representation}")
    if not samples:
        raise ValueError("Cannot tabularize an empty sample list")

    rows: list[np.ndarray] = []
    metadata: list[dict[str, int]] = []
    for sample in samples:
        latest = np.zeros(n_variables, dtype=np.float32)
        available = np.zeros(n_variables, dtype=np.float32)
        latest_recency = np.zeros(n_variables, dtype=np.float32)
        raw_delta = np.zeros(n_variables, dtype=np.float32)
        delta_available = np.zeros(n_variables, dtype=np.float32)
        counts = np.zeros(n_variables, dtype=np.float32)

        value = np.asarray(sample["value"], dtype=np.float32)
        variable_id = np.asarray(sample["variable_id"], dtype=np.int64)
        recency = np.asarray(sample["recency_hours"], dtype=np.float32)
        for variable_index in range(n_variables):
            positions = np.flatnonzero(variable_id == variable_index + 1)
            if not len(positions):
                continue
            available[variable_index] = 1.0
            counts[variable_index] = float(len(positions))
            latest_position = positions[-1]
            latest[variable_index] = value[latest_position]
            latest_recency[variable_index] = np.log1p(max(0.0, float(recency[latest_position])))
            if len(positions) >= 2:
                raw_delta[variable_index] = value[positions[-1]] - value[positions[0]]
                delta_available[variable_index] = 1.0

        stage = np.zeros(3, dtype=np.float32)
        stage[int(sample["stage_id"])] = 1.0
        parts = [
            np.asarray(sample["static"], dtype=np.float32),
            np.asarray(sample["static_missing"], dtype=np.float32),
            stage,
            latest,
            available,
            latest_recency,
        ]
        if representation == "current_delta":
            parts.extend([raw_delta, delta_available, np.log1p(counts)])
        row = np.concatenate(parts).astype(np.float32)
        if not np.isfinite(row).all():
            raise ValueError("Non-finite value in tabular baseline features")
        rows.append(row)
        metadata.append({
            "patient_id": int(sample["patient_id"]),
            "stage_id": int(sample["stage_id"]),
            "center_id": int(sample["center_id"]),
            "target": int(sample["target"]),
        })

    n_static = len(np.asarray(samples[0]["static"]))
    return TabularMatrix(
        features=np.stack(rows),
        target=np.asarray([row["target"] for row in metadata], dtype=np.int64),
        metadata=pd.DataFrame(metadata),
        feature_names=_feature_names(n_variables, n_static, representation),
    )


def tabularize_dataset(dataset, representation: str = "current_delta") -> TabularMatrix:
    samples = [dataset[index] for index in range(len(dataset))]
    return tabularize_samples(
        samples,
        n_variables=dataset.preprocessor.n_variables,
        representation=representation,
    )
