"""Query-anchored dense adapter for baselines that require T-by-V input."""

from __future__ import annotations

import numpy as np
import torch


def collate_dense_landmarks(
    samples: list[dict],
    bin_hours: float = 1.0 / 6.0,
    max_steps: int = 128,
) -> dict[str, torch.Tensor]:
    """Convert pre-query events to a dense tensor without crossing the query.

    GRU-D and Raindrop were designed for a time-by-variable matrix. Events are
    therefore assigned to bins anchored at the landmark query. Native-event
    models (Event Transformer and OPAL) do not call this adapter.
    """
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    if bin_hours <= 0 or max_steps < 1:
        raise ValueError("bin_hours and max_steps must be positive")

    n_variables = int(len(samples[0]["variable_available"]))
    prepared: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for sample in samples:
        recency = np.asarray(sample["recency_hours"], dtype=np.float32)
        variable_id = np.asarray(sample["variable_id"], dtype=np.int64)
        values = np.asarray(sample["value"], dtype=np.float32)
        if np.any(recency < -1e-6):
            raise ValueError("Future event detected before dense collation")
        valid = (variable_id > 0) & np.isfinite(values) & np.isfinite(recency)

        if valid.any():
            bucket_id = np.floor(np.maximum(recency[valid], 0.0) / bin_hours).astype(int)
            buckets = sorted(np.unique(bucket_id).tolist(), reverse=True)[-max_steps:]
            bucket_to_row = {bucket: row for row, bucket in enumerate(buckets)}
            dense = np.zeros((len(buckets), n_variables), dtype=np.float32)
            observed = np.zeros_like(dense)

            valid_positions = np.flatnonzero(valid)
            # Oldest to newest; the newest duplicate value in a bin is retained.
            order = valid_positions[np.argsort(-recency[valid_positions])]
            for position in order:
                bucket = int(np.floor(max(0.0, float(recency[position])) / bin_hours))
                if bucket not in bucket_to_row:
                    continue
                row = bucket_to_row[bucket]
                column = int(variable_id[position]) - 1
                dense[row, column] = float(values[position])
                observed[row, column] = 1.0

            oldest_bucket = buckets[0]
            time = np.asarray(
                [(oldest_bucket - bucket) * bin_hours for bucket in buckets],
                dtype=np.float32,
            )
        else:
            # Keep one valid all-missing step so recurrent/attention modules do
            # not receive an all-padding sequence. Static context remains usable.
            dense = np.zeros((1, n_variables), dtype=np.float32)
            observed = np.zeros_like(dense)
            time = np.zeros(1, dtype=np.float32)

        deltas = np.zeros_like(dense)
        for step in range(1, len(time)):
            interval = max(0.0, float(time[step] - time[step - 1]))
            deltas[step] = interval + (1.0 - observed[step - 1]) * deltas[step - 1]
        prepared.append((dense, observed, deltas, time))

    max_length = max(item[0].shape[0] for item in prepared)
    batch_size = len(samples)
    dense_values = torch.zeros(batch_size, max_length, n_variables, dtype=torch.float32)
    observed_mask = torch.zeros_like(dense_values)
    deltas = torch.zeros_like(dense_values)
    time_hours = torch.zeros(batch_size, max_length, dtype=torch.float32)
    padding = torch.ones(batch_size, max_length, dtype=torch.bool)
    for row, (dense, observed, delta, time) in enumerate(prepared):
        length = len(time)
        dense_values[row, :length] = torch.as_tensor(dense)
        observed_mask[row, :length] = torch.as_tensor(observed)
        deltas[row, :length] = torch.as_tensor(delta)
        time_hours[row, :length] = torch.as_tensor(time)
        padding[row, :length] = False

    batch = {
        "dense_values": dense_values,
        "observed_mask": observed_mask,
        "deltas": deltas,
        "time_hours": time_hours,
        "temporal_padding_mask": padding,
        "static": torch.as_tensor(np.stack([s["static"] for s in samples]), dtype=torch.float32),
        "static_missing": torch.as_tensor(
            np.stack([s["static_missing"] for s in samples]), dtype=torch.float32
        ),
    }
    for key in ["stage_id", "center_id", "target", "patient_id"]:
        batch[key] = torch.as_tensor([sample[key] for sample in samples], dtype=torch.long)
    return batch
