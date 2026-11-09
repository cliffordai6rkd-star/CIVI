"""Physical-unit low-pass filtering before teacher windowing and normalization."""

import numpy as np


def resolve_wrench_filter(config=None):
    config = dict(config or {})
    cutoff = float(config.get("cutoff_hz", 2.0))
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("wrench_filter.cutoff_hz must be positive and finite")
    return {"enabled": bool(config.get("enabled", False)), "cutoff_hz": cutoff,
            "apply_to_future": bool(config.get("apply_to_future", True))}


def lowpass_wrench(values, intervals_s, cutoff_hz, *, initial=None):
    """Use the training Dataset's y += (1-exp(-2*pi*fc*dt)) * (x-y).

    Keep float64 state, as in the Dataset, and cast only the model inputs to
    float32. Without an initial state the first sample initializes the filter.
    Forecast callers pass the current state; this function never mutates it.
    Values and intervals are already validated by the streaming adapter.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = -np.expm1(-2.0 * np.pi * cutoff_hz * np.asarray(intervals_s))
    filtered = np.empty_like(values)
    previous = initial
    for index, (value, weight) in enumerate(zip(values, weights)):
        previous = value.copy() if previous is None else previous + weight * (value - previous)
        filtered[index] = previous
    return filtered
