"""Common streaming API for FDP and future DP / pi0 / pi0.5 adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class InferenceObservation:
    """One synchronized sensor frame; timestamps are seconds within an episode.

    Images are RGB (uint8 HWC or float CHW in [0,1]). Joint q/dq/tau use
    rad, rad/s and N.m; wrench uses [Fx,Fy,Fz,Mx,My,Mz] in N and N.m.
    Individual policy adapters declare which fields they require. No field
    contains future observations, future actions, or ground-truth targets.
    """

    timestamp_s: float
    images: Mapping[str, np.ndarray]
    q: np.ndarray | None = None
    dq: np.ndarray | None = None
    tau: np.ndarray | None = None
    wrench: np.ndarray | None = None
    instruction: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceSpec:
    sample_rate_hz: float
    observation_steps: int
    image_shapes: Mapping[str, tuple[int, int, int]]
    action_shape: tuple[int, int]
    action_format: str
    required_fields: tuple[str, ...]


@dataclass(frozen=True)
class InferenceResult:
    """No actions are produced during warmup; action coordinates are explicit."""

    ready: bool
    observation_timestamp_s: float
    actions: np.ndarray | None
    action_timestamps_s: np.ndarray | None
    action_format: str
    info: Mapping[str, Any] = field(default_factory=dict)


class InferenceBase(ABC):
    """One instance per environment/episode stream.

    Call observe() for EVERY sensor frame, including between action replans.
    predict() without an observation replans from the latest buffered frames;
    predict(observation) combines those two operations. reset() starts a new
    episode and must discard temporal state and pending action context.
    The interface leaves model loading, modalities and action format to the
    adapter, so future DP / pi0 / pi0.5 implementations share the same loop.
    """

    @property
    @abstractmethod
    def spec(self) -> InferenceSpec:
        raise NotImplementedError

    @abstractmethod
    def reset(self, *, seed: int | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def observe(self, observation: InferenceObservation) -> bool:
        """Ingest one frame; return whether enough context is available."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, observation: InferenceObservation | None = None) -> InferenceResult:
        raise NotImplementedError

    def __call__(self, observation: InferenceObservation) -> InferenceResult:
        return self.predict(observation)

    def close(self) -> None:
        """Override for adapters owning external devices or services."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
