"""Transformation port definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from core.domain.transformations import TransformationChain
from core.domain.value_objects import TimeSeriesSlice


@dataclass(frozen=True)
class TransformationHandle:
    """Opaque handle returned by adapters representing fitted transformers."""

    identifier: str


@dataclass(frozen=True)
class TransformationResult:
    """Output of applying a transformation chain."""

    transformed_slice: TimeSeriesSlice
    handle: TransformationHandle


@runtime_checkable
class TransformationPort(Protocol):
    """Port abstraction for fit/transform/inverse operations."""

    def fit_transform(
        self,
        data: TimeSeriesSlice,
        chain: TransformationChain,
    ) -> TransformationResult:
        ...

    def inverse_transform(
        self,
        handle: TransformationHandle,
        forecast_values: np.ndarray,
    ) -> np.ndarray:
        ...

