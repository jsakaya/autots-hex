"""Simple in-memory transformation adapter for testing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.domain.transformations import TransformationChain
from core.domain.value_objects import TimeSeriesSlice
from ports.transformation_port import (
    TransformationHandle,
    TransformationPort,
    TransformationResult,
)


class IdentityTransformationAdapter(TransformationPort):
    """Adapter that performs no transformation, useful for tests."""

    def fit_transform(
        self,
        data: TimeSeriesSlice,
        chain: TransformationChain,
    ) -> TransformationResult:
        df = data.to_dataframe()
        return TransformationResult(
            transformed_slice=TimeSeriesSlice.from_dataframe(df),
            handle=TransformationHandle(identifier="identity"),
        )

    def inverse_transform(
        self,
        handle: TransformationHandle,
        forecast_values: np.ndarray,
    ) -> np.ndarray:
        if handle.identifier != "identity":
            raise ValueError("handle not produced by this adapter")
        return forecast_values


class SimpleScalerAdapter(TransformationPort):
    """Adapter applying per-series standardization for demonstration purposes."""

    def __init__(self) -> None:
        self._scalers: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def fit_transform(
        self,
        data: TimeSeriesSlice,
        chain: TransformationChain,
    ) -> TransformationResult:
        # only honors empty or single scaler step for demonstration
        df = data.to_dataframe()
        means = df.mean(axis=0).to_numpy()
        stds = df.std(axis=0, ddof=0).replace(0, 1).to_numpy()
        normalized = (df.to_numpy() - means) / stds
        handle_id = f"scaler-{id(data)}"
        self._scalers[handle_id] = (means, stds)
        transformed = TimeSeriesSlice(df.index.copy(), normalized, data.series_ids)
        return TransformationResult(
            transformed_slice=transformed,
            handle=TransformationHandle(identifier=handle_id),
        )

    def inverse_transform(
        self,
        handle: TransformationHandle,
        forecast_values: np.ndarray,
    ) -> np.ndarray:
        means, stds = self._scalers.get(handle.identifier, (None, None))
        if means is None:
            raise ValueError("Unknown transformation handle")
        return forecast_values * stds + means

