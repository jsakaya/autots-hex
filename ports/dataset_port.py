"""Dataset port definitions for hexagonal architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

from core.domain.dataset_service import DatasetConversionConfig, DatasetService
from core.domain.value_objects import DatasetProfile, Frequency, TimeSeriesSlice


@dataclass(frozen=True)
class DatasetRequest:
    """Payload describing a dataset to load."""

    dataframe: pd.DataFrame
    format: str = "wide"  # "wide" or "long"
    date_col: str | None = None
    value_col: str | None = None
    id_col: str | None = None
    frequency: Frequency | None = None


@dataclass(frozen=True)
class DatasetResponse:
    """Canonical dataset representation returned by adapters."""

    slice: TimeSeriesSlice
    profile: DatasetProfile


@runtime_checkable
class DatasetPort(Protocol):
    """Port for acquiring canonical datasets."""

    def load(self, request: DatasetRequest) -> DatasetResponse:
        ...


class InMemoryDatasetAdapter(DatasetPort):
    """Adapter that serves datasets from in-memory DataFrames."""

    def __init__(self, config: DatasetConversionConfig | None = None) -> None:
        self._service = DatasetService(config=config)

    def load(self, request: DatasetRequest) -> DatasetResponse:
        fmt = request.format.lower()
        if fmt not in {"wide", "long"}:
            raise ValueError(f"Unsupported dataset format: {request.format}")

        if fmt == "wide":
            slice_, profile = self._service.from_wide(
                request.dataframe, frequency=request.frequency
            )
        else:
            if not all([request.date_col, request.value_col, request.id_col]):
                raise ValueError(
                    "Long format requires date_col, value_col, and id_col to be specified"
                )
            slice_, profile = self._service.from_long(
                request.dataframe,
                date_col=request.date_col,
                value_col=request.value_col,
                id_col=request.id_col,
                frequency=request.frequency,
            )
        self._service.validate_slice(slice_)
        return DatasetResponse(slice=slice_, profile=profile)

