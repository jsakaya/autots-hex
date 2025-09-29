"""Domain value objects for dataset representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd


class FrequencyError(ValueError):
    """Raised when frequency alias cannot be interpreted."""


@dataclass(frozen=True, slots=True)
class Frequency:
    """Represents a pandas-compatible frequency alias."""

    alias: str
    _offset: pd.DateOffset = field(init=False, repr=False)

    def __post_init__(self) -> None:  # noqa: D401 (docs in class docstring)
        if not self.alias:
            raise FrequencyError("Frequency alias must be a non-empty string")
        try:
            offset = pd.tseries.frequencies.to_offset(self.alias)
        except Exception as exc:  # pragma: no cover - pandas provides message
            raise FrequencyError(f"Unknown frequency alias: {self.alias}") from exc
        object.__setattr__(self, 'alias', offset.freqstr)
        object.__setattr__(self, '_offset', offset)

    @property
    def offset(self) -> pd.DateOffset:
        """Return the pandas DateOffset instance for this alias."""
        return self._offset

    def __str__(self) -> str:
        return self.alias


class ForecastHorizonError(ValueError):
    """Raised when a forecast horizon is invalid."""


@dataclass(frozen=True, slots=True)
class ForecastHorizon:
    """Represents the number of periods to forecast."""

    steps: int

    def __post_init__(self) -> None:
        if not isinstance(self.steps, int):
            raise ForecastHorizonError("Forecast horizon steps must be an integer")
        if self.steps <= 0:
            raise ForecastHorizonError("Forecast horizon must be positive")

    def __int__(self) -> int:
        return self.steps

    def __str__(self) -> str:
        return str(self.steps)


class TimeSeriesSliceError(ValueError):
    """Raised when constructing a time series slice fails."""


@dataclass(frozen=True, slots=True)
class TimeSeriesSlice:
    """Immutable representation of multivariate time-series data."""

    timestamps: pd.DatetimeIndex
    values: np.ndarray
    series_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.timestamps, pd.DatetimeIndex):
            raise TimeSeriesSliceError("timestamps must be a pandas.DatetimeIndex")
        if self.timestamps.has_duplicates:
            raise TimeSeriesSliceError("timestamps contain duplicates")
        if not self.timestamps.is_monotonic_increasing:
            raise TimeSeriesSliceError("timestamps must be sorted ascending")
        if not isinstance(self.values, np.ndarray):
            raise TimeSeriesSliceError("values must be a numpy.ndarray")
        if self.values.ndim != 2:
            raise TimeSeriesSliceError("values must be a 2D array")
        if self.values.shape[0] != len(self.timestamps):
            raise TimeSeriesSliceError(
                "values row count must equal number of timestamps"
            )
        if len(self.series_ids) != self.values.shape[1]:
            raise TimeSeriesSliceError(
                "number of series_ids must match number of value columns"
            )
        if len(set(self.series_ids)) != len(self.series_ids):
            raise TimeSeriesSliceError("series_ids must be unique")

    @property
    def n_timesteps(self) -> int:
        """Return number of rows in the slice."""
        return len(self.timestamps)

    @property
    def n_series(self) -> int:
        """Return number of series in the slice."""
        return len(self.series_ids)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the slice as a pandas DataFrame."""
        return pd.DataFrame(self.values, index=self.timestamps, columns=self.series_ids)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        dtype: np.dtype | None = np.float64,
    ) -> "TimeSeriesSlice":
        """Build a slice from a wide-format dataframe."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TimeSeriesSliceError("DataFrame index must be a DatetimeIndex")
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        values = df.to_numpy(dtype=dtype, copy=True)
        series_ids = tuple(str(col) for col in df.columns)
        return cls(df.index.copy(), values, series_ids)

    def select_series(self, ids: Iterable[str]) -> "TimeSeriesSlice":
        """Return a new slice containing only the specified series."""
        indices = [self.series_ids.index(str(series)) for series in ids]
        values = self.values[:, indices]
        series_ids = tuple(self.series_ids[i] for i in indices)
        return TimeSeriesSlice(self.timestamps.copy(), values.copy(), series_ids)

    def trim_time(self, start: datetime | None, end: datetime | None) -> "TimeSeriesSlice":
        """Return a slice trimmed to [start, end]."""
        mask = np.ones(self.n_timesteps, dtype=bool)
        if start is not None:
            mask &= self.timestamps >= pd.Timestamp(start)
        if end is not None:
            mask &= self.timestamps <= pd.Timestamp(end)
        timestamps = self.timestamps[mask]
        values = self.values[mask]
        return TimeSeriesSlice(timestamps, values.copy(), self.series_ids)


class DatasetProfileError(ValueError):
    """Raised when dataset profiling fails."""


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """Summary statistics about a time-series dataset."""

    start: pd.Timestamp
    end: pd.Timestamp
    frequency: Frequency
    n_timesteps: int
    n_series: int
    series_ids: Tuple[str, ...]
    missing_ratio: float

    @classmethod
    def from_slice(
        cls,
        data: TimeSeriesSlice,
        *,
        frequency: Frequency | None = None,
    ) -> "DatasetProfile":
        if data.n_timesteps == 0:
            raise DatasetProfileError("Cannot profile empty dataset")
        freq = frequency or infer_frequency_from_index(data.timestamps)
        missing_ratio = float(np.isnan(data.values).sum() / data.values.size)
        return cls(
            start=data.timestamps[0],
            end=data.timestamps[-1],
            frequency=freq,
            n_timesteps=data.n_timesteps,
            n_series=data.n_series,
            series_ids=data.series_ids,
            missing_ratio=missing_ratio,
        )


def infer_frequency_from_index(index: pd.DatetimeIndex) -> Frequency:
    """Infer a pandas frequency alias and wrap it as a Frequency."""
    try:
        alias = pd.infer_freq(index)
    except ValueError as exc:  # pragma: no cover - pandas raises ValueError for short index
        raise FrequencyError("Unable to infer frequency from index") from exc
    if alias is None:
        raise FrequencyError("Unable to infer frequency from index")
    return Frequency(alias)


def validate_forecast_horizon(horizon: ForecastHorizon, data: TimeSeriesSlice) -> None:
    """Ensure the requested forecast horizon is supported by the dataset."""
    if data.n_timesteps < 2:
        raise ForecastHorizonError("Dataset must contain at least two timesteps")
    if int(horizon) > data.n_timesteps:
        raise ForecastHorizonError(
            "Forecast horizon cannot exceed available timesteps in training data"
        )
