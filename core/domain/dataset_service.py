"""Utilities for converting raw datasets into domain value objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Tuple

import numpy as np
import pandas as pd

from core.domain.value_objects import (
    DatasetProfile,
    Frequency,
    TimeSeriesSlice,
    TimeSeriesSliceError,
    infer_frequency_from_index,
)

AggrFunc = Literal["first", "mean", "sum", "median"]


class DatasetConversionError(ValueError):
    """Raised when raw data cannot be converted to the canonical representation."""


@dataclass(frozen=True)
class DatasetConversionConfig:
    """Configuration for dataset conversion."""

    aggregate_duplicates: AggrFunc = "first"
    sort_index: bool = True
    dtype: np.dtype = np.float64


def _validate_long_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise DatasetConversionError(
            f"Long-format dataframe is missing required columns: {missing}"
        )


def _aggregate_duplicates(df: pd.DataFrame, aggfunc: AggrFunc) -> pd.DataFrame:
    if not df.index.has_duplicates:
        return df
    if aggfunc == "first":
        return df[~df.index.duplicated(keep="first")]
    if aggfunc == "mean":
        return df.groupby(level=0).mean()
    if aggfunc == "sum":
        return df.groupby(level=0).sum()
    if aggfunc == "median":
        return df.groupby(level=0).median()
    raise DatasetConversionError(f"Unsupported aggregate function: {aggfunc}")


def _normalize_wide_dataframe(
    df: pd.DataFrame,
    config: DatasetConversionConfig,
) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DatasetConversionError("Wide-format dataframe index must be DatetimeIndex")
    normalized = df.copy()
    if config.sort_index and not normalized.index.is_monotonic_increasing:
        normalized = normalized.sort_index()
    normalized = _aggregate_duplicates(normalized, config.aggregate_duplicates)
    return normalized


def convert_long_to_wide(
    df: pd.DataFrame,
    *,
    date_col: str,
    value_col: str,
    id_col: str,
    aggfunc: AggrFunc = "first",
) -> pd.DataFrame:
    """Pivot long-format data into wide format."""
    _validate_long_columns(df, [date_col, value_col, id_col])
    pivot = df[[date_col, value_col, id_col]].copy()
    try:
        pivot[date_col] = pd.to_datetime(pivot[date_col])
    except Exception as exc:
        raise DatasetConversionError("date column could not be parsed to datetime") from exc
    pivot = pivot.pivot_table(
        index=date_col,
        columns=id_col,
        values=value_col,
        aggfunc=aggfunc,
    )
    pivot.index = pd.DatetimeIndex(pivot.index)
    if pivot.columns.hasnans:
        raise DatasetConversionError("Series identifiers cannot be NaN")
    pivot.columns = [str(col) for col in pivot.columns]
    return pivot


class DatasetService:
    """Service that converts external datasets into domain objects."""

    def __init__(self, config: DatasetConversionConfig | None = None) -> None:
        self.config = config or DatasetConversionConfig()

    def from_wide(
        self,
        df: pd.DataFrame,
        *,
        frequency: Frequency | None = None,
    ) -> Tuple[TimeSeriesSlice, DatasetProfile]:
        """Convert wide-format dataframe into canonical slice and profile."""
        normalized = _normalize_wide_dataframe(df, self.config)
        slice_ = TimeSeriesSlice.from_dataframe(normalized, dtype=self.config.dtype)
        freq = frequency or infer_frequency_from_index(slice_.timestamps)
        profile = DatasetProfile.from_slice(slice_, frequency=freq)
        return slice_, profile

    def from_long(
        self,
        df: pd.DataFrame,
        *,
        date_col: str,
        value_col: str,
        id_col: str,
        frequency: Frequency | None = None,
        aggfunc: AggrFunc | None = None,
    ) -> Tuple[TimeSeriesSlice, DatasetProfile]:
        """Convert long-format dataframe into canonical slice and profile."""
        pivot = convert_long_to_wide(
            df,
            date_col=date_col,
            value_col=value_col,
            id_col=id_col,
            aggfunc=aggfunc or self.config.aggregate_duplicates,
        )
        return self.from_wide(pivot, frequency=frequency)

    def validate_slice(self, slice_: TimeSeriesSlice) -> None:
        """Run basic sanity checks before downstream consumption."""
        if slice_.n_series == 0 or slice_.n_timesteps == 0:
            raise DatasetConversionError("Dataset slice contains no data")
        if np.all(np.isnan(slice_.values)):
            raise DatasetConversionError("Dataset contains only NaN values")
        # Check for all-NaN series
        nan_series = [sid for sid, col in zip(slice_.series_ids, slice_.values.T) if np.all(np.isnan(col))]
        if nan_series:
            raise DatasetConversionError(
                f"Series contain only NaNs and must be removed: {nan_series}"
            )
        # Validate monotonic timestamp spacing by ensuring deltas are positive
        deltas = np.diff(slice_.timestamps.view(np.int64))
        if np.any(deltas <= 0):  # pragma: no cover - safeguard; should be caught earlier
            raise TimeSeriesSliceError("Timestamps must be strictly increasing")

