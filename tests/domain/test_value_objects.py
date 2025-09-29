from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from core.domain.value_objects import (
    DatasetProfile,
    DatasetProfileError,
    ForecastHorizon,
    ForecastHorizonError,
    Frequency,
    FrequencyError,
    TimeSeriesSlice,
    TimeSeriesSliceError,
    infer_frequency_from_index,
    validate_forecast_horizon,
)


def _sample_dataframe(rows: int = 10) -> pd.DataFrame:
    date_range = pd.date_range("2024-01-01", periods=rows, freq="D")
    data = {
        "series_a": np.arange(rows, dtype=float),
        "series_b": np.arange(rows, dtype=float) * 2,
    }
    return pd.DataFrame(data, index=date_range)


class TestFrequency:
    def test_valid_alias(self) -> None:
        freq = Frequency("D")
        assert str(freq) == "D"
        assert freq.offset.n == 1

    @pytest.mark.parametrize("alias", ["", "???", None])
    def test_invalid_alias(self, alias: str | None) -> None:
        with pytest.raises(FrequencyError):
            Frequency(alias)  # type: ignore[arg-type]


class TestForecastHorizon:
    def test_valid_horizon(self) -> None:
        horizon = ForecastHorizon(7)
        assert int(horizon) == 7
        assert str(horizon) == "7"

    @pytest.mark.parametrize("value", [0, -1, 1.5, "5"])
    def test_invalid_horizon(self, value: object) -> None:
        with pytest.raises(ForecastHorizonError):
            ForecastHorizon(value)  # type: ignore[arg-type]


class TestTimeSeriesSlice:
    def test_from_dataframe_roundtrip(self) -> None:
        df = _sample_dataframe()
        slice_ = TimeSeriesSlice.from_dataframe(df)
        assert slice_.n_timesteps == len(df)
        assert slice_.n_series == df.shape[1]
        pd.testing.assert_frame_equal(slice_.to_dataframe(), df)

    def test_validation_errors(self) -> None:
        df = _sample_dataframe()
        duplicates = df.copy()
        duplicates.index = duplicates.index.repeat(2)[: len(df)]
        with pytest.raises(TimeSeriesSliceError):
            TimeSeriesSlice(duplicates.index, duplicates.to_numpy(), tuple(df.columns))

        wrong_columns = df.copy()
        with pytest.raises(TimeSeriesSliceError):
            TimeSeriesSlice(df.index, wrong_columns.to_numpy(), ("a",))

    def test_select_series(self) -> None:
        df = _sample_dataframe()
        slice_ = TimeSeriesSlice.from_dataframe(df)
        subset = slice_.select_series(["series_b"])
        assert subset.series_ids == ("series_b",)
        assert subset.values.shape == (df.shape[0], 1)

    def test_trim_time(self) -> None:
        df = _sample_dataframe()
        slice_ = TimeSeriesSlice.from_dataframe(df)
        trimmed = slice_.trim_time(datetime(2024, 1, 3), datetime(2024, 1, 5))
        assert trimmed.n_timesteps == 3
        assert trimmed.timestamps[0] == pd.Timestamp("2024-01-03")
        assert trimmed.timestamps[-1] == pd.Timestamp("2024-01-05")


class TestDatasetProfile:
    def test_profile_from_slice(self) -> None:
        df = _sample_dataframe()
        slice_ = TimeSeriesSlice.from_dataframe(df)
        profile = DatasetProfile.from_slice(slice_)
        assert profile.n_series == 2
        assert profile.n_timesteps == len(df)
        assert 0 <= profile.missing_ratio <= 1
        assert str(profile.frequency) == "D"

    def test_profile_requires_data(self) -> None:
        empty_df = _sample_dataframe(rows=1).iloc[0:0]
        with pytest.raises(DatasetProfileError):
            DatasetProfile.from_slice(TimeSeriesSlice.from_dataframe(empty_df))


class TestHelpers:
    def test_infer_frequency(self) -> None:
        index = pd.date_range("2024-01-01", periods=5, freq="H")
        freq = infer_frequency_from_index(index)
        assert str(freq).lower() == "h"

    def test_infer_frequency_failure(self) -> None:
        irregular = pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-03"),
                pd.Timestamp("2024-01-04"),
            ]
        )
        with pytest.raises(FrequencyError):
            infer_frequency_from_index(irregular)

    def test_validate_forecast_horizon(self) -> None:
        slice_ = TimeSeriesSlice.from_dataframe(_sample_dataframe())
        validate_forecast_horizon(ForecastHorizon(3), slice_)
        with pytest.raises(ForecastHorizonError):
            validate_forecast_horizon(ForecastHorizon(slice_.n_timesteps + 1), slice_)

        tiny_slice = TimeSeriesSlice.from_dataframe(_sample_dataframe(rows=1))
        with pytest.raises(ForecastHorizonError):
            validate_forecast_horizon(ForecastHorizon(1), tiny_slice)
