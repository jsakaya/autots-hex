import numpy as np
import pandas as pd
import pytest

from core.domain.dataset_service import (
    DatasetConversionConfig,
    DatasetConversionError,
    DatasetService,
    convert_long_to_wide,
)
from core.domain.value_objects import Frequency, TimeSeriesSlice


@pytest.fixture
def wide_df() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "s1": [1, 2, np.nan, 4],
            "s2": [0.5, np.nan, 1.5, 2.0],
        },
        index=index,
    )


class TestConvertLongToWide:
    def test_basic_conversion(self) -> None:
        data = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
                "id": ["a", "b", "a"],
                "value": [1, 2, 3],
            }
        )
        wide = convert_long_to_wide(
            data, date_col="date", value_col="value", id_col="id"
        )
        assert isinstance(wide.index, pd.DatetimeIndex)
        assert list(sorted(wide.columns)) == ["a", "b"]
        assert wide.loc[pd.Timestamp("2024-01-01"), "a"] == 1

    def test_missing_columns(self) -> None:
        with pytest.raises(DatasetConversionError):
            convert_long_to_wide(
                pd.DataFrame({"date": ["2024-01-01"], "value": [1]}),
                date_col="date",
                value_col="value",
                id_col="series",
            )

    def test_invalid_dates(self) -> None:
        with pytest.raises(DatasetConversionError):
            convert_long_to_wide(
                pd.DataFrame(
                    {"date": ["bad"], "value": [1], "series": ["a"]}
                ),
                date_col="date",
                value_col="value",
                id_col="series",
            )

    def test_aggregate_duplicates(self) -> None:
        data = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-01"],
                "id": ["a", "a"],
                "value": [1, 3],
            }
        )
        wide = convert_long_to_wide(
            data,
            date_col="date",
            value_col="value",
            id_col="id",
            aggfunc="mean",
        )
        assert wide.loc[pd.Timestamp("2024-01-01"), "a"] == 2


class TestDatasetServiceWide:
    def test_from_wide(self, wide_df: pd.DataFrame) -> None:
        service = DatasetService()
        slice_, profile = service.from_wide(wide_df)
        assert isinstance(slice_, TimeSeriesSlice)
        assert profile.n_series == 2
        assert profile.n_timesteps == 4
        assert str(profile.frequency).lower() == "d"

    def test_duplicate_index_handled(self) -> None:
        df = pd.DataFrame(
            {
                "value": [1, 2, 3, 4],
            },
            index=pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03"]
            ),
        )
        service = DatasetService()
        slice_, _ = service.from_wide(df)
        assert slice_.n_timesteps == 3
        assert slice_.timestamps.is_unique

    def test_invalid_index(self) -> None:
        df = pd.DataFrame({"value": [1, 2, 3]})
        service = DatasetService()
        with pytest.raises(DatasetConversionError):
            service.from_wide(df)


class TestDatasetServiceLong:
    def test_from_long(self) -> None:
        data = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
                "series": ["a", "b", "a", "b"],
                "value": [1, 2, 3, 4],
            }
        )
        service = DatasetService()
        slice_, profile = service.from_long(
            data, date_col="date", value_col="value", id_col="series"
        )
        assert slice_.n_series == 2
        assert profile.n_timesteps == 2

    def test_custom_frequency(self) -> None:
        data = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "series": ["a", "a"],
                "value": [1, 2],
            }
        )
        service = DatasetService()
        freq = Frequency("W")
        _, profile = service.from_long(
            data,
            date_col="date",
            value_col="value",
            id_col="series",
            frequency=freq,
        )
        assert str(profile.frequency).startswith("W")


class TestValidation:
    def test_validate_slice_success(self, wide_df: pd.DataFrame) -> None:
        service = DatasetService()
        slice_, _ = service.from_wide(wide_df)
        service.validate_slice(slice_)

    def test_validate_slice_all_nan_series(self) -> None:
        index = pd.date_range("2024-01-01", periods=3, freq="D")
        slice_ = TimeSeriesSlice(index, np.array([[np.nan], [np.nan], [np.nan]]), ("s1",))
        service = DatasetService()
        with pytest.raises(DatasetConversionError):
            service.validate_slice(slice_)

    def test_validate_slice_no_data(self) -> None:
        index = pd.DatetimeIndex([], dtype="datetime64[ns]")
        slice_ = TimeSeriesSlice(index, np.empty((0, 0)), tuple())
        service = DatasetService()
        with pytest.raises(DatasetConversionError):
            service.validate_slice(slice_)
