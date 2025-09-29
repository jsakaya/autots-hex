import numpy as np
import pandas as pd
import pytest

from ports.dataset_port import DatasetRequest, InMemoryDatasetAdapter


@pytest.fixture
def adapter() -> InMemoryDatasetAdapter:
    return InMemoryDatasetAdapter()


def test_load_wide(adapter: InMemoryDatasetAdapter) -> None:
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [2, 3, 4],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    response = adapter.load(DatasetRequest(dataframe=df))
    assert response.slice.n_series == 2
    assert response.profile.n_timesteps == 3


def test_load_long(adapter: InMemoryDatasetAdapter) -> None:
    data = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
            "series": ["a", "a", "b", "b"],
            "value": [1, 2, 3, 4],
        }
    )
    response = adapter.load(
        DatasetRequest(
            dataframe=data,
            format="long",
            date_col="date",
            value_col="value",
            id_col="series",
        )
    )
    assert response.slice.n_series == 2
    assert response.profile.n_timesteps == 2


def test_missing_columns(adapter: InMemoryDatasetAdapter) -> None:
    data = pd.DataFrame({"date": ["2024-01-01"], "value": [1]})
    request = DatasetRequest(dataframe=data, format="long", date_col="date", value_col="value")
    with pytest.raises(ValueError):
        adapter.load(request)


def test_invalid_format(adapter: InMemoryDatasetAdapter) -> None:
    df = pd.DataFrame({"a": [1]}, index=pd.date_range("2024-01-01", periods=1, freq="D"))
    with pytest.raises(ValueError):
        adapter.load(DatasetRequest(dataframe=df, format="json"))


def test_validation_failure(adapter: InMemoryDatasetAdapter) -> None:
    df = pd.DataFrame({"a": [np.nan, np.nan]}, index=pd.date_range("2024-01-01", periods=2, freq="D"))
    with pytest.raises(ValueError):
        adapter.load(DatasetRequest(dataframe=df))
