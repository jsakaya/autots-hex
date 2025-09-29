import numpy as np
import pandas as pd
import pytest

from adapters.secondary.memory.transformation_adapter import (
    IdentityTransformationAdapter,
    SimpleScalerAdapter,
)
from core.domain.transformations import TransformationChain
from core.domain.value_objects import TimeSeriesSlice
from ports.transformation_port import TransformationHandle


def sample_slice() -> TimeSeriesSlice:
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [2.0, 4.0, 6.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    return TimeSeriesSlice.from_dataframe(df)


class TestIdentityAdapter:
    def test_fit_transform_returns_same_data(self) -> None:
        adapter = IdentityTransformationAdapter()
        chain = TransformationChain.empty()
        result = adapter.fit_transform(sample_slice(), chain)
        assert np.array_equal(result.transformed_slice.values, sample_slice().values)
        assert result.handle.identifier == "identity"

    def test_inverse_transform(self) -> None:
        adapter = IdentityTransformationAdapter()
        values = np.array([[1.0, 2.0]])
        out = adapter.inverse_transform(TransformationHandle("identity"), values)
        assert np.array_equal(out, values)

    def test_wrong_handle(self) -> None:
        adapter = IdentityTransformationAdapter()
        with pytest.raises(ValueError):
            adapter.inverse_transform(TransformationHandle("other"), np.zeros((1, 1)))


class TestSimpleScalerAdapter:
    def test_scaler_roundtrip(self) -> None:
        adapter = SimpleScalerAdapter()
        result = adapter.fit_transform(sample_slice(), TransformationChain.empty())
        assert np.allclose(result.transformed_slice.values.mean(axis=0), 0, atol=1e-8)
        inverted = adapter.inverse_transform(
            result.handle,
            result.transformed_slice.values,
        )
        assert np.allclose(inverted, sample_slice().values)

    def test_unknown_handle(self) -> None:
        adapter = SimpleScalerAdapter()
        with pytest.raises(ValueError):
            adapter.inverse_transform(TransformationHandle("missing"), np.zeros((1, 1)))
