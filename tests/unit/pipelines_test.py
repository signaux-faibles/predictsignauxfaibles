# pylint: disable=missing-function-docstring
import pandas as pd
import pytest

from predictsignauxfaibles.pipelines import (
    ALL_PIPELINES,
    DEFAULT_PIPELINE,
    run_pipeline,
    MissingDataError,
)

from predictsignauxfaibles.preprocessors import Preprocessor
from predictsignauxfaibles.redressements import Redressement

from tests.fake_data.dataframes import df_test_full


@pytest.mark.parametrize("pipeline", ALL_PIPELINES)
def test_all_pipeline(pipeline):
    for step in pipeline:
        assert isinstance(step, (Preprocessor, Redressement))
        assert hasattr(step, "name")
        assert hasattr(step, "function")
        assert hasattr(step, "input")
        assert hasattr(step, "output")
        assert isinstance(step.name, str)
        assert callable(step.function)
        assert isinstance(step.input, list)
        assert isinstance(step.output, list) or step.output is None


def test_default_pipeline_sucess():
    data_in = df_test_full.copy()
    data_out = run_pipeline(data_in, pipeline=DEFAULT_PIPELINE)
    assert data_out.shape == (3, 11)
    # acoss
    assert (data_out["avg_delta_dette_par_effectif"] == [0, 50, -0.04]).all()
    # paydex yoy
    assert (data_out["paydex_yoy"] == [-81, 0, 9_985]).all()
    # paydex group
    expected = [pd.Interval(0, 15), pd.Interval(15, 30), pd.Interval(90, float("inf"))]
    assert (data_out["paydex_group"] == expected).all()


def test_default_pipeline_missing_input():
    data_in = df_test_full.copy()
    data_in.drop(columns=("code_naf"), inplace=True)
    with pytest.raises(MissingDataError):
        run_pipeline(data_in, pipeline=DEFAULT_PIPELINE)
