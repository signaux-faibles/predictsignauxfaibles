# pylint: disable=missing-function-docstring
import pandas as pd
import pytest
from predictsignauxfaibles.preprocessors import (
    PIPELINE,
    Preprocessor,
    acoss_make_avg_delta_dette_par_effectif,
    paydex_make_groups,
    paydex_make_yoy,
    remove_administrations,
    run_pipeline,
    MissingDataError,
)

from tests.fake_data.dataframes import (
    df_test_acoss,
    df_test_paydex,
    df_test_full,
    df_test_code_naf,
)


def test_final_pipeline():
    for preprocessor in PIPELINE:
        assert isinstance(preprocessor, Preprocessor)
        assert hasattr(preprocessor, "name")
        assert hasattr(preprocessor, "function")
        assert hasattr(preprocessor, "input")
        assert hasattr(preprocessor, "output")
        assert isinstance(preprocessor.name, str)
        assert callable(preprocessor.function)
        assert isinstance(preprocessor.input, list)
        assert isinstance(preprocessor.output, list) or preprocessor.output is None


def test_acoss_make_avg_delta_dette_par_effectif():
    data_in = df_test_acoss.copy()
    data_out = acoss_make_avg_delta_dette_par_effectif(data_in)
    assert (data_out["avg_delta_dette_par_effectif"] == [0, 50, -0.04]).all()
    assert len(data_out.columns) == len(df_test_acoss.columns) + 1


def test_paydex_make_yoy():
    data_in = df_test_paydex.copy()
    data_out = paydex_make_yoy(data_in)
    assert (data_out["paydex_yoy"] == [-81, 0, 9_985]).all()
    assert len(data_out.columns) == len(df_test_paydex.columns) + 1


def test_paydex_make_group():
    data_in = df_test_paydex.copy()
    data_out = paydex_make_groups(data_in)
    expected = [pd.Interval(0, 15), pd.Interval(15, 30), pd.Interval(90, float("inf"))]
    assert (data_out["paydex_group"] == expected).all()
    assert len(data_out.columns) == len(df_test_paydex.columns) + 1


def test_remove_administrations():
    data_in = df_test_code_naf.copy()
    data_out = remove_administrations(data_in)
    assert (data_out["code_naf"] == ["A", "B", "E"]).all()
    assert len(data_in) - len(data_out) == 2


def test_full_pipeline_sucess():
    data_in = df_test_full.copy()
    data_out = run_pipeline(data_in, pipeline=PIPELINE)
    assert data_out.shape == (3, 11)
    # acoss
    assert (data_out["avg_delta_dette_par_effectif"] == [0, 50, -0.04]).all()
    # paydex yoy
    assert (data_out["paydex_yoy"] == [-81, 0, 9_985]).all()
    # paydex group
    expected = [pd.Interval(0, 15), pd.Interval(15, 30), pd.Interval(90, float("inf"))]
    assert (data_out["paydex_group"] == expected).all()


def test_full_pipeline_missing_input():
    data_in = df_test_full.copy()
    data_in.drop(columns=("code_naf"), inplace=True)
    with pytest.raises(MissingDataError):
        run_pipeline(data_in, pipeline=PIPELINE)
