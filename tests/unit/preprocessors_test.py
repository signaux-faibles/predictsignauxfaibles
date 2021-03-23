# pylint: disable=missing-function-docstring
import pandas as pd
from predictsignauxfaibles.preprocessors import (
    PIPELINE,
    Preprocessor,
    acoss_make_avg_delta_dette_par_effectif,
    paydex_make_groups,
    paydex_make_yoy,
)

from tests.fake_data.dataframes import df_test_acoss, df_test_paydex


def test_final_pipeline():
    for preprocessor in PIPELINE:
        assert isinstance(preprocessor, Preprocessor)
        assert hasattr(preprocessor, "function")
        assert hasattr(preprocessor, "input")
        assert hasattr(preprocessor, "output")
        assert isinstance(preprocessor.input, list)
        assert isinstance(preprocessor.output, list) or preprocessor.output is None


def test_acoss_make_avg_delta_dette_par_effectif():
    data_in = df_test_acoss.copy()
    data_out = acoss_make_avg_delta_dette_par_effectif(data_in)
    assert (data_out["avg_delta_dette_par_effectif"] == (0, 50, -0.04)).all()
    assert len(data_out.columns) == len(df_test_acoss.columns) + 1


def test_paydex_make_yoy():
    data_in = df_test_paydex.copy()
    data_out = paydex_make_yoy(data_in)
    assert (data_out["paydex_yoy"] == (-81, 0, 9_985)).all()
    assert len(data_out.columns) == len(df_test_paydex.columns) + 1


def test_paydex_make_group():
    data_in = df_test_paydex.copy()
    data_out = paydex_make_groups(data_in)
    expected = [pd.Interval(0, 15), pd.Interval(15, 30), pd.Interval(90, float("inf"))]
    assert (data_out["paydex_group"] == expected).all()
    assert len(data_out.columns) == len(df_test_paydex.columns) + 1
