# pylint: disable=missing-function-docstring
import pandas as pd
from predictsignauxfaibles.preprocessors import (
    acoss_make_avg_delta_dette_par_effectif,
    paydex_make_groups,
    paydex_make_yoy,
    remove_administrations,
)

from tests.fake_data.dataframes import (
    df_test_acoss,
    df_test_paydex,
    df_test_code_naf,
)


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
