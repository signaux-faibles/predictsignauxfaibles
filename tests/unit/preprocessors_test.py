# pylint: disable=missing-function-docstring
from predictsignauxfaibles.preprocessors import (
    PIPELINE,
    Preprocessor,
    acoss_make_avg_delta_dette_par_effectif,
)

from tests.fake_data.dataframes import df_test_acoss


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
