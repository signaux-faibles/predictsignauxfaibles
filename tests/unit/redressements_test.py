# pylint: disable=missing-function-docstring
import pandas as pd
from predictsignauxfaibles.redressements import redressement_urssaf_covid

from tests.fake_data.dataframes import df_test_redressement_urssaf


def test_redressement_urssaf_covid():
    data_in = df_test_redressement_urssaf.copy()
    data_out = redressement_urssaf_covid(data_in)
    assert (data_out["group_final_regle_urssaf"] == ["orange", "vert", "rouge"]).all()
