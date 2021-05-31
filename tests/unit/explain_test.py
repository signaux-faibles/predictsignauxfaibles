import pandas as pd

from predictsignauxfaibles.explain import (
    list_concerning_contributions,
    list_reassuring_contributions,
    group_retards_paiement,
)

from tests.fake_data.dataframes import (
    df_test_explain_prod,
)


def test_list_concerning_contributions():
    """
    Check whether concerning contributions are correctly computed
    and returned in the right order
    """
    data_in = df_test_explain_prod.copy()
    concerning_contributions = data_in.apply(
        lambda s: list_concerning_contributions(s, thr=0.1), axis=1
    )

    assert concerning_contributions[0] == ["paydex_yoy"]
    assert concerning_contributions[1] == ["ratio_dette", "paydex_yoy"]
    assert concerning_contributions[2] == []


def test_list_reassuring_contributions():
    """
    Check whether reassuring contributions are correctly computed
    and returned in the right order
    """
    data_in = df_test_explain_prod.copy()
    reassuring_contributions = data_in.apply(
        lambda s: list_reassuring_contributions(s, thr=0.1), axis=1
    )

    assert reassuring_contributions[0] == []
    assert reassuring_contributions[1] == []
    assert reassuring_contributions[2] == ["ratio_dette", "paydex_yoy"]


def test_group_retards_paiement():
    """
    group_retards_paiement should replace in df_micro all columns
    that belong to group "retards_paiement" and substitute them
    with column "retards_paiement" containing the sum of
    individual contributions
    """
    df_micro = df_test_explain_prod.copy()
    multi_columns = [
        ("sante_financiere", "equilibre_financier"),
        ("retards_paiement", "paydex_yoy"),
        ("retards_paiement", "paydex_other_feat"),
        ("dette_urssaf", "ratio_dette"),
    ]
    df_micro.columns = pd.MultiIndex.from_tuples(
        multi_columns, names=["Group", "Feature"]
    )

    df_macro = df_micro.groupby(by="Group", axis=1).sum()
    df_remapped = group_retards_paiement(df_micro, df_macro, multi_columns)

    assert ("retards_paiement", "retards_paiement") in df_remapped.columns
    assert ("retards_paiement", "paydex_yoy") not in df_remapped.columns
    assert ("retards_paiement", "paydex_other_feat") not in df_remapped.columns

    assert (
        df_remapped.loc[0][("retards_paiement", "retards_paiement")]
        == df_micro.loc[0][("retards_paiement", "paydex_yoy")]
        + df_micro.loc[0][("retards_paiement", "paydex_other_feat")]
    )
