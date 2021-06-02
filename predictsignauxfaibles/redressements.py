from collections import namedtuple

import pandas as pd

Redressement = namedtuple("Redressement", ["name", "function", "input", "output"])


def redressement_urssaf_covid(data: pd.DataFrame):
    """
    Règle experte
    """

    # compute change in social debt as a proportion of average cotisations over the past 12months
    data["montant_part_ouvriere_latest"].fillna(0, inplace=True)
    data["montant_part_patronale_latest"].fillna(0, inplace=True)
    data["montant_part_ouvriere_july2020"].fillna(0, inplace=True)
    data["montant_part_patronale_july2020"].fillna(0, inplace=True)
    data["dette_sociale_july2020"] = (
        data.montant_part_ouvriere_july2020 + data.montant_part_patronale_july2020
    )
    data["dette_sociale_latest"] = (
        data.montant_part_ouvriere_latest + data.montant_part_patronale_latest
    )

    data["delta_dette"] = (
        data.dette_sociale_latest - data.dette_sociale_july2020
    ) / data.cotisation_moy12m_latest

    tol = 0.2  # tolerate a change smaller than 20%

    assert "alert" in data.columns, "'alert' was not found in data.columns"
    data["alert_pre_redressements"] = data["alert"]

    def rule(dataframe):
        """
        Expert rule to apply
        """
        value = dataframe["delta_dette"]
        group = dataframe["alert_pre_redressements"]
        if value > tol:
            if group == "Pas d'alerte":
                return "Alerte seuil F2"
            if group == "Alerte seuil F2":
                return "Alerte seuil F1"
            if group == "Alerte seuil F1":
                return "Alerte seuil F1"
        return group

    data["alert"] = data.apply(rule, axis=1)

    return data


ALL_REDRESSEMENTS = {
    "urssaf_2020": Redressement(
        "Redressement URSSAF evolution dette Juillet 2020",
        redressement_urssaf_covid,
        input=[
            "montant_part_ouvriere_latest",
            "montant_part_patronale_latest",
            "montant_part_ouvriere_july2020",
            "montant_part_patronale_july2020",
            "cotisation_moy12m_latest",
            "group_final",
        ],
        output=["group_final_regle_urssaf"],
    ),
}
