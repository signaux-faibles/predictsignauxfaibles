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

    def rule(dataframe):
        """
        Expert rule to apply
        """
        value = dataframe["delta_dette"]
        group = dataframe["group_final"]
        if value > tol:
            if group == "vert":
                return "orange"
            if group == "orange":
                return "rouge"
            if group == "rouge":
                return "rouge"
        return group

    data["group_final_regle_urssaf"] = data.apply(rule, axis=1)

    return data
