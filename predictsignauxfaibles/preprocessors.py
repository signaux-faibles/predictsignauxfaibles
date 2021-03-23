from collections import namedtuple
import numpy as np
import pandas as pd


def remove_administrations(data: pd.DataFrame):
    """
    Remove observations with NAF Code 'O' (administrations) or 'P' education
    """
    data = data.query("code_naf != 'O' and code_naf != 'P'")
    return data


def paydex_make_yoy(data: pd.DataFrame):
    """
    TODO: this should be in opensignauxfaibles/reducejs2
    Compute a new column for the dataset containing the year-over-year
    """
    data["paydex_yoy"] = data["paydex_nb_jours"] - data["paydex_nb_jours_past_12"]
    return data


def paydex_make_groups(data: pd.DataFrame):
    """
    Cut paydex into bins
    """
    data["paydex_group"] = pd.cut(
        data["paydex_nb_jours"], bins=(-float("inf"), 15, 30, 60, 90, float("inf"))
    )
    return data


def acoss_make_avg_delta_dette_par_effectif(data: pd.DataFrame):
    """
    Compute the average change in social debt / effectif
    """

    data["dette_par_effectif"] = (
        (data["montant_part_ouvriere"] + data["montant_part_patronale"])
        / data["effectif"]
    ).replace([np.nan, np.inf, -np.inf], 0)

    data["dette_par_effectif_past_3"] = (
        (data["montant_part_ouvriere_past_3"] + data["montant_part_patronale_past_3"])
        / data["effectif"]
    ).replace([np.nan, np.inf, -np.inf], 0)

    data["avg_delta_dette_par_effectif"] = (
        data["dette_par_effectif"] - data["dette_par_effectif_past_3"]
    ) / 3

    columns_to_drop = ["dette_par_effectif", "dette_par_effectif_past_3"]
    data.drop(columns=columns_to_drop, axis=1, inplace=True)
    return data


Preprocessor = namedtuple("Preprocessor", ["function", "input", "output"])

PIPELINE = [
    Preprocessor(remove_administrations, input=["code_naf"], output=None),
    Preprocessor(
        paydex_make_yoy,
        input=["paydex_nb_jours", "paydex_nb_jours_past_12"],
        output=["paydex_yoy"],
    ),
    Preprocessor(
        paydex_make_groups, input=["paydex_nb_jours"], output=["paydex_group"]
    ),
    Preprocessor(
        acoss_make_avg_delta_dette_par_effectif,
        input=[
            "effectif",
            "montant_part_patronale",
            "montant_part_ouvriere",
            "montant_part_patronale_past_3",
            "montant_part_ouvriere_past_3",
        ],
        output=["avg_delta_dette_par_effectif"],
    ),
]
