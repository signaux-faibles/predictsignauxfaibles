from collections import namedtuple
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
]
