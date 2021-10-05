from collections import namedtuple
import pandas as pd
import numpy as np

Preprocessor = namedtuple("Preprocessor", ["name", "function", "input", "output"])


def remove_administrations(data: pd.DataFrame) -> pd.DataFrame:
    """Removes observations with NAF Code 'O' (administrations) or 'P' (education).

    Args:
        data: A pd.DataFrame object with a "code_naf" column.

    Returns:
        The filtered pd.DataFrame.

    """
    data = data.query("code_naf != 'O' and code_naf != 'P'")
    return data


def paydex_make_yoy(data: pd.DataFrame) -> pd.DataFrame:
    """Computes a new column for the dataset containing the year-over-year

    Args:
        data: A pd.DataFrame object with "paydex_nb_jours" and
        "paydex_nb_jours_past_12" columns.

    Returns:
        The pd.DataFrame with a new "paydex_yoy" column.

    """
    # TODO: this should be in opensignauxfaibles/reducejs2
    data["paydex_yoy"] = data["paydex_nb_jours"] - data["paydex_nb_jours_past_12"]
    return data


def paydex_make_groups(data: pd.DataFrame) -> pd.DataFrame:
    """Cut paydex into bins.

    Args:
        data: A pd.DataFrame object.

    Returns:
        The pd.DataFrame with a new "paydex_group" column.

    """
    data["paydex_group"] = pd.cut(
        data["paydex_nb_jours"], bins=(-float("inf"), 0, 15, 30, 60, 90, float("inf"))
    )
    return data


def acoss_make_avg_delta_dette_par_effectif(data: pd.DataFrame) -> pd.DataFrame:
    """Compute the average change in social debt / effectif.

    Args:
        data: A pd.DataFrame object.

    Returns:
        The pd.DataFrame with a new `avg_delta_dette_par_effectif` column.

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
