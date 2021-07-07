from collections import namedtuple
from typing import List, Callable
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from predictsignauxfaibles.data import SFDataset

Preprocessor = namedtuple("Preprocessor", ["name", "function", "input", "output"])


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
    Output column : 'paydex_yoy'
    """
    data["paydex_yoy"] = data["paydex_nb_jours"] - data["paydex_nb_jours_past_12"]
    return data


def paydex_make_groups(data: pd.DataFrame):
    """
    Cut paydex into bins
    Output column : 'paydex_group'
    """
    data["paydex_group"] = pd.cut(
        data["paydex_nb_jours"], bins=(-float("inf"), 0, 15, 30, 60, 90, float("inf"))
    )
    return data


def acoss_make_avg_delta_dette_par_effectif(data: pd.DataFrame):
    """
    Compute the average change in social debt / effectif
    Output column : 'avg_delta_dette_par_effectif'
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


def apply_log(number: float) -> float:
    """
    Apply the transformation number -> log(number + 1)
    """
    return math.log(number + 1)


def apply_sqrt(number: float) -> float:
    """
    Apply the transformation number -> sqrt(number)
    """
    return math.sqrt(number)


def get_featuring(
    features: List[str], funcs: List[Callable[[float], float]]
) -> List[dict]:
    """
    For each feature in 'features' taken separately, provide the transformation among
    'funcs' to apply to improve the prediction performance with a LogisticRegression
    (outcome = feature). For significant results, this should be bootstrap but it
    is enough to test the transformations in the model used in the project in a
    second step.
    """
    dataset = SFDataset(
        date_min="2015-01-01",
        date_max="2020-06-30",
        fields=["outcome"] + features,
        sample_size=1000,
    )
    dataset.fetch_data()

    res = [
        get_featuring_unitary(dataset.data, feat, f) for feat in features for f in funcs
    ]

    res_as_df = pd.DataFrame(res)
    res_as_df = res_as_df[res_as_df["is_relevant"]]

    return res_as_df.sort_values("score_after", ascending=False).drop_duplicates(
        ["feature"]
    )


def get_featuring_unitary(
    data: pd.DataFrame, feat: str, func: Callable[[float], float]
) -> dict:
    """
    Apply the transformation 'func' to the feature 'feat', build a LogisticRegression
    (outcome = feature) with and without the transformation and determine if it was
    relevant.
    """
    data = data[["outcome", feat]].copy()

    # handle missing value
    data.dropna(inplace=True)

    if len(data[feat]) == 0:
        return {
            "feature": feat,
            "func": func.__name__,
            "score_before": np.nan,
            "score_after": np.nan,
            "is_relevant": False,
        }

    # handle non-numeric
    if not all([type(x) in [int, float] for x in data[feat]]):
        return {
            "feature": feat,
            "func": func.__name__,
            "score_before": np.nan,
            "score_after": np.nan,
            "is_relevant": False,
        }

    # handle negative values
    if any(data[feat] < 0):
        return {
            "feature": feat,
            "func": func.__name__,
            "score_before": np.nan,
            "score_after": np.nan,
            "is_relevant": False,
        }

    response = data.outcome

    # handle singular class in the response
    if len(response.unique()) == 1:
        return {
            "feature": feat,
            "func": func.__name__,
            "score_before": np.nan,
            "score_after": np.nan,
            "is_relevant": False,
        }
    feat_values = np.array(data[feat]).reshape(-1, 1)

    # Logistic without featuring
    model = LogisticRegression()
    model.fit(feat_values, response)
    score_before = model.score(feat_values, response)

    # Logistic with featuring
    feat_values = np.array(list(map(func, feat_values))).reshape(-1, 1)
    model = LogisticRegression()
    model.fit(feat_values, response)
    score_after = model.score(feat_values, response)

    return {
        "feature": feat,
        "func": func.__name__,
        "score_before": score_before,
        "score_after": score_after,
        "is_relevant": score_after > score_before,
    }
