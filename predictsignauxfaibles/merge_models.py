# pylint: disable=all
from collections import namedtuple
import logging
from typing import List

import pandas as pd

ModelPreds = namedtuple("ModelPreds", ["name", "predictions"])


def merge_columns(
    df_main: pd.DataFrame, df_supp: pd.DataFrame, col_names: str, supp_model_name: str
):
    suffixes = ("_main", "_supp")
    merged = pd.merge(
        df_main,
        df_supp,
        left_on="siret",
        right_on="siret",
        how="outer",
        suffixes=suffixes,
        indicator=True,
    )
    assert "_merge" in merged.columns, "'_merge' indicator cannot be found in DataFrame"
    merged.loc[merged["_merge"] == "right_only", "which_model"] = supp_model_name

    for col_name in col_names:
        assert (
            col_name + suffixes[0] in merged.columns
        ), f"{col_name+suffixes[0]} cannot be found in DataFrame"
        assert (
            col_name + suffixes[1] in merged.columns
        ), f"{col_name+suffixes[1]} cannot be found in DataFrame"

        merged[col_name] = merged[col_name + suffixes[0]]
        merged.loc[merged["_merge"] == "right_only", col_name] = merged.loc[
            merged["_merge"] == "right_only", col_name + suffixes[1]
        ]

    return merged[["siret", "which_model"] + col_names]


def merge_models(model_list: List[namedtuple], cols_to_merge: List[str]):
    """
    Builds a single list of predicted probabilities based on several models,
    listed by decreasing order of priority.
    For a given SIRET, if no prediction is found for the first model in the list,
    a prediction for the next model will be considered, etc
    Arguments:
        model_list: list
            A list of pandas DataFrame containing, at least, the following columns: siren, predicted_probability
    """
    try:
        main_model = model_list.pop(0)
        merged = main_model.predictions
        merged["which_model"] = main_model.name
    except IndexError:
        logging.error("model_list appears to be empty")
    if model_list == []:
        logging.warning("model_list contains a single model")
        return merged

    for model in model_list:
        merged = merge_columns(
            df_main=merged,
            df_supp=model.predictions,
            col_names=cols_to_merge,
            supp_model_name=model.name,
        )

    return merged
