from collections import namedtuple
import logging
from typing import List

import numpy as np
import pandas as pd

Preprocessor = namedtuple("Preprocessor", ["name", "function", "input", "output"])


class MissingDataError(Exception):
    """
    Custom error type for `run_pipeline`
    """


def run_pipeline(data: pd.DataFrame, pipeline: List[namedtuple]):
    """
    Run a pipeline of Preprocessor objects on a dataframe
    """
    logging.info("Checking that input columns are all there.")
    for preprocessor in pipeline:
        if not set(preprocessor.input).issubset(data.columns):
            missing_cols = set(preprocessor.input) - set(data.columns)
            error_message = (
                f"Missing variables {missing_cols} in order to run {preprocessor.name}."
            )
            raise MissingDataError(error_message)

    logging.info("Running pipeline on data.")
    data = data.copy()
    for i, preprocessor in enumerate(pipeline):
        logging.info(f"STEP {i+1}: {preprocessor.name}")
        data = preprocessor.function(data)
        if preprocessor.output is None:
            continue
        if not set(preprocessor.output).issubset(data.columns):
            missing_output_cols = set(preprocessor.output) - set(data.columns)
            warning_message = f"STEP {i+1}: function {preprocessor.function.__name__} \
did not produce expected output {missing_output_cols}"
            logging.warning(warning_message)
            continue

    return data


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
        data["paydex_nb_jours"], bins=(-float("inf"), 0, 15, 30, 60, 90, float("inf"))
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


PIPELINE = [
    Preprocessor(
        "Remove Administrations",
        function=remove_administrations,
        input=["code_naf"],
        output=None,
    ),
    Preprocessor(
        "Make `paydex_yoy`",
        function=paydex_make_yoy,
        input=["paydex_nb_jours", "paydex_nb_jours_past_12"],
        output=["paydex_yoy"],
    ),
    Preprocessor(
        "Make `paydex_group`",
        function=paydex_make_groups,
        input=["paydex_nb_jours"],
        output=["paydex_group"],
    ),
    Preprocessor(
        "Make `avg_delta_dette_par_effectif`",
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
