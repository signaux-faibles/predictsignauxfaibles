from collections import namedtuple
import logging
from typing import List
import pandas as pd

from predictsignauxfaibles.preprocessors import (
    Preprocessor,
    remove_administrations,
    paydex_make_groups,
    paydex_make_yoy,
    acoss_make_avg_delta_dette_par_effectif,
)


class MissingDataError(Exception):
    """
    Custom error type for `run_pipeline`
    """


def run_pipeline(data: pd.DataFrame, pipeline: List[namedtuple]):
    """
    Run a pipeline of Preprocessor objects on a dataframe
    Args:
        pipeline: a list of Preprocessor objects (see predictsignauxfaibles.preprocessors)
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


# Pipelines

DEFAULT_PIPELINE = [
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

SMALL_PIPELINE = [
    Preprocessor(
        "Remove Administrations",
        function=remove_administrations,
        input=["code_naf"],
        output=None,
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

# This is useful for automatic testing
ALL_PIPELINES = [DEFAULT_PIPELINE, SMALL_PIPELINE]
