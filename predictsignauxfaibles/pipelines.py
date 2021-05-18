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

from predictsignauxfaibles.redressements import (
    Redressement,
    redressement_urssaf_covid,
)


class MissingDataError(Exception):
    """
    Custom error type for `run_pipeline`
    """


def run_pipeline(data: pd.DataFrame, pipeline: List[namedtuple]):
    """
    Run a pipeline of Preprocessor or Redressement objects (aka "steps") on a dataframe
    Args:
        pipeline: a list of Preprocessor or Redressement objects
          (see predictsignauxfaibles.preprocessors or predictsignauxfaibles.redressements)
    """
    logging.info("Checking that input columns are all there.")
    for step in pipeline:
        if not set(step.input).issubset(data.columns):
            missing_cols = set(step.input) - set(data.columns)
            error_message = (
                f"Missing variables {missing_cols} in order to run {step.name}."
            )
            raise MissingDataError(error_message)

    logging.info("Running pipeline on data.")
    data = data.copy()
    for i, step in enumerate(pipeline):
        logging.info(f"STEP {i+1}: {step.name}")
        data = step.function(data)
        if step.output is None:
            continue
        if not set(step.output).issubset(data.columns):
            missing_output_cols = set(step.output) - set(data.columns)
            warning_message = f"STEP {i+1}: function {step.function.__name__} \
did not produce expected output {missing_output_cols}"
            logging.warning(warning_message)
            continue

    return data


# Pipelines

# Preprocessors

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

# Redressements

REDRESSEMENTS_PIPELINE = [
    Redressement(
        "Redressement URSSAF evolution dette Juillet 2020",
        redressement_urssaf_covid,
        input=["ratio_dette", "ratio_dette_july2020", "group_final"],
        output=["group_final_regle_urssaf"],
    ),
]

# This is useful for automatic testing
ALL_PIPELINES = [DEFAULT_PIPELINE, SMALL_PIPELINE, REDRESSEMENTS_PIPELINE]
