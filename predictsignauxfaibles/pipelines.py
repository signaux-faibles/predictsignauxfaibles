import logging
from collections import namedtuple
from typing import List

import pandas as pd

from predictsignauxfaibles.preprocessors import (
    Preprocessor,
    acoss_make_avg_delta_dette_par_effectif,
    paydex_make_groups,
    paydex_make_yoy,
    remove_administrations,
)
from predictsignauxfaibles.redressements import (
    Redressement,
    prepare_redressement_urssaf_covid,
    redressement_urssaf_covid,
)


class MissingDataError(Exception):
    """Custom error type for `run_pipeline`."""


def run_pipeline(data: pd.DataFrame, pipeline: List[namedtuple]):
    """Runs a pipeline on a pd.DataFrame.

    The pipeline can contain Preprocessor or Redressement objects (aka "steps").

    Args:
        data: The data to process.
        pipeline: A list of Preprocessor or Redressement objects.
          (see `predictsignauxfaibles.preprocessors` or
          `predictsignauxfaibles.redressements`).

    Returns:
        A processed DataFrame.

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


### Pipelines

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
        "Prepare redressement URSSAF",
        prepare_redressement_urssaf_covid,
        input=["siret", "siren"],
        output=[
            "montant_part_ouvriere_latest",
            "montant_part_patronale_latest",
            "montant_part_ouvriere_july2020",
            "montant_part_patronale_july2020",
            "cotisation_moy12m_latest",
        ],
    ),
    Redressement(
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
]

# This is useful for automatic testing

ALL_PIPELINES = [DEFAULT_PIPELINE, SMALL_PIPELINE, REDRESSEMENTS_PIPELINE]
