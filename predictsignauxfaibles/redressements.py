import logging
from collections import namedtuple

import pandas as pd

from predictsignauxfaibles.data import SFDataset

Redressement = namedtuple("Redressement", ["name", "function", "input", "output"])


def redressement_urssaf_covid(data: pd.DataFrame):
    """Executes post-processing "expert rule" decision.

    The expert rule is based on URSSAF debt data.

    Args:
        data: The data to post-process.

    Returns:
        The pd.DataFrame with a new "group_final_regle urssaf" column containing
        post-processed alert levels.

    """
    # Compute changes in social debt as a proportion of average cotisations over the
    # past 12months
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
        """Expert rule to apply."""
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


def prepare_redressement_urssaf_covid(data: pd.DataFrame):
    """Fetches and prepares data needed for `redressement_urssaf_covid`.

    Args:
        data: The data to post-process.

    Returns:
        The pd.DataFrame with new data.

    """
    latest_data = "2021-01-01"
    logging.info(f"Fetching latest ({latest_data}) URSSAF data for redressement")
    latest = (
        SFDataset(
            date_min=latest_data,
            date_max=latest_data[:-2] + "28",
            fields=[
                "siren",
                "siret",
                "montant_part_ouvriere",
                "montant_part_patronale",
                "cotisation_moy12m",
            ],
            sample_size=1_000_000,
        )
        .fetch_data()
        .raise_if_empty()
    )

    logging.info("Fetching july URSSAF data for redressement")
    july2020 = (
        SFDataset(
            date_min="2020-07-01",
            date_max="2020-07-31",
            fields=[
                "siren",
                "siret",
                "montant_part_ouvriere",
                "montant_part_patronale",
            ],
            sample_size=1_000_000,
        )
        .fetch_data()
        .raise_if_empty()
    )

    data.set_index(["siret", "siren"], inplace=True)
    latest.data.set_index(["siret", "siren"], inplace=True)
    july2020.data.set_index(["siret", "siren"], inplace=True)

    data = data.join(july2020.data, rsuffix="_july2020")
    data = data.join(latest.data, lsuffix="_july2020", rsuffix="_latest")

    return data
