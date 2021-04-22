from collections import namedtuple
import json

import pandas as pd

Redressement = namedtuple("Redressement", ["name", "function", "input", "output"])


def redressement_urssaf_covid(data: pd.DataFrame):
    """
    Règle experte
    """
    # compute delta dette since july 2020
    data["delta_ratio_dette"] = data["ratio_dette"] - data["ratio_dette_july2020"]

    tol = 0.1  # tolerate a change smaller than 10%

    def rule(dataframe, tol):  # pylint: disable=too-many-return-statements
        """
        Redressement rule
        """
        value = dataframe["delta_ratio_dette"]
        group = dataframe["group_final"]  # original model's predicted label
        if value < -tol:
            # after - before < 0 --> l'entreprise va "mieux"
            if group == "vert":
                return "vert"
            if group == "orange":
                return "vert"
            if group == "rouge":
                return "orange"
        if value > tol:
            # after - before > 0 --> l'entreprise va "moins bien"
            if group == "vert":
                return "orange"
            if group == "orange":
                return "rouge"
            if group == "rouge":
                return "rouge"
        return group

    data["group_final_regle_urssaf"] = data.apply(rule, axis=1, tol=tol)

    return data


def redressement_secteur_covid(data: pd.DataFrame):
    """
    Règle experte
    """
    with open("data/secteurs_covid.json", "rb") as file:
        secteurs_covid = json.loads(file.read())

    mapping = {
        secteur: group
        for group in secteurs_covid.keys()
        for secteur in {act["codeActivite"] for act in secteurs_covid[group]}
    }

    data["secteur_covid"] = data.code_ape.apply(lambda ape: mapping.get(ape, "non"))

    def rule(dataframe):  # pylint: disable=too-many-return-statements
        """
        Redressement rule
        """
        secteur_covid = dataframe["secteur_covid"]
        group = dataframe["group_final"]
        if secteur_covid in {"s1", "s1Possible"}:
            if group == "vert":
                return "rouge"
            if group == "orange":
                return "rouge"
            if group == "rouge":
                return "rouge"
        if secteur_covid in {"s1bis", "s1bisPossible", "s2"}:
            if group == "vert":
                return "orange"
            if group == "orange":
                return "rouge"
            if group == "rouge":
                return "rouge"
        return group

    data["group_final_secteur_covid"] = data.apply(rule, axis=1)

    return data
