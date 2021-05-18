from collections import namedtuple

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
