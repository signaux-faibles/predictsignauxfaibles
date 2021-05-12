# pylint: disable=all
import pandas as pd


def merge_models(model_list: list):
    """
    Builds a single list of predicted probabilities based on several models,
    listed by decreasing order of priority.
    For a given SIRET, if no prediction is found for the first model in the list,
    a prediction for the next model will be considered, etc
    Arguments:
        model_list: list
            A list of pandas DataFrame containing, at least, the following columns: siren, predicted_probability
    """
    merged = model_list.pop()
    for model_id in range(len(model_list)):
        model = model_list.pop()
        merged = pd.merge(
            model,
            merged,
            left_on="siret",
            right_on="siret",
            how="outer",
            suffixes=("_main", "_supp"),
        )
        merged["predicted_probability"] = merged["predicted_probability_main"].fillna(
            merged["predicted_probability_supp"]
        )
        merged = merged[["siret", "predicted_probability"]]
    return merged


def assign_flag(pred: float, t_rouge: float, t_orange: float):
    if pred > t_rouge:
        return "Alerte seuil F1"
    elif pred > t_orange:
        return "Alerte seuil F2"
    return "Pas d'alerte"


def make_alert(preds: pd.DataFrame, t_rouge: float, t_orange: float):
    """
    Generates red/orange/green flags based on two thresholds
    """
    assert "predicted_probability" in preds.columns.tolist()
    preds["alert"] = preds["predicted_probability"].apply(
        lambda x: assign_flag(x, t_rouge, t_orange)
    )

    num_rouge = sum(preds["predicted_probability"] > t_rouge)
    num_orange = sum(preds["predicted_probability"] > t_orange)
    num_orange -= num_rouge
    print(f"{num_rouge} rouge ({round(num_rouge/preds.shape[0] * 100, 2)}%)")
    print(f"{num_orange} orange ({round(num_orange/preds.shape[0] * 100, 2)}%)")

    return preds
