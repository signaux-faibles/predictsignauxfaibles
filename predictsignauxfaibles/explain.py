from types import ModuleType

import numpy as np
import pandas as pd

from predictsignauxfaibles.data import SFDataset
from predictsignauxfaibles.utils import (
    sigmoid,
    make_multi_columns,
)


def list_concerning_contributions(entry: pd.Series, thr: float = 0.07):
    """
    From a record containing dot-products of each feature with the weight vector,
    computes a list of features that contribute to a high risk of failure
    """
    masked = entry.where(entry >= 4 * thr)
    masked.sort_values(ascending=False, inplace=True)
    return masked[~masked.isnull()].index.tolist()


def list_reassuring_contributions(entry: pd.Series, thr: float = 0.07):
    """
    From a record containing dot-products of each feature with the weight vector,
    computes a list of features that contribute to a low risk of failure
    """
    masked = entry.where(entry <= -4 * thr)
    masked.sort_values(ascending=True, inplace=True)
    return masked[~masked.isnull()].index.tolist()


def group_retards_paiement(
    micro_df: pd.DataFrame, macro_df: pd.DataFrame, group_feat_tuples: list
):
    """
    Replaces micro contributions for group "retards_paiement"
    with aggregated contribution
    """
    remapped_df = micro_df.copy()
    remapped_df[[("retards_paiement", "retards_paiement")]] = macro_df[
        ["retards_paiement"]
    ]
    remapped_df.drop(
        columns=[
            (GROUP, FEAT)
            for (GROUP, FEAT) in group_feat_tuples
            if GROUP == "retards_paiement"
        ],
        inplace=True,
    )
    return remapped_df


def explain(
    sf_data: SFDataset, conf: ModuleType, thresh_micro: float = 0.07
):  # pylint: disable=too-many-locals
    """
    Provides the relative contribution of each features to the risk score,
    as well as relative contributions for each group of features,
    as defined in a the model configuration file.
    This relative contribution of feature $i$ to the score
    for company $s$, for reglog parameter $\beta$ is defined by:
    expl_i(s) = frac{beta_i * s_i}{|beta_0| + sum_{j=1}^{N}{|{beta_1}_j||s_j|}}
    expl_0(s) = frac{beta_0}{|beta_0| + sum_{j=1}^{N}{|{beta_1}_j||s_j|}}
    Arguments:
        sf_data: SFDataset
            A SFDataset containing predictions produced by a logistic regression
        conf: ModuleType
            The model configuration file used for predictions
    """

    ## COLUMNS NAMES REGISTRATION & MAPPING
    ## ====================================
    raveled_data = sf_data.data.copy()

    # Create a list of tuples that we'll use to multi-index our columns
    # including categorical features
    multi_columns_full = make_multi_columns(raveled_data, conf)
    multi_columns_full.append(("model_offset", "model_offset"))

    # Our model's mapper uses a OneHotEncoder to generate binary variables
    # from categorical ones. It creates new columns that we must register and
    # map to our groups in order to compute the total contribution of each
    # group to our risk prediction
    model_pp = conf.MODEL_PIPELINE
    (_, mapper) = model_pp.steps[0]
    unraveled_data = mapper.transform(sf_data.data)
    unraveled_data = np.hstack((unraveled_data, np.ones((len(sf_data), 1))))

    ## COMPUTING CONTRIBUTIONS FROM OUR LOGISTIC REGRESSION
    ## ====================================================
    (_, logreg) = model_pp.steps[1]
    coefs = np.append(logreg.coef_[0], logreg.intercept_)

    ## ABSOLUTE CONTRIBUTIONS are used to select the features
    ## that significantly contribute to our risk score
    ## ------------------------------------------------------
    feats_contr = np.multiply(coefs, unraveled_data)

    micro_prod = pd.DataFrame(
        feats_contr,
        columns=multi_columns_full,
        index=sf_data.data.index,
    ).drop([("model_offset", "model_offset")], axis=1, inplace=False)
    micro_prod.columns = pd.MultiIndex.from_tuples(
        micro_prod.columns, names=["Group", "Feature"]
    )

    macro_prod = micro_prod.groupby(by="Group", axis=1).sum()

    micro_prod = group_retards_paiement(micro_prod, macro_prod, multi_columns_full)

    micro_select_concerning = micro_prod.apply(
        lambda s: list_concerning_contributions(s, thr=thresh_micro), axis=1
    )
    micro_select_reassuring = micro_prod.apply(
        lambda s: list_reassuring_contributions(s, thr=thresh_micro), axis=1
    )
    micro_select = micro_select_concerning.to_frame(name="select_concerning").join(
        micro_select_reassuring.to_frame(name="select_reassuring")
    )
    sf_data.data["expl_selection"] = micro_select.apply(lambda s: s.to_dict(), axis=1)

    ## OFFSET ABSOLUTE CONTRIBUTIONS are used for radar plots
    ## and to select contributive micro-variables to show in front-end
    ## ---------------------------------------------------------------
    offset_feats_contr = feats_contr - (logreg.intercept_ / coefs.size)

    micro_radar = pd.DataFrame(
        offset_feats_contr,
        columns=multi_columns_full,
        index=sf_data.data.index,
    ).drop([("model_offset", "model_offset")], axis=1, inplace=False)
    micro_radar.columns = pd.MultiIndex.from_tuples(
        micro_radar.columns, names=["Group", "Feature"]
    )

    ## Aggregating contributions at the group level
    # and applying sigmoid provides the radar score for each group
    macro_radar = micro_radar.groupby(by="Group", axis=1).sum().applymap(sigmoid)

    sf_data.data["macro_radar"] = macro_radar.apply(lambda x: x.to_dict(), axis=1)

    ## RELATIVE CONTRIBUTIONS are used to provide explanations
    ## as full-text on the front-end
    ## -------------------------------------------------------
    norm_feats_contr = (
        feats_contr / np.dot(np.absolute(coefs), np.absolute(unraveled_data.T))[:, None]
    )

    micro_expl = pd.DataFrame(
        norm_feats_contr,
        columns=multi_columns_full,
        index=sf_data.data.index,
    )
    micro_expl.columns = pd.MultiIndex.from_tuples(
        micro_expl.columns, names=["Group", "Feature"]
    )

    macro_expl = micro_expl.groupby(by="Group", axis=1).sum()
    micro_expl = group_retards_paiement(micro_expl, macro_expl, multi_columns_full)

    # Aggregating contributions at the group level
    sf_data.data["macro_expl"] = macro_expl.apply(lambda x: x.to_dict(), axis=1)

    # Flatten micro_expl and store the contribution of each feature
    micro_expl.columns = micro_expl.columns.droplevel()
    sf_data.data["micro_expl"] = micro_expl.apply(lambda x: x.to_dict(), axis=1)

    return sf_data
