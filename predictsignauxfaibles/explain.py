from types import ModuleType

import numpy as np
import pandas as pd

from predictsignauxfaibles.data import SFDataset
from predictsignauxfaibles.utils import (
    sigmoid,
    map_cat_feature_to_categories,
    make_multi_columns,
)


def list_concerning_contributions(entry: pd.Series, thr=0.07):
    """
    From a record containing dot-products of each feature with the weight vector,
    computes a list of features that contribute to a high risk of failure
    """
    masked = entry.where(entry >= 4 * thr)
    masked.sort_values(ascending=False, inplace=True)
    return masked[~masked.isnull()].index.tolist()


def list_reassuring_contributions(entry: pd.Series, thr=0.07):
    """
    From a record containing dot-products of each feature with the weight vector,
    computes a list of features that contribute to a low risk of failure
    """
    masked = entry.where(entry <= -4 * thr)
    masked.sort_values(ascending=True, inplace=True)
    return masked[~masked.isnull()].index.tolist()


def group_retards_paiement(
    micro_df: pd.DataFrame, macro_df: pd.DataFrame, feat_groups: dict
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
            ("retards_paiement", FEAT) for FEAT in feat_groups["retards_paiement"]
        ],
        inplace=True,
    )
    return remapped_df


def explain(
    sf_data: SFDataset, conf: ModuleType, thresh_micro: float = 0.07
):  # pylint: disable=too-many-statements, too-many-locals
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

    # Creating a multi-indexed-columns version of our dataset
    # where features are listed in the same order as in conf.FEATURE_GROUPS
    multi_columns = [
        (group, feat)
        for (group, feats) in conf.FEATURE_GROUPS.items()
        for feat in feats
    ]
    data = pd.DataFrame(sf_data.data[[feat for (group, feat) in multi_columns]])
    data.columns = pd.MultiIndex.from_tuples(multi_columns, names=["Group", "Feature"])

    # Mapping categorical vairables to their oh-encoded level variables
    cat_mapping = map_cat_feature_to_categories(data, conf)

    # The reverse mapping will help us as well
    cat_to_group = {
        cat_feat: key[0]
        for (key, cat_feats) in cat_mapping.items()
        for cat_feat in cat_feats
    }

    # Create a list of tuples that we'll use to multi-index our columns
    # including categorical features
    multi_columns = make_multi_columns(data, conf)
    multi_columns.append(("model_offset", "model_offset"))

    ## COLUMNS NAMES REGISTRATION & MAPPING
    ## ====================================
    # Our model's mapper uses a OneHotEncoder to generate binary variables
    # from categorical ones. It creates new columns that we must register and
    # map to our groups in order to compute the total contribution of each
    # group to our risk prediction
    model_pp = conf.MODEL_PIPELINE

    (_, mapper) = model_pp.steps[0]
    flat_data = data.copy()
    flat_data.columns = [feat for (group, feat) in data.columns]
    mapped_data = mapper.transform(flat_data)
    mapped_data = np.hstack((mapped_data, np.ones((len(sf_data), 1))))

    # Correctly naming each column of transformed_names_
    mapper.transformed_names_[: -len(conf.TO_SCALE)] = [
        (cat_to_group[cat_feat], cat_feat)
        for cat_feat in mapper.transformed_names_[: -len(conf.TO_SCALE)]
    ]
    mapper.transformed_names_[-len(conf.TO_SCALE) : -1] = [
        (group, feat)
        for (group, feats) in conf.FEATURE_GROUPS.items()
        for feat in feats
        if feat in conf.TO_SCALE
    ]
    mapper.transformed_names_[-1] = ("model_offset", "model_offset")

    ## COMPUTING CONTRIBUTIONS FROM OUR LOGISTIC REGRESSION
    ## ====================================================
    (_, logreg) = model_pp.steps[1]
    coefs = np.append(logreg.coef_[0], logreg.intercept_)

    ## ABSOLUTE CONTRIBUTIONS are used to select the features
    ## that significantly contribute to our risk score
    ## ------------------------------------------------------
    feats_contr = np.multiply(coefs, mapped_data)
    micro_prod = pd.DataFrame(
        feats_contr, index=data.index, columns=mapper.transformed_names_
    )
    micro_prod = micro_prod[multi_columns].drop(
        [("model_offset", "model_offset")], axis=1, inplace=False
    )
    micro_prod.columns = pd.MultiIndex.from_tuples(
        micro_prod.columns, names=["Group", "Feature"]
    )

    macro_prod = micro_prod.groupby(by="Group", axis=1).sum()

    micro_prod = group_retards_paiement(micro_prod, macro_prod, conf.FEATURE_GROUPS)

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
    offset_feats_contr = feats_contr - logreg.intercept_ / coefs.size
    micro_radar = pd.DataFrame(
        offset_feats_contr, index=data.index, columns=mapper.transformed_names_
    )
    micro_radar = micro_radar[multi_columns]
    micro_radar.columns = pd.MultiIndex.from_tuples(
        micro_radar.columns, names=["Group", "Feature"]
    )
    ## Aggregating contributions at the group level
    # and applying sigmoid provides the radar score for each group
    macro_radar = micro_radar.groupby(by="Group", axis=1).sum().apply(sigmoid)
    macro_radar.drop(columns=["model_offset"], inplace=True)

    sf_data.data["macro_radar"] = macro_radar.apply(lambda x: x.to_dict(), axis=1)

    ## RELATIVE CONTRIBUTIONS are used to provide explanations
    ## as full-text on the front-end
    ## -------------------------------------------------------
    norm_feats_contr = (
        feats_contr / np.dot(np.absolute(coefs), np.absolute(mapped_data.T))[:, None]
    )

    micro_expl = pd.DataFrame(
        norm_feats_contr, index=data.index, columns=mapper.transformed_names_
    )
    micro_expl = micro_expl[multi_columns]
    micro_expl.columns = pd.MultiIndex.from_tuples(
        micro_expl.columns, names=["Group", "Feature"]
    )
    macro_expl = micro_expl.groupby(by="Group", axis=1).sum()
    micro_expl = group_retards_paiement(micro_expl, macro_expl, conf.FEATURE_GROUPS)

    # Aggregating contributions at the group level
    sf_data.data["macro_expl"] = macro_expl.apply(lambda x: x.to_dict(), axis=1)

    # Flatten micro_expl and store the contribution of each feature
    micro_expl.columns = micro_expl.columns.droplevel()
    sf_data.data["micro_expl"] = micro_expl.apply(lambda x: x.to_dict(), axis=1)

    return sf_data
