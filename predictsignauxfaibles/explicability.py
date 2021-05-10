from types import ModuleType

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from predictsignauxfaibles.data import SFDataset


def explain(sf_data: SFDataset, conf: ModuleType):  # pylint: disable=too-many-locals
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
    multi_columns = [
        (group, feat)
        for (group, feats) in conf.FEATURE_GROUPS.items()
        for feat in feats
    ]
    flat_data = pd.DataFrame(sf_data.data[[feat for (group, feat) in multi_columns]])
    data = pd.DataFrame(sf_data.data[[feat for (group, feat) in multi_columns]])
    data.columns = multi_columns
    data.columns = pd.MultiIndex.from_tuples(data.columns, names=["Group", "Feature"])

    # Mapping categorical vairables to their oh-encoded level variables
    cat_mapping = {}
    for (group, feats) in conf.FEATURE_GROUPS.items():
        for feat in feats:
            if feat not in conf.TO_ONEHOT_ENCODE:
                continue
            feat_oh = OneHotEncoder()
            feat_oh.fit(
                flat_data[
                    [
                        feat,
                    ]
                ]
            )
            cat_names = feat_oh.get_feature_names().tolist()
            cat_mapping[(group, feat)] = [feat + "_" + name for name in cat_names]

    cat_to_group = {
        cat_feat: key[0]
        for (key, cat_feats) in cat_mapping.items()
        for cat_feat in cat_feats
    }

    model_pp = conf.MODEL_PIPELINE

    (_, mapper) = model_pp.steps[0]
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
    (_, logreg) = model_pp.steps[1]
    coefs = np.append(logreg.coef_[0], logreg.intercept_)

    feats_contr = np.multiply(coefs, mapped_data)
    norm_feats_contr = (
        feats_contr / np.dot(np.absolute(coefs), np.absolute(mapped_data.T))[:, None]
    )

    multi_columns = []
    for (group, feats) in conf.FEATURE_GROUPS.items():
        for feat in feats:
            if (group, feat) in cat_mapping.keys():
                for cat_feat in cat_mapping[(group, feat)]:
                    multi_columns.append((group, cat_feat))
            else:
                multi_columns.append((group, feat))
    multi_columns.append(("model_offset", "model_offset"))

    micro_expl = pd.DataFrame(
        norm_feats_contr, index=data.index, columns=mapper.transformed_names_
    )
    micro_expl = micro_expl[multi_columns]
    micro_expl.columns = pd.MultiIndex.from_tuples(
        micro_expl.columns, names=["Group", "Feature"]
    )

    macro_expl = micro_expl.apply(lambda x: x.groupby(by="Group").sum(), axis=1)

    sf_data.data["macro_expl"] = macro_expl.apply(lambda x: x.to_dict(), axis=1)
    micro_expl.columns = micro_expl.columns.droplevel()
    sf_data.data["micro_expl"] = micro_expl.apply(lambda x: x.to_dict(), axis=1)

    return sf_data