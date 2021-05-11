from types import ModuleType

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from predictsignauxfaibles.data import SFDataset
from predictsignauxfaibles.utils import sigmoid


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
    # Creating a flat version of our group-feature hierarchy
    flat_data = pd.DataFrame(sf_data.data[[feat for (group, feat) in multi_columns]])

    # Creating a multi-indexed-columns version of our dataset
    # where features are listed in the same order as in conf.FEATURE_GROUPS
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

    # The reverse mapping with help us as well
    cat_to_group = {
        cat_feat: key[0]
        for (key, cat_feats) in cat_mapping.items()
        for cat_feat in cat_feats
    }

    # Finally, let's create a list of tuples that we'll use
    # to multi-index our columns...
    multi_columns = []
    for (group, feats) in conf.FEATURE_GROUPS.items():
        for feat in feats:
            if (group, feat) in cat_mapping.keys():
                for cat_feat in cat_mapping[(group, feat)]:
                    multi_columns.append((group, cat_feat))
            else:
                multi_columns.append((group, feat))
    multi_columns.append(("model_offset", "model_offset"))

    ## COLUMNS NAMES REGISTRATION & MAPPING
    # Our model's mapper uses a OneHotEncoder to generate binary variables
    # from categorical ones. It creates new columns that we must register and
    # map to our groups in order to compute the total contribution of each
    # group to our risk prediction
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

    ## COMPUTING CONTRIBUTIONS FROM OUR LOGISTIC REGRESSION
    (_, logreg) = model_pp.steps[1]
    coefs = np.append(logreg.coef_[0], logreg.intercept_)

    ## ABSOLUTE CONTRIBUTIONS are used for radar plots
    feats_contr = np.multiply(coefs, mapped_data)
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
    macro_radar = micro_radar.apply(
        lambda x: sigmoid(x.groupby(by="Group").sum()), axis=1
    )
    sf_data.data["macro_radar"] = macro_radar.apply(lambda x: x.to_dict(), axis=1)

    ## RELATIVE CONTRIBUTIONS are used for explanations
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
    macro_expl = micro_expl.apply(lambda x: x.groupby(by="Group").sum(), axis=1)

    # Aggregating contributions at the group level
    sf_data.data["macro_expl"] = macro_expl.apply(lambda x: x.to_dict(), axis=1)

    # Flatten micro_expl and store the contribution of each feature
    micro_expl.columns = micro_expl.columns.droplevel()
    sf_data.data["micro_expl"] = micro_expl.apply(lambda x: x.to_dict(), axis=1)

    return sf_data
