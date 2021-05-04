import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys
import logging
from types import ModuleType

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from predictsignauxfaibles.config import OUTPUT_FOLDER, IGNORE_NA
from predictsignauxfaibles.pipelines import run_pipeline
from predictsignauxfaibles.utils import set_if_not_none
from predictsignauxfaibles.data import SFDataset

sys.path.append("../")

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level="INFO")

# Mute logs from sklean_pandas
logging.getLogger("sklearn_pandas").setLevel(logging.WARNING)

ARGS_TO_ATTRS = {
    "train_spl_size": ("train", "sample_size"),
    "test_spl_size": ("test", "sample_size"),
    "predict_spl_size": ("predict", "sample_size"),
    "train_proportion_positive_class": ("train", "proportion_positive_class"),
    "train_from": ("train", "date_min"),
    "train_to": ("train", "date_max"),
    "test_from": ("test", "date_min"),
    "test_to": ("test", "date_max"),
    "predict_on": ("predict", "date_min"),
}


def load_conf(args: argparse.Namespace):  # pylint: disable=redefined-outer-name
    """
    Loads a model configuration from a argparse.Namespace
    containing a model name
    Args:
        conf_args: a argparse.Namespace object containing attributes
            model_name
    """
    conf_filepath = Path("models") / args.model_name / "model_conf.py"
    if not conf_filepath.exists():
        raise ValueError(f"{conf_filepath} does not exist")

    spec = importlib.util.spec_from_file_location(
        f"models.{args.model_name}.model_conf", conf_filepath
    )
    model_conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_conf)  # pylint: disable=wrong-import-position
    return model_conf


def get_train_test_predict_datasets(args_ns: argparse.Namespace, conf: ModuleType):
    """
    Configures train, test and predict dataset from user-provided options when pertaining.
    Args:
      args_ns: a argpast Namespace object containing
      the custom attributes to be used for training, testing and/or prediction
      conf: the model configuration module containing default parameters,
      to be overwritten by the content of args_ns
    Returns:
      train: a SFDataset object containing the training data
      test: a SFDataset object containing the test data
      predict: a SFDataset object containing the data to predict on
    """
    datasets = {
        "train": conf.TRAIN_DATASET,
        "test": conf.TEST_DATASET,
        "predict": conf.PREDICT_DATASET,
    }

    args_dict = vars(args_ns)
    for (arg, dest) in ARGS_TO_ATTRS.items():
        set_if_not_none(datasets[dest[0]], dest[1], args_dict[arg])

    if args_ns.predict_on is not None:
        datasets["predict"].date_max = args_ns.predict_on[:-2] + "28"

    return datasets["train"], datasets["test"], datasets["predict"]


def make_stats(train: SFDataset, test: SFDataset, predict: SFDataset):
    """
    Initialises a dictionary containing model run stats for logging purposes
    Args:
      train: a SFDataset object containing the training data
      test: a SFDataset object containing the test data
      predict: a SFDataset object containing the data to predict on
    Returns:
      stats: a dictionnary containing model run parameters
    """
    stats = {}
    datasets = {"train": train, "test": test, "predict": predict}

    for (arg, dest) in ARGS_TO_ATTRS.items():
        stats[arg] = getattr(datasets[dest[0]], dest[1])

    return stats


def evaluate(
    model: Pipeline, dataset: SFDataset, beta: float
):  # To be turned into a SFModel method when refactoring models
    """
    Returns evaluation metrics of model evaluated on df
    Args:
        model: a sklearn-like model with a predict method
        df: dataset
    """
    balanced_accuracy = balanced_accuracy_score(
        dataset.data["outcome"], model.predict(dataset.data)
    )
    fbeta = fbeta_score(dataset.data["outcome"], model.predict(dataset.data), beta=beta)
    return {"balanced_accuracy": balanced_accuracy, "fbeta": fbeta}


def explain(sf_data: SFDataset, conf: ModuleType): #pylint: disable=too-many-locals
    """
    Provides the relative contribution to the score intensity
    (ie, the term within the sigmoid, which is any real number
    that will later be brought to [0,1] by the sigmoid)
    Arguments:
        lr: LogisticRegression
            The LogReg that has been used to train the model
        etab: pd.Series
            A Series containing all features of the etablissement that were input to the LogReg
        feat_groups: dict
            A dictionnary mapping macro features to lists of features.
            Each key is a macro variable name, associated to a group of model features.
            The contribution of all features in the value list are considered together.
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

    expl = pd.DataFrame(
        norm_feats_contr, index=data.index, columns=mapper.transformed_names_
    )
    expl = expl[multi_columns]
    expl.columns = pd.MultiIndex.from_tuples(expl.columns, names=["Group", "Feature"])

    group_expls = expl.apply(lambda x: x.groupby(by="Group").sum(), axis=1)
    group_expls.columns = [("expl", group) for group in group_expls.columns]
    group_expls.columns = pd.MultiIndex.from_tuples(
        group_expls.columns, names=["expl", "feat_group"]
    )
    expl.drop([("model_offset", "model_offset")], axis=1, inplace=True)
    expl = expl.merge(group_expls, left_index=True, right_index=True)
    expl.columns = pd.MultiIndex.from_tuples(expl.columns, names=["group", "feat"])

    return expl


def run(
    args,
):  # pylint: disable=too-many-statements,too-many-locals
    """
    Run model
    """
    conf = load_conf(args)
    logging.info(
        f"Running Model {conf.MODEL_ID} (commit {conf.MODEL_GIT_SHA}) ENV={conf.ENV}"
    )
    model_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    train, test, predict = get_train_test_predict_datasets(args, conf)
    model_stats = make_stats(train, test, predict)
    model_stats["run_on"] = model_id

    step = "[TRAIN]"
    model_stats["train"] = {}
    logging.info(f"{step} - Fetching train set ({train.sample_size} samples)")
    train.fetch_data().raise_if_empty()

    logging.info(f"{step} - Data preprocessing")
    train.replace_missing_data().remove_na(ignore=IGNORE_NA)
    train.data = run_pipeline(train.data, conf.TRANSFO_PIPELINE)

    logging.info(f"{step} - Training on {len(train)} observations.")
    fit = conf.MODEL_PIPELINE.fit(train.data, train.data["outcome"])

    eval_metrics = evaluate(fit, train, conf.EVAL_BETA)
    balanced_accuracy_train = eval_metrics.get("balanced_accuracy")
    fbeta_train = eval_metrics.get("fbeta")
    logging.info(f"{step} - Balanced_accuracy: {balanced_accuracy_train}")
    logging.info(f"{step} - F{conf.EVAL_BETA} score: {fbeta_train}")
    model_stats["train"]["balanced_accuracy"] = balanced_accuracy_train
    model_stats["train"]["Fbeta"] = fbeta_train

    step = "[TEST]"
    model_stats["test"] = {}
    logging.info(f"{step} - Fetching test set ({test.sample_size} samples)")
    test.fetch_data().raise_if_empty()

    train_siren_set = train.data["siren"].unique().tolist()
    test.remove_siren(train_siren_set)

    logging.info(f"{step} - Data preprocessing")
    test.replace_missing_data().remove_na(ignore=IGNORE_NA).remove_strong_signals()
    test.data = run_pipeline(test.data, conf.TRANSFO_PIPELINE)
    logging.info(f"{step} - Testing on {len(test)} observations.")

    eval_metrics = evaluate(fit, test, conf.EVAL_BETA)
    balanced_accuracy_test = eval_metrics.get("balanced_accuracy")
    fbeta_test = eval_metrics.get("fbeta")
    logging.info(f"{step} - Balanced_accuracy: {balanced_accuracy_test}")
    logging.info(f"{step} - F{conf.EVAL_BETA} score: {fbeta_test}")
    model_stats["test"]["balanced_accuracy"] = balanced_accuracy_test
    model_stats["test"]["Fbeta"] = fbeta_test

    step = "[PREDICT]"
    model_stats["predict"] = {}

    logging.info(f"{step} - Fetching predict set")
    predict.fetch_data().raise_if_empty()
    logging.info(f"{step} - Data preprocessing")
    predict.replace_missing_data()
    predict.remove_na(ignore=IGNORE_NA)
    predict.data = run_pipeline(predict.data, conf.TRANSFO_PIPELINE)
    logging.info(f"{step} - Predicting on {len(predict)} observations.")
    predictions = fit.predict_proba(predict.data)
    predict.data["predicted_probability"] = predictions[:, 1]
    expl = explain(predict, conf)
    predict = predict.merge(expl, left_index=True, right_index=True)

    logging.info(f"{step} - Exporting prediction data to csv")

    run_path = Path(OUTPUT_FOLDER) / model_id
    run_path.mkdir(parents=True, exist_ok=True)

    export_destination = f"predictions-{model_id}.csv"
    predict.data[
        [
            "siren",
            "siret",
            "predicted_probability",
            "fail_expl",
            "nofail_expl",
            "group_expl",
        ]
    ].to_csv(run_path / export_destination, index=False)

    with open(run_path / f"stats-{model_id}.json", "w") as stats_file:
        stats_file.write(json.dumps(model_stats))


def make_parser():
    """
    Builds a parser object with all arguments to run a custom version of prediction
    """
    parser = argparse.ArgumentParser("main.py", description="Run model prediction")

    parser.add_argument(
        "--model_name",
        type=str,
        default="default",
        help="The model to use for prediction. If not provided, models 'default' will be used",
    )

    train_args = parser.add_argument_group("Train dataset")
    train_args.add_argument(
        "--train_sample",
        type=int,
        dest="train_spl_size",
        help="Train the model on data from this date",
    )
    train_args.add_argument(
        "--oversampling",
        type=float,
        dest="train_proportion_positive_class",
        help="""
        Enforces the ratio of positive observations
        (entreprises en defaillance) to be the specified ratio
        """,
    )
    train_args.add_argument(
        "--train_from",
        type=str,
        help="Train the model on data from this date",
    )
    train_args.add_argument(
        "--train_to",
        type=str,
        help="Train the model on data up to this date",
    )

    test_args = parser.add_argument_group("Test dataset")
    test_args.add_argument(
        "--test_sample",
        type=int,
        dest="test_spl_size",
        help="The sample size to test the model on",
    )
    test_args.add_argument(
        "--test_from",
        type=str,
        help="Test the model on data from this date",
    )
    test_args.add_argument(
        "--test_to",
        type=str,
        help="Test the model on data up to this date",
    )

    predict_args = parser.add_argument_group("Predict dataset")
    predict_args.add_argument(
        "--predict_sample",
        type=int,
        dest="predict_spl_size",
        help="The sample size to predict on",
    )
    predict_args.add_argument(
        "--predict_on",
        type=str,
        help="""
        Predict on all companies for the month specified.
        To predict on April 2021, provide any date such as '2021-04-01'
        """,
    )

    return parser


if __name__ == "__main__":
    model_args = make_parser().parse_args()
    run(model_args)
