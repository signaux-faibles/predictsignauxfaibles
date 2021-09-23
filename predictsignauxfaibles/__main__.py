import argparse
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Mapping, Tuple

import pandas as pd

from predictsignauxfaibles.config import IGNORE_NA, OUTPUT_FOLDER
from predictsignauxfaibles.data import SFDataset
from predictsignauxfaibles.evaluate import evaluate
from predictsignauxfaibles.explain import explain
from predictsignauxfaibles.pipelines import run_pipeline
from predictsignauxfaibles.utils import EmptyFileError, load_conf, set_if_not_none

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


def get_train_test_predict_datasets(
    args_ns: argparse.Namespace, conf: ModuleType
) -> Tuple[SFDataset, SFDataset, SFDataset]:
    """Prepares datasets for processing.

    Configures `train`, `test` and `predict` datasets pertaining to the user-provided
    options.

    Args:
        args_ns: a argpast Namespace object containing the custom attributes to be used
          for training, testing and/or prediction.
        conf: the model configuration module containing default parameters, to be
          overwritten by the content of args_ns.

    Returns:
        Tuple:
        - train: a SFDataset object containing the training data.
        - test: a SFDataset object containing the test data.
        - predict: a SFDataset object containing the data to predict on.

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

    if args_ns.predict_siretlist_path is not None:
        predict_siret_list = (
            pd.read_csv(
                args_ns.predict_siretlist_path,
                names=["siret"],
                header=0,
                index_col=False,
            )
            .siret.astype(str)
            .tolist()
        )

        if predict_siret_list == []:
            raise EmptyFileError(
                f"File {args_ns.predict_siretlist_path} appears to be empty"
            )

        set_if_not_none(datasets["predict"], "sirets", predict_siret_list)

    return datasets["train"], datasets["test"], datasets["predict"]


def make_stats(
    train: SFDataset, test: SFDataset, predict: SFDataset
) -> Mapping[str, SFDataset]:
    """Initializes a dictionary containing model run stats for logging purposes.

    Args:
        train: a SFDataset object containing the training data.
        test: a SFDataset object containing the test data.
        predict: a SFDataset object containing the data to predict on.

    Returns:
        A dictionary containing model run parameters.

    """
    stats = {}
    datasets = {"train": train, "test": test, "predict": predict}

    for (arg, dest) in ARGS_TO_ATTRS.items():
        stats[arg] = getattr(datasets[dest[0]], dest[1])

    return stats


def run(
    args,
):  # pylint: disable=too-many-statements,too-many-locals
    """Runs a model."""
    conf = load_conf(args.model_name)
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

    export_columns = [
        "siren",
        "siret",
        "predicted_probability",
    ]

    if args.predict_explain:
        logging.info(f"{step} - Computing score explanations")
        predict = explain(predict, conf)
        export_columns += [
            "expl_selection",
            "macro_expl",
            "micro_expl",
            "macro_radar",
        ]

    logging.info(f"{step} - Exporting prediction data to csv")

    run_path = Path(OUTPUT_FOLDER) / f"{args.model_name}_{model_id}"
    run_path.mkdir(parents=True, exist_ok=True)

    export_destination = "predictions.csv"

    predict.data[export_columns].to_csv(run_path / export_destination, index=False)

    with open(run_path / "stats.json", "w") as stats_file:
        stats_file.write(json.dumps(model_stats))

    if args.save_model:
        for comp_id, model_component in enumerate(conf.MODEL_PIPELINE.steps):
            comp_filename = f"model_comp{comp_id}.pickle"
            pickle.dump(model_component, open(run_path / comp_filename, "wb"))


def make_parser() -> argparse.ArgumentParser:
    """Builds a CLI parser object that fetches all learning / prediction parameters."""
    parser = argparse.ArgumentParser("main.py", description="Run model prediction")

    parser.add_argument(
        "--model_name",
        type=str,
        default="default",
        help="""
        The model to use for prediction. If not provided, 'default' model will be
        used.
        """,
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="If this option is provided, model parameters will be saved",
    )

    train_args = parser.add_argument_group("Train dataset")
    train_args.add_argument(
        "--train_sample",
        type=int,
        dest="train_spl_size",
        help="The sample size to train the model on.",
    )
    train_args.add_argument(
        "--oversampling",
        type=float,
        dest="train_proportion_positive_class",
        help="""
        Enforces the ratio of positive observations
        ("entreprises en défaillance") to be the specified ratio
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
        "--predict_siret_list",
        type=str,
        dest="predict_siretlist_path",
        help="""
        Path to a file containing a list of SIRETs that the model will predict on.
        The input file must contain one SIRET per line, and must not include a header.
        If more than one column is present in the file, SIRETs should be in the first column,
        and columns should be comma-separated. Subsequent columns will be ignored.
        In particular, no index different from SIRETs should be included as first column.
        """,
    )
    predict_args.add_argument(
        "--predict_on",
        type=str,
        help="""
        Predict on all companies for the specified month.
        For example, to predict for April 2021, provide any date such as '2021-04-01'
        """,
    )
    predict_args.add_argument(
        "--predict_explain",
        action="store_true",
        help="""
        If provided, the contribution of features to model predictions will be computed and added to the output
        """,
    )

    return parser


if __name__ == "__main__":
    model_args = make_parser().parse_args()
    run(model_args)
