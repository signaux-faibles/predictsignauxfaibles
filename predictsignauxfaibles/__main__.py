import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys
import logging
from types import ModuleType

from sklearn.metrics import fbeta_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline

from predictsignauxfaibles.config import OUTPUT_FOLDER, IGNORE_NA
from predictsignauxfaibles.pipelines import run_pipeline
from predictsignauxfaibles.utils import set_if_not_none
from predictsignauxfaibles.data import SFDataset
from predictsignauxfaibles.explainability import explain

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
    logging.info(f"{step} - Computing score explanations")
    predict = explain(predict, conf)

    logging.info(f"{step} - Exporting prediction data to csv")

    run_path = Path(OUTPUT_FOLDER) / model_id
    run_path.mkdir(parents=True, exist_ok=True)

    export_destination = f"predictions-{model_id}.csv"
    predict.data[
        [
            "siren",
            "siret",
            "predicted_probability",
            "expl_selection",
            "macro_expl",
            "micro_expl",
            "macro_radar",
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
