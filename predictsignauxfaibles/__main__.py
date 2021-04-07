import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys
import logging

from sklearn.metrics import fbeta_score
from sklearn.metrics import balanced_accuracy_score
from predictsignauxfaibles.config import OUTPUT_FOLDER
from predictsignauxfaibles.pipelines import run_pipeline
from predictsignauxfaibles.data import OversampledSFDataset, SFDataset

sys.path.append("../")

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level="INFO")

# Mute logs from sklean_pandas
logging.getLogger("sklearn_pandas").setLevel(logging.WARNING)


def load_conf(args):  # pylint: disable=redefined-outer-name
    """
    Loads a model configuration from a argparse.Namespace
    containing a model name and a configuration filename
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


def evaluate(
    model, dataset
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
    fbeta = fbeta_score(
        dataset.data["outcome"], model.predict(dataset.data), beta=conf.EVAL_BETA
    )
    return {"balanced_accuracy": balanced_accuracy, "fbeta": fbeta}


def run():  # pylint: disable=redefined-outer-name,too-many-statements,too-many-locals
    """
    Run model
    """
    logging.info(f"Running Model {conf.MODEL_ID} (commit {conf.MODEL_GIT_SHA})")
    model_stats = {}
    model_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_stats["run_on"] = model_id

    step = "[TRAIN]"
    model_stats["train"] = {}
    train_dataset = OversampledSFDataset(
        conf.TRAIN_OVERSAMPLING,
        date_min=conf.TRAIN_FROM,
        date_max=conf.TRAIN_TO,
        fields=conf.VARIABLES,
        sample_size=conf.TRAIN_SAMPLE_SIZE,
    )
    logging.info(f"{step} - Fetching train set")
    train_dataset.fetch_data()

    logging.info(f"{step} - Data preprocessing")
    train_dataset.replace_missing_data().remove_na(ignore=["time_til_outcome"])
    train_dataset.data = run_pipeline(train_dataset.data, conf.TRANSFO_PIPELINE)

    logging.info(f"{step} - Training on {len(train_dataset)} observations.")
    fit = conf.MODEL_PIPELINE.fit(train_dataset.data, train_dataset.data["outcome"])

    eval_metrics = evaluate(fit, train_dataset)
    balanced_accuracy_train = eval_metrics.get("balanced_accuracy")
    fbeta_train = eval_metrics.get("fbeta")
    logging.info(f"{step} - Balanced_accuracy: {balanced_accuracy_train}")
    logging.info(f"{step} - F{conf.EVAL_BETA} score: {fbeta_train}")
    model_stats["train"]["balanced_accuracy"] = balanced_accuracy_train
    model_stats["train"]["Fbeta"] = fbeta_train

    step = "[TEST]"
    model_stats["test"] = {}
    test_dataset = SFDataset(
        date_min=conf.TEST_FROM,
        date_max=conf.TEST_TO,
        fields=conf.VARIABLES,
        sample_size=conf.TEST_SAMPLE_SIZE,
    )
    logging.info(f"{step} - Fetching test set")
    test_dataset.fetch_data()

    train_siren_set = train_dataset.data["siren"].unique().tolist()
    test_dataset.remove_siren(train_siren_set)

    logging.info(f"{step} - Data preprocessing")
    test_dataset.replace_missing_data().remove_na(
        ignore=["time_til_outcome"]
    ).remove_strong_signals()
    test_dataset.data = run_pipeline(test_dataset.data, conf.TRANSFO_PIPELINE)
    logging.info(f"{step} - Testing on {len(test_dataset)} observations.")

    eval_metrics = evaluate(fit, test_dataset)
    balanced_accuracy_test = eval_metrics.get("balanced_accuracy")
    fbeta_test = eval_metrics.get("fbeta")
    logging.info(f"{step} - Balanced_accuracy: {balanced_accuracy_test}")
    logging.info(f"{step} - F{conf.EVAL_BETA} score: {fbeta_test}")
    model_stats["test"]["balanced_accuracy"] = balanced_accuracy_test
    model_stats["test"]["Fbeta"] = fbeta_test

    step = "[PREDICT]"
    model_stats["predict"] = {}

    predict_dataset = SFDataset(
        date_min=conf.PREDICT_ON,
        date_max=conf.PREDICT_ON[:-2] + "28",
        fields=conf.VARIABLES,
        sample_size=conf.PREDICT_SAMPLE_SIZE,
    )
    logging.info(f"{step} - Fetching predict set")
    predict_dataset.fetch_data()
    logging.info(f"{step} - Data preprocessing")
    predict_dataset.replace_missing_data()
    predict_dataset.remove_na(ignore=["time_til_outcome", "outcome"])
    predict_dataset.data = run_pipeline(predict_dataset.data, conf.TRANSFO_PIPELINE)
    logging.info(f"{step} - Predicting on {len(predict_dataset)} observations.")
    predictions = fit.predict_proba(predict_dataset.data)
    predict_dataset.data["predicted_probability"] = predictions[:, 1]

    logging.info(f"{step} - Exporting prediction data to csv")

    run_path = Path(OUTPUT_FOLDER) / model_id
    run_path.mkdir(parents=True, exist_ok=True)

    export_destination = f"predictions-{model_id}.csv"
    predict_dataset.data[["siren", "siret", "predicted_probability"]].to_csv(
        run_path / export_destination, index=False
    )

    with open(run_path / f"stats-{model_id}.json", "w") as stats_file:
        stats_file.write(json.dumps(model_stats))


parser = argparse.ArgumentParser("main.py", description="Run model prediction")

parser.add_argument(
    "--model_name",
    type=str,
    default="default",
    help="The model to use for prediction. If not provided, models/default will be used",
)

conf_args = parser.parse_args()

conf = load_conf(conf_args)


if __name__ == "__main__":
    run()
