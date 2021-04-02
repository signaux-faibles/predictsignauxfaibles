# pylint: disable=invalid-name
import logging
import sys

from os import path
import argparse
import importlib.util

from datetime import datetime
from dateutil.relativedelta import relativedelta

from sklearn.metrics import balanced_accuracy_score
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
            model_conf_path
    """
    conf_filepath = path.join("models", args.model_name, args.model_conf_path + ".py")

    spec = importlib.util.spec_from_file_location(
        f"models.{args.model_name}.{args.model_conf_path}", conf_filepath
    )
    model_conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_conf)  # pylint: disable=wrong-import-position
    return model_conf


def run(args):  # pylint: disable=redefined-outer-name
    """
    Run model
    """
    logging.info(f"Running Model {conf.MODEL_ID} (commit {conf.MODEL_GIT_SHA})")

    step = "[TRAIN]"
    TRAIN_DATASET = OversampledSFDataset(
        args.train_proportion_positive_class,
        date_min=args.train_from,
        date_max=args.train_to,
        fields=conf.VARIABLES,
        sample_size=args.train_spl_size,
    )
    logging.info(f"{step} - Fetching train set")
    TRAIN_DATASET.fetch_data()

    logging.info(f"{step} - Data preprocessing")
    TRAIN_DATASET.replace_missing_data().remove_na(ignore=["time_til_outcome"])
    TRAIN_DATASET.data = run_pipeline(TRAIN_DATASET.data, conf.TRANSFO_PIPELINE)

    logging.info(f"{step} - Training on {len(TRAIN_DATASET)} observations.")
    fit = conf.MODEL_PIPELINE.fit(TRAIN_DATASET.data, TRAIN_DATASET.data["outcome"])

    step = "[TEST]"
    TEST_DATASET = SFDataset(
        date_min=args.test_from,
        date_max=args.test_to,
        fields=conf.VARIABLES,
        sample_size=args.test_spl_size,
    )
    logging.info(f"{step} - Fetching test set")
    TEST_DATASET.fetch_data()

    train_siren_set = TRAIN_DATASET.data["siren"].unique().tolist()
    TEST_DATASET.remove_siren(train_siren_set)

    logging.info(f"{step} - Data preprocessing")
    TEST_DATASET.replace_missing_data().remove_na(
        ignore=["time_til_outcome"]
    ).remove_strong_signals()
    TEST_DATASET.data = run_pipeline(TEST_DATASET.data, conf.TRANSFO_PIPELINE)
    logging.info(f"{step} - Testing on {len(TEST_DATASET)} observations.")

    balanced_accuracy = balanced_accuracy_score(
        TEST_DATASET.data["outcome"], fit.predict(TEST_DATASET.data)
    )
    logging.info(f"{step} - Test balanced_accuracy: {balanced_accuracy}")

    step = "[PREDICT]"
    args.predict_on = args.predict_on
    predict_from = args.predict_on + relativedelta(day=1)
    predict_to = args.predict_on + relativedelta(months=+1)
    predict_to = predict_to + relativedelta(days=-1)

    PREDICT_DATASET = SFDataset(
        date_min=predict_from, date_max=predict_to, fields=conf.VARIABLES
    )
    logging.info(f"{step} - Fetching predict set")
    PREDICT_DATASET.fetch_data()
    logging.info(f"{step} - Data preprocessing")
    PREDICT_DATASET.replace_missing_data()
    PREDICT_DATASET.remove_na(ignore=["time_til_outcome", "outcome"])
    PREDICT_DATASET.data = run_pipeline(PREDICT_DATASET.data, conf.TRANSFO_PIPELINE)
    logging.info(f"{step} - Predicting on {len(PREDICT_DATASET)} observations.")
    predictions = fit.predict_proba(PREDICT_DATASET.data)
    PREDICT_DATASET.data["predicted_probability"] = predictions[:, 1]

    logging.info(f"{step} - Exporting data to csv")
    model_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    export_destination = f"predictions-{model_id}.csv"
    PREDICT_DATASET.data[["siren", "siret", "predicted_probability"]].to_csv(
        export_destination, index=False
    )


parser = argparse.ArgumentParser("main.py", description="Run model prediction")

config_help = """
The name of the model configuration file that you wish to use, relative to predictsignauxfaibles/models/<YOUR_MODEL_NAME>/".
A model configuration file describes the model tp be trained, some training and scoring parameters, as well as default train, test and predict dates.
Any additional argument provided to this script will override the default variables declared in the configuration file.
By default, will use model_config.py
"""
parser.add_argument(
    "--model_name",
    type=str,
    default="default",
    help="The model to use for prediction. If not provided, models/default will be used",
)
parser.add_argument(
    "--config_file",
    type=str,
    dest="model_conf_path",
    default="model_conf",
    help=config_help,
)
conf_args = parser.parse_args()

conf = load_conf(conf_args)

parser.add_argument(
    "--train_sample",
    type=int,
    dest="train_spl_size",
    default=conf.TRAIN_SAMPLE_SIZE,
    help="Train the model on data from this date",
)
parser.add_argument(
    "--test_sample",
    type=int,
    dest="test_spl_size",
    default=conf.TEST_SAMPLE_SIZE,
    help="Train the model on data from this date",
)
parser.add_argument(
    "--oversampling",
    type=float,
    dest="train_proportion_positive_class",
    default=conf.TRAIN_OVERSAMPLING,
    help="""
    Enforces the ratio of positive observations
    (entreprises en défaillance) to be the specified ratio
    """,
)

parser.add_argument(
    "--train_from",
    type=str,
    default=conf.TRAIN_FROM,
    help="Train the model on data from this date",
)
parser.add_argument(
    "--train_to",
    type=str,
    default=conf.TRAIN_TO,
    help="Train the model on data up to this date",
)
parser.add_argument(
    "--test_from",
    type=str,
    default=conf.TEST_FROM,
    help="Test the model on data from this date",
)
parser.add_argument(
    "--test_to",
    type=str,
    default=conf.TEST_TO,
    help="Test the model on data up to this date",
)
parser.add_argument(
    "--predict_on",
    type=str,
    default=conf.PREDICT_ON,
    help="""
    Predict on all companies for the month specified
    To predict on April 2021, provide any date such as '01-04-2021'
    """,
)

args = parser.parse_args()


if __name__ == "__main__":
    run(args)
