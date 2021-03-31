# pylint: disable=protected-access
from datetime import datetime
import logging
import sys

from sklearn.metrics import balanced_accuracy_score

from predictsignauxfaibles.pipelines import run_pipeline

sys.path.append("../")
import models.default.model_conf as conf  # pylint: disable=wrong-import-position

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level="INFO")

# Mute logs from sklean_pandas
logging.getLogger("sklearn_pandas").setLevel(logging.WARNING)


def run():
    """
    Run model
    """
    logging.info(f"Running Model {conf.MODEL_ID} (commit {conf.MODEL_GIT_SHA})")

    step = "[TRAIN]"
    logging.info(f"{step} - Fetching train set")
    conf.TRAIN_DATASET.fetch_data()

    logging.info(f"{step} - Data preprocessing")
    conf.TRAIN_DATASET._replace_missing_data()
    conf.TRAIN_DATASET._remove_na(ignore=["time_til_outcome"])
    conf.TRAIN_DATASET.data = run_pipeline(
        conf.TRAIN_DATASET.data, conf.TRANSFO_PIPELINE
    )

    logging.info(f"{step} - Training on {len(conf.TRAIN_DATASET)} observations.")
    fit = conf.MODEL_PIPELINE.fit(
        conf.TRAIN_DATASET.data, conf.TRAIN_DATASET.data["outcome"]
    )

    step = "[TEST]"
    logging.info(f"{step} - Fetching test set")
    conf.TEST_DATASET.fetch_data()

    logging.info(f"{step} - Data preprocessing")
    conf.TEST_DATASET._replace_missing_data()
    conf.TEST_DATASET._remove_na(ignore=["time_til_outcome"])
    conf.TEST_DATASET._remove_strong_signals()
    conf.TEST_DATASET.data = run_pipeline(conf.TEST_DATASET.data, conf.TRANSFO_PIPELINE)
    logging.info(f"{step} - Testing on {len(conf.TEST_DATASET)} observations.")

    balanced_accuracy = balanced_accuracy_score(
        conf.TEST_DATASET.data["outcome"], fit.predict(conf.TEST_DATASET.data)
    )
    logging.info(f"{step} - Test balanced_accuracy: {balanced_accuracy}")

    step = "[PREDICT]"
    logging.info(f"{step} - Fetching predict set")
    conf.PREDICT_DATASET.fetch_data()
    logging.info(f"{step} - Data preprocessing")
    conf.PREDICT_DATASET._replace_missing_data()
    conf.PREDICT_DATASET._remove_na(ignore=["time_til_outcome", "outcome"])
    conf.PREDICT_DATASET.data = run_pipeline(
        conf.PREDICT_DATASET.data, conf.TRANSFO_PIPELINE
    )
    logging.info(f"{step} - Predicting on {len(conf.PREDICT_DATASET)} observations.")
    predictions = fit.predict_proba(conf.PREDICT_DATASET.data)
    conf.PREDICT_DATASET.data["predicted_probability"] = predictions[:, 1]

    logging.info(f"{step} - Exporting data to csv")
    model_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    export_destination = f"predictions-{model_id}.csv"
    conf.PREDICT_DATASET.data[["siren", "siret", "predicted_probability"]].to_csv(
        export_destination, index=False
    )


if __name__ == "__main__":
    run()
