from datetime import datetime
import subprocess

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn_pandas import DataFrameMapper

from predictsignauxfaibles.config import DEFAULT_DATA_VALUES, IGNORE_NA
from predictsignauxfaibles.data import OversampledSFDataset, SFDataset
from predictsignauxfaibles.pipelines import DEFAULT_PIPELINE
from predictsignauxfaibles.utils import check_feature

# Model Information
MODEL_ID = "Modèle de Mars - V1"
MODEL_RUN_DATE = datetime.today()
MODEL_GIT_SHA = str(
    subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]), encoding="utf-8"
).rstrip("\n")

# Variables disponibles en base :
# https://github.com/signaux-faibles/opensignauxfaibles/master/js/reduce.algo2/docs/variables.json
VARIABLES = [
    "",
]

# ces variables sont toujours requêtées
VARIABLES += ["outcome", "periode", "siret", "siren", "time_til_outcome", "code_naf"]

# Model-specific préprocessing
PIPELINE = DEFAULT_PIPELINE

# features
FEATURES = [
    "",
]

for feature in FEATURES:
    if not check_feature(feature):
        raise ValueError(
            f"Feature '{feature}' is not in VARIABLES nor created by the PIPELINE"
        )

# model
TO_ONEHOT_ENCODE = ["paydex_group"]
TO_SCALE = list(set(FEATURES) - set(TO_ONEHOT_ENCODE))

mapper = DataFrameMapper(
    [
        (TO_ONEHOT_ENCODE, [OneHotEncoder()]),
        (TO_SCALE, [StandardScaler()]),
    ],
)

MODEL_PIPELINE = Pipeline(
    [("transform_dataframe", mapper), ("fit_model", LogisticRegression())]
)

# Train Dataset
TRAIN_FROM = "2016-01-01"
TRAIN_TO = "2018-06-30"
TRAIN_SAMPLE_SIZE = 1_000_000
TRAIN_OVERSAMPLING = 0.2

TRAIN_DATASET = OversampledSFDataset(
    TRAIN_OVERSAMPLING,
    date_min=TRAIN_FROM,
    date_max=TRAIN_TO,
    fields=VARIABLES,
    sample_size=TRAIN_SAMPLE_SIZE,
)

# Test Dataset
TEST_FROM = "2018-07-01"
TEST_TO = "2018-10-31"
TEST_SAMPLE_SIZE = 5_000_000

TEST_DATASET = SFDataset(
    date_min=TEST_FROM, date_max=TEST_TO, fields=VARIABLES, sample_size=TEST_SAMPLE_SIZE
)

# Predict Dataset
PREDICT_FROM = "2020-02-01"
PREDICT_TO = "2020-02-28"

PREDICT_DATASET = SFDataset(
    date_min=PREDICT_FROM, date_max=PREDICT_TO, fields=VARIABLES, sample_size=5_000_000
)
