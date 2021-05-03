from datetime import datetime
import logging
import os
import subprocess

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn_pandas import DataFrameMapper

from predictsignauxfaibles.data import SFDataset, OversampledSFDataset
from predictsignauxfaibles.pipelines import DEFAULT_PIPELINE
from predictsignauxfaibles.utils import check_feature

# ENV (default is "develop", can be set to "prod")
ENV = os.getenv("ENV", "develop")


# Model Information
MODEL_ID = "202103_logreg_full"
MODEL_RUN_DATE = datetime.today()
MODEL_GIT_SHA = str(
    subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]), encoding="utf-8"
).rstrip("\n")

# Variables disponibles en base :
# https://github.com/signaux-faibles/opensignauxfaibles/master/js/reduce.algo2/docs/variables.json
VARIABLES = [
    "financier_court_terme",
    "interets",
    "ca",
    "equilibre_financier",
    "endettement",
    "degre_immo_corporelle",
    "liquidite_reduite",
    "poids_bfr_exploitation",
    "productivite_capital_investi",
    "rentabilite_economique",
    "rentabilite_nette",
    "cotisation",
    "cotisation_moy12m",
    "montant_part_ouvriere",
    "montant_part_ouvriere_past_1",
    "montant_part_ouvriere_past_12",
    "montant_part_ouvriere_past_2",
    "montant_part_ouvriere_past_3",
    "montant_part_ouvriere_past_6",
    "montant_part_patronale",
    "montant_part_patronale_past_1",
    "montant_part_patronale_past_12",
    "montant_part_patronale_past_2",
    "montant_part_patronale_past_3",
    "montant_part_patronale_past_6",
    "ratio_dette",
    "ratio_dette_moy12m",
    "effectif",
    "apart_heures_consommees_cumulees",
    "apart_heures_consommees",
    "paydex_nb_jours",
    "paydex_nb_jours_past_12",
]

# ces variables sont toujours requêtées
VARIABLES += ["outcome", "periode", "siret", "siren", "time_til_outcome", "code_naf"]

# Model-specific préprocessing
TRANSFO_PIPELINE = DEFAULT_PIPELINE

# features
FEATURES = [
    "apart_heures_consommees_cumulees",
    "apart_heures_consommees",
    "ratio_dette",
    "avg_delta_dette_par_effectif",
    "paydex_group",
    "paydex_yoy",
    "financier_court_terme",
    "interets",
    "ca",
    "equilibre_financier",
    "endettement",
    "degre_immo_corporelle",
    "liquidite_reduite",
    "poids_bfr_exploitation",
    "productivite_capital_investi",
    "rentabilite_economique",
    "rentabilite_nette",
]

for feature in FEATURES:
    if not check_feature(feature, VARIABLES, TRANSFO_PIPELINE):
        raise ValueError(
            f"Feature '{feature}' is not in VARIABLES nor created by the PIPELINE"
        )

FEATURE_GROUPS = {
    "sante_financiere": [
        "financier_court_terme",
        "interets",
        "ca",
        "equilibre_financier",
        "endettement",
        "degre_immo_corporelle",
        "liquidite_reduite",
        "poids_bfr_exploitation",
        "productivite_capital_investi",
        "rentabilite_economique",
        "rentabilite_nette",
    ],
    "activite_partielle": [
        "apart_heures_consommees_cumulees",
        "apart_heures_consommees",
    ],
    "retards_paiement": [
        "paydex_group",
        "paydex_yoy",
    ],
    "dette_urssaf": [
        "ratio_dette",
        "avg_delta_dette_par_effectif",
    ]
}
        
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
TRAIN_SAMPLE_SIZE = 1_000_000 if ENV == "prod" else 5_000
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
TEST_SAMPLE_SIZE = 250_000 if ENV == "prod" else 5_000
TEST_DATASET = SFDataset(
    date_min=TEST_FROM,
    date_max=TEST_TO,
    fields=VARIABLES,
    sample_size=TEST_SAMPLE_SIZE,
)

# Predict Dataset
PREDICT_ON = "2020-02-01"
PREDICT_SAMPLE_SIZE = 1_000_000_000 if ENV == "prod" else 5_000
PREDICT_DATASET = SFDataset(
    date_min=PREDICT_ON,
    date_max=PREDICT_ON[:-2] + "28",
    fields=VARIABLES,
    sample_size=PREDICT_SAMPLE_SIZE,
)

# Evaluation parameters
EVAL_BETA = 2

if __name__ == "__main__":
    logging.getLogger().setLevel("INFO")
    logging.info(f"ENV : {ENV}")
    logging.info(f"Model {MODEL_ID}")
    logging.info(f"Run on {MODEL_RUN_DATE}")
    logging.info(f"Current commit: {MODEL_GIT_SHA}")
