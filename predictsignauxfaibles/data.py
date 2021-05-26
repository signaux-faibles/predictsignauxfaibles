import logging

import pandas as pd
from numpy import nan as NAN
from numpy.random import rand, randint
from pymongo import MongoClient, monitoring
from pymongo.cursor import Cursor

import predictsignauxfaibles.config as config
from predictsignauxfaibles.logging import CommandLogger
from predictsignauxfaibles.utils import MongoDBQuery
from predictsignauxfaibles.decorators import is_random


monitoring.register(CommandLogger())  # loglevel is debug


class SFDataset:
    """
    Retrieve a Signaux Faibles dataset.
    All arguments are optional. The default is a random sample of 1000 observations.

    NB: filtering on categorical variables (e.g. SIREN) may cause slow queries
     as our database is not optimized for such queries.

    Args:
        date_min: first period to include, in the 'YYYY-MM-DD' format. Default is first.
        date_max: first period to exclude, in the 'YYYY-MM-DD' format Default is latest.
        fields: which fields of the Features collection to retrieve. Default is all.
        sample_size: max number of (siret x period) rows to retrieve. Default is 1000.
        outcome: restrict query to firms that fall in a specific outcome (True / False)
        sirets: a list of SIRET to select
        sirens: a list of SIREN to select
        min_effectif: min number of employees for firm to be in the sample (defaults to your .env)
        **categorical_filters can be any filter in the form `field = ["a", "b", "c"]`
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        date_min: str = "1970-01-01",
        date_max: str = "3000-01-01",
        fields: list = None,
        sample_size: int = 1_000,
        min_effectif: int = None,
        outcome: bool = None,
        sirets: list = None,
        sirens: list = None,
        **categorical_filters,
    ):
        self._mongo_client = MongoClient(host=config.MONGODB_PARAMS.url)
        self._mongo_database = self._mongo_client.get_database(config.MONGODB_PARAMS.db)
        self._mongo_collection = self._mongo_database.get_collection(
            config.MONGODB_PARAMS.collection
        )
        self.data = None
        self.date_min = date_min
        self.date_max = date_max
        self.fields = fields
        self.sample_size = sample_size
        self.min_effectif = config.MIN_EFFECTIF if not min_effectif else min_effectif
        self.sirets = sirets
        self.sirens = sirens
        self.outcome = outcome

        if categorical_filters or self.sirens or self.sirets:
            logging.warning(
                "Queries using additional filters usually take longer (see function docstring)"
            )
        self.categorical_filters = categorical_filters

        self.mongo_pipeline = MongoDBQuery()

    def _connect_to_mongo(self):
        if self._mongo_client is None:
            logging.debug("opening connection")
            self._mongo_client = MongoClient(host=config.MONGODB_PARAMS.url)
            self._mongo_database = self._mongo_client.get_database(
                config.MONGODB_PARAMS.db
            )
            self._mongo_collection = self._mongo_database.get_collection(
                config.MONGODB_PARAMS.collection
            )

    def _disconnect_from_mongo(self):
        if self._mongo_client is not None:
            logging.debug("closing connection")
            self._mongo_client.close()
            self._mongo_client = None
            self._mongo_database = None
            self._mongo_collection = None

    def _make_pipeline(self):
        """
        Build MongoDB Aggregate pipeline for dataset
         and store it in the `mongo_pipeline` attribute
        """
        self.mongo_pipeline.reset()

        self.mongo_pipeline.add_standard_match(
            self.date_min,
            self.date_max,
            self.min_effectif,
            sirets=self.sirets,
            sirens=self.sirens,
            categorical_filters=self.categorical_filters,
            outcome=self.outcome,
        )
        self.mongo_pipeline.add_sort()
        self.mongo_pipeline.add_limit(self.sample_size)
        self.mongo_pipeline.add_replace_root()

        if self.fields is not None:
            self.mongo_pipeline.add_projection(self.fields)

        logging.debug(f"MongoDB Aggregate query: {self.mongo_pipeline.to_pipeline()}")

    def fetch_data(self, warn: bool = True):
        """
        Retrieve query from MongoDB database using the Aggregate framework
        Store the resulting data in the `data` attribute
        Args:
            warn : emmit a warning if fetch_data is overwriting some already existing data
        """
        self._make_pipeline()

        if warn and self.data is not None:
            logging.warning("Dataset object was not empty. Overriding...")

        try:
            self._connect_to_mongo()
            cursor = self._mongo_collection.aggregate(self.mongo_pipeline.to_pipeline())
        except Exception as exception:  # pylint: disable=broad-except
            raise exception
        finally:
            self._disconnect_from_mongo()

        try:
            self.data = self.__cursor_to_df(cursor)
        except Exception as exception:
            raise exception
        finally:
            cursor.close()

        # create and fill missing fields with NAs
        if self.fields is not None:
            if not set(self.fields).issubset(set(self.data.columns)):
                missing = set(self.fields) - set(self.data.columns)
                logging.info(
                    f"Creating missing columns {missing} and filling them with NAs."
                )
                for feat in missing:
                    self.data[feat] = NAN

        # force SIREN and SIRET to be strings
        if "siren" in self.data.columns:
            self.data.siren = self.data.siren.astype(str)
        if "siret" in self.data.columns:
            self.data.siret = self.data.siret.astype(str)

        return self

    def raise_if_empty(self):
        """
        Check that dataset is filled with data
        """
        if len(self) == 0:
            raise EmptyDataset("Dataset is empty !")
        return self

    def explain(self):
        """
        Explain MongoDB query plan for Dataset
         This is useful for debugging a long running MongoDB job.
        """
        self._make_pipeline()
        return self._mongo_database.command(
            "aggregate",
            self._mongo_collection.name,
            pipeline=self.mongo_pipeline.pipeline,
            explain=True,
        )

    def remove_strong_signals(self):
        """
        Strong signals is when a firm is already in default (time_til_outcome <= 0)
        """
        assert (
            "time_til_outcome" in self.data.columns
        ), "The `time_til_outcome` column is needed in order to remove strong signals."

        self.data = self.data[~(self.data["time_til_outcome"] <= 0)]
        self.data.reset_index(drop=True, inplace=True)
        return self

    def replace_missing_data(self, defaults_map: dict = None):
        """
        Replace missing data with defaults defined in project config
        Args:
            defaults_map: a dictionnary in the {column_name: default_value} format
        """
        if defaults_map is None:
            defaults_map = config.DEFAULT_DATA_VALUES

        for feature, default_value in defaults_map.items():
            logging.debug(f"Column {feature} defaulting to value {default_value}")

        for column in defaults_map:
            try:
                self.data[column] = self.data[column].fillna(defaults_map.get(column))
            except KeyError:
                logging.debug(f"Column {column} not in dataset")
                continue
        self.data.reset_index(drop=True, inplace=True)
        return self

    def remove_na(self, ignore: list):
        """
        Remove all observations with missing values.
        Args:
            ignore: a list of column names to ignore when dropping NAs
        """

        cols_drop_na = set(self.data.columns).difference(set(ignore))

        logging.info("Removing NAs from dataset.")
        for feature in cols_drop_na:
            logging.debug(
                f"Rows with NAs in field {feature} will be dropped, unless default val is provided"
            )

        for feature in ignore:
            logging.debug(f"Rows with NAs in field {feature} will NOT be dropped")

        logging.info(f"Number of observations before: {len(self.data.index)}")
        self.data.dropna(subset=cols_drop_na, inplace=True)
        logging.info(f"Number of observations after: {len(self.data.index)}")
        self.data.reset_index(drop=True, inplace=True)
        return self

    def remove_siren(self, siren_list: list):
        """
        Removes all observations corresponding to
        one of the siren provided as input.
        Args:
            siren_list: a list of siren to be removed from the data.
        """
        orig_length = len(self)
        self.data = self.data[~self.data["siren"].isin(siren_list)]
        post_length = len(self)
        logging.info(
            f"Removed {orig_length - post_length} from data based on SIREN blacklist"
        )
        self.data.reset_index(drop=True, inplace=True)
        return self

    def __repr__(self):
        out = f"""
Signaux Faibles Dataset
----------------------------------------------------
{self.data.head() if isinstance(self.data, pd.DataFrame) else "Empty Dataset"}
[...]
----------------------------------------------------
Number of observations = {len(self) if isinstance(self.data, pd.DataFrame) else "0"}
        """
        return out

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        """
        Length of SFDataset is the length of its dataframe
        """
        return len(self.data) if isinstance(self.data, pd.DataFrame) else 0

    @staticmethod
    def __cursor_to_df(cursor: Cursor):
        """
        Extract data from a MongoDB cursor into a Pandas dataframe
        """
        return pd.DataFrame(cursor)


class OversampledSFDataset(SFDataset):
    """
    Helper class to oversample a SFDataset
    Args:
        proportion_positive_class: the desired proportion of firms for which outcome = True
    All other arguments are documented in the SFDataset docstring
    """

    def __init__(self, proportion_positive_class: float, **kwargs):
        super().__init__(**kwargs)
        assert (
            0 <= proportion_positive_class <= 1
        ), "proportion_positive_class must be between 0 and 1"
        self.proportion_positive_class = proportion_positive_class

    @is_random
    def fetch_data(self):  # pylint: disable=arguments-differ
        """
        Retrieve query from MongoDB database using the Aggregate framework
        Store the resulting data in the `data` attribute
        """
        # compute the number of lines to fetch with outcome = True
        n_obs_true = round(self.proportion_positive_class * self.sample_size)
        n_obs_false = self.sample_size - n_obs_true

        # fetch true
        self.sample_size = n_obs_true
        self.outcome = True
        super().fetch_data(warn=False)
        true_data = self.data

        # fetch false
        self.sample_size = n_obs_false
        self.outcome = False
        super().fetch_data(warn=False)
        false_data = self.data
        full_data = true_data.append(false_data)
        self.data = full_data.sample(frac=1).reset_index(drop=True)
        return self


class EmptyDataset(Exception):
    """
    Custom error for empty datasets
    """


def build_synthetic_dataset(
    base_dataset: SFDataset,
    cont_variables: list,
    cat_variables: list,
    group_size: int = 5,
):
    """
    Génère un set de données synthétique, contenant des établissements fictifs.
    Les établissements synthétiquessont générés à partir d'établissements de nature similaire,
    en aggrégant les données de X établissements à minima
    pour générer un seul établissement synthétique.
    Chaque synthétique est fabriqué à partir de <group_size>
    établissements d'un même sous-secteur (APE 3).
    Args:
        base_dataset: SFDataset
            The dataset to build a synthetic extract from
        cont_variables: list
            The list of continuous variables from collection Features to include in the extract
        cat_variables: list
            The list of categorical variables from collection Features to include in the extract
        group_size: int
            The number of établissements required to build a synthetic SIRET from
    """
    # Finding subsectors with enough SIRET to build a synthetic
    subsectors_count = base_dataset.data.groupby("code_ape_niveau3").agg(
        siret_count=("siret", "count")
    )
    repr_subsectors = subsectors_count[
        subsectors_count["siret_count"] > group_size
    ].index.tolist()
    models = base_dataset.data[
        base_dataset.data["code_ape_niveau3"].isin(repr_subsectors)
    ]

    # Building a random ranking by ape3 that we will use to generate synthetics
    models["ranker"] = rand(models.shape[0])
    models["within_ape3_id"] = models.groupby("code_ape_niveau3")["ranker"].rank()
    models["within_ape3_group_id"] = models["within_ape3_id"] % group_size

    # Filtering synthesis set
    cont_agg_dct = {cont_var: "mean" for cont_var in cont_variables}
    dflt_agg_dct = {"periode": "max"}

    agg_dct = dict(cont_agg_dct, **dflt_agg_dct)

    within_ape3_ref = models.groupby(
        ["code_ape_niveau3", "within_ape3_group_id"]
    ).within_ape3_id.idxmin()
    synthetic_cont = models.groupby(["code_ape_niveau3", "within_ape3_group_id"]).agg(
        agg_dct
    )

    cat_references = models.loc[within_ape3_ref]
    cat_references.index = pd.MultiIndex.from_frame(
        cat_references[["code_ape_niveau3", "within_ape3_group_id"]]
    )
    cat_references.drop(
        ["code_ape_niveau3", "within_ape3_group_id"], axis=1, inplace=True
    )

    synthetic = pd.merge(
        synthetic_cont,
        cat_references[cat_variables],
        on=["code_ape_niveau3", "within_ape3_group_id"],
        how="inner",
    )

    synthetic["siret"] = randint(1e13, 1e14 - 1, len(synthetic)).astype(str)
    synthetic["siren"] = synthetic["siret"].apply(lambda siret: siret[:9])

    synthetic.set_index("siret", inplace=True, drop=False)
    synthetic = synthetic[
        ["siret", "siren", "periode", "outcome"] + cat_variables + cont_variables
    ]

    return synthetic
