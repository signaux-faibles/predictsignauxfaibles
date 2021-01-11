# TODO: find better name for module ?

from abc import ABC, abstractmethod
import logging
from typing import List, Tuple

from pygam import LogisticGAM
import pandas as pd
from sklearn.model_selection import train_test_split

from lib.data import SFDataset


class SFModel(ABC):
    """
    Main abstract base class for a Signaux Faibles model.
    Used to define the main interfaces that a model must implement.
    """

    def __init__(self, features: List, target: str):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model = None
        self.features = features
        self.target = target
        self._is_trained = False
        self._performance = None

    @classmethod
    def from_config_file(cls, path_to_file: str):
        """
        Instantiate a SFModel instance using a standardized config file template (e.g. a .yaml file)
        """
        # parse information needed to instantiate SFModel from config file
        logging.info(f"Instantiating SFModel with config found in {path_to_file}")
        # return the instantiated SFModel object
        return cls()  # pylint: disable=no-value-for-parameter

    @abstractmethod
    def train(self):
        """
        Train a model using training data.
        """

    @abstractmethod
    def evaluate(self):
        """
        Evaluate a model's performance using testing data.
        """

    @abstractmethod
    def predict(self, new_data: pd.DataFrame):
        """
        Given new data in a DataFrame and a trained model, make prediction for each new observation.
        """

    def save_model(self):
        """
        Serialize the model to a format that can be saved and version-controlled for easy reloading.
        `pickle` could work, but I think some much better alternatives exist.
        We should brainstorm this.
        """
        logging.info("Saving model.")
        logging.info("Model saved in /here/model.pickle")
        return self

    def _split_train_test(self, dataset: pd.DataFrame) -> Tuple:  # FIXME
        features = dataset[self.features]
        target = dataset[[self.target]]
        return train_test_split(features, target, test_size=0.3)


class SFModelGAM(SFModel):
    """
    Generalised Additive Model (GAM)
    From package `pygam`
    """

    def __init__(self, dataset: SFDataset, features: List, target: str = "outcome"):
        super().__init__(features, target)
        self.x_train, self.x_test, self.y_train, self.y_test = self._split_train_test(
            dataset.data
        )
        self.model = LogisticGAM()

    def train(self):
        """
        Train a GAM model on the data
        """
        self.model.fit(self.x_train, self.y_train)
        self._is_trained = True
        return self

    def evaluate(self):
        """
        Evaluate model using pyGAM
        """
        self._performance = self.model.accuracy(self.x_test, self.y_test)
        return self._performance

    def predict(self, new_data: pd.DataFrame):
        """
        Use the trained model to make predictions on new data
        """
        return self.model.predict(new_data)
