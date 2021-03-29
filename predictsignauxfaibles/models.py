from abc import ABC, abstractmethod
import logging
from typing import List

from pygam import LogisticGAM
import pandas as pd

from predictsignauxfaibles.data import SFDataset
from predictsignauxfaibles.decorators import is_random


class SFModel(ABC):
    """
    Main abstract base class for a Signaux Faibles model.
    Used to define the main interfaces that a model must implement.
    """

    def __init__(self, dataset: SFDataset, features: List, target: str):
        self.dataset = dataset
        self.model = None
        self.features = features
        self.target = target
        self.X = self.dataset.data[self.features]  # pylint: disable=invalid-name
        self.y = self.dataset.data[[self.target]]  # pylint: disable=invalid-name
        self._is_trained = False
        self._performance = None

    @abstractmethod
    def train(self):
        """
        Train a model using training data.
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

    def __repr__(self):
        return f"""
SFModel wrapping {self.model}
Predicting target "{self.target}" with {len(self.features)} features
        """

    def __str__(self):
        return self.__repr__()


class SFModelGAM(SFModel):
    """
    Generalised Additive Model (GAM)
    From package `pygam`
    """

    def __init__(self, dataset: SFDataset, features: List, target: str = "outcome"):
        super().__init__(dataset, features, target)
        self.model = LogisticGAM()

    @is_random
    def train(self):
        """
        Train a GAM model on the data
        """
        self.model.fit(self.X, self.y)
        self._is_trained = True
        return self

    def predict(self, new_data: pd.DataFrame):
        """
        Use the trained model to make predictions on new data
        """
        return self.model.predict(new_data)

    def predict_proba(self, new_data: pd.DataFrame):
        """
        Use the trained model to make predictions on new data
        """
        return self.model.predict_proba(new_data)
