import json
from typing import List

import numpy as np
import pandas as pd


class ExperimentResult:
    """
    Encapsulates the results of the training of a model.

    Specifically it stores the target and prediction tensors for both training and validation data.
    This information can be used at a later point for calculating different error metrics and plotting.
    """

    def __init__(
            self,
            model_class: str,
            input_features: List[str],
            target_features: List[str],
            lookback_horizon: int,
            prediction_horizon: int,
            metrics_train: dict,
            metrics_test: dict,
            train_index: pd.Series,
            train_target: np.ndarray,
            train_prediction: np.ndarray,
            test_index: pd.Series,
            test_target: np.ndarray,
            test_prediction: np.ndarray,
    ):

        if not train_prediction.shape == train_target.shape:
            raise RuntimeError(
                f"Unable to create result object. "
                f"The training predictions and training target tensors do not have the same shape. "
                f"Shape of predictions: '{train_prediction.shape}', shape of targets: '{train_target.shape}'"
            )

        if not test_prediction.shape == test_target.shape:
            raise RuntimeError(
                f"Unable to create result object. "
                f"The testing predictions and testing target tensors do not have the same shape. "
                f"Shape of predictions: '{test_prediction.shape}', shape of targets: '{test_target.shape}'"
            )

        self.model_class = model_class
        self.input_features = input_features
        self.target_features = target_features

        self.lookback_horizon = lookback_horizon
        self.prediction_horizon = prediction_horizon

        self.metrics_train = metrics_train
        self.metrics_test = metrics_test

        self.train_index = train_index
        self.train_target = train_target
        self.train_prediction = train_prediction

        self.test_index = test_index
        self.test_target = test_target
        self.test_prediction = test_prediction

    def to_dict(self) -> dict:
        return {
            "model_class": self.model_class,
            "input_features": self.input_features,
            "target_features": self.target_features,
            "lookback_horizon": self.lookback_horizon,
            "prediction_horizon": self.prediction_horizon,
            "metrics_train": self.metrics_train,
            "metrics_test": self.metrics_test,
        }

    def to_file(self, file_path):
        file_path = str(file_path)
        if not file_path.endswith('.json'):
            file_path = f"{file_path}.json"

        with open(file_path, "w") as fp:
            results = self.to_dict()
            json.dump(results, fp, indent=4)
