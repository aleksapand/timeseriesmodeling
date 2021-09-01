# supresses tensorflow warnings
from datamodels.validation import metrics
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd

from typing import List, Tuple

from tensorflow import keras

import dataprocessors
import datamodels as dm

from experiment import ExperimentResult

all_features = [
    'energy', 
    'water', 
    'occupants',
    'temperature',
    'datetime',
]

model_to_class = {
    "NeuralNetwork": dm.NeuralNetwork,
    "ConvolutionNetwork": dm.ConvolutionNetwork,
    "EncoderDecoderLSTM": dm.EncoderDecoderLSTM,
    "VanillaLstm": dm.VanillaLSTM,
    "LinearRegression": dm.LinearRegression,
    "RandomForest": dm.RandomForestRegression,
    "SVR": dm.SupportVectorRegression,
}

normalization_to_class = {
    None: dm.processing.IdentityScaler,
    "standardize": dm.processing.Standardizer,
    "normalize": dm.processing.Normalizer,
    "standardize-robust": dm.processing.RobustStandardizer,
}

def select_data(
    data: pd.DataFrame,
    input_features: List[str],
    target_features: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not input_features:
        input_features = all_features
    if not target_features:
        target_features = ['energy']

    feature_data = {
        'energy': data['energy'],
        'water': data['water'],
        'occupants': data['registrations'],
        'temperature': data['temperature'],
        'daylight': data['daylight'],
        'datetime': data[['weekday', 'holiday','uni_holiday','time']],
    }
    
    input_data = pd.concat(
        [dataprocessors.numeric.interpolate(feature_data[feature]) for feature in input_features],
        axis=1
    )
    
    target_data = pd.concat(
        [dataprocessors.numeric.interpolate(feature_data[feature]) for feature in target_features],
        axis=1
    )
    
    return input_data, target_data


def run_experiment(
    data: pd.DataFrame,
    model_class: str,
    normalization: str,
    input_features: List[str],
    target_features: List[str],
    lookback_horizon: int,
    prediction_horizon: int,
    training_split: float,
    epochs: int,
    batch_size: int,
    validation_split: float,
) -> ExperimentResult:
    if any([f for f in input_features if f in target_features]) and prediction_horizon == 0:
        raise RuntimeError(
            f"The features that are predicted, i.e. {target_features} "
            f"by the network are also present in features fed into the network: i.e. {input_features}."
            f"prediction horizon = 0 "
            f"⟹ we are trying to 'predict' the value of a feature already in the input to the model"
        )

    """
    Split data

    """
    input_data, target_data = select_data(data, input_features, target_features)
    
    input_train, input_test = dataprocessors.shape.split(
        input_data, training_split
    )
    target_train, target_test = dataprocessors.shape.split(
        target_data, training_split
    )

    index_train = input_train.iloc[lookback_horizon + prediction_horizon:].index.to_series()
    index_test = input_test.iloc[lookback_horizon + prediction_horizon:].index.to_series()

    """
    Reshape data

    """
    x_train, y_train = dm.processing.shape.split_into_target_segments(
        features=input_train.to_numpy(),
        targets=target_train.to_numpy(),
        lookback_horizon=lookback_horizon,
        prediction_horizon=prediction_horizon,
    )

    x_test, y_test = dm.processing.shape.split_into_target_segments(
        features=input_test.to_numpy(),
        targets=target_test.to_numpy(),
        lookback_horizon=lookback_horizon,
        prediction_horizon=prediction_horizon,
    )

    """
    Instantiate model

    """

    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=30, restore_best_weights=True
    )

    def network_train_function(model, x_train, y_train):
        return model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping_callback],
        )

    model = model_to_class[model_class](
        x_scaler_class=normalization_to_class[normalization],
        train_function=network_train_function,
    )


    """
    Train and predict

    """
    print(
        f'training {model_class} model with\n',
        f'input features: {input_features}, shaped: {x_train.shape}\n',
        f'target features: {target_features}, shaped: {y_train.shape}'
    )

    model.train(x_train, y_train)

    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    metrics_train = dm.validation.metrics.all_metrics(y_train, y_pred_train)
    metrics_test = dm.validation.metrics.all_metrics(y_test, y_pred_test)


    return ExperimentResult(
        model_class,
        input_features,
        target_features,
        lookback_horizon,
        prediction_horizon,
        metrics_train,
        metrics_test,
        index_train,
        y_train,
        y_pred_train,
        index_test,
        y_test,
        y_pred_test
    )