from tensorflow import keras
from tensorflow.keras import layers

from datamodels import NeuralNetwork


def build_model(input_shape: tuple, target_shape: tuple) -> keras.Model:
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(units=16),
            layers.RepeatVector(target_shape[0]),
            layers.LSTM(units=16, activation="relu", return_sequences=True),
            layers.TimeDistributed(layers.Dense(units=8, activation="relu")),
            layers.TimeDistributed(layers.Dense(units=target_shape[0])),
            layers.Flatten(),
        ]
    )


class EncoderDecoderLSTM(NeuralNetwork):
    def __init__(self, build_function=build_model, **kwargs):
        super().__init__(build_function=build_function, **kwargs)
