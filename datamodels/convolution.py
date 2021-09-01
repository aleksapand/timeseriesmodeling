from tensorflow import keras
from tensorflow.keras import layers

from datamodels import NeuralNetwork


def build_model(input_shape: tuple, target_shape: tuple) -> keras.Model:
    return keras.Sequential(
        [
            layers.Input(input_shape),
            layers.Conv1D(filters=32,
                          kernel_size=3,
                          activation='relu'),
            layers.Dense(units=32, activation='relu'),
            layers.Flatten(),
            layers.Dense(target_shape[0]),
        ]
    )


class ConvolutionNetwork(NeuralNetwork):
    def __init__(self, build_function=build_model, **kwargs):
        super().__init__(build_function=build_function, **kwargs)
