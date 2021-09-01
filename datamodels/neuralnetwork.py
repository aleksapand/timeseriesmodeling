import json

import matplotlib.pyplot as plt
from matplotlib import ticker
from tensorflow import keras
from tensorflow.keras import layers

from datamodels import Model


def build_model(input_shape: tuple, target_shape: tuple) -> keras.Model:
    """
    Do NOT change it here if you want to use a different build function.
    Create your own build function that matches this function's signature and pass it to the network's
    constructor.

    :param input_shape: shape of one input sample, e.g. (lookback, num_features)
    :param target_shape: shape of one target vector, e.g. 1 for a single target value
    :return: a keras model
    """
    hidden_layer_size = 64
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Flatten(),
            layers.Dense(units=hidden_layer_size, activation='relu'),
            layers.Dense(units=hidden_layer_size, activation='relu'),
            layers.Dense(units=target_shape[0]),
        ]
    )


def compile_model(model: keras.Model):
    """
    Do NOT change it here if you want to use a different build function.
    Create your own compile function that matches this function's signature and pass it to the network's
    constructor.

    :param model: a keras model
    """
    optimizer = keras.optimizers.RMSprop()
    model.compile(loss='mse', optimizer=optimizer)


def train_model(model, x_train, y_train) -> keras.callbacks.History:
    """
    Do NOT change it here if you want to use a different build function.
    Create your own train function that matches this function's signature and pass it to the network's
    constructor.

    :param x_train, y_train: feature and target vectors
    """

    return model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=24,
        validation_split=0.2
    )


class NeuralNetwork(Model):
    """
    The NeuralNetwork class acts as a wrapper for networks.
    A neural network can be customized by passing a custom function for any of the three parameters in the constructor.
    Per default the sample implementation above are used to build, compile and train the model.

    Do NOT change the functions here if you want a different Network, instead in the file where you use the network add
    the three functions and pass it to the constructor in this class.
    """

    def __init__(
            self,
            build_function=build_model,
            compile_function=compile_model,
            train_function=train_model,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.build_function = build_function
        self.compile_function = compile_function
        self.train_function = train_function

        self.model = None
        self.history = None

    def train_model(self, x_train, y_train, **kwargs):
        input_shape = x_train.shape[1:]
        target_shape = y_train.shape[1:]

        self.model = self.build_function(input_shape, target_shape)

        self.compile_function(self.model)

        print(self.model.summary())

        self.history = self.train_function(self.model, x_train, y_train).history

    def plot_loss(self):
        if not self.history:
            raise RuntimeError('You must train the model first.')

        fig, ax = plt.subplots()
        plt.title('Model loss')

        loss = self.history['loss']
        ax.plot(loss, label='training loss')

        if 'val_loss' in self.history:
            val_loss = self.history['val_loss']
            ax.plot(val_loss, label='validation loss')

        plt.xlabel('epochs')
        plt.ylabel('loss')
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=1.0))

        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    def predict_model(self, x):
        return self.model.predict(x)

    def save(self, path='sequential_models/sequential_model'):
        super(NeuralNetwork, self).save(path)
        with open(f'{path}/architecture.json', 'w') as file:
            json.dump(self.model.to_json(), file)

        self.model.save_weights(f'{path}/weights.h5')

    def load_model(self, path='sequential_models/sequential_model'):
        super(NeuralNetwork, self).load_model(path)
        with open(f'{path}/architecture.json', 'r') as file:
            self.model = keras.models.model_from_json(json.load(file))

        self.model.load_weights(f'{path}/weights.h5')
