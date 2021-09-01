import pickle

import numpy as np

from datamodels import Model


class SupportVectorRegression(Model):

    def __init__(self, parameters=None, **kwargs):
        super().__init__(**kwargs)

        if parameters is None:
            parameters = {
                "kernel": "rbf",
                "C": 100,
                "gamma": "auto",
                "epsilon": 0.01
            }

        from sklearn.svm import SVR
        self.model = SVR(**parameters)

    def reshape_data(self, x):
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        return x

    def train_model(self, x_train, y_train, **kwargs):
        if y_train.shape[1] == 1:
            y_train = y_train.ravel()

        self.model.fit(x_train, y_train)

    def predict_model(self, x):
        y = self.model.predict(x)
        if y.ndim == 1:
            y = y[:, np.newaxis]
        return y

    def save(self, path="models/support_vector_regression.pickle"):
        with open(path, "wb") as file:
            pickle.dump(self.model, file)

    def load_model(self, path="models/support_vector_regression.pickle"):
        with open(path, "rb") as file:
            self.model = pickle.load(file)
