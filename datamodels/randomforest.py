import pickle

import numpy as np

from datamodels import Model


class RandomForestRegression(Model):

    def __init__(self, parameters=None, **kwargs):
        super().__init__(**kwargs)

        if parameters is None:
            parameters = {
                "bootstrap": True,
                "max_depth": None,
                "max_features": "auto",
                "min_samples_leaf": 1,
                "min_samples_split": 2,
                "n_estimators": 100,
            }

        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(**parameters)

    def train_model(self, x_train, y_train, **kwargs):
        if y_train.shape[1] == 1:
            y_train = y_train.ravel()

        self.model.fit(x_train, y_train)

    def reshape_data(self, x):
        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], -1)
        return x

    def predict_model(self, x):
        y = self.model.predict(x)
        if y.ndim == 1:
            y = y[:, np.newaxis]
        return y

    def save(self, path="random_forest/random_forest"):
        super(RandomForestRegression, self).save(path)
        with open(f'{path}/model.pickle', 'wb') as file:
            pickle.dump(self.model, file)

    def load_model(self, path="random_forest/random_forest"):
        super(RandomForestRegression, self).load_model(path)
        with open(f'{path}/model.pickle', 'rb') as file:
            self.model = pickle.load(file)
