from datamodels import Model
import pickle


class LinearRegression(Model):

    def __init__(self, parameters=None, **kwargs):
        super().__init__(**kwargs)

        if parameters is None:
            parameters = {"n_jobs": -1}

        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression(**parameters)

    def reshape_data(self, x):
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        return x

    def train_model(self, x_train, y_train, **kwargs):
        self.model.fit(x_train, y_train)

    def predict_model(self, x):
        return self.model.predict(x)

    def save(self, path="data/models/DUMMY.txt"):
        super(LinearRegression, self).save(path)
        with open(f'{path}/model.pickle', 'wb') as file:
            pickle.dump(self.model, file)

    def load_model(self, path="data/models/DUMMY.txt"):
        super(LinearRegression, self).load_model(path)
        with open(f'{path}/model.pickle', 'rb') as file:
            self.model = pickle.load(file)
