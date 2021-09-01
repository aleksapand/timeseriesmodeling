import numpy as np

from datamodels import Model


class Residual:

    def __init__(self, model: Model):
        self.model = model

    def train(self, x_train, y_train):
        if not x_train.shape[1] == 1:
            raise ValueError('Residual models can only handle input for the current timestep (i.e. loockback = 0)')

        y_train = np.diff(y_train, axis=0)
        x_train = x_train[1:]
        self.model.train(x_train, y_train)

    def predict(self, x, y_0):
        ys = y_0

        for xn in x:
            y_current = ys[-1]

            xm = np.append([y_current], xn, axis=0)
            xm = np.reshape(xm, (1, 1, xm.shape[0]))
            ym = self.model.predict(xm)
            y_next = y_current + ym[0]
            ys = np.append(ys, y_next, axis=0)

        return ys[1:][..., np.newaxis]
