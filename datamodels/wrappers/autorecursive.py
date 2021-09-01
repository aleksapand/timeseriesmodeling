import numpy as np

from datamodels import Model


class AutoRecursive:
    """
    This is a wrapper for autorecursive prediction.

    It takes a trained Model and, given an initial set of inputs,
    predicts a variable number of outputs into the future. This ONLY works for models where the number of
    input features MATCHES the number of output features, because the model output is used as the input, recursively.

    """

    def __init__(self, model: Model):
        self.model = model

    def predict(self, x, outputs):
        if not x.ndim == 2:
            raise ValueError('This wrapper takes only one sample, '
                             'i.e. one set of input features, shape: (lookback + 1, input_features)')

        x = np.expand_dims(x, 0)
        y = self.model.predict(x)

        if not x.shape[-1] == y.shape[-1]:
            raise ValueError('This model does not support autorecursive prediction because the number of input and'
                             f'output features does not match. in: {x.shape[-1]}, out: {y.shape[-1]}.')

        ys = y
        for _ in range(1, outputs):
            x = np.append(x[0, 1:], [ys[-1]], axis=0)
            x = np.expand_dims(x, 0)
            y = self.model.predict(x)
            ys = np.append(ys, y, axis=0)
        return ys
