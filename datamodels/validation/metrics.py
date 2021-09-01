import numpy as np

from datamodels.processing.shape import prevent_zeros


def cvrmse(y_true, y_pred):
    mse = np.square(np.subtract(y_true, y_pred)).mean()
    return np.sqrt(mse) / prevent_zeros(y_true.mean())


def mape(y_true, y_pred):
    return (100 * np.abs(y_true - y_pred) / prevent_zeros(y_true)).mean()


def all_metrics(y_true, y_pred):
    return {
        'CV-RMS': cvrmse(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
    }
