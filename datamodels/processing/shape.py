from typing import Tuple

import numpy as np
import copy


def prevent_zeros(value):
    """
    this ensures that the value does not contain zeros.

    it can be used to prevent division by zero.

    :param value: scalar or array-like
    :return: value without zeros
    """
    if np.isscalar(value):
        return value if value != 0 else 1.

    corrected_value = copy.deepcopy(value)
    corrected_value[corrected_value == 0] = 1.0
    return corrected_value


def split(data: np.ndarray, frac: float) -> Tuple[np.ndarray, np.ndarray]:
    if not 0 <= frac <= 1:
        raise ValueError('Invalid fraction, must be between 0 and 1')

    index = int(frac * data.shape[0])

    if data.ndim == 1:
        first = data[:index]
        second = data[index:]
        return first, second

    if data.ndim > 1:
        first = data[:index, :]
        second = data[index:, :]
        return first, second


def split_into_target_segments(
        features: np.ndarray,
        targets: np.ndarray,
        lookback_horizon: int,
        prediction_horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param features: an array containing the features, this can be previous system states (i.e. targets) and/or
                     other inputs.

    :param targets: an array containing one ore more target values.

    :param lookback_horizon: length of the lookback. resulting feature segments will be of of size lookback + 1
                             for 0, this is: (f_t_0 ... f_t_n)
                             for 1, it is: ((f_t_0 ... f_t_n), (f_t-1_0 ... f_t-1_n))

    :param prediction_horizon: prediction happens at a single point in time, the prediction horizon is
                               the offset between the end of the feature segment and the target, i.e.
                               feature segments always contain samples from lookback horizon to t=0,
                               if prediction_horizon is set to 0 target and feature segment both contain the value
                               associated with the current time step.

    :return: two numpy arrays:
     a) the feature_batches: an array of arrays of size lookback_horizon x (number of features + label value)
     b) the labels: an array of single number labels
    """

    if not features.ndim == 2:
        raise RuntimeError(f'features must have shape (seq_length, num_features), '
                           f'but has {features.shape}')

    if not targets.ndim == 2:
        raise RuntimeError(f'targets must have shape (seq_length, num_target_features), '
                           f'but has {targets.shape}')

    if not features.shape[0] == targets.shape[0]:
        raise RuntimeError(f'features and targets must have the same length.\n'
                           f'features has: {features.shape}, targets has: {targets.shape}')

    samples = features.shape[0]
    start_index = lookback_horizon
    end_index = samples - prediction_horizon

    if not start_index < end_index:
        raise RuntimeError(f'there are not enough samples in features and targets for these horizons.\n'
                           f'samples: {samples}\n'
                           f'lookback_horizon: {lookback_horizon}, predicion_horizon {prediction_horizon}\n'
                           f'you need at least {lookback_horizon + prediction_horizon + 1} sample(s).')

    feature_list = []
    target_list = []

    for i in range(start_index, end_index):
        feature_list.append(features[i - lookback_horizon: i + 1])
        target_list.append(targets[i + prediction_horizon])

    feature_segments = np.array(feature_list)
    target_segments = np.array(target_list)

    return feature_segments, target_segments
