from abc import abstractmethod

import numpy as np

from datamodels.processing.shape import prevent_zeros


class DataScaler:

    @abstractmethod
    def fit(self, distribution):
        """
        for statistical models it is often necessary to scale datasets.

        to properly evaluate statistical models it is important that the validation data does not bias
        the training process, i.e. information about the validation/test data must not be used during the
        training phase. dataset metrics such as min, max, std, etc. are information about the data which is why
        validation data is typically scales with metrics from the training set.

        :param distribution: the distribution that supplies the metrics for scaling
        """
        raise NotImplementedError()

    @abstractmethod
    def transform(self, data):
        raise NotImplementedError()

    @abstractmethod
    def inverse_transform(self, data):
        raise NotImplementedError()


class IdentityScaler(DataScaler):

    def fit(self, distribution):
        return self

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class Normalizer(DataScaler):

    def __init__(self):
        self.min = None
        self.max = None
        self.scale = None

    def fit(self, distribution):
        self.min = np.nanmin(distribution, axis=0)
        self.max = np.nanmax(distribution, axis=0)
        self.scale = prevent_zeros(self.max - self.min)
        return self

    def transform(self, data):
        """
        data is scaled such that it is 0 <= data <= 1.
        in a feature vector this puts all features to the same scale.

        this is useful for data that is not Gaussian distributed and might help with convergence.
        features are more consistent, however it can be disadvantageous if the scale between features is
        important.

        :param data: array-like, pd.DataFrame, numpy-array
        :return: data, normalized between 0 and 1
        """
        assert self.min is not None and self.max is not None and self.scale is not None, \
            "cannot transform data, you must call .fit(distribution) first."

        return (data - self.min) / self.scale

    def inverse_transform(self, data):
        assert self.min is not None and self.max is not None and self.scale is not None, \
            "cannot transform data, you must call .fit(distribution) first."

        return data * self.scale + self.min


class Standardizer(DataScaler):

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, distribution):
        self.mean = np.nanmean(distribution, axis=0)
        self.std = prevent_zeros(np.nanstd(distribution, axis=0))
        return self

    def transform(self, data):
        """
        data is scaled to 0 mean and 1 standard deviation.
        mostly helpful when data follows a Gaussian distribution. for PCA features have to be centered around the
        mean.

        :param data: array-like, pd.DataFrame, numpy-array
        :return: data, scaled such that mean=0 and std=1
        """
        assert self.mean is not None and self.std is not None, \
            "cannot transform data, you must call .fit(distribution) first."

        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        assert self.mean is not None and self.std is not None, \
            "cannot transform data, you must call .fit(distribution) first."

        return data * self.std + self.mean


class RobustStandardizer(DataScaler):

    def __init__(self):
        self.median = None
        self.scale = None

    def fit(self, distribution):
        self.median = np.nanmedian(distribution, axis=0)
        q25 = np.nanquantile(distribution, .25, axis=0)
        q75 = np.nanquantile(distribution, .75, axis=0)
        self.scale = prevent_zeros(q75 - q25)
        return self

    def transform(self, data):
        """
        mean and variance are often influenced by outliers. using median and interquanitle range instead often
        improves standardization results.

        :param data: array-like, pd.DataFrame, numpy-array
        :return: standardized data, scaled by range between 1st and 3rd quantile
        """
        assert self.median is not None and self.scale is not None, \
            "cannot transform data, you must call .fit(distribution) first."

        return (data - self.median) / self.scale

    def inverse_transform(self, data):
        assert self.median is not None and self.scale is not None, \
            "cannot transform data, you must call .fit(distribution) first."

        return data * self.scale + self.median
