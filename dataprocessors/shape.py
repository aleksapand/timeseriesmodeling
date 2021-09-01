from typing import Any, Tuple

import pandas as pd


def split(data: Any, frac: float) -> Tuple[Any, Any]:
    if not 0 <= frac <= 1:
        raise ValueError('Invalid fraction, must be between 0 and 1')

    index = int(frac * len(data.index))

    if isinstance(data, pd.Series):
        first = data.iloc[:index]
        second = data.iloc[index:]
        return first, second
    if isinstance(data, pd.DataFrame):
        first = data.iloc[:index, :]
        second = data.iloc[index:, :]
        return first, second

    raise Exception('array type not supported.')


def split_randomized(df: pd.DataFrame, frac: float, random_state) -> Tuple[pd.DataFrame, pd.DataFrame]:
    first = df.sample(frac=frac, random_state=random_state)
    second = df.drop(first.index)
    return first, second

