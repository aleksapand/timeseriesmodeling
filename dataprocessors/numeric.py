import numpy as np
import pandas as pd


def drop_non_numeric_values(data: pd.DataFrame) -> None:
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)


def downcast_to_int(data: pd.DataFrame) -> pd.DataFrame:
    for column in data.columns:
        data[column] = data[column].astype(int)
    return data


def encode_cyclic_data(df: pd.DataFrame, column_id: str, max_val: int, drop=True) -> None:
    df[column_id + '_sin'] = np.sin(2 * np.pi * df[column_id] / max_val)
    df[column_id + '_cos'] = np.cos(2 * np.pi * df[column_id] / max_val)
    if drop:
        df.drop(columns=[column_id], inplace=True)


def aggregate(data: pd.DataFrame, target_column: str, rule: str) -> pd.DataFrame:
    data.replace(np.nan, -np.inf, inplace=True)

    aggregator = {}
    for column in data.columns:
        if column is target_column:
            aggregator[column] = 'sum'
        else:
            aggregator[column] = 'first'

    data = data.resample(rule).agg(aggregator)

    data.replace(-np.inf, np.nan, inplace=True)
    return data


def interpolate(df: pd.DataFrame, method='linear') -> pd.DataFrame:
    # replace inf with NaN, so they are interpolated
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.interpolate(method=method, limit_direction='both')

    assert not df.isnull().values.any(), 'well that is awkward. maybe the entire dataframe contained NaNs only?'
    return df
