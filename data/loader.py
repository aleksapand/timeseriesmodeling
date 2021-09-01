import pandas as pd

from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent


def load_data(
    file_path='data.csv'):

    file_path = get_project_root().joinpath(file_path)
    data = pd.read_csv(file_path, sep=None, engine='python')
    data['datetime'] = pd.to_datetime(data['datetime'], dayfirst=True)
    data.set_index('datetime', inplace=True)
    return data