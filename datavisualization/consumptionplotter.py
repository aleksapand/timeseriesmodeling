from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter, HourLocator, DayLocator
from matplotlib.ticker import Locator
from pandas.tseries.offsets import MonthBegin


class ConsumptionPlotter:

    @staticmethod
    def get_nan_filled_interval_data(data, interval_start, interval_end, index_frequency='1H'):
        data_index = pd.date_range(interval_start, interval_end, freq=index_frequency)
        df = pd.DataFrame(0, index=data_index, columns=data.columns)

        interval_data = data[interval_start: interval_end]
        return df.add(interval_data)

    @staticmethod
    def get_weekly_dfs(data: pd.DataFrame) -> list:
        sundays = data.resample('W').asfreq().index.to_series()
        # shift data by -1 so that the first value of the day, in our encoding at 01:00 (corresponding to the
        # consumption between 00:00) is timestamped with 00:00,
        data = data.shift(-1)

        dfs = []

        for sunday in sundays:
            monday = sunday - timedelta(days=6)
            sunday = sunday + timedelta(hours=23)

            dfs.append(ConsumptionPlotter.get_nan_filled_interval_data(data, monday, sunday))

        return dfs

    @staticmethod
    def get_monthly_dfs(data: pd.DataFrame) -> list:
        last_days_in_month = data.resample('M').asfreq().index.to_series()
        # shift data by -1 so that the first value of the day, in our encoding at 01:00 (corresponding to the
        # consumption between 00:00) is timestamped with 00:00,
        data = data.shift(-1)

        dfs = []

        for last_day_in_month in last_days_in_month:
            first_day_in_month = last_day_in_month - MonthBegin()
            last_day_in_month = last_day_in_month + timedelta(hours=23)

            dfs.append(ConsumptionPlotter.get_nan_filled_interval_data(data, first_day_in_month, last_day_in_month))

        return dfs

    @staticmethod
    def plot_line(data: pd.DataFrame,
                  title=None,
                  major_locator: Locator = None,
                  major_formatter=None,
                  minor_locator: Locator = None,
                  minor_formatter=None,
                  save_path=None):

        if np.isnan(data.to_numpy()).all():
            print('Nothing to print, all values in dataframe are Nan.')
            return

        fig, ax = plt.subplots()

        for column in data.columns:
            ax.plot(data[column], label=column)

        if major_locator: ax.xaxis.set_major_locator(major_locator)
        if major_formatter: ax.xaxis.set_major_formatter(major_formatter)

        if minor_locator: ax.xaxis.set_minor_locator(minor_locator)
        if minor_formatter: ax.xaxis.set_minor_formatter(minor_formatter)

        ax.set_xlim(data.index.min(), data.index.max())

        plt.title(title)
        plt.legend(loc='best')
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)

        plt.show()

    @staticmethod
    def plot_weekly(data, max_weeks=None):
        for i, df in enumerate(ConsumptionPlotter.get_weekly_dfs(data)):
            if max_weeks and i >= max_weeks:
                break
            monday = df.index.to_series().iloc[0]
            ConsumptionPlotter.plot_line(df,
                                         title=monday.strftime('%B %Y'),
                                         minor_locator=HourLocator(byhour=[4, 8, 12, 16, 20]),
                                         major_formatter=DateFormatter('%a\n %d.%m'))

    @staticmethod
    def plot_monthly(data, max_months=None):
        for i, df in enumerate(ConsumptionPlotter.get_monthly_dfs(data)):
            if max_months and i >= max_months:
                break

            first_day_in_month = df.index.to_series().iloc[0]
            ConsumptionPlotter.plot_line(df,
                                         title=first_day_in_month.strftime('%B %Y'),
                                         major_locator=DayLocator(interval=1),
                                         major_formatter=DateFormatter('%d'))

    @staticmethod
    def plot_training_and_test_data(train_df: pd.DataFrame, test_df: pd.DataFrame, 
        yaxis_formatter=None, xaxis_minor_formatter=HourLocator(byhour=[4, 8, 12, 16, 20]), 
        show=True
    ):
        fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
        margin = timedelta(hours=3)

        ax1.set_title("Predictions Training")
        for column in train_df.columns:
            ax1.plot(train_df[column], label=column)
        ax1.set_xlim(train_df.index.min() - margin, train_df.index.max() + margin)

        ax1.legend()

        ax2.set_title("Predictions Test")
        for column in test_df.columns:
            ax2.plot(test_df[column], label=column)
        ax2.set_xlim(test_df.index.min() - margin, test_df.index.max() + margin)

        ax2.legend()

        for axis in [ax1, ax2]:
            axis.xaxis.set_major_locator(DayLocator(interval=1))
            axis.xaxis.set_major_formatter(DateFormatter("%a\n %d.%m"))
            
            if xaxis_minor_formatter:
                axis.xaxis.set_minor_locator(xaxis_minor_formatter)
            if yaxis_formatter:
                axis.yaxis.set_major_formatter(yaxis_formatter)

        plt.tight_layout()
        if show:
            plt.show()
