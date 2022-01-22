# supresses tensorflow warnings
from datavisualization import ConsumptionPlotter
import training
import data.loader as loader
from cli_parser import Parser
from datetime import datetime
from os import makedirs
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import pandas as pd
import json
from experiment import ExperimentResult
from numpy.lib.function_base import append
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def append_to_results(json_file_path, result: ExperimentResult, result_name: str):
    if json_file_path.is_file():
        with open(json_file_path, 'r') as fp:
            json_dict = json.load(fp)
    else:
        json_dict = {}

    with open(json_file_path, 'w') as fp:
        json_dict[result_name] = result.to_dict()
        json.dump(json_dict, fp, indent=4)


def run():

    parser = Parser(
        list(training.model_to_class.keys()),
        training.all_features,
        list(training.normalization_to_class.keys()))
    args = parser.parse_args()

    models = args.models
    lookback_horizons = args.lookback_horizons
    prediction_horizons = args.prediction_horizons
    normalization = args.normalization
    input_features = args.features
    target_features = ['energy']
    training_split = args.training_split
    epochs = args.epochs
    batch_size = args.batch_size
    validation_split = args.validation_split
    plot_image_format = args.plot_image_format

    results_dir_root = 'results'
    results_dir_run = Path(results_dir_root).joinpath(
        datetime.now().strftime("%Y%m%d-%H%M%S"))

    makedirs(results_dir_run)

    data = loader.load_data()

    experiments = [
        (m, l, p)
        for m in models
        for l in lookback_horizons
        for p in prediction_horizons
    ]

    for i, (model_class, lookback_horizon, prediction_horizon) in enumerate(experiments):

        result = training.run_experiment(
            data,
            model_class,
            normalization,
            input_features,
            target_features=target_features,
            lookback_horizon=lookback_horizon,
            prediction_horizon=prediction_horizon,
            training_split=training_split,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split)

        print('experiment successful, saving results.')

        result_name = f'experiment_{i}'
        """
        Write to results

        """
        append_to_results(results_dir_run.joinpath(
            'results.json'), result, result_name)

        """
        Generate and save plot
        
        """
        train_set = pd.DataFrame({
            'true': result.train_target.flatten(),
            'predicted': result.train_prediction.flatten()},
            index=result.train_index)

        test_set = pd.DataFrame({
            'true': result.test_target.flatten(),
            'predicted': result.test_prediction.flatten()},
            index=result.test_index)

        ConsumptionPlotter.plot_training_and_test_data(
            train_set,
            test_set,
            yaxis_formatter=FormatStrFormatter("%.2f kw/h"),
            xaxis_minor_formatter=None,
            show=True)

        with open(results_dir_run.joinpath(f'{result_name}.{plot_image_format}'), "wb") as fp:
            plt.savefig(fp, format=plot_image_format)


if __name__ == '__main__':
    run()
