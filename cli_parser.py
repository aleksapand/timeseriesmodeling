import argparse


class Parser(argparse.ArgumentParser):

    def __init__(self, models, features, normalizers):
        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.add_argument(
            "--models",
            choices=models,
            required=False,
            help="if specified, run only these models, otherwise run all available models",
            default=[models[0]],
            type=str,
        )

        self.add_argument(
            "--features",
            choices=features,
            required=False,
            help="if specified, use only these features, otherwise run all available models",
            default=[features[0]],
            type=str,
        )

        self.add_argument(
            "--epochs",
            help="the maximum number of training algorithm passes being applied to the training partition of the dataset",
            default=10,
            type=int,
        )

        self.add_argument(
            "--normalization",
            choices=normalizers,
            required=False,
            help="if defined normalize input and output from model, otherwise use 'identity scaling'",
            default=None,
        )

        self.add_argument(
            "--lookback-horizons",
            dest="lookback_horizons",
            nargs="+",
            help="number of samples used in every step of the prediction",
            default=[24],
            type=int,
        )

        self.add_argument(
            "--prediction-horizons",
            dest="prediction_horizons",
            nargs="+",
            help="number of samples into the future predicted for 'lookback-horizon' samples fed into the model",
            type=int,
            default=[3],
        )

        self.add_argument(
            "--training_split",
            dest="training_split",
            help="fraction of the total data used for the training partition",
            default=0.8,
            type=float,
        )

        self.add_argument(
            "--validation-fraction",
            dest="validation_split",
            help="fraction of the TRAINING data reserved for validation. "
                 "i.e. the data will first be split into a training and validation set determined by 'training_split'. "
                 "The training fraction is then split using the defined fraction",
            default=0.2,
            type=float,
        )

        self.add_argument(
            "--batch-size",
            dest="batch_size",
            help="the batch size used during training",
            default=72,
            type=int,
        )

        self.add_argument(
            "--result-dir",
            dest="result_dir",
            help="directory into which the results of the experiments are stored",
            default="results",
        )

        plot_args = self.add_argument_group("Plotting")
        plot_args.add_argument(
            "--plot-weeks",
            dest="num_weeks",
            type=int,
            help="number of weeks to plot",
            required=False,
            default=1,
        )

        plot_args.add_argument(
            "--export-format",
            dest="plot_image_format",
            choices=["pdf", "png", "svg"],
            help="the format of the exported figures",
            default="png",
        )