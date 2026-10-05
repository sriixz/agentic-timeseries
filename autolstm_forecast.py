import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from epiweeks import Week
from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoLSTM, OptunaOptions
from neuralforecast.models import LSTM


DATA_PATH = Path("data/target-hospital-admissions.csv")
CONFIG_PATH = Path("configs/generated_autolstm_config.json")
RESULTS_DIR = Path("forecast_results")
PLOTS_DIR = Path("plots")


def load_experiment_config():
    """
    Load the experiment specification and Optuna search
    space from JSON.
    """

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    required_fields = {
        "model",
        "location",
        "target",
        "train_end",
        "test_start",
        "test_end",
        "horizon",
        "validation_size",
        "num_samples",
        "frequency",
        "search_space",
    }

    missing_fields = (
        required_fields
        - set(config.keys())
    )

    if missing_fields:
        raise ValueError(
            "Configuration is missing required fields: "
            f"{sorted(missing_fields)}"
        )

    if config["model"] != "LSTM":
        raise ValueError(
            "This trainer currently supports only LSTM. "
            f"Received: {config['model']}"
        )

    return config


def prepare_national_series(experiment_config):
    """
    Load the configured FluSight target and location in
    NeuralForecast format.
    """

    location = experiment_config["location"]
    target = experiment_config["target"]

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"],
    )

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' "
            "was not found in the dataset."
        )

    location_df = (
        df[
            df["location_name"]
            == location
        ]
        .copy()
        .sort_values("date")
    )

    if location_df.empty:
        raise ValueError(
            f"No observations found for location: "
            f"{location}"
        )

    location_df = location_df.dropna(
        subset=[target]
    )

    nf_df = pd.DataFrame(
        {
            "unique_id": location,
            "ds": location_df["date"],
            "y": location_df[target],
        }
    )

    return nf_df.reset_index(drop=True)


def split_data(
    full_df,
    experiment_config,
):
    """
    Split the data using dates supplied by the
    experiment configuration.
    """

    train_end = pd.Timestamp(
        experiment_config["train_end"]
    )

    test_start = pd.Timestamp(
        experiment_config["test_start"]
    )

    test_end = pd.Timestamp(
        experiment_config["test_end"]
    )

    train_df = full_df[
        full_df["ds"] <= train_end
    ].copy()

    test_df = full_df[
        (
            full_df["ds"] >= test_start
        )
        & (
            full_df["ds"] <= test_end
        )
    ].copy()

    evaluation_df = full_df[
        full_df["ds"] <= test_end
    ].copy()

    if train_df.empty:
        raise ValueError(
            "Training split is empty."
        )

    if test_df.empty:
        raise ValueError(
            "Test split is empty."
        )

    return (
        train_df,
        test_df,
        evaluation_df,
    )


def suggest_parameter(
    trial,
    parameter_name,
    specification,
):
    """
    Convert one JSON search-space specification into
    the corresponding Optuna suggestion.

    Supported types:
    - fixed
    - categorical
    - int
    - float
    """

    parameter_type = specification["type"]

    if parameter_type == "fixed":
        return specification["value"]

    if parameter_type == "categorical":
        return trial.suggest_categorical(
            parameter_name,
            specification["values"],
        )

    if parameter_type == "int":
        kwargs = {}

        if "step" in specification:
            kwargs["step"] = (
                specification["step"]
            )

        if "log" in specification:
            kwargs["log"] = (
                specification["log"]
            )

        return trial.suggest_int(
            parameter_name,
            specification["low"],
            specification["high"],
            **kwargs,
        )

    if parameter_type == "float":
        kwargs = {}

        if "log" in specification:
            kwargs["log"] = (
                specification["log"]
            )

        if "step" in specification:
            kwargs["step"] = (
                specification["step"]
            )

        return trial.suggest_float(
            parameter_name,
            specification["low"],
            specification["high"],
            **kwargs,
        )

    raise ValueError(
        f"Unsupported search-space type "
        f"'{parameter_type}' for "
        f"'{parameter_name}'."
    )


def build_autolstm_config_factory(
    experiment_config,
):
    """
    Build the callable required by NeuralForecast's
    Optuna backend.

    The search space itself comes entirely from the
    external JSON configuration.
    """

    search_space = (
        experiment_config["search_space"]
    )

    def build_autolstm_config(trial):
        model_config = {}

        for (
            parameter_name,
            specification,
        ) in search_space.items():

            model_config[
                parameter_name
            ] = suggest_parameter(
                trial,
                parameter_name,
                specification,
            )

        return model_config

    return build_autolstm_config


def tune_autolstm(
    train_df,
    experiment_config,
):
    """
    Tune AutoLSTM using the search space supplied by
    the external configuration file.

    The final test period remains excluded from
    hyperparameter selection.
    """

    horizon = experiment_config["horizon"]

    validation_size = (
        experiment_config[
            "validation_size"
        ]
    )

    num_samples = (
        experiment_config[
            "num_samples"
        ]
    )

    frequency = (
        experiment_config[
            "frequency"
        ]
    )

    optuna_config = (
        build_autolstm_config_factory(
            experiment_config
        )
    )

    print(
        "\n--- AUTOLSTM OPTUNA TUNING ---"
    )

    print(
        f"Training observations: "
        f"{len(train_df)}"
    )

    print(
        "Training date range: "
        f"{train_df['ds'].min().date()} "
        "to "
        f"{train_df['ds'].max().date()}"
    )

    print(
        f"Validation size: "
        f"{validation_size} weeks"
    )

    print(
        f"Optuna trials: "
        f"{num_samples}"
    )

    print(
        f"Configuration source: "
        f"{CONFIG_PATH}"
    )

    optuna_options = OptunaOptions(
        create_study_kwargs={
            "study_name":
                "autolstm_flu_weekly_rate",
        },
        study_kwargs={
            "gc_after_trial": True,
        },
    )

    auto_model = AutoLSTM(
        h=horizon,
        backend="optuna",
        config=optuna_config,
        num_samples=num_samples,
        refit_with_val=False,
        optuna_options=optuna_options,
        verbose=True,
        alias="AutoLSTM",
    )

    nf = NeuralForecast(
        models=[auto_model],
        freq=frequency,
    )

    nf.fit(
        df=train_df,
        val_size=validation_size,
    )

    fitted_auto = nf.models[0]
    study = fitted_auto.results
    best_trial = study.best_trial

    best_config = (
        best_trial.user_attrs[
            "ALL_PARAMS"
        ].copy()
    )

    best_validation_loss = (
        best_trial.value
    )

    print(
        "\n--- OPTUNA BEST TRIAL ---"
    )

    print(
        f"Best trial number: "
        f"{best_trial.number}"
    )

    print(
        "Best validation loss: "
        f"{best_validation_loss:.6f}"
    )

    print(
        "\nBest configuration:"
    )

    for key, value in sorted(
        best_config.items()
    ):
        print(
            f"{key}: {value}"
        )

    return (
        best_config,
        best_validation_loss,
        study,
    )


def clean_best_config(best_config):
    """
    Remove AutoLSTM-only or duplicated fields before
    creating a standard LSTM instance.
    """

    config = best_config.copy()

    for field in [
        "h",
        "loss",
        "valid_loss",
        "alias",
    ]:
        config.pop(
            field,
            None,
        )

    config.setdefault(
        "scaler_type",
        "standard",
    )

    return config


def build_fixed_lstm(
    best_config,
    experiment_config,
):
    """
    Convert the winning AutoLSTM configuration into
    a standard LSTM for fixed-weight evaluation.
    """

    config = clean_best_config(
        best_config
    )

    model = LSTM(
        h=experiment_config["horizon"],
        **config,
    )

    return model


def fit_fixed_model(
    train_df,
    best_config,
    experiment_config,
):
    """
    Fit the selected LSTM once on all pre-test data.
    """

    print(
        "\n--- FITTING SELECTED LSTM ON "
        "PRE-TEST DATA ---"
    )

    model = build_fixed_lstm(
        best_config,
        experiment_config,
    )

    nf = NeuralForecast(
        models=[model],
        freq=experiment_config[
            "frequency"
        ],
    )

    nf.fit(
        df=train_df
    )

    return nf


def evaluate_fixed_model(
    nf,
    evaluation_df,
    test_df,
):
    """
    Evaluate forecasts across the untouched test
    period without updating model weights.
    """

    test_size = len(
        test_df
    )

    print(
        "\n--- FIXED-WEIGHT TEST EVALUATION ---"
    )

    print(
        f"Test observations: "
        f"{test_size}"
    )

    print(
        "Test date range: "
        f"{test_df['ds'].min().date()} "
        "to "
        f"{test_df['ds'].max().date()}"
    )

    cv_df = nf.cross_validation(
        df=evaluation_df,
        n_windows=None,
        test_size=test_size,
        step_size=1,
        refit=False,
        verbose=False,
    )

    prediction_columns = [
        column
        for column in cv_df.columns
        if column
        not in {
            "unique_id",
            "ds",
            "cutoff",
            "y",
        }
    ]

    if len(prediction_columns) != 1:
        raise ValueError(
            "Expected exactly one forecast "
            "column, found: "
            f"{prediction_columns}"
        )

    prediction_column = (
        prediction_columns[0]
    )

    cv_df = cv_df.rename(
        columns={
            prediction_column:
                "forecast"
        }
    )

    cv_df["horizon_weeks"] = (
        (
            cv_df["ds"]
            - cv_df["cutoff"]
        ).dt.days
        // 7
    )

    cv_df["absolute_error"] = (
        cv_df["forecast"]
        - cv_df["y"]
    ).abs()

    cv_df["squared_error"] = (
        cv_df["forecast"]
        - cv_df["y"]
    ) ** 2

    metrics = (
        cv_df
        .groupby(
            "horizon_weeks",
            as_index=False,
        )
        .agg(
            forecast_count=(
                "forecast",
                "count",
            ),
            mae=(
                "absolute_error",
                "mean",
            ),
            mse=(
                "squared_error",
                "mean",
            ),
        )
    )

    metrics["rmse"] = np.sqrt(
        metrics["mse"]
    )

    metrics = metrics[
        [
            "horizon_weeks",
            "forecast_count",
            "mae",
            "rmse",
        ]
    ]

    return (
        cv_df,
        metrics,
    )


def build_trial_table(study):
    """
    Create a compact table of Optuna trials.
    """

    rows = []

    for trial in study.trials:
        row = {
            "trial_number":
                trial.number,
            "validation_loss":
                trial.value,
            "state":
                str(trial.state),
        }

        for key, value in (
            trial.params.items()
        ):
            row[
                f"param_{key}"
            ] = value

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def save_results(
    cv_df,
    metrics,
    study,
    best_config,
    best_validation_loss,
):
    """
    Save forecasts, metrics, Optuna trials, and the
    winning configuration.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecast_path = (
        RESULTS_DIR
        / "autolstm_fixed_forecasts.csv"
    )

    metrics_path = (
        RESULTS_DIR
        / "autolstm_fixed_metrics_by_horizon.csv"
    )

    trials_path = (
        RESULTS_DIR
        / "autolstm_optuna_trials.csv"
    )

    config_path = (
        RESULTS_DIR
        / "autolstm_best_config.csv"
    )

    cv_df.to_csv(
        forecast_path,
        index=False,
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    trial_df = build_trial_table(
        study
    )

    trial_df.to_csv(
        trials_path,
        index=False,
    )

    config_rows = [
        {
            "parameter":
                key,
            "value":
                value,
        }
        for key, value
        in sorted(
            best_config.items()
        )
    ]

    config_rows.append(
        {
            "parameter":
                "best_validation_loss",
            "value":
                best_validation_loss,
        }
    )

    pd.DataFrame(
        config_rows
    ).to_csv(
        config_path,
        index=False,
    )

    return {
        "forecasts":
            forecast_path,
        "metrics":
            metrics_path,
        "trials":
            trials_path,
        "best_config":
            config_path,
    }


def plot_test_forecasts(
    evaluation_df,
    cv_df,
    experiment_config,
):
    """
    Plot observed values and one-week-ahead forecasts
    across the configured test period using epiweek
    labels on the x-axis.
    """

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_path = (
        PLOTS_DIR
        / "autolstm_fixed_us_forecast_test_period.png"
    )

    test_start = pd.Timestamp(
        experiment_config["test_start"]
    )

    test_end = pd.Timestamp(
        experiment_config["test_end"]
    )

    test_actuals = evaluation_df[
        (
            evaluation_df["ds"]
            >= test_start
        )
        & (
            evaluation_df["ds"]
            <= test_end
        )
    ].copy()

    horizon_one = cv_df[
        cv_df["horizon_weeks"] == 1
    ].copy()

    plt.figure(
        figsize=(11, 6)
    )

    plt.plot(
        test_actuals["ds"],
        test_actuals["y"],
        label="Observed weekly rate",
    )

    plt.plot(
        horizon_one["ds"],
        horizon_one["forecast"],
        label="AutoLSTM 1-week forecast",
    )

    tick_dates = (
        test_actuals["ds"]
        .iloc[::4]
        .tolist()
    )

    if (
        len(test_actuals) > 0
        and test_actuals["ds"].iloc[-1]
        not in tick_dates
    ):
        tick_dates.append(
            test_actuals["ds"].iloc[-1]
        )

    tick_labels = []

    for date_value in tick_dates:
        epiweek = Week.fromdate(
            date_value.date()
        )

        tick_labels.append(
            f"{epiweek.year}-EW{epiweek.week:02d}"
        )

    plt.xticks(
        tick_dates,
        tick_labels,
        rotation=45,
        ha="right",
    )

    plt.xlabel(
        "Epidemiological week"
    )

    plt.ylabel(
        "Weekly hospitalization rate"
    )

    plt.title(
        "AutoLSTM Fixed-Weight US Flu Forecast "
        "During Test Period"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        plot_path,
        dpi=150,
    )

    plt.close()

    return plot_path


def plot_metrics(metrics):
    """
    Plot MAE and RMSE by forecast horizon.
    """

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_path = (
        PLOTS_DIR
        / "autolstm_fixed_metrics_by_horizon.png"
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        metrics["horizon_weeks"],
        metrics["mae"],
        marker="o",
        label="MAE",
    )

    plt.plot(
        metrics["horizon_weeks"],
        metrics["rmse"],
        marker="o",
        label="RMSE",
    )

    plt.xticks(
        metrics["horizon_weeks"]
    )

    plt.xlabel(
        "Forecast horizon (weeks)"
    )

    plt.ylabel(
        "Error"
    )

    plt.title(
        "AutoLSTM Fixed-Weight Error "
        "by Forecast Horizon"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        plot_path,
        dpi=150,
    )

    plt.close()

    return plot_path


def print_experiment_config(
    experiment_config,
):
    """
    Print the external experiment configuration before
    training begins.
    """

    print(
        "\n--- EXPERIMENT CONFIGURATION ---"
    )

    print(
        f"Model: "
        f"{experiment_config['model']}"
    )

    print(
        f"Location: "
        f"{experiment_config['location']}"
    )

    print(
        f"Target: "
        f"{experiment_config['target']}"
    )

    print(
        f"Training through: "
        f"{experiment_config['train_end']}"
    )

    print(
        "Test period: "
        f"{experiment_config['test_start']} "
        "to "
        f"{experiment_config['test_end']}"
    )

    print(
        f"Horizon: "
        f"{experiment_config['horizon']} weeks"
    )

    print(
        "\nSearch space:"
    )

    for (
        parameter,
        specification,
    ) in experiment_config[
        "search_space"
    ].items():

        print(
            f"  {parameter}: "
            f"{specification}"
        )


def main():
    experiment_config = (
        load_experiment_config()
    )

    print_experiment_config(
        experiment_config
    )

    print(
        "\n--- LOADING FLUSIGHT DATA ---"
    )

    full_df = prepare_national_series(
        experiment_config
    )

    print(
        f"Observations: "
        f"{len(full_df)}"
    )

    print(
        "Date range: "
        f"{full_df['ds'].min().date()} "
        "to "
        f"{full_df['ds'].max().date()}"
    )

    print(
        "\nForecast target: "
        f"{experiment_config['target']}"
    )

    (
        train_df,
        test_df,
        evaluation_df,
    ) = split_data(
        full_df,
        experiment_config,
    )

    print(
        "\n--- TRAIN / TEST SPLIT ---"
    )

    print(
        "Last pre-test training date: "
        f"{train_df['ds'].max().date()}"
    )

    print(
        "First test observation: "
        f"{test_df['ds'].min().date()}"
    )

    print(
        "Last test observation: "
        f"{test_df['ds'].max().date()}"
    )

    (
        best_config,
        best_validation_loss,
        study,
    ) = tune_autolstm(
        train_df,
        experiment_config,
    )

    fixed_nf = fit_fixed_model(
        train_df,
        best_config,
        experiment_config,
    )

    (
        cv_df,
        metrics,
    ) = evaluate_fixed_model(
        fixed_nf,
        evaluation_df,
        test_df,
    )

    print(
        "\n--- AUTOLSTM FIXED-WEIGHT METRICS ---"
    )

    print(
        metrics.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    paths = save_results(
        cv_df,
        metrics,
        study,
        best_config,
        best_validation_loss,
    )

    forecast_plot = (
        plot_test_forecasts(
            evaluation_df,
            cv_df,
            experiment_config,
        )
    )

    metrics_plot = (
        plot_metrics(
            metrics
        )
    )

    print(
        "\n--- SAVED OUTPUTS ---"
    )

    for label, path in paths.items():
        print(
            f"{label}: {path}"
        )

    print(
        f"forecast_plot: "
        f"{forecast_plot}"
    )

    print(
        f"metrics_plot: "
        f"{metrics_plot}"
    )


if __name__ == "__main__":
    main()