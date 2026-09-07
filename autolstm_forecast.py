from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoLSTM, OptunaOptions
from neuralforecast.models import LSTM


DATA_PATH = Path("data/target-hospital-admissions.csv")
RESULTS_DIR = Path("forecast_results")
PLOTS_DIR = Path("plots")

TRAIN_END = pd.Timestamp("2025-09-30")
TEST_START = pd.Timestamp("2025-10-01")
TEST_END = pd.Timestamp("2026-05-31")

HORIZON = 4
VALIDATION_SIZE = 16
NUM_SAMPLES = 5

FREQ = "W-SAT"


def prepare_national_weekly_rate():
    """
    Load national US weekly hospitalization-rate data
    in NeuralForecast format.
    """

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"],
    )

    us_df = (
        df[
            df["location_name"] == "US"
        ]
        .copy()
        .sort_values("date")
    )

    us_df = us_df.dropna(
        subset=["weekly_rate"]
    )

    nf_df = pd.DataFrame(
        {
            "unique_id": "US",
            "ds": us_df["date"],
            "y": us_df["weekly_rate"],
        }
    )

    return nf_df.reset_index(drop=True)


def split_data(full_df):
    """
    Keep the same train/test split used by the
    manual LSTM baseline.
    """

    train_df = full_df[
        full_df["ds"] <= TRAIN_END
    ].copy()

    test_df = full_df[
        (full_df["ds"] >= TEST_START)
        & (full_df["ds"] <= TEST_END)
    ].copy()

    evaluation_df = full_df[
        full_df["ds"] <= TEST_END
    ].copy()

    return (
        train_df,
        test_df,
        evaluation_df,
    )


def build_autolstm_config(trial):
    """
    Optuna search space for AutoLSTM.

    NeuralForecast's Optuna backend requires config
    to be a callable that accepts an Optuna trial
    and returns a model configuration dictionary.
    """

    return {
        "input_size": trial.suggest_categorical(
            "input_size",
            [12, 16, 24, 32, 52],
        ),
        "inference_input_size": -1,
        "encoder_hidden_size": trial.suggest_categorical(
            "encoder_hidden_size",
            [16, 32, 64],
        ),
        "encoder_n_layers": trial.suggest_int(
            "encoder_n_layers",
            1,
            3,
        ),
        "context_size": trial.suggest_categorical(
            "context_size",
            [5, 10],
        ),
        "decoder_hidden_size": trial.suggest_categorical(
            "decoder_hidden_size",
            [16, 32, 64],
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            1e-4,
            1e-2,
            log=True,
        ),
        "max_steps": trial.suggest_categorical(
            "max_steps",
            [300, 500, 1000],
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size",
            [16, 32],
        ),
        "random_seed": trial.suggest_int(
            "random_seed",
            1,
            20,
        ),
        "scaler_type": "standard",
    }


def tune_autolstm(train_df):
    """
    Tune AutoLSTM using Optuna on pre-test data only.

    The final Oct 2025-May 2026 test period is not
    included in hyperparameter selection.
    """

    print(
        "\n--- AUTOLSTM OPTUNA TUNING ---"
    )

    print(
        f"Training observations: {len(train_df)}"
    )

    print(
        "Training date range: "
        f"{train_df['ds'].min().date()} "
        "to "
        f"{train_df['ds'].max().date()}"
    )

    print(
        f"Validation size: {VALIDATION_SIZE} weeks"
    )

    print(
        f"Optuna trials: {NUM_SAMPLES}"
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
        h=HORIZON,
        backend="optuna",
        config=build_autolstm_config,
        num_samples=NUM_SAMPLES,
        refit_with_val=False,
        optuna_options=optuna_options,
        verbose=True,
        alias="AutoLSTM",
    )

    nf = NeuralForecast(
        models=[auto_model],
        freq=FREQ,
    )

    nf.fit(
        df=train_df,
        val_size=VALIDATION_SIZE,
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

    config.pop(
        "h",
        None,
    )

    config.pop(
        "loss",
        None,
    )

    config.pop(
        "valid_loss",
        None,
    )

    config.pop(
        "alias",
        None,
    )

    config.setdefault(
        "scaler_type",
        "standard",
    )

    return config


def build_fixed_lstm(best_config):
    """
    Convert the winning AutoLSTM configuration into
    a normal LSTM for fixed-weight test evaluation.
    """

    config = clean_best_config(
        best_config
    )

    model = LSTM(
        h=HORIZON,
        **config,
    )

    return model


def fit_fixed_model(
    train_df,
    best_config,
):
    """
    Fit the selected LSTM once on all pre-test data.
    """

    print(
        "\n--- FITTING SELECTED LSTM ON "
        "PRE-TEST DATA ---"
    )

    model = build_fixed_lstm(
        best_config
    )

    nf = NeuralForecast(
        models=[model],
        freq=FREQ,
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
    Evaluate 1-4 week forecasts across the untouched
    test period without updating model weights.

    Later forecast origins may use observations that
    would have been available by that origin, but
    refit=False prevents weight updates.
    """

    test_size = len(
        test_df
    )

    print(
        "\n--- FIXED-WEIGHT TEST EVALUATION ---"
    )

    print(
        f"Test observations: {test_size}"
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
            "Expected exactly one forecast column, "
            f"found: {prediction_columns}"
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
    Save a compact table of Optuna trials.
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
):
    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_path = (
        PLOTS_DIR
        / "autolstm_fixed_us_forecast_test_period.png"
    )

    test_actuals = evaluation_df[
        (
            evaluation_df["ds"]
            >= TEST_START
        )
        & (
            evaluation_df["ds"]
            <= TEST_END
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

    plt.xlabel(
        "Date"
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
        "AutoLSTM Fixed-Weight Error by Forecast Horizon"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        plot_path,
        dpi=150,
    )

    plt.close()

    return plot_path


def main():
    print(
        "--- LOADING NATIONAL FLUSIGHT DATA ---"
    )

    full_df = (
        prepare_national_weekly_rate()
    )

    print(
        f"Observations: {len(full_df)}"
    )

    print(
        "Date range: "
        f"{full_df['ds'].min().date()} "
        "to "
        f"{full_df['ds'].max().date()}"
    )

    print(
        "\nForecast target: weekly_rate"
    )

    (
        train_df,
        test_df,
        evaluation_df,
    ) = split_data(
        full_df
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
        train_df
    )

    fixed_nf = fit_fixed_model(
        train_df,
        best_config,
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
        f"forecast_plot: {forecast_plot}"
    )

    print(
        f"metrics_plot: {metrics_plot}"
    )


if __name__ == "__main__":
    main()