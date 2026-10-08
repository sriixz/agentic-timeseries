import json
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
import pandas as pd
from epiweeks import Week
from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoLSTM
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import LSTM
from optuna.trial import FixedTrial

from configs.generated_autolstm_config import autolstm_config


DATA_PATH = Path("data/target-hospital-admissions.csv")
EXPERIMENT_PATH = Path("configs/generated_autolstm_experiment.json")

RESULT_DIR = Path("forecast_results")
PLOT_DIR = Path("plots")


def load_experiment_config() -> dict:
    if not EXPERIMENT_PATH.exists():
        raise FileNotFoundError(
            f"Experiment metadata not found: {EXPERIMENT_PATH}\n"
            "Run python config_agent.py first."
        )

    with EXPERIMENT_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    required = {
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
    }

    missing = required - set(config)

    if missing:
        raise ValueError(
            f"Experiment config missing required keys: {sorted(missing)}"
        )

    if config["model"] != "LSTM":
        raise ValueError("Only LSTM is currently supported.")

    return config


def load_forecast_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"FluSight data not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {
        "date",
        "location",
        config["target"],
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset missing required columns: {sorted(missing)}"
        )

    df["date"] = pd.to_datetime(df["date"])

    location = str(config["location"])
    target = config["target"]

    scoped = df[
        df["location"].astype(str) == location
    ].copy()

    scoped = scoped.dropna(
        subset=["date", target]
    )

    scoped = scoped.sort_values("date")

    model_df = pd.DataFrame(
        {
            "unique_id": location,
            "ds": scoped["date"],
            "y": scoped[target].astype(float),
        }
    )

    train_end = pd.Timestamp(config["train_end"])
    test_start = pd.Timestamp(config["test_start"])
    test_end = pd.Timestamp(config["test_end"])

    train_df = model_df[
        model_df["ds"] <= train_end
    ].copy()

    test_df = model_df[
        (model_df["ds"] >= test_start)
        & (model_df["ds"] <= test_end)
    ].copy()

    if train_df.empty:
        raise ValueError("Training dataset is empty.")

    if test_df.empty:
        raise ValueError("Held-out test dataset is empty.")

    return train_df, test_df


def run_autolstm_search(
    train_df: pd.DataFrame,
    config: dict,
) -> tuple[NeuralForecast, optuna.study.Study]:
    horizon = int(config["horizon"])
    validation_size = int(config["validation_size"])
    num_samples = int(config["num_samples"])
    frequency = config["frequency"]

    print("\nRunning AutoLSTM / Optuna search")
    print("=================================")
    print(f"Horizon: {horizon}")
    print(f"Validation size: {validation_size}")
    print(f"Optuna trials: {num_samples}")
    print(
        "Config source: "
        "configs/generated_autolstm_config.py"
    )

    auto_model = AutoLSTM(
        h=horizon,
        loss=MAE(),
        valid_loss=MAE(),
        config=autolstm_config,
        backend="optuna",
        num_samples=num_samples,
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

    if not hasattr(fitted_auto, "results"):
        raise RuntimeError(
            "AutoLSTM finished, but no Optuna study was found."
        )

    study = fitted_auto.results

    return nf, study


def extract_best_config(
    study: optuna.study.Study,
) -> dict:
    """
    Re-run the generated config callable using Optuna's FixedTrial.

    The FixedTrial contains exactly the parameter values selected by
    the winning Optuna trial. This reconstructs the complete LSTM
    configuration, including fixed parameters such as scaler_type.
    """
    best_trial = study.best_trial

    fixed_trial = FixedTrial(best_trial.params)

    best_config = autolstm_config(fixed_trial)

    return best_config


def save_optuna_results(
    study: optuna.study.Study,
    best_config: dict,
) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    trials_path = RESULT_DIR / "autolstm_optuna_trials.csv"

    study.trials_dataframe().to_csv(
        trials_path,
        index=False,
    )

    best_path = RESULT_DIR / "autolstm_best_config.json"

    best_output = {
        "trial_number": study.best_trial.number,
        "validation_loss": study.best_value,
        "config": best_config,
    }

    with best_path.open("w", encoding="utf-8") as file:
        json.dump(
            best_output,
            file,
            indent=2,
        )

    print(f"\nSaved: {trials_path}")
    print(f"Saved: {best_path}")


def run_fixed_lstm_evaluation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_config: dict,
    config: dict,
) -> pd.DataFrame:
    horizon = int(config["horizon"])
    frequency = config["frequency"]

    print("\nRunning fixed LSTM held-out evaluation")
    print("======================================")

    model = LSTM(
        h=horizon,
        **best_config,
    )

    nf = NeuralForecast(
        models=[model],
        freq=frequency,
    )

    full_df = pd.concat(
        [train_df, test_df],
        ignore_index=True,
    )

    cv = nf.cross_validation(
    df=full_df,
    n_windows=None,
    test_size=len(test_df),
    step_size=1,
    refit=False,
)

    cv = cv.sort_values(
        ["cutoff", "ds"]
    ).reset_index(drop=True)

    cv["horizon"] = (
        cv.groupby("cutoff")
        .cumcount()
        .add(1)
    )

    cv = cv[
        cv["horizon"] <= horizon
    ].copy()

    test_start = pd.Timestamp(config["test_start"])
    test_end = pd.Timestamp(config["test_end"])

    cv = cv[
        (cv["ds"] >= test_start)
        & (cv["ds"] <= test_end)
    ].copy()

    prediction_columns = [
        column
        for column in cv.columns
        if column
        not in {
            "unique_id",
            "ds",
            "cutoff",
            "y",
            "horizon",
        }
    ]

    if not prediction_columns:
        raise RuntimeError(
            "Could not identify the forecast column."
        )

    forecast_column = prediction_columns[0]

    cv = cv.rename(
        columns={
            forecast_column: "forecast"
        }
    )

    return cv


def calculate_metrics(
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for horizon, group in forecasts.groupby("horizon"):
        error = group["forecast"] - group["y"]

        mae = error.abs().mean()
        rmse = (error.pow(2).mean()) ** 0.5

        rows.append(
            {
                "horizon": int(horizon),
                "forecast_count": len(group),
                "mae": float(mae),
                "rmse": float(rmse),
            }
        )

    metrics = pd.DataFrame(rows)

    metrics = metrics.sort_values(
        "horizon"
    ).reset_index(drop=True)

    return metrics


def epiweek_label(date_value) -> str:
    timestamp = pd.Timestamp(date_value)
    week = Week.fromdate(timestamp.date())

    return f"{week.year}-EW{week.week:02d}"


def plot_test_forecasts(
    forecasts: pd.DataFrame,
) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    horizons = sorted(
        forecasts["horizon"].unique()
    )

    plt.figure(figsize=(14, 7))

    actual = (
        forecasts[
            forecasts["horizon"] == horizons[0]
        ][["ds", "y"]]
        .drop_duplicates("ds")
        .sort_values("ds")
    )

    plt.plot(
        actual["ds"],
        actual["y"],
        marker="o",
        label="Actual",
        linewidth=2,
    )

    for horizon in horizons:
        subset = forecasts[
            forecasts["horizon"] == horizon
        ].sort_values("ds")

        plt.plot(
            subset["ds"],
            subset["forecast"],
            marker="o",
            label=f"{horizon}-week forecast",
            alpha=0.8,
        )

    dates = actual["ds"].tolist()

    tick_dates = dates[::4]

    if dates and dates[-1] not in tick_dates:
        tick_dates.append(dates[-1])

    plt.xticks(
        tick_dates,
        [
            epiweek_label(date_value)
            for date_value in tick_dates
        ],
        rotation=45,
        ha="right",
    )

    plt.title(
        "AutoLSTM-Selected Fixed LSTM: "
        "Held-Out FluSight Forecasts"
    )
    plt.xlabel("Epidemiological week")
    plt.ylabel("Weekly hospitalization rate")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    output_path = (
        PLOT_DIR
        / "autolstm_fixed_us_forecast_test_period.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


def plot_metrics(
    metrics: pd.DataFrame,
) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 6))

    plt.plot(
        metrics["horizon"],
        metrics["mae"],
        marker="o",
        label="MAE",
    )

    plt.plot(
        metrics["horizon"],
        metrics["rmse"],
        marker="o",
        label="RMSE",
    )

    plt.xticks(
        metrics["horizon"]
    )

    plt.xlabel("Forecast horizon (weeks)")
    plt.ylabel("Error")
    plt.title(
        "AutoLSTM-Selected Fixed LSTM "
        "Error by Forecast Horizon"
    )
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    output_path = (
        PLOT_DIR
        / "autolstm_fixed_metrics_by_horizon.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


def save_forecast_results(
    forecasts: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    forecast_path = (
        RESULT_DIR
        / "autolstm_fixed_test_forecasts.csv"
    )

    metric_path = (
        RESULT_DIR
        / "autolstm_fixed_metrics_by_horizon.csv"
    )

    forecasts.to_csv(
        forecast_path,
        index=False,
    )

    metrics.to_csv(
        metric_path,
        index=False,
    )

    print(f"Saved: {forecast_path}")
    print(f"Saved: {metric_path}")


def print_best_trial(
    study: optuna.study.Study,
    best_config: dict,
) -> None:
    print("\nBest AutoLSTM trial")
    print("===================")
    print(f"Trial: {study.best_trial.number}")
    print(
        f"Validation loss: "
        f"{study.best_value:.6f}"
    )

    print("\nSelected configuration:")

    for key, value in best_config.items():
        print(f"  {key}: {value}")


def print_metrics(
    metrics: pd.DataFrame,
) -> None:
    print("\nHeld-out metrics")
    print("================")

    for row in metrics.itertuples():
        print(
            f"{row.horizon}-week: "
            f"MAE={row.mae:.4f}, "
            f"RMSE={row.rmse:.4f}, "
            f"N={row.forecast_count}"
        )


def main() -> None:
    config = load_experiment_config()

    train_df, test_df = load_forecast_data(config)

    print("Agent-Generated AutoLSTM Experiment")
    print("===================================")
    print(f"Location: {config['location']}")
    print(f"Target: {config['target']}")
    print(
        f"Training observations: "
        f"{train_df['ds'].min().date()} "
        f"through {train_df['ds'].max().date()}"
    )
    print(
        f"Held-out observations: "
        f"{test_df['ds'].min().date()} "
        f"through {test_df['ds'].max().date()}"
    )

    _, study = run_autolstm_search(
        train_df=train_df,
        config=config,
    )

    best_config = extract_best_config(study)

    print_best_trial(
        study,
        best_config,
    )

    save_optuna_results(
        study,
        best_config,
    )

    forecasts = run_fixed_lstm_evaluation(
        train_df=train_df,
        test_df=test_df,
        best_config=best_config,
        config=config,
    )

    metrics = calculate_metrics(forecasts)

    save_forecast_results(
        forecasts,
        metrics,
    )

    plot_test_forecasts(forecasts)
    plot_metrics(metrics)

    print_metrics(metrics)

    print(
        "\nWorkflow complete:"
        "\n  natural-language request"
        "\n  -> GPT config agent"
        "\n  -> generated AutoLSTM callable"
        "\n  -> AutoLSTM / Optuna"
        "\n  -> winning configuration"
        "\n  -> fixed LSTM"
        "\n  -> held-out evaluation"
    )


if __name__ == "__main__":
    main()