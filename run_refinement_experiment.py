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


DATA_PATH = Path("data/target-hospital-admissions.csv")

EXPERIMENT_PATH = Path(
    "configs/generated_autolstm_experiment.json"
)

BASELINE_CONFIG_PATH = Path(
    "forecast_results/autolstm_best_config.json"
)

RESULT_DIR = Path("forecast_results")
PLOT_DIR = Path("plots")


LOOKBACK_CANDIDATES = [8, 12, 16]


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_experiment() -> dict:
    experiment = load_json(EXPERIMENT_PATH)

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

    missing = required - set(experiment)

    if missing:
        raise ValueError(
            f"Experiment missing keys: {sorted(missing)}"
        )

    return experiment


def load_baseline_config() -> dict:
    output = load_json(BASELINE_CONFIG_PATH)

    if "config" not in output:
        raise ValueError(
            "Baseline config file does not contain 'config'."
        )

    return output["config"]


def build_refined_config_callable(
    baseline_config: dict,
):
    """
    Keep the baseline winning configuration fixed except
    for input_size.

    The only AutoLSTM search-space change in this refinement
    is the lookback window:
        [8, 12, 16]
    """

    def refined_autolstm_config(trial):
        return {
            "input_size": trial.suggest_categorical(
                "input_size",
                LOOKBACK_CANDIDATES,
            ),
            "inference_input_size": (
                baseline_config[
                    "inference_input_size"
                ]
            ),
            "encoder_hidden_size": (
                baseline_config[
                    "encoder_hidden_size"
                ]
            ),
            "encoder_n_layers": (
                baseline_config[
                    "encoder_n_layers"
                ]
            ),
            "context_size": (
                baseline_config[
                    "context_size"
                ]
            ),
            "decoder_hidden_size": (
                baseline_config[
                    "decoder_hidden_size"
                ]
            ),
            "learning_rate": (
                baseline_config[
                    "learning_rate"
                ]
            ),
            "max_steps": (
                baseline_config[
                    "max_steps"
                ]
            ),
            "batch_size": (
                baseline_config[
                    "batch_size"
                ]
            ),
            "random_seed": (
                baseline_config[
                    "random_seed"
                ]
            ),
            "scaler_type": (
                baseline_config[
                    "scaler_type"
                ]
            ),
        }

    return refined_autolstm_config


def load_forecast_data(
    experiment: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"FluSight dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    target = experiment["target"]
    location = str(experiment["location"])

    required = {
        "date",
        "location",
        target,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset missing columns: {sorted(missing)}"
        )

    df["date"] = pd.to_datetime(df["date"])

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

    train_end = pd.Timestamp(
        experiment["train_end"]
    )

    test_start = pd.Timestamp(
        experiment["test_start"]
    )

    test_end = pd.Timestamp(
        experiment["test_end"]
    )

    train_df = model_df[
        model_df["ds"] <= train_end
    ].copy()

    test_df = model_df[
        (model_df["ds"] >= test_start)
        & (model_df["ds"] <= test_end)
    ].copy()

    if train_df.empty:
        raise ValueError(
            "Training dataset is empty."
        )

    if test_df.empty:
        raise ValueError(
            "Held-out test dataset is empty."
        )

    return train_df, test_df


def run_refined_search(
    train_df: pd.DataFrame,
    experiment: dict,
    refined_config,
) -> optuna.study.Study:

    horizon = int(experiment["horizon"])
    validation_size = int(
        experiment["validation_size"]
    )

    num_samples = int(
        experiment["num_samples"]
    )

    frequency = experiment["frequency"]

    print("\nControlled Refinement Search")
    print("============================")
    print(
        "Only tuned parameter: input_size"
    )
    print(
        f"Candidates: {LOOKBACK_CANDIDATES}"
    )
    print(
        "All other LSTM parameters are fixed "
        "to the baseline winning configuration."
    )
    print(
        f"Optuna trials: {num_samples}"
    )

    auto_model = AutoLSTM(
        h=horizon,
        loss=MAE(),
        valid_loss=MAE(),
        config=refined_config,
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
            "AutoLSTM completed without an "
            "accessible Optuna study."
        )

    return fitted_auto.results


def reconstruct_best_config(
    study: optuna.study.Study,
    refined_config,
) -> dict:

    fixed_trial = FixedTrial(
        study.best_trial.params
    )

    return refined_config(fixed_trial)


def run_fixed_evaluation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_config: dict,
    experiment: dict,
) -> pd.DataFrame:

    horizon = int(experiment["horizon"])
    frequency = experiment["frequency"]

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

    test_start = pd.Timestamp(
        experiment["test_start"]
    )

    test_end = pd.Timestamp(
        experiment["test_end"]
    )

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
            "Could not identify forecast column."
        )

    forecast_column = prediction_columns[0]

    cv = cv.rename(
        columns={
            forecast_column: "raw_forecast"
        }
    )

    # Refinement change #2:
    # hospitalization rates cannot be negative.
    #
    # Preserve raw output for auditability, then create
    # a corrected nonnegative forecast.
    cv["forecast"] = cv[
        "raw_forecast"
    ].clip(lower=0)

    cv["was_clipped"] = (
        cv["raw_forecast"] < 0
    )

    return cv


def calculate_metrics(
    forecasts: pd.DataFrame,
    forecast_column: str,
) -> pd.DataFrame:

    rows = []

    for horizon, group in forecasts.groupby(
        "horizon"
    ):
        error = (
            group[forecast_column]
            - group["y"]
        )

        mae = error.abs().mean()

        rmse = (
            error.pow(2).mean()
        ) ** 0.5

        bias = error.mean()

        actual_peak = group.loc[
            group["y"].idxmax()
        ]

        forecast_peak = group.loc[
            group[forecast_column].idxmax()
        ]

        peak_magnitude_error = (
            float(
                forecast_peak[
                    forecast_column
                ]
            )
            - float(actual_peak["y"])
        )

        peak_timing_error_weeks = (
            (
                forecast_peak["ds"]
                - actual_peak["ds"]
            ).days
            / 7.0
        )

        negative_count = int(
            (
                group[forecast_column] < 0
            ).sum()
        )

        rows.append(
            {
                "horizon": int(horizon),
                "forecast_count": len(group),
                "mae": float(mae),
                "rmse": float(rmse),
                "bias": float(bias),
                "peak_magnitude_error": (
                    peak_magnitude_error
                ),
                "peak_timing_error_weeks": (
                    peak_timing_error_weeks
                ),
                "negative_forecast_count": (
                    negative_count
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("horizon")
        .reset_index(drop=True)
    )


def save_results(
    study: optuna.study.Study,
    best_config: dict,
    forecasts: pd.DataFrame,
    raw_metrics: pd.DataFrame,
    corrected_metrics: pd.DataFrame,
) -> None:

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    study.trials_dataframe().to_csv(
        RESULT_DIR
        / "refined_autolstm_optuna_trials.csv",
        index=False,
    )

    with (
        RESULT_DIR
        / "refined_autolstm_best_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "trial_number": (
                    study.best_trial.number
                ),
                "validation_loss": (
                    study.best_value
                ),
                "config": best_config,
                "lookback_candidates": (
                    LOOKBACK_CANDIDATES
                ),
                "nonnegative_handling": (
                    "clip forecasts at zero"
                ),
            },
            file,
            indent=2,
        )

    forecasts.to_csv(
        RESULT_DIR
        / "refined_autolstm_test_forecasts.csv",
        index=False,
    )

    raw_metrics.to_csv(
        RESULT_DIR
        / "refined_autolstm_raw_metrics.csv",
        index=False,
    )

    corrected_metrics.to_csv(
        RESULT_DIR
        / "refined_autolstm_corrected_metrics.csv",
        index=False,
    )


def epiweek_label(date_value) -> str:
    timestamp = pd.Timestamp(date_value)

    week = Week.fromdate(
        timestamp.date()
    )

    return (
        f"{week.year}-EW{week.week:02d}"
    )


def plot_forecasts(
    forecasts: pd.DataFrame,
) -> None:

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    horizons = sorted(
        forecasts["horizon"].unique()
    )

    plt.figure(
        figsize=(14, 7)
    )

    actual = (
        forecasts[
            forecasts["horizon"]
            == horizons[0]
        ][["ds", "y"]]
        .drop_duplicates("ds")
        .sort_values("ds")
    )

    plt.plot(
        actual["ds"],
        actual["y"],
        marker="o",
        linewidth=2,
        label="Actual",
    )

    for horizon in horizons:
        subset = forecasts[
            forecasts["horizon"]
            == horizon
        ].sort_values("ds")

        plt.plot(
            subset["ds"],
            subset["forecast"],
            marker="o",
            alpha=0.8,
            label=(
                f"{horizon}-week forecast"
            ),
        )

    dates = actual["ds"].tolist()
    tick_dates = dates[::4]

    if (
        dates
        and dates[-1]
        not in tick_dates
    ):
        tick_dates.append(
            dates[-1]
        )

    plt.xticks(
        tick_dates,
        [
            epiweek_label(date)
            for date in tick_dates
        ],
        rotation=45,
        ha="right",
    )

    plt.title(
        "Refined AutoLSTM: "
        "Held-Out FluSight Forecasts"
    )

    plt.xlabel(
        "Epidemiological week"
    )

    plt.ylabel(
        "Weekly hospitalization rate"
    )

    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    output_path = (
        PLOT_DIR
        / "refined_autolstm_forecast.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


def print_results(
    study: optuna.study.Study,
    best_config: dict,
    forecasts: pd.DataFrame,
    raw_metrics: pd.DataFrame,
    corrected_metrics: pd.DataFrame,
) -> None:

    print("\nBest Refinement Trial")
    print("=====================")

    print(
        f"Trial: "
        f"{study.best_trial.number}"
    )

    print(
        f"Validation loss: "
        f"{study.best_value:.6f}"
    )

    print(
        f"Selected input_size: "
        f"{best_config['input_size']}"
    )

    total_clipped = int(
        forecasts["was_clipped"].sum()
    )

    print(
        f"Negative raw forecasts clipped: "
        f"{total_clipped}"
    )

    print("\nRaw Forecast Metrics")
    print("====================")

    for row in raw_metrics.itertuples():
        print(
            f"{row.horizon}-week: "
            f"MAE={row.mae:.4f}, "
            f"RMSE={row.rmse:.4f}, "
            f"bias={row.bias:.4f}, "
            f"peak error="
            f"{row.peak_magnitude_error:.4f}, "
            f"peak timing="
            f"{row.peak_timing_error_weeks:+.1f}, "
            f"negatives="
            f"{row.negative_forecast_count}"
        )

    print("\nCorrected Forecast Metrics")
    print("==========================")

    for row in corrected_metrics.itertuples():
        print(
            f"{row.horizon}-week: "
            f"MAE={row.mae:.4f}, "
            f"RMSE={row.rmse:.4f}, "
            f"bias={row.bias:.4f}, "
            f"peak error="
            f"{row.peak_magnitude_error:.4f}, "
            f"peak timing="
            f"{row.peak_timing_error_weeks:+.1f}, "
            f"negatives="
            f"{row.negative_forecast_count}"
        )


def main() -> None:
    experiment = load_experiment()

    baseline_config = (
        load_baseline_config()
    )

    refined_config = (
        build_refined_config_callable(
            baseline_config
        )
    )

    train_df, test_df = (
        load_forecast_data(
            experiment
        )
    )

    study = run_refined_search(
        train_df=train_df,
        experiment=experiment,
        refined_config=refined_config,
    )

    best_config = (
        reconstruct_best_config(
            study=study,
            refined_config=refined_config,
        )
    )

    forecasts = (
        run_fixed_evaluation(
            train_df=train_df,
            test_df=test_df,
            best_config=best_config,
            experiment=experiment,
        )
    )

    raw_metrics = (
        calculate_metrics(
            forecasts=forecasts,
            forecast_column=(
                "raw_forecast"
            ),
        )
    )

    corrected_metrics = (
        calculate_metrics(
            forecasts=forecasts,
            forecast_column="forecast",
        )
    )

    save_results(
        study=study,
        best_config=best_config,
        forecasts=forecasts,
        raw_metrics=raw_metrics,
        corrected_metrics=(
            corrected_metrics
        ),
    )

    plot_forecasts(forecasts)

    print_results(
        study=study,
        best_config=best_config,
        forecasts=forecasts,
        raw_metrics=raw_metrics,
        corrected_metrics=(
            corrected_metrics
        ),
    )


if __name__ == "__main__":
    main()