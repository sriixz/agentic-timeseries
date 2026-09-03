from pathlib import Path
import math

import matplotlib.pyplot as plt
import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM

from tools import prepare_flu_data


PLOTS_DIR = Path("plots")
RESULTS_DIR = Path("forecast_results")

TRAIN_END = pd.Timestamp("2025-09-30")
TEST_START = pd.Timestamp("2025-10-01")
TEST_END = pd.Timestamp("2026-05-31")

HORIZON = 4


def prepare_us_series():
    """
    Prepare the US national weekly hospitalization-rate
    series in NeuralForecast format:

        unique_id | ds | y
    """

    df = prepare_flu_data()

    us_df = (
        df[df["location_name"] == "US"]
        .copy()
        .sort_values("date")
    )

    us_df = us_df[
        [
            "date",
            "weekly_rate",
        ]
    ].copy()

    us_df = us_df.dropna(
        subset=["weekly_rate"]
    )

    us_df = us_df.rename(
        columns={
            "date": "ds",
            "weekly_rate": "y",
        }
    )

    us_df["unique_id"] = "US"

    us_df = us_df[
        [
            "unique_id",
            "ds",
            "y",
        ]
    ]

    return us_df


def split_train_test(df):
    """
    Training:
        observations through September 2025

    Testing:
        October 2025 through May 2026
    """

    train_df = df[
        df["ds"] <= TRAIN_END
    ].copy()

    test_df = df[
        (
            df["ds"] >= TEST_START
        )
        & (
            df["ds"] <= TEST_END
        )
    ].copy()

    return train_df, test_df


def build_model():
    """
    Build the NeuralForecast LSTM baseline.

    h=4 produces forecasts from one through
    four weeks ahead.
    """

    model = LSTM(
        h=HORIZON,
        input_size=12,
        max_steps=300,
        scaler_type="standard",
        encoder_hidden_size=64,
        decoder_hidden_size=64,
    )

    nf = NeuralForecast(
        models=[model],
        freq="W-SAT",
    )

    return nf


def run_fixed_model_evaluation(
    full_df,
    test_df,
):
    """
    Fit the LSTM once before the held-out test period
    and evaluate it over October 2025 through May 2026.

    The dataframe supplied to NeuralForecast ends at
    the requested test-period boundary. This ensures
    test_size refers exactly to the October-May holdout.

    refit=False means the learned model parameters
    remain fixed across all evaluation windows.
    """

    test_size = len(test_df)

    # IMPORTANT:
    # Do not pass observations after May 2026 into
    # cross-validation. Otherwise NeuralForecast would
    # define the holdout relative to the later data.
    evaluation_df = full_df[
        full_df["ds"] <= TEST_END
    ].copy()

    print(
        "\n--- FITTING FIXED LSTM AND "
        "RUNNING TEST WINDOWS ---"
    )

    print(
        f"Forecast horizon: {HORIZON} weeks"
    )

    print(
        f"Test size: {test_size} weeks"
    )

    print(
        f"Evaluation data ends: "
        f"{evaluation_df['ds'].max().date()}"
    )

    nf = build_model()

    cv_df = nf.cross_validation(
        df=evaluation_df,
        n_windows=None,
        test_size=test_size,
        step_size=1,
        refit=False,
        verbose=False,
    )

    cv_df = cv_df.reset_index(
        drop=True
    )

    return cv_df


def convert_cv_to_results(cv_df):
    """
    Convert NeuralForecast cross-validation output
    into an explicit forecast table.

    Output columns include:

        forecast_origin
        target_date
        horizon
        actual
        prediction
        error
        absolute_error
        squared_error
    """

    results = []

    for _, row in cv_df.iterrows():
        target_date = pd.Timestamp(
            row["ds"]
        )

        cutoff = pd.Timestamp(
            row["cutoff"]
        )

        if not (
            TEST_START
            <= target_date
            <= TEST_END
        ):
            continue

        day_difference = (
            target_date - cutoff
        ).days

        horizon = int(
            day_difference / 7
        )

        if horizon not in range(
            1,
            HORIZON + 1,
        ):
            continue

        actual = float(
            row["y"]
        )

        prediction = float(
            row["LSTM"]
        )

        error = (
            prediction - actual
        )

        results.append(
            {
                "forecast_origin": cutoff,
                "target_date": target_date,
                "horizon": horizon,
                "actual": actual,
                "prediction": prediction,
                "error": error,
                "absolute_error": abs(
                    error
                ),
                "squared_error": (
                    error ** 2
                ),
            }
        )

    results_df = pd.DataFrame(
        results
    )

    if results_df.empty:
        raise ValueError(
            "No forecast results were produced."
        )

    results_df = results_df.sort_values(
        [
            "forecast_origin",
            "horizon",
        ]
    ).reset_index(
        drop=True
    )

    return results_df


def summarize_metrics(results_df):
    """
    Compute MAE and RMSE separately for
    1-, 2-, 3-, and 4-week-ahead forecasts.
    """

    rows = []

    for horizon in range(
        1,
        HORIZON + 1,
    ):
        horizon_df = results_df[
            results_df["horizon"]
            == horizon
        ]

        if horizon_df.empty:
            continue

        mae = horizon_df[
            "absolute_error"
        ].mean()

        rmse = math.sqrt(
            horizon_df[
                "squared_error"
            ].mean()
        )

        rows.append(
            {
                "horizon_weeks": horizon,
                "forecast_count": len(
                    horizon_df
                ),
                "mae": round(
                    float(mae),
                    4,
                ),
                "rmse": round(
                    float(rmse),
                    4,
                ),
            }
        )

    return pd.DataFrame(rows)


def plot_horizon_one(results_df):
    """
    Plot actual test-period values against the
    raw 1-week-ahead LSTM forecasts.

    Predictions are not clipped, so negative model
    outputs remain visible.
    """

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    horizon_one = (
        results_df[
            results_df["horizon"] == 1
        ]
        .copy()
        .sort_values("target_date")
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        horizon_one["target_date"],
        horizon_one["actual"],
        label="Actual",
        linewidth=2,
    )

    ax.plot(
        horizon_one["target_date"],
        horizon_one["prediction"],
        label="LSTM 1-week forecast",
        linewidth=2,
    )

    ax.axhline(
        0,
        linewidth=1,
        linestyle="--",
    )

    ax.set_title(
        "Fixed LSTM: US FluSight Test Forecasts "
        "(Oct 2025-May 2026)"
    )

    ax.set_xlabel("Date")

    ax.set_ylabel(
        "Weekly hospitalization rate"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    output_path = (
        PLOTS_DIR
        / "lstm_fixed_us_forecast_test_period.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path


def plot_metrics(metrics_df):
    """
    Plot MAE and RMSE against forecast horizon.
    """

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        metrics_df["horizon_weeks"],
        metrics_df["mae"],
        marker="o",
        label="MAE",
    )

    ax.plot(
        metrics_df["horizon_weeks"],
        metrics_df["rmse"],
        marker="o",
        label="RMSE",
    )

    ax.set_title(
        "LSTM Forecast Error by Horizon"
    )

    ax.set_xlabel(
        "Forecast horizon (weeks)"
    )

    ax.set_ylabel(
        "Error in weekly hospitalization rate"
    )

    ax.set_xticks(
        [1, 2, 3, 4]
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    output_path = (
        PLOTS_DIR
        / "lstm_fixed_metrics_by_horizon.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path


def save_results(
    results_df,
    metrics_df,
):
    """
    Save forecasts and horizon-level metrics.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecasts_path = (
        RESULTS_DIR
        / "lstm_fixed_forecasts.csv"
    )

    metrics_path = (
        RESULTS_DIR
        / "lstm_fixed_metrics_by_horizon.csv"
    )

    results_df.to_csv(
        forecasts_path,
        index=False,
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    return (
        forecasts_path,
        metrics_path,
    )


def main():
    print(
        "\n--- PREPARING US NATIONAL SERIES ---"
    )

    full_df = prepare_us_series()

    train_df, test_df = split_train_test(
        full_df
    )

    print(
        f"Training observations: "
        f"{len(train_df)}"
    )

    print(
        f"Training range: "
        f"{train_df['ds'].min().date()} "
        f"to "
        f"{train_df['ds'].max().date()}"
    )

    print(
        f"Test observations: "
        f"{len(test_df)}"
    )

    print(
        f"Test range: "
        f"{test_df['ds'].min().date()} "
        f"to "
        f"{test_df['ds'].max().date()}"
    )

    print(
        "\nModel parameters will be fitted "
        "once before the test period."
    )

    print(
        "refit=False: model weights remain "
        "fixed during evaluation."
    )

    cv_df = run_fixed_model_evaluation(
        full_df,
        test_df,
    )

    results_df = convert_cv_to_results(
        cv_df
    )

    print(
        "\n--- COMPUTING METRICS ---"
    )

    metrics_df = summarize_metrics(
        results_df
    )

    print(
        metrics_df.to_string(
            index=False
        )
    )

    forecasts_path, metrics_path = (
        save_results(
            results_df,
            metrics_df,
        )
    )

    forecast_plot = plot_horizon_one(
        results_df
    )

    metrics_plot = plot_metrics(
        metrics_df
    )

    print(
        "\n--- SAVED OUTPUTS ---"
    )

    print(
        f"Forecasts: {forecasts_path}"
    )

    print(
        f"Metrics:   {metrics_path}"
    )

    print(
        f"Forecast plot: {forecast_plot}"
    )

    print(
        f"Metrics plot:  {metrics_plot}"
    )

    print(
        "\n--- EXPERIMENT SUMMARY ---"
    )

    print(
        "Target: US national weekly "
        "hospitalization rate"
    )

    print(
        "Training period: "
        "2022 through September 2025"
    )

    print(
        "Test period: "
        "October 2025 through May 2026"
    )

    print(
        "Forecast horizons: "
        "1 through 4 weeks"
    )

    print(
        "Model refitting during test: No"
    )


if __name__ == "__main__":
    main()