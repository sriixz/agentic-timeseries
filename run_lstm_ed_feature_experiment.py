from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import LSTM


HOSPITAL_PATH = Path(
    "data/target-hospital-admissions.csv"
)

ED_PATH = Path(
    "data/target-ed-visits-prop.csv"
)

RESULT_DIR = Path("forecast_results")
PLOT_DIR = Path("plots")


LOCATION = "US"

TARGET_COLUMN = "weekly_rate"
ED_COLUMN = "ed_visits_prop"


TRAIN_END = pd.Timestamp(
    "2025-09-30"
)

TEST_START = pd.Timestamp(
    "2025-10-01"
)

TEST_END = pd.Timestamp(
    "2026-05-31"
)


HORIZON = 4
INPUT_SIZE = 12

MAX_STEPS = 1000
RANDOM_SEED = 2


# Same hand-fixed refined LSTM family we have
# already been using in the peak experiments.
ENCODER_HIDDEN_SIZE = 32
ENCODER_N_LAYERS = 3

DECODER_HIDDEN_SIZE = 64

LEARNING_RATE = 0.00156345

BATCH_SIZE = 32


def load_hospital_data() -> pd.DataFrame:
    if not HOSPITAL_PATH.exists():
        raise FileNotFoundError(
            f"Hospitalization file not found: "
            f"{HOSPITAL_PATH}"
        )

    df = pd.read_csv(
        HOSPITAL_PATH,
        dtype={
            "location": str,
        },
    )

    required = {
        "date",
        "location",
        TARGET_COLUMN,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Hospitalization data missing columns: "
            f"{sorted(missing)}"
        )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df["location"] = (
        df["location"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["location"]
        == LOCATION
    ].copy()

    if df.empty:
        raise ValueError(
            f"No hospitalization data found "
            f"for location {LOCATION}."
        )

    df = (
        df[
            [
                "date",
                TARGET_COLUMN,
            ]
        ]
        .dropna()
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return df


def load_ed_data() -> pd.DataFrame:
    if not ED_PATH.exists():
        raise FileNotFoundError(
            f"ED-visits file not found: "
            f"{ED_PATH}"
        )

    df = pd.read_csv(
        ED_PATH,
        dtype={
            "location": str,
        },
    )

    required = {
        "date",
        "location",
        "value",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "ED-visits data missing columns: "
            f"{sorted(missing)}"
        )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df["location"] = (
        df["location"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["location"]
        == LOCATION
    ].copy()

    if df.empty:
        raise ValueError(
            f"No ED-visits data found "
            f"for location {LOCATION}."
        )

    df = (
        df[
            [
                "date",
                "value",
            ]
        ]
        .rename(
            columns={
                "value": ED_COLUMN,
            }
        )
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    missing_count = int(
        df[ED_COLUMN]
        .isna()
        .sum()
    )

    if missing_count > 0:
        raise ValueError(
            f"ED-visits series contains "
            f"{missing_count} missing values "
            f"for {LOCATION}."
        )

    return df


def build_model_data() -> pd.DataFrame:
    hospital_df = (
        load_hospital_data()
    )

    ed_df = (
        load_ed_data()
    )

    merged = hospital_df.merge(
        ed_df,
        on="date",
        how="inner",
        validate="one_to_one",
    )

    merged = merged[
        merged["date"]
        <= TEST_END
    ].copy()

    merged = (
        merged
        .sort_values("date")
        .reset_index(drop=True)
    )

    if merged.empty:
        raise ValueError(
            "No overlapping hospitalization "
            "and ED-visits observations."
        )

    expected_dates = pd.date_range(
        start=merged[
            "date"
        ].min(),
        end=merged[
            "date"
        ].max(),
        freq="W-SAT",
    )

    actual_dates = pd.DatetimeIndex(
        merged["date"]
    )

    missing_dates = (
        expected_dates
        .difference(actual_dates)
    )

    if len(missing_dates) > 0:
        raise ValueError(
            "Merged weekly series contains "
            f"{len(missing_dates)} missing dates. "
            f"First few: "
            f"{list(missing_dates[:5])}"
        )

    return pd.DataFrame(
        {
            "unique_id":
                LOCATION,

            "ds":
                merged["date"],

            "y":
                merged[
                    TARGET_COLUMN
                ].astype(float),

            ED_COLUMN:
                merged[
                    ED_COLUMN
                ].astype(float),
        }
    )


def build_baseline_model() -> LSTM:
    return LSTM(
        h=HORIZON,

        input_size=INPUT_SIZE,

        loss=MAE(),
        valid_loss=MAE(),

        encoder_n_layers=(
            ENCODER_N_LAYERS
        ),

        encoder_hidden_size=(
            ENCODER_HIDDEN_SIZE
        ),

        decoder_hidden_size=(
            DECODER_HIDDEN_SIZE
        ),

        max_steps=MAX_STEPS,

        learning_rate=(
            LEARNING_RATE
        ),

        batch_size=BATCH_SIZE,

        scaler_type="standard",

        random_seed=(
            RANDOM_SEED
        ),

        alias="LSTMBaseline",
    )


def build_ed_model() -> LSTM:
    return LSTM(
        h=HORIZON,

        input_size=INPUT_SIZE,

        loss=MAE(),
        valid_loss=MAE(),

        hist_exog_list=[
            ED_COLUMN
        ],

        encoder_n_layers=(
            ENCODER_N_LAYERS
        ),

        encoder_hidden_size=(
            ENCODER_HIDDEN_SIZE
        ),

        decoder_hidden_size=(
            DECODER_HIDDEN_SIZE
        ),

        max_steps=MAX_STEPS,

        learning_rate=(
            LEARNING_RATE
        ),

        batch_size=BATCH_SIZE,

        scaler_type="standard",

        random_seed=(
            RANDOM_SEED
        ),

        alias="LSTMED",
    )


def run_experiment(
    model_df: pd.DataFrame,
) -> pd.DataFrame:
    train_df = model_df[
        model_df["ds"]
        <= TRAIN_END
    ].copy()

    test_df = model_df[
        (
            model_df["ds"]
            >= TEST_START
        )
        & (
            model_df["ds"]
            <= TEST_END
        )
    ].copy()

    if train_df.empty:
        raise ValueError(
            "Training data is empty."
        )

    if test_df.empty:
        raise ValueError(
            "Test data is empty."
        )

    print(
        "\nLSTM ED-Visits Experiment"
    )

    print(
        "========================="
    )

    print(
        f"Location: {LOCATION}"
    )

    print(
        "Feature: historical "
        "ED-visits proportion"
    )

    print(
        f"Data overlap begins: "
        f"{model_df['ds'].min().date()}"
    )

    print(
        f"Training observations: "
        f"{len(train_df)}"
    )

    print(
        f"Held-out observations: "
        f"{len(test_df)}"
    )

    print(
        f"Input size: "
        f"{INPUT_SIZE}"
    )

    print(
        f"Horizon: "
        f"{HORIZON}"
    )

    print(
        f"Training steps/model: "
        f"{MAX_STEPS}"
    )

    print(
        "\nImportant: baseline and "
        "ED-feature LSTM use the "
        "exact same dates."
    )

    print(
        "ED visits are supplied only "
        "as historical exogenous data."
    )

    models = [
        build_baseline_model(),
        build_ed_model(),
    ]

    nf = NeuralForecast(
        models=models,
        freq="W-SAT",
    )

    cv = nf.cross_validation(
        df=model_df,

        n_windows=None,

        test_size=len(
            test_df
        ),

        step_size=1,

        refit=False,
    )

    cv = (
        cv
        .sort_values(
            [
                "cutoff",
                "ds",
            ]
        )
        .reset_index(drop=True)
    )

    cv[
        "horizon"
    ] = (
        cv.groupby(
            "cutoff"
        )
        .cumcount()
        .add(1)
    )

    cv = cv[
        cv["horizon"]
        <= HORIZON
    ].copy()

    cv = cv[
        (
            cv["ds"]
            >= TEST_START
        )
        & (
            cv["ds"]
            <= TEST_END
        )
    ].copy()

    return cv


def calculate_metrics(
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for horizon, group in (
        forecasts.groupby(
            "horizon"
        )
    ):
        actual = (
            group["y"]
            .astype(float)
        )

        baseline = (
            group[
                "LSTMBaseline"
            ]
            .astype(float)
        )

        ed = (
            group[
                "LSTMED"
            ]
            .astype(float)
        )

        baseline_error = (
            baseline
            - actual
        )

        ed_error = (
            ed
            - actual
        )

        baseline_mae = float(
            baseline_error
            .abs()
            .mean()
        )

        baseline_rmse = float(
            np.sqrt(
                np.mean(
                    baseline_error
                    ** 2
                )
            )
        )

        ed_mae = float(
            ed_error
            .abs()
            .mean()
        )

        ed_rmse = float(
            np.sqrt(
                np.mean(
                    ed_error
                    ** 2
                )
            )
        )

        rows.append(
            {
                "horizon":
                    int(horizon),

                "baseline_mae":
                    baseline_mae,

                "ed_mae":
                    ed_mae,

                "mae_improvement_pct":
                    (
                        (
                            baseline_mae
                            - ed_mae
                        )
                        / baseline_mae
                        * 100.0
                    ),

                "baseline_rmse":
                    baseline_rmse,

                "ed_rmse":
                    ed_rmse,

                "rmse_improvement_pct":
                    (
                        (
                            baseline_rmse
                            - ed_rmse
                        )
                        / baseline_rmse
                        * 100.0
                    ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "horizon"
        )
        .reset_index(
            drop=True
        )
    )


def calculate_clipped_metrics(
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for horizon, group in (
        forecasts.groupby(
            "horizon"
        )
    ):
        actual = (
            group["y"]
            .astype(float)
        )

        baseline = (
            group[
                "LSTMBaseline"
            ]
            .astype(float)
            .clip(lower=0.0)
        )

        ed = (
            group[
                "LSTMED"
            ]
            .astype(float)
            .clip(lower=0.0)
        )

        baseline_error = (
            baseline
            - actual
        )

        ed_error = (
            ed
            - actual
        )

        baseline_mae = float(
            baseline_error
            .abs()
            .mean()
        )

        baseline_rmse = float(
            np.sqrt(
                np.mean(
                    baseline_error
                    ** 2
                )
            )
        )

        ed_mae = float(
            ed_error
            .abs()
            .mean()
        )

        ed_rmse = float(
            np.sqrt(
                np.mean(
                    ed_error
                    ** 2
                )
            )
        )

        rows.append(
            {
                "horizon":
                    int(horizon),

                "baseline_mae_clipped":
                    baseline_mae,

                "ed_mae_clipped":
                    ed_mae,

                "baseline_rmse_clipped":
                    baseline_rmse,

                "ed_rmse_clipped":
                    ed_rmse,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "horizon"
        )
        .reset_index(
            drop=True
        )
    )


def calculate_peak_diagnostics(
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for horizon, group in (
        forecasts.groupby(
            "horizon"
        )
    ):
        group = (
            group
            .sort_values("ds")
            .copy()
        )

        actual_peak_idx = (
            group["y"]
            .idxmax()
        )

        baseline_peak_idx = (
            group[
                "LSTMBaseline"
            ]
            .idxmax()
        )

        ed_peak_idx = (
            group[
                "LSTMED"
            ]
            .idxmax()
        )

        actual_peak_date = (
            group.loc[
                actual_peak_idx,
                "ds",
            ]
        )

        actual_peak_value = float(
            group.loc[
                actual_peak_idx,
                "y",
            ]
        )

        baseline_peak_date = (
            group.loc[
                baseline_peak_idx,
                "ds",
            ]
        )

        baseline_peak_value = float(
            group.loc[
                baseline_peak_idx,
                "LSTMBaseline",
            ]
        )

        ed_peak_date = (
            group.loc[
                ed_peak_idx,
                "ds",
            ]
        )

        ed_peak_value = float(
            group.loc[
                ed_peak_idx,
                "LSTMED",
            ]
        )

        rows.append(
            {
                "horizon":
                    int(horizon),

                "actual_peak_date":
                    actual_peak_date,

                "actual_peak_value":
                    actual_peak_value,

                "baseline_peak_date":
                    baseline_peak_date,

                "baseline_peak_error":
                    (
                        baseline_peak_value
                        - actual_peak_value
                    ),

                "baseline_peak_timing_weeks":
                    int(
                        (
                            baseline_peak_date
                            - actual_peak_date
                        ).days
                        / 7
                    ),

                "ed_peak_date":
                    ed_peak_date,

                "ed_peak_error":
                    (
                        ed_peak_value
                        - actual_peak_value
                    ),

                "ed_peak_timing_weeks":
                    int(
                        (
                            ed_peak_date
                            - actual_peak_date
                        ).days
                        / 7
                    ),
            }
        )

    return pd.DataFrame(rows)


def print_metrics(
    metrics: pd.DataFrame,
) -> None:
    print(
        "\nForecast Evaluation"
    )

    print(
        "==================="
    )

    for row in (
        metrics.itertuples()
    ):
        print(
            f"\n{row.horizon}-week horizon"
        )

        print(
            "  Baseline MAE: "
            f"{row.baseline_mae:.4f}"
        )

        print(
            "  ED MAE:       "
            f"{row.ed_mae:.4f}"
        )

        print(
            "  MAE change:   "
            f"{row.mae_improvement_pct:+.1f}%"
        )

        print(
            "  Baseline RMSE: "
            f"{row.baseline_rmse:.4f}"
        )

        print(
            "  ED RMSE:       "
            f"{row.ed_rmse:.4f}"
        )

        print(
            "  RMSE change:   "
            f"{row.rmse_improvement_pct:+.1f}%"
        )


def print_peak_diagnostics(
    peak_df: pd.DataFrame,
) -> None:
    print(
        "\nPeak Diagnostics"
    )

    print(
        "================"
    )

    for row in (
        peak_df.itertuples()
    ):
        print(
            f"\n{row.horizon}-week horizon"
        )

        print(
            "  Baseline peak error: "
            f"{row.baseline_peak_error:+.4f}"
        )

        print(
            "  Baseline timing error: "
            f"{row.baseline_peak_timing_weeks:+d} weeks"
        )

        print(
            "  ED peak error: "
            f"{row.ed_peak_error:+.4f}"
        )

        print(
            "  ED timing error: "
            f"{row.ed_peak_timing_weeks:+d} weeks"
        )


def plot_four_week(
    forecasts: pd.DataFrame,
) -> None:
    four_week = (
        forecasts[
            forecasts["horizon"]
            == 4
        ]
        .sort_values("ds")
        .copy()
    )

    if four_week.empty:
        raise ValueError(
            "No 4-week forecasts found."
        )

    fig, ax = plt.subplots(
        figsize=(
            13,
            7,
        )
    )

    ax.plot(
        four_week["ds"],
        four_week["y"],
        marker="o",
        linewidth=2,
        label="Actual",
    )

    ax.plot(
        four_week["ds"],
        four_week[
            "LSTMBaseline"
        ],
        marker="o",
        label="LSTM baseline",
    )

    ax.plot(
        four_week["ds"],
        four_week[
            "LSTMED"
        ],
        marker="o",
        label=(
            "LSTM + historical ED visits"
        ),
    )

    ax.set_title(
        "U.S. LSTM: Baseline vs "
        "Historical ED-Visits Feature "
        "(4-Week Ahead)"
    )

    ax.set_xlabel(
        "Target week"
    )

    ax.set_ylabel(
        "Weekly hospitalization rate"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    output_path = (
        PLOT_DIR
        / "lstm_ed_feature_us_4w.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"\nSaved: {output_path}"
    )


def main():
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_df = (
        build_model_data()
    )

    forecasts = (
        run_experiment(
            model_df=model_df
        )
    )

    metrics = (
        calculate_metrics(
            forecasts
        )
    )

    clipped_metrics = (
        calculate_clipped_metrics(
            forecasts
        )
    )

    peak_df = (
        calculate_peak_diagnostics(
            forecasts
        )
    )

    print_metrics(
        metrics
    )

    print_peak_diagnostics(
        peak_df
    )

    forecast_path = (
        RESULT_DIR
        / "lstm_ed_feature_us_forecasts.csv"
    )

    metrics_path = (
        RESULT_DIR
        / "lstm_ed_feature_us_metrics.csv"
    )

    clipped_path = (
        RESULT_DIR
        / "lstm_ed_feature_us_clipped_metrics.csv"
    )

    peak_path = (
        RESULT_DIR
        / "lstm_ed_feature_us_peak_metrics.csv"
    )

    forecasts.to_csv(
        forecast_path,
        index=False,
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    clipped_metrics.to_csv(
        clipped_path,
        index=False,
    )

    peak_df.to_csv(
        peak_path,
        index=False,
    )

    plot_four_week(
        forecasts
    )

    print(
        f"Saved: {forecast_path}"
    )

    print(
        f"Saved: {metrics_path}"
    )

    print(
        f"Saved: {clipped_path}"
    )

    print(
        f"Saved: {peak_path}"
    )


if __name__ == "__main__":
    main()