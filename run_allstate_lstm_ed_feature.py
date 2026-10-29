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


# Refined LSTM settings used in the
# earlier peak-feature experiments.
ENCODER_HIDDEN_SIZE = 32
ENCODER_N_LAYERS = 3
CONTEXT_SIZE = 10

DECODER_HIDDEN_SIZE = 64

LEARNING_RATE = 0.00156345
BATCH_SIZE = 32


SUMMARY_PATH = (
    RESULT_DIR
    / "allstate_lstm_ed_summary.csv"
)

FAILURE_PATH = (
    RESULT_DIR
    / "allstate_lstm_ed_failures.csv"
)

FORECAST_PATH = (
    RESULT_DIR
    / "allstate_lstm_ed_forecasts.csv"
)


def load_source_data():
    if not HOSPITAL_PATH.exists():
        raise FileNotFoundError(
            f"Missing hospitalization file: "
            f"{HOSPITAL_PATH}"
        )

    if not ED_PATH.exists():
        raise FileNotFoundError(
            f"Missing ED-visits file: "
            f"{ED_PATH}"
        )

    hospital_df = pd.read_csv(
        HOSPITAL_PATH,
        dtype={
            "location": str,
        },
    )

    ed_df = pd.read_csv(
        ED_PATH,
        dtype={
            "location": str,
        },
    )

    hospital_df["date"] = (
        pd.to_datetime(
            hospital_df["date"]
        )
    )

    ed_df["date"] = (
        pd.to_datetime(
            ed_df["date"]
        )
    )

    hospital_df["location"] = (
        hospital_df["location"]
        .astype(str)
        .str.strip()
    )

    ed_df["location"] = (
        ed_df["location"]
        .astype(str)
        .str.strip()
    )

    return (
        hospital_df,
        ed_df,
    )


def get_location_table(
    hospital_df,
) -> pd.DataFrame:
    locations = (
        hospital_df[
            [
                "location",
                "location_name",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    # State/DC experiment only.
    # Keep US out of the all-state ranking.
    locations = locations[
        locations["location"]
        != "US"
    ].copy()

    return (
        locations
        .sort_values(
            "location_name"
        )
        .reset_index(
            drop=True
        )
    )


def build_location_data(
    hospital_df,
    ed_df,
    location,
):
    hospital = hospital_df[
        hospital_df["location"]
        == location
    ][
        [
            "date",
            TARGET_COLUMN,
        ]
    ].copy()

    ed = ed_df[
        ed_df["location"]
        == location
    ][
        [
            "date",
            "value",
        ]
    ].copy()

    if hospital.empty:
        raise ValueError(
            "No hospitalization data."
        )

    if ed.empty:
        raise ValueError(
            "No ED-visits data."
        )

    hospital = (
        hospital
        .dropna(
            subset=[
                TARGET_COLUMN
            ]
        )
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
    )

    ed = (
        ed
        .rename(
            columns={
                "value":
                    ED_COLUMN
            }
        )
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
    )

    merged = hospital.merge(
        ed,
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
            "No overlapping observations."
        )

    missing_ed = int(
        merged[
            ED_COLUMN
        ].isna().sum()
    )

    if missing_ed:
        raise ValueError(
            f"{missing_ed} missing ED values."
        )

    expected_dates = (
        pd.date_range(
            start=merged[
                "date"
            ].min(),
            end=merged[
                "date"
            ].max(),
            freq="W-SAT",
        )
    )

    actual_dates = (
        pd.DatetimeIndex(
            merged["date"]
        )
    )

    gaps = (
        expected_dates
        .difference(
            actual_dates
        )
    )

    if len(gaps) > 0:
        raise ValueError(
            f"{len(gaps)} missing weekly dates."
        )

    model_df = pd.DataFrame(
        {
            "unique_id":
                location,

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

    train_count = int(
        (
            model_df["ds"]
            <= TRAIN_END
        ).sum()
    )

    test_count = int(
        (
            (
                model_df["ds"]
                >= TEST_START
            )
            & (
                model_df["ds"]
                <= TEST_END
            )
        ).sum()
    )

    if train_count < (
        INPUT_SIZE
        + HORIZON
        + 10
    ):
        raise ValueError(
            "Insufficient training history."
        )

    if test_count < HORIZON:
        raise ValueError(
            "Insufficient test history."
        )

    return model_df


def build_baseline_model():
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

        context_size=(
            CONTEXT_SIZE
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


def build_ed_model():
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

        context_size=(
            CONTEXT_SIZE
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


def run_location(
    model_df,
):
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

    nf = NeuralForecast(
        models=[
            build_baseline_model(),
            build_ed_model(),
        ],
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
        .reset_index(
            drop=True
        )
    )

    cv["horizon"] = (
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


def metric_values(
    actual,
    prediction,
):
    error = (
        prediction
        - actual
    )

    mae = float(
        error
        .abs()
        .mean()
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error
                ** 2
            )
        )
    )

    return (
        mae,
        rmse,
    )


def summarize_location(
    location,
    location_name,
    forecasts,
):
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

        (
            baseline_mae,
            baseline_rmse,
        ) = metric_values(
            actual,
            baseline,
        )

        (
            ed_mae,
            ed_rmse,
        ) = metric_values(
            actual,
            ed,
        )

        baseline_clipped = (
            baseline
            .clip(lower=0.0)
        )

        ed_clipped = (
            ed
            .clip(lower=0.0)
        )

        (
            baseline_mae_clipped,
            baseline_rmse_clipped,
        ) = metric_values(
            actual,
            baseline_clipped,
        )

        (
            ed_mae_clipped,
            ed_rmse_clipped,
        ) = metric_values(
            actual,
            ed_clipped,
        )

        actual_peak_idx = (
            actual.idxmax()
        )

        baseline_peak_idx = (
            baseline.idxmax()
        )

        ed_peak_idx = (
            ed.idxmax()
        )

        actual_peak_date = (
            group.loc[
                actual_peak_idx,
                "ds",
            ]
        )

        actual_peak_value = float(
            actual.loc[
                actual_peak_idx
            ]
        )

        baseline_peak_date = (
            group.loc[
                baseline_peak_idx,
                "ds",
            ]
        )

        baseline_peak_value = float(
            baseline.loc[
                baseline_peak_idx
            ]
        )

        ed_peak_date = (
            group.loc[
                ed_peak_idx,
                "ds",
            ]
        )

        ed_peak_value = float(
            ed.loc[
                ed_peak_idx
            ]
        )

        rows.append(
            {
                "location":
                    location,

                "location_name":
                    location_name,

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

                "baseline_mae_clipped":
                    baseline_mae_clipped,

                "ed_mae_clipped":
                    ed_mae_clipped,

                "baseline_rmse_clipped":
                    baseline_rmse_clipped,

                "ed_rmse_clipped":
                    ed_rmse_clipped,

                "actual_peak_date":
                    actual_peak_date,

                "actual_peak_value":
                    actual_peak_value,

                "baseline_peak_error":
                    (
                        baseline_peak_value
                        - actual_peak_value
                    ),

                "ed_peak_error":
                    (
                        ed_peak_value
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

    return pd.DataFrame(
        rows
    )


def save_incremental(
    summaries,
    forecasts,
    failures,
):
    if summaries:
        pd.concat(
            summaries,
            ignore_index=True,
        ).to_csv(
            SUMMARY_PATH,
            index=False,
        )

    if forecasts:
        pd.concat(
            forecasts,
            ignore_index=True,
        ).to_csv(
            FORECAST_PATH,
            index=False,
        )

    pd.DataFrame(
        failures
    ).to_csv(
        FAILURE_PATH,
        index=False,
    )


def plot_ranking(
    summary_df,
):
    four_week = summary_df[
        summary_df["horizon"]
        == 4
    ].copy()

    four_week = (
        four_week
        .sort_values(
            "rmse_improvement_pct"
        )
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(
        figsize=(
            11,
            15,
        )
    )

    y = np.arange(
        len(four_week)
    )

    ax.barh(
        y,
        four_week[
            "rmse_improvement_pct"
        ],
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        four_week[
            "location_name"
        ]
    )

    ax.axvline(
        0,
        linewidth=1,
    )

    ax.set_xlabel(
        "4-week RMSE improvement from ED feature (%)"
    )

    ax.set_ylabel(
        "State"
    )

    ax.set_title(
        "LSTM + Historical ED Visits: "
        "4-Week RMSE Improvement by State"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    fig.tight_layout()

    path = (
        PLOT_DIR
        / "allstate_lstm_ed_4w_rmse_improvement.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {path}"
    )


def plot_peak_examples(
    summary_df,
    forecast_df,
):
    four_week_summary = (
        summary_df[
            summary_df["horizon"]
            == 4
        ]
        .sort_values(
            "rmse_improvement_pct",
            ascending=False,
        )
        .copy()
    )

    best = (
        four_week_summary
        .head(3)
        .copy()
    )

    worst = (
        four_week_summary
        .tail(3)
        .sort_values(
            "rmse_improvement_pct"
        )
        .copy()
    )

    chosen = pd.concat(
        [
            best,
            worst,
        ],
        ignore_index=True,
    )

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(
            14,
            13,
        )
    )

    axes = (
        axes
        .flatten()
    )

    for ax, row in zip(
        axes,
        chosen.itertuples(),
    ):
        location = row.location

        group = forecast_df[
            (
                forecast_df[
                    "location"
                ]
                == location
            )
            & (
                forecast_df[
                    "horizon"
                ]
                == 4
            )
        ].copy()

        group["ds"] = (
            pd.to_datetime(
                group["ds"]
            )
        )

        group = (
            group
            .sort_values("ds")
        )

        actual_peak_idx = (
            group["y"]
            .idxmax()
        )

        peak_date = (
            group.loc[
                actual_peak_idx,
                "ds",
            ]
        )

        window_start = (
            peak_date
            - pd.Timedelta(
                weeks=3
            )
        )

        window_end = (
            peak_date
            + pd.Timedelta(
                weeks=3
            )
        )

        window = group[
            (
                group["ds"]
                >= window_start
            )
            & (
                group["ds"]
                <= window_end
            )
        ].copy()

        ax.plot(
            window["ds"],
            window["y"],
            marker="o",
            linewidth=2,
            label="Actual",
        )

        ax.plot(
            window["ds"],
            window[
                "LSTMBaseline"
            ],
            marker="o",
            label="Baseline",
        )

        ax.plot(
            window["ds"],
            window[
                "LSTMED"
            ],
            marker="o",
            label="ED feature",
        )

        ax.axvline(
            peak_date,
            linestyle="--",
            linewidth=1,
        )

        ax.set_title(
            f"{row.location_name}\n"
            f"RMSE change: "
            f"{row.rmse_improvement_pct:+.1f}%"
        )

        ax.grid(
            alpha=0.25
        )

        ax.tick_params(
            axis="x",
            rotation=35,
        )

    handles, labels = (
        axes[0]
        .get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
    )

    fig.suptitle(
        "4-Week-Ahead Forecasts Near "
        "the Observed Peak\n"
        "Best and Worst ED-Feature States",
        y=0.995,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.95,
        ]
    )

    path = (
        PLOT_DIR
        / "allstate_lstm_ed_peak_examples.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {path}"
    )


def print_four_week_summary(
    summary_df,
):
    four_week = summary_df[
        summary_df["horizon"]
        == 4
    ].copy()

    four_week = (
        four_week
        .sort_values(
            "rmse_improvement_pct",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    improved = int(
        (
            four_week[
                "rmse_improvement_pct"
            ]
            > 0
        ).sum()
    )

    total = len(
        four_week
    )

    print(
        "\nAll-State 4-Week Summary"
    )

    print(
        "========================"
    )

    print(
        f"Eligible states: "
        f"{total}"
    )

    print(
        f"ED feature improved RMSE: "
        f"{improved}/{total}"
    )

    print(
        "Average relative RMSE change: "
        f"{four_week['rmse_improvement_pct'].mean():+.1f}%"
    )

    print(
        "\nTop 5 improvements"
    )

    print(
        four_week[
            [
                "location_name",
                "baseline_rmse",
                "ed_rmse",
                "rmse_improvement_pct",
            ]
        ]
        .head(5)
        .to_string(
            index=False
        )
    )

    print(
        "\nBottom 5"
    )

    print(
        four_week[
            [
                "location_name",
                "baseline_rmse",
                "ed_rmse",
                "rmse_improvement_pct",
            ]
        ]
        .tail(5)
        .sort_values(
            "rmse_improvement_pct"
        )
        .to_string(
            index=False
        )
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

    (
        hospital_df,
        ed_df,
    ) = load_source_data()

    locations = (
        get_location_table(
            hospital_df
        )
    )

    summaries = []
    all_forecasts = []
    failures = []

    total_locations = len(
        locations
    )

    print(
        "\nAll-State LSTM ED Experiment"
    )

    print(
        "============================"
    )

    print(
        f"Candidate locations: "
        f"{total_locations}"
    )

    for index, row in enumerate(
        locations.itertuples(),
        start=1,
    ):
        location = row.location

        location_name = (
            row.location_name
        )

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"[{index}/{total_locations}] "
            f"{location_name} "
            f"({location})"
        )

        print(
            "=" * 60
        )

        try:
            model_df = (
                build_location_data(
                    hospital_df=(
                        hospital_df
                    ),
                    ed_df=ed_df,
                    location=location,
                )
            )

            forecasts = (
                run_location(
                    model_df
                )
            )

            forecasts[
                "location"
            ] = location

            forecasts[
                "location_name"
            ] = location_name

            summary = (
                summarize_location(
                    location=location,
                    location_name=(
                        location_name
                    ),
                    forecasts=(
                        forecasts
                    ),
                )
            )

            summaries.append(
                summary
            )

            all_forecasts.append(
                forecasts
            )

            four_week = (
                summary[
                    summary[
                        "horizon"
                    ]
                    == 4
                ]
                .iloc[0]
            )

            print(
                "4-week RMSE: "
                f"{four_week['baseline_rmse']:.4f}"
                " -> "
                f"{four_week['ed_rmse']:.4f}"
                " "
                f"({four_week['rmse_improvement_pct']:+.1f}%)"
            )

        except Exception as exc:
            print(
                f"SKIPPED: {exc}"
            )

            failures.append(
                {
                    "location":
                        location,

                    "location_name":
                        location_name,

                    "reason":
                        str(exc),
                }
            )

        # Save after every location.
        save_incremental(
            summaries=summaries,
            forecasts=all_forecasts,
            failures=failures,
        )

    if not summaries:
        raise RuntimeError(
            "No locations completed successfully."
        )

    summary_df = pd.concat(
        summaries,
        ignore_index=True,
    )

    forecast_df = pd.concat(
        all_forecasts,
        ignore_index=True,
    )

    print_four_week_summary(
        summary_df
    )

    plot_ranking(
        summary_df
    )

    plot_peak_examples(
        summary_df=summary_df,
        forecast_df=forecast_df,
    )

    print(
        f"\nSaved: {SUMMARY_PATH}"
    )

    print(
        f"Saved: {FORECAST_PATH}"
    )

    print(
        f"Saved: {FAILURE_PATH}"
    )


if __name__ == "__main__":
    main()