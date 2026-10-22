import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from epiweeks import Week
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import Informer
from neuralforecast.utils import PredictionIntervals


DATA_PATH = Path(
    "data/target-hospital-admissions.csv"
)

EXPERIMENT_PATH = Path(
    "configs/generated_autolstm_experiment.json"
)

PEAK_PROBABILITY_PATH = Path(
    "analysis_results/us_peak_probability_by_epiweek.csv"
)

RESULT_DIR = Path("forecast_results")
PLOT_DIR = Path("plots")


FEATURE_COLUMN = "peak_probability_next_4w"

WINDOW_WEEKS = 4

INPUT_SIZE = 12
MAX_STEPS = 1000
RANDOM_SEED = 2

LEVELS = [90, 95]

# Reduced from 8 for a faster first conformal experiment.
CONFORMAL_WINDOWS = 4


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def seasonal_epiweek_position(
    date: pd.Timestamp,
) -> int:
    week = Week.fromdate(
        date.date()
    ).week

    if week >= 31:
        return week

    return week + 53


def epiweek_label(
    date_value,
) -> str:
    timestamp = pd.Timestamp(
        date_value
    )

    week = Week.fromdate(
        timestamp.date()
    )

    return (
        f"{week.year}-EW{week.week:02d}"
    )


def load_experiment() -> dict:
    experiment = load_json(
        EXPERIMENT_PATH
    )

    required = {
        "target",
        "train_end",
        "test_start",
        "test_end",
        "horizon",
        "frequency",
    }

    missing = (
        required
        - set(experiment)
    )

    if missing:
        raise ValueError(
            "Experiment metadata missing keys: "
            f"{sorted(missing)}"
        )

    return experiment


def build_next4w_feature() -> pd.DataFrame:
    if not PEAK_PROBABILITY_PATH.exists():
        raise FileNotFoundError(
            "Peak probability file not found: "
            f"{PEAK_PROBABILITY_PATH}"
        )

    probability_df = pd.read_csv(
        PEAK_PROBABILITY_PATH
    )

    required = {
        "seasonal_position",
        "peak_probability",
    }

    missing = (
        required
        - set(
            probability_df.columns
        )
    )

    if missing:
        raise ValueError(
            "Peak probability data missing columns: "
            f"{sorted(missing)}"
        )

    probability_df = (
        probability_df
        .copy()
        .sort_values(
            "seasonal_position"
        )
    )

    lookup = dict(
        zip(
            probability_df[
                "seasonal_position"
            ].astype(int),
            probability_df[
                "peak_probability"
            ].astype(float),
        )
    )

    values = []

    for position in probability_df[
        "seasonal_position"
    ].astype(int):

        next4w_probability = sum(
            lookup.get(
                position + offset,
                0.0,
            )
            for offset
            in range(
                1,
                WINDOW_WEEKS + 1,
            )
        )

        values.append(
            float(
                next4w_probability
            )
        )

    return pd.DataFrame(
        {
            "seasonal_position":
                probability_df[
                    "seasonal_position"
                ].astype(int),

            FEATURE_COLUMN:
                values,
        }
    )


def load_data(
    experiment: dict,
    feature_df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        dtype={
            "location": str,
        },
    )

    target = experiment[
        "target"
    ]

    required = {
        "date",
        "location",
        target,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Dataset missing columns: "
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

    scoped = df[
        df["location"]
        == "US"
    ].copy()

    if scoped.empty:
        raise ValueError(
            "No U.S. national series found."
        )

    scoped = scoped.dropna(
        subset=[
            "date",
            target,
        ]
    )

    scoped = scoped.sort_values(
        "date"
    )

    scoped[
        "seasonal_position"
    ] = scoped[
        "date"
    ].apply(
        seasonal_epiweek_position
    )

    scoped = scoped.merge(
        feature_df,
        on="seasonal_position",
        how="left",
        validate="many_to_one",
    )

    missing_feature_count = int(
        scoped[
            FEATURE_COLUMN
        ].isna().sum()
    )

    if missing_feature_count > 0:
        raise ValueError(
            "Missing peak feature values: "
            f"{missing_feature_count}"
        )

    model_df = pd.DataFrame(
        {
            "unique_id":
                "US",

            "ds":
                scoped["date"],

            "y":
                scoped[
                    target
                ].astype(float),

            FEATURE_COLUMN:
                scoped[
                    FEATURE_COLUMN
                ].astype(float),
        }
    )

    train_end = pd.Timestamp(
        experiment[
            "train_end"
        ]
    )

    test_start = pd.Timestamp(
        experiment[
            "test_start"
        ]
    )

    test_end = pd.Timestamp(
        experiment[
            "test_end"
        ]
    )

    train_df = model_df[
        model_df["ds"]
        <= train_end
    ].copy()

    test_df = model_df[
        (
            model_df["ds"]
            >= test_start
        )
        & (
            model_df["ds"]
            <= test_end
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

    return (
        train_df,
        test_df,
    )


def build_model(
    horizon: int,
) -> Informer:
    return Informer(
        h=horizon,

        input_size=INPUT_SIZE,

        loss=MAE(),

        futr_exog_list=[
            FEATURE_COLUMN
        ],

        max_steps=MAX_STEPS,

        random_seed=RANDOM_SEED,

        scaler_type="standard",

        hidden_size=64,

        n_head=4,

        encoder_layers=2,

        decoder_layers=1,

        dropout=0.05,

        learning_rate=1e-4,

        batch_size=32,
    )


def run_conformal_cross_validation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    experiment: dict,
) -> pd.DataFrame:
    horizon = int(
        experiment[
            "horizon"
        ]
    )

    model = build_model(
        horizon
    )

    nf = NeuralForecast(
        models=[
            model
        ],
        freq=experiment[
            "frequency"
        ],
    )

    full_df = pd.concat(
        [
            train_df,
            test_df,
        ],
        ignore_index=True,
    )

    prediction_intervals = (
        PredictionIntervals(
            n_windows=CONFORMAL_WINDOWS,
            method="conformal_error",
            step_size=1,
        )
    )

    print(
        "\nConformal configuration"
    )

    print(
        "======================="
    )

    print(
        f"Method: "
        f"{prediction_intervals.method}"
    )

    print(
        f"Calibration windows: "
        f"{prediction_intervals.n_windows}"
    )

    print(
        f"Calibration step size: "
        f"{prediction_intervals.step_size}"
    )

    print(
        "Cross-validation refit: True"
    )

    cv = nf.cross_validation(
        df=full_df,

        n_windows=None,

        test_size=len(
            test_df
        ),

        step_size=1,

        # Required when prediction_intervals
        # are supplied to cross_validation.
        refit=True,

        prediction_intervals=(
            prediction_intervals
        ),

        level=LEVELS,
    )

    cv = cv.sort_values(
        [
            "cutoff",
            "ds",
        ]
    ).reset_index(
        drop=True
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
        <= horizon
    ].copy()

    test_start = pd.Timestamp(
        experiment[
            "test_start"
        ]
    )

    test_end = pd.Timestamp(
        experiment[
            "test_end"
        ]
    )

    cv = cv[
        (
            cv["ds"]
            >= test_start
        )
        & (
            cv["ds"]
            <= test_end
        )
    ].copy()

    return cv


def find_column(
    columns,
    suffix: str,
) -> str:
    matches = [
        column
        for column
        in columns
        if column.endswith(
            suffix
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one column ending "
            f"in '{suffix}', found: {matches}"
        )

    return matches[0]


def standardize_columns(
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    print(
        "\nConformal forecast columns"
    )

    print(
        "=========================="
    )

    for column in forecasts.columns:
        print(
            column
        )

    point_candidates = [
        column
        for column
        in forecasts.columns
        if (
            column.startswith(
                "Informer"
            )
            and "-lo-" not in column
            and "-hi-" not in column
        )
    ]

    if len(point_candidates) != 1:
        raise RuntimeError(
            "Could not uniquely identify "
            "Informer point forecast column: "
            f"{point_candidates}"
        )

    point_col = point_candidates[0]

    lo90_col = find_column(
        forecasts.columns,
        "-lo-90",
    )

    hi90_col = find_column(
        forecasts.columns,
        "-hi-90",
    )

    lo95_col = find_column(
        forecasts.columns,
        "-lo-95",
    )

    hi95_col = find_column(
        forecasts.columns,
        "-hi-95",
    )

    forecasts = forecasts.rename(
        columns={
            point_col:
                "point_forecast",

            lo90_col:
                "lo_90",

            hi90_col:
                "hi_90",

            lo95_col:
                "lo_95",

            hi95_col:
                "hi_95",
        }
    )

    return forecasts


def calculate_metrics(
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for horizon, group in (
        forecasts.groupby(
            "horizon"
        )
    ):
        error = (
            group[
                "point_forecast"
            ]
            - group["y"]
        )

        row = {
            "horizon":
                int(horizon),

            "point_mae":
                float(
                    error.abs().mean()
                ),

            "point_rmse":
                float(
                    (
                        error.pow(2)
                        .mean()
                    )
                    ** 0.5
                ),
        }

        for level in LEVELS:
            lower = group[
                f"lo_{level}"
            ]

            upper = group[
                f"hi_{level}"
            ]

            covered = (
                (
                    group["y"]
                    >= lower
                )
                & (
                    group["y"]
                    <= upper
                )
            )

            width = (
                upper
                - lower
            )

            row[
                f"coverage_{level}"
            ] = float(
                covered.mean()
            )

            row[
                f"mean_width_{level}"
            ] = float(
                width.mean()
            )

        rows.append(
            row
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "horizon"
        )
        .reset_index(
            drop=True
        )
    )


def print_metrics(
    metrics: pd.DataFrame,
) -> None:
    print(
        "\nConformal Forecast Evaluation"
    )

    print(
        "============================="
    )

    for row in (
        metrics.itertuples()
    ):
        print(
            f"\n{row.horizon}-week horizon"
        )

        print(
            "  Point MAE: "
            f"{row.point_mae:.4f}"
        )

        print(
            "  Point RMSE: "
            f"{row.point_rmse:.4f}"
        )

        print(
            "  90% interval coverage: "
            f"{row.coverage_90 * 100:.1f}%"
        )

        print(
            "  90% mean interval width: "
            f"{row.mean_width_90:.4f}"
        )

        print(
            "  95% interval coverage: "
            f"{row.coverage_95 * 100:.1f}%"
        )

        print(
            "  95% mean interval width: "
            f"{row.mean_width_95:.4f}"
        )


def plot_four_week(
    forecasts: pd.DataFrame,
) -> None:
    four_week = (
        forecasts[
            forecasts[
                "horizon"
            ]
            == 4
        ]
        .sort_values(
            "ds"
        )
        .copy()
    )

    if four_week.empty:
        raise ValueError(
            "No four-week forecasts found."
        )

    fig, ax = plt.subplots(
        figsize=(
            13,
            7,
        )
    )

    ax.fill_between(
        four_week["ds"],
        four_week["lo_95"],
        four_week["hi_95"],
        alpha=0.18,
        label=(
            "95% conformal interval"
        ),
    )

    ax.fill_between(
        four_week["ds"],
        four_week["lo_90"],
        four_week["hi_90"],
        alpha=0.28,
        label=(
            "90% conformal interval"
        ),
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
            "point_forecast"
        ],
        marker="o",
        label=(
            "Informer + peak feature"
        ),
    )

    dates = (
        four_week[
            "ds"
        ]
        .tolist()
    )

    tick_dates = dates[::4]

    if (
        dates
        and dates[-1]
        not in tick_dates
    ):
        tick_dates.append(
            dates[-1]
        )

    ax.set_xticks(
        tick_dates,
        [
            epiweek_label(
                date
            )
            for date
            in tick_dates
        ],
    )

    ax.tick_params(
        axis="x",
        rotation=45,
    )

    ax.set_title(
        "U.S. Informer + Peak Feature: "
        "4-Week-Ahead Conformal Forecast"
    )

    ax.set_xlabel(
        "Target epidemiological week"
    )

    ax.set_ylabel(
        "Weekly hospitalization rate"
    )

    ax.legend()

    ax.grid(
        alpha=0.25,
    )

    fig.tight_layout()

    output_path = (
        PLOT_DIR
        / (
            "informer_peak_feature_"
            "conformal_us_4w.png"
        )
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

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

    experiment = (
        load_experiment()
    )

    feature_df = (
        build_next4w_feature()
    )

    train_df, test_df = (
        load_data(
            experiment=experiment,
            feature_df=feature_df,
        )
    )

    print(
        "\nConformal Informer Experiment"
    )

    print(
        "============================"
    )

    print(
        f"Train observations: "
        f"{len(train_df)}"
    )

    print(
        f"Test observations: "
        f"{len(test_df)}"
    )

    print(
        f"Input size: "
        f"{INPUT_SIZE}"
    )

    print(
        f"Forecast horizon: "
        f"{experiment['horizon']}"
    )

    print(
        "Requested intervals: "
        "90%, 95%"
    )

    print(
        f"Calibration windows: "
        f"{CONFORMAL_WINDOWS}"
    )

    forecasts = (
        run_conformal_cross_validation(
            train_df=train_df,
            test_df=test_df,
            experiment=experiment,
        )
    )

    forecasts = (
        standardize_columns(
            forecasts
        )
    )

    metrics = (
        calculate_metrics(
            forecasts
        )
    )

    print_metrics(
        metrics
    )

    forecast_path = (
        RESULT_DIR
        / (
            "informer_peak_feature_"
            "conformal_forecasts.csv"
        )
    )

    metrics_path = (
        RESULT_DIR
        / (
            "informer_peak_feature_"
            "conformal_metrics.csv"
        )
    )

    forecasts.to_csv(
        forecast_path,
        index=False,
    )

    metrics.to_csv(
        metrics_path,
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


if __name__ == "__main__":
    main()