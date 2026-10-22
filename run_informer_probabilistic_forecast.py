import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from epiweeks import Week
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.models import Informer


DATA_PATH = Path("data/target-hospital-admissions.csv")

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
            "Peak probability file missing columns: "
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

    next4w_values = []

    for position in probability_df[
        "seasonal_position"
    ].astype(int):

        probability = sum(
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

        next4w_values.append(
            float(
                probability
            )
        )

    return pd.DataFrame(
        {
            "seasonal_position":
                probability_df[
                    "seasonal_position"
                ].astype(int),

            FEATURE_COLUMN:
                next4w_values,
        }
    )


def load_data(
    experiment: dict,
    feature_df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    df = pd.read_csv(
        DATA_PATH,
        dtype={
            "location": str,
        },
    )

    target = experiment[
        "target"
    ]

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df["location"] = (
        df["location"]
        .astype(str)
        .str.strip()
    )

    scoped = df[
        df["location"] == "US"
    ].copy()

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

    if scoped[
        FEATURE_COLUMN
    ].isna().any():
        raise ValueError(
            "Missing peak-probability feature values."
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

    return (
        train_df,
        test_df,
    )


def run_probabilistic_informer(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    experiment: dict,
) -> pd.DataFrame:
    loss = MQLoss(
        level=LEVELS
    )

    model = Informer(
        h=int(
            experiment[
                "horizon"
            ]
        ),

        input_size=INPUT_SIZE,

        loss=loss,

        valid_loss=loss,

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

    cv = nf.cross_validation(
        df=full_df,
        n_windows=None,
        test_size=len(
            test_df
        ),
        step_size=1,
        refit=False,
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
        <= int(
            experiment[
                "horizon"
            ]
        )
    ].copy()

    cv = cv[
        (
            cv["ds"]
            >= pd.Timestamp(
                experiment[
                    "test_start"
                ]
            )
        )
        & (
            cv["ds"]
            <= pd.Timestamp(
                experiment[
                    "test_end"
                ]
            )
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
            f"Expected one column ending in "
            f"'{suffix}', found: {matches}"
        )

    return matches[0]


def standardize_columns(
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    print(
        "\nProbabilistic forecast columns"
    )

    print(
        "=============================="
    )

    for column in forecasts.columns:
        print(
            column
        )

    median_col = find_column(
        forecasts.columns,
        "-median",
    )

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
            median_col:
                "median_forecast",

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


def calculate_interval_metrics(
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for horizon, group in forecasts.groupby(
        "horizon"
    ):
        group = group.copy()

        point_error = (
            group[
                "median_forecast"
            ]
            - group["y"]
        )

        row = {
            "horizon":
                int(horizon),

            "median_mae":
                float(
                    point_error.abs().mean()
                ),

            "median_rmse":
                float(
                    (
                        point_error.pow(2)
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
                (group["y"] >= lower)
                & (group["y"] <= upper)
            )

            width = (
                upper - lower
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
        "\nProbabilistic Forecast Evaluation"
    )

    print(
        "================================"
    )

    for row in metrics.itertuples():
        print(
            f"\n{row.horizon}-week horizon"
        )

        print(
            "  Median MAE: "
            f"{row.median_mae:.4f}"
        )

        print(
            "  Median RMSE: "
            f"{row.median_rmse:.4f}"
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


def plot_four_week_intervals(
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
        label="95% prediction interval",
    )

    ax.fill_between(
        four_week["ds"],
        four_week["lo_90"],
        four_week["hi_90"],
        alpha=0.28,
        label="90% prediction interval",
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
            "median_forecast"
        ],
        marker="o",
        label=(
            "Informer + peak feature "
            "median"
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
        "4-Week-Ahead Probabilistic Forecast"
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
        / "informer_peak_feature_probabilistic_us_4w.png"
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
        "\nProbabilistic Informer Experiment"
    )

    print(
        "================================"
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
        "Prediction intervals: "
        "90%, 95%"
    )

    forecasts = (
        run_probabilistic_informer(
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
        calculate_interval_metrics(
            forecasts
        )
    )

    print_metrics(
        metrics
    )

    forecast_path = (
        RESULT_DIR
        / "informer_peak_feature_probabilistic_forecasts.csv"
    )

    metrics_path = (
        RESULT_DIR
        / "informer_peak_feature_probabilistic_metrics.csv"
    )

    forecasts.to_csv(
        forecast_path,
        index=False,
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    plot_four_week_intervals(
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