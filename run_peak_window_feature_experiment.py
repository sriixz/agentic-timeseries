import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from epiweeks import Week
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import LSTM


DATA_PATH = Path(
    "data/target-hospital-admissions.csv"
)

EXPERIMENT_PATH = Path(
    "configs/generated_autolstm_experiment.json"
)

REFINED_CONFIG_PATH = Path(
    "forecast_results/refined_autolstm_best_config.json"
)

PEAK_PROBABILITY_PATH = Path(
    "analysis_results/us_peak_probability_by_epiweek.csv"
)

RESULT_DIR = Path("forecast_results")
PLOT_DIR = Path("plots")


POINT_FEATURE = (
    "peak_probability_shifted_4w"
)

WINDOW_FEATURE = (
    "peak_probability_next_4w"
)

WINDOW_WEEKS = 4


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
    """
    Convert epiweek to continuous seasonal position.

    EW52 -> 52
    EW53 -> 53
    EW01 -> 54
    EW02 -> 55
    """
    week = Week.fromdate(
        date.date()
    ).week

    if week >= 31:
        return week

    return week + 53


def load_experiment() -> dict:
    experiment = load_json(
        EXPERIMENT_PATH
    )

    required = {
        "location",
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


def load_refined_config() -> dict:
    output = load_json(
        REFINED_CONFIG_PATH
    )

    if "config" not in output:
        raise ValueError(
            "Refined config file does not contain "
            "a 'config' field."
        )

    return output["config"]


def build_peak_features() -> pd.DataFrame:
    """
    Load the original national peak-probability curve.

    Keep the existing exact 4-week shifted feature and
    create a new feature:

        P(peak within next 4 weeks)

      = P(t+1) + P(t+2) + P(t+3) + P(t+4)

    Because peak_probability is a normalized discrete
    probability distribution, this sum is directly
    interpretable as probability mass in the next
    four-week window.
    """
    if not PEAK_PROBABILITY_PATH.exists():
        raise FileNotFoundError(
            "Peak-probability file not found: "
            f"{PEAK_PROBABILITY_PATH}"
        )

    df = pd.read_csv(
        PEAK_PROBABILITY_PATH
    )

    required = {
        "seasonal_position",
        "peak_probability",
        POINT_FEATURE,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Peak-probability file missing columns: "
            f"{sorted(missing)}"
        )

    probability_lookup = dict(
        zip(
            df["seasonal_position"].astype(int),
            df["peak_probability"].astype(float),
        )
    )

    window_values = []

    for position in df[
        "seasonal_position"
    ].astype(int):

        probability_next_4w = sum(
            probability_lookup.get(
                position + offset,
                0.0,
            )
            for offset in range(
                1,
                WINDOW_WEEKS + 1,
            )
        )

        window_values.append(
            float(
                probability_next_4w
            )
        )

    df[
        WINDOW_FEATURE
    ] = window_values

    return df[
        [
            "seasonal_position",
            "peak_probability",
            POINT_FEATURE,
            WINDOW_FEATURE,
        ]
    ].copy()


def load_forecasting_data(
    experiment: dict,
    features: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    target = experiment[
        "target"
    ]

    location = str(
        experiment["location"]
    )

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
        == location
    ].copy()

    if scoped.empty:
        raise ValueError(
            f"No rows found for location={location}"
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
        features,
        on="seasonal_position",
        how="left",
        validate="many_to_one",
    )

    feature_columns = [
        POINT_FEATURE,
        WINDOW_FEATURE,
    ]

    for column in feature_columns:
        missing_count = int(
            scoped[
                column
            ].isna().sum()
        )

        if missing_count > 0:
            raise ValueError(
                f"Missing {column} values "
                f"after merge: {missing_count}"
            )

    model_df = pd.DataFrame(
        {
            "unique_id": location,
            "ds": scoped["date"],
            "y": scoped[
                target
            ].astype(float),
            POINT_FEATURE: scoped[
                POINT_FEATURE
            ].astype(float),
            WINDOW_FEATURE: scoped[
                WINDOW_FEATURE
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


def identify_forecast_column(
    cv: pd.DataFrame,
) -> str:
    excluded = {
        "unique_id",
        "ds",
        "cutoff",
        "y",
        "horizon",
        POINT_FEATURE,
        WINDOW_FEATURE,
    }

    candidates = [
        column
        for column in cv.columns
        if column not in excluded
    ]

    if len(candidates) != 1:
        raise RuntimeError(
            "Could not uniquely identify forecast "
            f"column. Candidates: {candidates}"
        )

    return candidates[0]


def run_model(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
    experiment: dict,
    feature_column: str | None,
) -> pd.DataFrame:
    horizon = int(
        experiment["horizon"]
    )

    frequency = experiment[
        "frequency"
    ]

    model_kwargs = {
        **config,
        "h": horizon,
        "loss": MAE(),
    }

    if feature_column is not None:
        model_kwargs[
            "futr_exog_list"
        ] = [
            feature_column
        ]

    print(
        f"\nTraining {model_name}"
    )

    print(
        "=" * (
            len(model_name)
            + 9
        )
    )

    if feature_column is None:
        print(
            "Future exogenous feature: none"
        )
    else:
        print(
            "Future exogenous feature: "
            f"{feature_column}"
        )

    model = LSTM(
        **model_kwargs
    )

    nf = NeuralForecast(
        models=[
            model
        ],
        freq=frequency,
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

    forecast_column = (
        identify_forecast_column(
            cv
        )
    )

    cv = cv.rename(
        columns={
            forecast_column:
                "raw_forecast"
        }
    )

    cv[
        "forecast"
    ] = (
        cv[
            "raw_forecast"
        ]
        .clip(
            lower=0
        )
    )

    cv[
        "was_clipped"
    ] = (
        cv[
            "raw_forecast"
        ]
        < 0
    )

    cv[
        "model"
    ] = model_name

    return cv


def calculate_metrics(
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Use raw forecasts so the comparison measures
    what each model actually learned rather than
    the effect of clipping.
    """
    rows = []

    for horizon, group in (
        forecasts.groupby(
            "horizon"
        )
    ):
        error = (
            group[
                "raw_forecast"
            ]
            - group["y"]
        )

        mae = float(
            error.abs().mean()
        )

        rmse = float(
            error.pow(2)
            .mean()
            ** 0.5
        )

        bias = float(
            error.mean()
        )

        actual_peak = (
            group.loc[
                group[
                    "y"
                ].idxmax()
            ]
        )

        forecast_peak = (
            group.loc[
                group[
                    "raw_forecast"
                ].idxmax()
            ]
        )

        peak_magnitude_error = (
            float(
                forecast_peak[
                    "raw_forecast"
                ]
            )
            - float(
                actual_peak[
                    "y"
                ]
            )
        )

        peak_timing_error_weeks = (
            (
                forecast_peak[
                    "ds"
                ]
                - actual_peak[
                    "ds"
                ]
            ).days
            / 7.0
        )

        negative_count = int(
            (
                group[
                    "raw_forecast"
                ]
                < 0
            ).sum()
        )

        rows.append(
            {
                "horizon":
                    int(horizon),
                "forecast_count":
                    len(group),
                "mae":
                    mae,
                "rmse":
                    rmse,
                "bias":
                    bias,
                "peak_magnitude_error":
                    peak_magnitude_error,
                "peak_timing_error_weeks":
                    peak_timing_error_weeks,
                "negative_forecast_count":
                    negative_count,
            }
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


def build_comparison(
    baseline: pd.DataFrame,
    point: pd.DataFrame,
    window: pd.DataFrame,
) -> pd.DataFrame:
    baseline = (
        baseline
        .add_prefix(
            "baseline_"
        )
        .rename(
            columns={
                "baseline_horizon":
                    "horizon"
            }
        )
    )

    point = (
        point
        .add_prefix(
            "point_"
        )
        .rename(
            columns={
                "point_horizon":
                    "horizon"
            }
        )
    )

    window = (
        window
        .add_prefix(
            "window_"
        )
        .rename(
            columns={
                "window_horizon":
                    "horizon"
            }
        )
    )

    comparison = (
        baseline
        .merge(
            point,
            on="horizon",
            validate="one_to_one",
        )
        .merge(
            window,
            on="horizon",
            validate="one_to_one",
        )
    )

    return comparison


def print_comparison(
    comparison: pd.DataFrame,
) -> None:
    print(
        "\nThree-Way Peak Feature Comparison"
    )

    print(
        "================================="
    )

    for row in comparison.itertuples():
        print(
            f"\n{row.horizon}-week horizon"
        )

        print(
            "  MAE"
        )
        print(
            f"    baseline: "
            f"{row.baseline_mae:.4f}"
        )
        print(
            f"    point:    "
            f"{row.point_mae:.4f}"
        )
        print(
            f"    next-4w:  "
            f"{row.window_mae:.4f}"
        )

        print(
            "  RMSE"
        )
        print(
            f"    baseline: "
            f"{row.baseline_rmse:.4f}"
        )
        print(
            f"    point:    "
            f"{row.point_rmse:.4f}"
        )
        print(
            f"    next-4w:  "
            f"{row.window_rmse:.4f}"
        )

        print(
            "  Bias"
        )
        print(
            f"    baseline: "
            f"{row.baseline_bias:+.4f}"
        )
        print(
            f"    point:    "
            f"{row.point_bias:+.4f}"
        )
        print(
            f"    next-4w:  "
            f"{row.window_bias:+.4f}"
        )

        print(
            "  Peak magnitude error"
        )
        print(
            f"    baseline: "
            f"{row.baseline_peak_magnitude_error:+.4f}"
        )
        print(
            f"    point:    "
            f"{row.point_peak_magnitude_error:+.4f}"
        )
        print(
            f"    next-4w:  "
            f"{row.window_peak_magnitude_error:+.4f}"
        )

        print(
            "  Peak timing error"
        )
        print(
            f"    baseline: "
            f"{row.baseline_peak_timing_error_weeks:+.1f}"
        )
        print(
            f"    point:    "
            f"{row.point_peak_timing_error_weeks:+.1f}"
        )
        print(
            f"    next-4w:  "
            f"{row.window_peak_timing_error_weeks:+.1f}"
        )

        print(
            "  Negative forecasts"
        )
        print(
            f"    baseline: "
            f"{row.baseline_negative_forecast_count}"
        )
        print(
            f"    point:    "
            f"{row.point_negative_forecast_count}"
        )
        print(
            f"    next-4w:  "
            f"{row.window_negative_forecast_count}"
        )


def save_results(
    features: pd.DataFrame,
    baseline_forecasts: pd.DataFrame,
    point_forecasts: pd.DataFrame,
    window_forecasts: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    point_metrics: pd.DataFrame,
    window_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        RESULT_DIR
        / "us_peak_probability_with_next4w.csv",
        index=False,
    )

    baseline_forecasts.to_csv(
        RESULT_DIR
        / "peak_window_baseline_forecasts.csv",
        index=False,
    )

    point_forecasts.to_csv(
        RESULT_DIR
        / "peak_window_point_forecasts.csv",
        index=False,
    )

    window_forecasts.to_csv(
        RESULT_DIR
        / "peak_window_next4w_forecasts.csv",
        index=False,
    )

    baseline_metrics.to_csv(
        RESULT_DIR
        / "peak_window_baseline_metrics.csv",
        index=False,
    )

    point_metrics.to_csv(
        RESULT_DIR
        / "peak_window_point_metrics.csv",
        index=False,
    )

    window_metrics.to_csv(
        RESULT_DIR
        / "peak_window_next4w_metrics.csv",
        index=False,
    )

    comparison.to_csv(
        RESULT_DIR
        / "peak_window_three_way_comparison.csv",
        index=False,
    )

    print(
        "\nSaved three-way peak feature results."
    )


def plot_feature_curves(
    features: pd.DataFrame,
) -> None:
    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(
            13,
            7,
        )
    )

    ax.plot(
        features[
            "seasonal_position"
        ],
        features[
            POINT_FEATURE
        ],
        marker="o",
        markersize=3,
        label=(
            "Exact 4-week shifted probability"
        ),
    )

    ax.plot(
        features[
            "seasonal_position"
        ],
        features[
            WINDOW_FEATURE
        ],
        marker="o",
        markersize=3,
        label=(
            "Probability peak occurs within next 4 weeks"
        ),
    )

    ax.set_title(
        "U.S. Peak-Timing Features"
    )

    ax.set_xlabel(
        "Seasonal epiweek position"
    )

    ax.set_ylabel(
        "Peak probability"
    )

    ax.legend()

    ax.grid(
        alpha=0.25,
    )

    fig.tight_layout()

    output_path = (
        PLOT_DIR
        / "peak_point_vs_next4w_feature.png"
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
        f"Saved: {output_path}"
    )


def plot_forecast_comparison(
    baseline_forecasts: pd.DataFrame,
    point_forecasts: pd.DataFrame,
    window_forecasts: pd.DataFrame,
) -> None:
    horizons = sorted(
        baseline_forecasts[
            "horizon"
        ].unique()
    )

    for horizon in horizons:
        baseline = (
            baseline_forecasts[
                baseline_forecasts[
                    "horizon"
                ]
                == horizon
            ]
            .sort_values(
                "ds"
            )
        )

        point = (
            point_forecasts[
                point_forecasts[
                    "horizon"
                ]
                == horizon
            ]
            .sort_values(
                "ds"
            )
        )

        window = (
            window_forecasts[
                window_forecasts[
                    "horizon"
                ]
                == horizon
            ]
            .sort_values(
                "ds"
            )
        )

        fig, ax = plt.subplots(
            figsize=(
                13,
                7,
            )
        )

        ax.plot(
            baseline[
                "ds"
            ],
            baseline[
                "y"
            ],
            marker="o",
            linewidth=2,
            label="Actual",
        )

        ax.plot(
            baseline[
                "ds"
            ],
            baseline[
                "forecast"
            ],
            marker="o",
            label="Baseline",
        )

        ax.plot(
            point[
                "ds"
            ],
            point[
                "forecast"
            ],
            marker="o",
            label=(
                "Exact 4-week feature"
            ),
        )

        ax.plot(
            window[
                "ds"
            ],
            window[
                "forecast"
            ],
            marker="o",
            label=(
                "Next-4-week probability"
            ),
        )

        ax.set_title(
            "Peak Feature Comparison: "
            f"{horizon}-Week Horizon"
        )

        ax.set_xlabel(
            "Date"
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
                "peak_window_comparison_"
                f"{horizon}w.png"
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
            f"Saved: {output_path}"
        )


def main():
    experiment = (
        load_experiment()
    )

    config = (
        load_refined_config()
    )

    features = (
        build_peak_features()
    )

    train_df, test_df = (
        load_forecasting_data(
            experiment=experiment,
            features=features,
        )
    )

    print(
        "\nControlled Three-Way "
        "Peak-Feature Experiment"
    )

    print(
        "===================================="
    )

    print(
        "A. Baseline"
    )

    print(
        "B. Exact 4-week shifted probability"
    )

    print(
        "C. Probability of peak within "
        "next 4 weeks"
    )

    print(
        f"\nInput size: "
        f"{config['input_size']}"
    )

    baseline_forecasts = (
        run_model(
            model_name="baseline",
            train_df=train_df,
            test_df=test_df,
            config=config,
            experiment=experiment,
            feature_column=None,
        )
    )

    point_forecasts = (
        run_model(
            model_name="point_feature",
            train_df=train_df,
            test_df=test_df,
            config=config,
            experiment=experiment,
            feature_column=POINT_FEATURE,
        )
    )

    window_forecasts = (
        run_model(
            model_name="next4w_feature",
            train_df=train_df,
            test_df=test_df,
            config=config,
            experiment=experiment,
            feature_column=WINDOW_FEATURE,
        )
    )

    baseline_metrics = (
        calculate_metrics(
            baseline_forecasts
        )
    )

    point_metrics = (
        calculate_metrics(
            point_forecasts
        )
    )

    window_metrics = (
        calculate_metrics(
            window_forecasts
        )
    )

    comparison = (
        build_comparison(
            baseline=(
                baseline_metrics
            ),
            point=(
                point_metrics
            ),
            window=(
                window_metrics
            ),
        )
    )

    print_comparison(
        comparison
    )

    save_results(
        features=features,
        baseline_forecasts=(
            baseline_forecasts
        ),
        point_forecasts=(
            point_forecasts
        ),
        window_forecasts=(
            window_forecasts
        ),
        baseline_metrics=(
            baseline_metrics
        ),
        point_metrics=(
            point_metrics
        ),
        window_metrics=(
            window_metrics
        ),
        comparison=comparison,
    )

    plot_feature_curves(
        features
    )

    plot_forecast_comparison(
        baseline_forecasts=(
            baseline_forecasts
        ),
        point_forecasts=(
            point_forecasts
        ),
        window_forecasts=(
            window_forecasts
        ),
    )


if __name__ == "__main__":
    main()