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

PEAK_FEATURE_PATH = Path(
    "analysis_results/us_peak_probability_by_epiweek.csv"
)

RESULT_DIR = Path("forecast_results")
PLOT_DIR = Path("plots")

FEATURE_COLUMN = (
    "peak_probability_shifted_4w"
)


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
    Convert an epidemiological week to the same
    continuous seasonal position used by the
    peak-probability feature.

    Examples:
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
            "Refined config file does not "
            "contain a 'config' field."
        )

    return output["config"]


def load_peak_feature() -> pd.DataFrame:
    if not PEAK_FEATURE_PATH.exists():
        raise FileNotFoundError(
            "National peak probability file "
            f"not found: {PEAK_FEATURE_PATH}"
        )

    features = pd.read_csv(
        PEAK_FEATURE_PATH
    )

    required = {
        "seasonal_position",
        FEATURE_COLUMN,
    }

    missing = (
        required
        - set(features.columns)
    )

    if missing:
        raise ValueError(
            "Peak feature file missing columns: "
            f"{sorted(missing)}"
        )

    features = features[
        [
            "seasonal_position",
            FEATURE_COLUMN,
        ]
    ].copy()

    return features


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

    missing_feature_count = int(
        scoped[
            FEATURE_COLUMN
        ].isna().sum()
    )

    if missing_feature_count > 0:
        raise ValueError(
            "Missing peak feature values after merge: "
            f"{missing_feature_count}"
        )

    model_df = pd.DataFrame(
        {
            "unique_id": location,
            "ds": scoped["date"],
            "y": scoped[target].astype(float),
            FEATURE_COLUMN: scoped[
                FEATURE_COLUMN
            ].astype(float),
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
        FEATURE_COLUMN,
    }

    candidates = [
        column
        for column in cv.columns
        if column not in excluded
    ]

    if len(candidates) != 1:
        raise RuntimeError(
            "Could not uniquely identify "
            "forecast column. Candidates: "
            f"{candidates}"
        )

    return candidates[0]


def run_model(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
    experiment: dict,
    use_peak_feature: bool,
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

    if use_peak_feature:
        model_kwargs[
            "futr_exog_list"
        ] = [
            FEATURE_COLUMN
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

    if use_peak_feature:
        print(
            "Future exogenous feature: "
            f"{FEATURE_COLUMN}"
        )
    else:
        print(
            "Future exogenous feature: none"
        )

    model = LSTM(
        **model_kwargs
    )

    nf = NeuralForecast(
        models=[model],
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

    cv["horizon"] = (
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

    cv["forecast"] = (
        cv[
            "raw_forecast"
        ]
        .clip(
            lower=0
        )
    )

    cv["was_clipped"] = (
        cv[
            "raw_forecast"
        ]
        < 0
    )

    cv["model"] = model_name

    return cv


def calculate_metrics(
    forecasts: pd.DataFrame,
    forecast_column: str,
) -> pd.DataFrame:
    rows = []

    for horizon, group in (
        forecasts.groupby(
            "horizon"
        )
    ):
        error = (
            group[
                forecast_column
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
                    forecast_column
                ].idxmax()
            ]
        )

        peak_magnitude_error = (
            float(
                forecast_peak[
                    forecast_column
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
                    forecast_column
                ]
                < 0
            ).sum()
        )

        rows.append(
            {
                "horizon": int(
                    horizon
                ),
                "forecast_count": (
                    len(group)
                ),
                "mae": mae,
                "rmse": rmse,
                "bias": bias,
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
    baseline_metrics: pd.DataFrame,
    feature_metrics: pd.DataFrame,
) -> pd.DataFrame:
    baseline = (
        baseline_metrics
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

    feature = (
        feature_metrics
        .add_prefix(
            "feature_"
        )
        .rename(
            columns={
                "feature_horizon":
                    "horizon"
            }
        )
    )

    comparison = (
        baseline.merge(
            feature,
            on="horizon",
            validate="one_to_one",
        )
    )

    comparison[
        "mae_change"
    ] = (
        comparison[
            "feature_mae"
        ]
        - comparison[
            "baseline_mae"
        ]
    )

    comparison[
        "rmse_change"
    ] = (
        comparison[
            "feature_rmse"
        ]
        - comparison[
            "baseline_rmse"
        ]
    )

    comparison[
        "bias_change"
    ] = (
        comparison[
            "feature_bias"
        ]
        - comparison[
            "baseline_bias"
        ]
    )

    comparison[
        "peak_magnitude_abs_change"
    ] = (
        comparison[
            "feature_peak_magnitude_error"
        ].abs()
        - comparison[
            "baseline_peak_magnitude_error"
        ].abs()
    )

    comparison[
        "peak_timing_abs_change"
    ] = (
        comparison[
            "feature_peak_timing_error_weeks"
        ].abs()
        - comparison[
            "baseline_peak_timing_error_weeks"
        ].abs()
    )

    return comparison


def print_comparison(
    comparison: pd.DataFrame,
) -> None:
    print(
        "\nBaseline vs Peak-Feature LSTM"
    )
    print(
        "============================="
    )

    for row in (
        comparison.itertuples()
    ):
        print(
            f"\n{row.horizon}-week horizon"
        )

        print(
            "  MAE: "
            f"{row.baseline_mae:.4f} "
            "-> "
            f"{row.feature_mae:.4f}"
        )

        print(
            "  RMSE: "
            f"{row.baseline_rmse:.4f} "
            "-> "
            f"{row.feature_rmse:.4f}"
        )

        print(
            "  Bias: "
            f"{row.baseline_bias:+.4f} "
            "-> "
            f"{row.feature_bias:+.4f}"
        )

        print(
            "  Peak magnitude error: "
            f"{row.baseline_peak_magnitude_error:+.4f} "
            "-> "
            f"{row.feature_peak_magnitude_error:+.4f}"
        )

        print(
            "  Peak timing error: "
            f"{row.baseline_peak_timing_error_weeks:+.1f} "
            "-> "
            f"{row.feature_peak_timing_error_weeks:+.1f} "
            "weeks"
        )

        print(
            "  Negative forecasts: "
            f"{row.baseline_negative_forecast_count} "
            "-> "
            f"{row.feature_negative_forecast_count}"
        )


def save_results(
    baseline_forecasts: pd.DataFrame,
    feature_forecasts: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    feature_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline_forecasts.to_csv(
        RESULT_DIR
        / "peak_feature_baseline_forecasts.csv",
        index=False,
    )

    feature_forecasts.to_csv(
        RESULT_DIR
        / "peak_feature_model_forecasts.csv",
        index=False,
    )

    baseline_metrics.to_csv(
        RESULT_DIR
        / "peak_feature_baseline_metrics.csv",
        index=False,
    )

    feature_metrics.to_csv(
        RESULT_DIR
        / "peak_feature_model_metrics.csv",
        index=False,
    )

    comparison.to_csv(
        RESULT_DIR
        / "peak_feature_comparison.csv",
        index=False,
    )

    print(
        "\nSaved peak-feature experiment results."
    )


def plot_comparison(
    baseline_forecasts: pd.DataFrame,
    feature_forecasts: pd.DataFrame,
) -> None:
    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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

        feature = (
            feature_forecasts[
                feature_forecasts[
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
            label=(
                "Baseline LSTM"
            ),
        )

        ax.plot(
            feature[
                "ds"
            ],
            feature[
                "forecast"
            ],
            marker="o",
            label=(
                "LSTM + peak feature"
            ),
        )

        ax.set_title(
            "Peak Probability Feature Experiment: "
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
                "peak_feature_comparison_"
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
        load_peak_feature()
    )

    train_df, test_df = (
        load_forecasting_data(
            experiment=experiment,
            features=features,
        )
    )

    print(
        "\nControlled Peak-Feature Experiment"
    )
    print(
        "=================================="
    )

    print(
        "Baseline and feature models use "
        "the same refined configuration."
    )

    print(
        "Only difference: the feature model "
        "receives the shifted peak-probability "
        "curve as future exogenous input."
    )

    print(
        f"Input size: "
        f"{config['input_size']}"
    )

    baseline_forecasts = (
        run_model(
            model_name=(
                "baseline"
            ),
            train_df=train_df,
            test_df=test_df,
            config=config,
            experiment=experiment,
            use_peak_feature=False,
        )
    )

    feature_forecasts = (
        run_model(
            model_name=(
                "peak_feature"
            ),
            train_df=train_df,
            test_df=test_df,
            config=config,
            experiment=experiment,
            use_peak_feature=True,
        )
    )

    # Use raw forecasts for the main model comparison.
    # This isolates what the model itself learned,
    # rather than the effect of clipping negatives.
    baseline_metrics = (
        calculate_metrics(
            forecasts=baseline_forecasts,
            forecast_column=(
                "raw_forecast"
            ),
        )
    )

    feature_metrics = (
        calculate_metrics(
            forecasts=feature_forecasts,
            forecast_column=(
                "raw_forecast"
            ),
        )
    )

    comparison = (
        build_comparison(
            baseline_metrics=(
                baseline_metrics
            ),
            feature_metrics=(
                feature_metrics
            ),
        )
    )

    print_comparison(
        comparison
    )

    save_results(
        baseline_forecasts=(
            baseline_forecasts
        ),
        feature_forecasts=(
            feature_forecasts
        ),
        baseline_metrics=(
            baseline_metrics
        ),
        feature_metrics=(
            feature_metrics
        ),
        comparison=comparison,
    )

    plot_comparison(
        baseline_forecasts=(
            baseline_forecasts
        ),
        feature_forecasts=(
            feature_forecasts
        ),
    )


if __name__ == "__main__":
    main()