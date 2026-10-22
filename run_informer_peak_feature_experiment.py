import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from epiweeks import Week
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import Informer


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
        - set(probability_df.columns)
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
        df["location"]
        == "US"
    ].copy()

    if scoped.empty:
        raise ValueError(
            "No U.S. national data found."
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
        model_df[
            "ds"
        ]
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
        for column
        in cv.columns
        if column
        not in excluded
    ]

    if len(candidates) != 1:
        raise RuntimeError(
            "Could not uniquely identify forecast "
            f"column. Candidates: {candidates}"
        )

    return candidates[0]


def run_informer(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    experiment: dict,
    use_peak_feature: bool,
) -> pd.DataFrame:
    horizon = int(
        experiment[
            "horizon"
        ]
    )

    model_kwargs = {
        "h":
            horizon,
        "input_size":
            INPUT_SIZE,
        "loss":
            MAE(),
        "max_steps":
            MAX_STEPS,
        "random_seed":
            RANDOM_SEED,
        "scaler_type":
            "standard",

        # Keep this first experiment small and
        # deterministic rather than tuning a large
        # transformer search space.
        "hidden_size":
            64,
        "n_head":
            4,
        "encoder_layers":
            2,
        "decoder_layers":
            1,
        "dropout":
            0.05,
        "learning_rate":
            1e-4,
        "batch_size":
            32,
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

    model = Informer(
        **model_kwargs
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
        cv[
            "horizon"
        ]
        <= horizon
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
    rows = []

    for horizon, group in (
        forecasts.groupby(
            "horizon"
        )
    ):
        error = (
            group[
                "forecast"
            ]
            - group["y"]
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
                    "forecast"
                ].idxmax()
            ]
        )

        rows.append(
            {
                "horizon":
                    int(horizon),

                "mae":
                    float(
                        error.abs().mean()
                    ),

                "rmse":
                    float(
                        error.pow(2)
                        .mean()
                        ** 0.5
                    ),

                "bias":
                    float(
                        error.mean()
                    ),

                "peak_magnitude_error":
                    float(
                        forecast_peak[
                            "forecast"
                        ]
                        - actual_peak[
                            "y"
                        ]
                    ),

                "peak_timing_error_weeks":
                    float(
                        (
                            forecast_peak[
                                "ds"
                            ]
                            - actual_peak[
                                "ds"
                            ]
                        ).days
                        / 7.0
                    ),

                "negative_raw_forecasts":
                    int(
                        (
                            group[
                                "raw_forecast"
                            ]
                            < 0
                        ).sum()
                    ),
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


def print_comparison(
    baseline_metrics: pd.DataFrame,
    feature_metrics: pd.DataFrame,
) -> None:
    print(
        "\nInformer Peak Feature Comparison"
    )

    print(
        "================================"
    )

    merged = baseline_metrics.merge(
        feature_metrics,
        on="horizon",
        suffixes=(
            "_baseline",
            "_feature",
        ),
    )

    for row in (
        merged.itertuples()
    ):
        print(
            f"\n{row.horizon}-week horizon"
        )

        print(
            "  MAE: "
            f"{row.mae_baseline:.4f} "
            "-> "
            f"{row.mae_feature:.4f}"
        )

        print(
            "  RMSE: "
            f"{row.rmse_baseline:.4f} "
            "-> "
            f"{row.rmse_feature:.4f}"
        )

        print(
            "  Bias: "
            f"{row.bias_baseline:+.4f} "
            "-> "
            f"{row.bias_feature:+.4f}"
        )

        print(
            "  Peak magnitude error: "
            f"{row.peak_magnitude_error_baseline:+.4f} "
            "-> "
            f"{row.peak_magnitude_error_feature:+.4f}"
        )

        print(
            "  Peak timing error: "
            f"{row.peak_timing_error_weeks_baseline:+.1f} "
            "-> "
            f"{row.peak_timing_error_weeks_feature:+.1f} "
            "weeks"
        )

        print(
            "  Negative raw forecasts: "
            f"{row.negative_raw_forecasts_baseline} "
            "-> "
            f"{row.negative_raw_forecasts_feature}"
        )


def plot_four_week_comparison(
    baseline: pd.DataFrame,
    feature: pd.DataFrame,
) -> None:
    baseline_4w = (
        baseline[
            baseline[
                "horizon"
            ]
            == 4
        ]
        .sort_values(
            "ds"
        )
    )

    feature_4w = (
        feature[
            feature[
                "horizon"
            ]
            == 4
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
        baseline_4w[
            "ds"
        ],
        baseline_4w[
            "y"
        ],
        marker="o",
        linewidth=2,
        label="Actual",
    )

    ax.plot(
        baseline_4w[
            "ds"
        ],
        baseline_4w[
            "forecast"
        ],
        marker="o",
        label="Informer baseline",
    )

    ax.plot(
        feature_4w[
            "ds"
        ],
        feature_4w[
            "forecast"
        ],
        marker="o",
        label=(
            "Informer + next-4-week "
            "peak probability"
        ),
    )

    dates = baseline_4w[
        "ds"
    ].tolist()

    tick_dates = (
        dates[::4]
    )

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
        "U.S. Informer: 4-Week-Ahead "
        "Influenza Hospitalization Forecast"
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
        / "informer_peak_feature_us_4w.png"
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


def save_results(
    baseline: pd.DataFrame,
    feature: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    feature_metrics: pd.DataFrame,
) -> None:
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline.to_csv(
        RESULT_DIR
        / "informer_baseline_forecasts.csv",
        index=False,
    )

    feature.to_csv(
        RESULT_DIR
        / "informer_peak_feature_forecasts.csv",
        index=False,
    )

    baseline_metrics.to_csv(
        RESULT_DIR
        / "informer_baseline_metrics.csv",
        index=False,
    )

    feature_metrics.to_csv(
        RESULT_DIR
        / "informer_peak_feature_metrics.csv",
        index=False,
    )

    comparison = baseline_metrics.merge(
        feature_metrics,
        on="horizon",
        suffixes=(
            "_baseline",
            "_feature",
        ),
    )

    comparison.to_csv(
        RESULT_DIR
        / "informer_peak_feature_comparison.csv",
        index=False,
    )

    print(
        "\nSaved Informer experiment results."
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

    features = (
        build_next4w_feature()
    )

    train_df, test_df = (
        load_forecasting_data(
            experiment=experiment,
            features=features,
        )
    )

    print(
        "\nControlled Informer Peak-Feature Experiment"
    )

    print(
        "=========================================="
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
        f"Input size: {INPUT_SIZE}"
    )

    print(
        f"Forecast horizon: "
        f"{experiment['horizon']}"
    )

    baseline = (
        run_informer(
            model_name=(
                "informer_baseline"
            ),
            train_df=train_df,
            test_df=test_df,
            experiment=experiment,
            use_peak_feature=False,
        )
    )

    feature = (
        run_informer(
            model_name=(
                "informer_peak_feature"
            ),
            train_df=train_df,
            test_df=test_df,
            experiment=experiment,
            use_peak_feature=True,
        )
    )

    baseline_metrics = (
        calculate_metrics(
            baseline
        )
    )

    feature_metrics = (
        calculate_metrics(
            feature
        )
    )

    print_comparison(
        baseline_metrics=(
            baseline_metrics
        ),
        feature_metrics=(
            feature_metrics
        ),
    )

    save_results(
        baseline=baseline,
        feature=feature,
        baseline_metrics=(
            baseline_metrics
        ),
        feature_metrics=(
            feature_metrics
        ),
    )

    plot_four_week_comparison(
        baseline=baseline,
        feature=feature,
    )


if __name__ == "__main__":
    main()