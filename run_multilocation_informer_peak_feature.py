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

US_PROBABILITY_PATH = Path(
    "analysis_results/us_peak_probability_by_epiweek.csv"
)

STATE_PROBABILITY_PATH = Path(
    "analysis_results/state_peak_probability_by_epiweek.csv"
)

RESULT_DIR = Path("forecast_results")
PLOT_DIR = Path("plots")


FEATURE_COLUMN = "peak_probability_next_4w"

WINDOW_WEEKS = 4

INPUT_SIZE = 12
MAX_STEPS = 1000
RANDOM_SEED = 2


LOCATIONS = {
    "US": {
        "name": "United States",
        "location": "US",
    },
    "CA": {
        "name": "California",
        "location": "06",
    },
    "FL": {
        "name": "Florida",
        "location": "12",
    },
    "NY": {
        "name": "New York",
        "location": "36",
    },
    "TX": {
        "name": "Texas",
        "location": "48",
    },
}


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


def build_next4w_feature(
    probability_df: pd.DataFrame,
) -> pd.DataFrame:
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


def load_location_feature(
    state_abbr: str,
) -> pd.DataFrame:
    if state_abbr == "US":
        if not US_PROBABILITY_PATH.exists():
            raise FileNotFoundError(
                "National peak probability file "
                f"not found: {US_PROBABILITY_PATH}"
            )

        probability_df = pd.read_csv(
            US_PROBABILITY_PATH
        )

    else:
        if not STATE_PROBABILITY_PATH.exists():
            raise FileNotFoundError(
                "State peak probability file "
                f"not found: {STATE_PROBABILITY_PATH}"
            )

        all_states = pd.read_csv(
            STATE_PROBABILITY_PATH,
            dtype={
                "location": str,
            },
        )

        probability_df = all_states[
            all_states[
                "state_abbr"
            ]
            == state_abbr
        ].copy()

        if probability_df.empty:
            raise ValueError(
                "No peak probability curve "
                f"found for {state_abbr}."
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
            f"{state_abbr} probability data "
            f"missing columns: {sorted(missing)}"
        )

    return build_next4w_feature(
        probability_df
    )


def load_location_data(
    location_code: str,
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
        == location_code
    ].copy()

    if scoped.empty:
        raise ValueError(
            "No data found for location "
            f"{location_code}."
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
            f"{location_code}: "
            f"{missing_feature_count} "
            "feature values are missing."
        )

    model_df = pd.DataFrame(
        {
            "unique_id":
                location_code,

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
            f"{location_code}: "
            "training data is empty."
        )

    if test_df.empty:
        raise ValueError(
            f"{location_code}: "
            "test data is empty."
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
        for column
        in cv.columns
        if column
        not in excluded
    ]

    if len(candidates) != 1:
        raise RuntimeError(
            "Could not uniquely identify "
            "forecast column. Candidates: "
            f"{candidates}"
        )

    return candidates[0]


def run_informer(
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
        cv["horizon"]
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

    return cv


def calculate_metrics(
    forecasts: pd.DataFrame,
) -> dict:
    error = (
        forecasts[
            "forecast"
        ]
        - forecasts[
            "y"
        ]
    )

    actual_peak = (
        forecasts.loc[
            forecasts[
                "y"
            ].idxmax()
        ]
    )

    forecast_peak = (
        forecasts.loc[
            forecasts[
                "forecast"
            ].idxmax()
        ]
    )

    return {
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
                    forecasts[
                        "raw_forecast"
                    ]
                    < 0
                ).sum()
            ),
    }


def plot_four_week_forecast(
    state_abbr: str,
    state_name: str,
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
        f"{state_name}: "
        "Informer 4-Week-Ahead "
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
        / (
            "multilocation_informer_"
            f"{state_abbr.lower()}_4w.png"
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

    print(
        "\nMulti-Location Informer "
        "Peak-Feature Experiment"
    )

    print(
        "=========================================="
    )

    print(
        f"Input size: {INPUT_SIZE}"
    )

    print(
        "Locations: "
        + ", ".join(
            LOCATIONS.keys()
        )
    )

    summary_rows = []

    all_forecasts = []

    for state_abbr, info in (
        LOCATIONS.items()
    ):
        state_name = info[
            "name"
        ]

        location_code = info[
            "location"
        ]

        print(
            f"\n{state_name} ({state_abbr})"
        )

        print(
            "-" * (
                len(state_name)
                + len(state_abbr)
                + 3
            )
        )

        feature_df = (
            load_location_feature(
                state_abbr
            )
        )

        train_df, test_df = (
            load_location_data(
                location_code=(
                    location_code
                ),
                experiment=(
                    experiment
                ),
                feature_df=(
                    feature_df
                ),
            )
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
            "Training baseline..."
        )

        baseline = (
            run_informer(
                train_df=train_df,
                test_df=test_df,
                experiment=experiment,
                use_peak_feature=False,
            )
        )

        print(
            "Training peak-feature model..."
        )

        feature = (
            run_informer(
                train_df=train_df,
                test_df=test_df,
                experiment=experiment,
                use_peak_feature=True,
            )
        )

        baseline[
            "variant"
        ] = "baseline"

        feature[
            "variant"
        ] = "next4w_feature"

        baseline[
            "state_abbr"
        ] = state_abbr

        feature[
            "state_abbr"
        ] = state_abbr

        all_forecasts.extend(
            [
                baseline,
                feature,
            ]
        )

        baseline_4w = (
            baseline[
                baseline[
                    "horizon"
                ]
                == 4
            ]
            .copy()
        )

        feature_4w = (
            feature[
                feature[
                    "horizon"
                ]
                == 4
            ]
            .copy()
        )

        baseline_metrics = (
            calculate_metrics(
                baseline_4w
            )
        )

        feature_metrics = (
            calculate_metrics(
                feature_4w
            )
        )

        print(
            "4-week MAE: "
            f"{baseline_metrics['mae']:.4f} "
            "-> "
            f"{feature_metrics['mae']:.4f}"
        )

        print(
            "4-week RMSE: "
            f"{baseline_metrics['rmse']:.4f} "
            "-> "
            f"{feature_metrics['rmse']:.4f}"
        )

        print(
            "Peak magnitude error: "
            f"{baseline_metrics['peak_magnitude_error']:+.4f} "
            "-> "
            f"{feature_metrics['peak_magnitude_error']:+.4f}"
        )

        print(
            "Peak timing error: "
            f"{baseline_metrics['peak_timing_error_weeks']:+.1f} "
            "-> "
            f"{feature_metrics['peak_timing_error_weeks']:+.1f} "
            "weeks"
        )

        summary_rows.append(
            {
                "state_abbr":
                    state_abbr,

                "location_name":
                    state_name,

                "baseline_mae":
                    baseline_metrics[
                        "mae"
                    ],

                "feature_mae":
                    feature_metrics[
                        "mae"
                    ],

                "baseline_rmse":
                    baseline_metrics[
                        "rmse"
                    ],

                "feature_rmse":
                    feature_metrics[
                        "rmse"
                    ],

                "baseline_bias":
                    baseline_metrics[
                        "bias"
                    ],

                "feature_bias":
                    feature_metrics[
                        "bias"
                    ],

                "baseline_peak_magnitude_error":
                    baseline_metrics[
                        "peak_magnitude_error"
                    ],

                "feature_peak_magnitude_error":
                    feature_metrics[
                        "peak_magnitude_error"
                    ],

                "baseline_peak_timing_error_weeks":
                    baseline_metrics[
                        "peak_timing_error_weeks"
                    ],

                "feature_peak_timing_error_weeks":
                    feature_metrics[
                        "peak_timing_error_weeks"
                    ],

                "baseline_negative_raw_forecasts":
                    baseline_metrics[
                        "negative_raw_forecasts"
                    ],

                "feature_negative_raw_forecasts":
                    feature_metrics[
                        "negative_raw_forecasts"
                    ],
            }
        )

        plot_four_week_forecast(
            state_abbr=state_abbr,
            state_name=state_name,
            baseline=baseline,
            feature=feature,
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        RESULT_DIR
        / (
            "multilocation_informer_"
            "4w_summary.csv"
        )
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    forecasts = pd.concat(
        all_forecasts,
        ignore_index=True,
    )

    forecasts_path = (
        RESULT_DIR
        / (
            "multilocation_informer_"
            "forecasts.csv"
        )
    )

    forecasts.to_csv(
        forecasts_path,
        index=False,
    )

    print(
        f"\nSaved: {summary_path}"
    )

    print(
        f"Saved: {forecasts_path}"
    )

    print(
        "\n4-Week Informer Summary"
    )

    print(
        "======================="
    )

    for row in (
        summary.itertuples()
    ):
        print(
            f"{row.state_abbr}: "
            f"RMSE "
            f"{row.baseline_rmse:.3f} "
            "-> "
            f"{row.feature_rmse:.3f}, "
            f"peak error "
            f"{row.baseline_peak_magnitude_error:+.3f} "
            "-> "
            f"{row.feature_peak_magnitude_error:+.3f}, "
            f"timing "
            f"{row.baseline_peak_timing_error_weeks:+.1f} "
            "-> "
            f"{row.feature_peak_timing_error_weeks:+.1f}"
        )


if __name__ == "__main__":
    main()