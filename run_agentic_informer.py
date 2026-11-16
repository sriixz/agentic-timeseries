import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import Informer

from configs.generated_informer_config import (
    informer_config,
)


# ============================================================
# Paths
# ============================================================

HOSPITAL_PATH = Path(
    "data/target-hospital-admissions.csv"
)

ED_PATH = Path(
    "data/target-ed-visits-prop.csv"
)

RESULT_DIR = Path(
    "forecast_results"
)

BEST_CONFIG_PATH = (
    RESULT_DIR
    / "agentic_informer_best_config.json"
)

TRIALS_PATH = (
    RESULT_DIR
    / "agentic_informer_trials.csv"
)

METRICS_PATH = (
    RESULT_DIR
    / "agentic_informer_metrics_by_horizon.csv"
)

FORECAST_PATH = (
    RESULT_DIR
    / "agentic_informer_test_forecasts.csv"
)

COMPARISON_PATH = (
    RESULT_DIR
    / "agentic_informer_manual_comparison.csv"
)


# ============================================================
# Experiment
# ============================================================

LOCATION = "US"

TARGET_COLUMN = "weekly_rate"

RAW_ED_COLUMN = "ed_visits_prop"

ED_LAG_COLUMN = "ed_visits_lag4"

ED_LAG_WEEKS = 4

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

VALIDATION_WEEKS = 12

NUM_TRIALS = 8

OPTUNA_SEED = 2


# ============================================================
# Manual Informer configuration
#
# This matches the hand-set Informer + ED setup used in
# the existing experiment.
# ============================================================

MANUAL_CONFIG = {
    "input_size": 12,
    "hidden_size": 64,
    "n_head": 4,
    "encoder_layers": 2,
    "decoder_layers": 1,
    "dropout": 0.05,
    "learning_rate": 1e-4,
    "max_steps": 1000,
    "batch_size": 32,
    "random_seed": 2,
    "scaler_type": "standard",
}


# ============================================================
# Data
# ============================================================

def load_hospital_data():

    df = pd.read_csv(
        HOSPITAL_PATH,
        dtype={
            "location": str,
        },
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
            f"No hospitalization data "
            f"for {LOCATION}."
        )

    return (
        df[
            [
                "date",
                TARGET_COLUMN,
            ]
        ]
        .dropna()
        .sort_values(
            "date"
        )
        .drop_duplicates(
            subset=[
                "date"
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )


def load_ed_data():

    df = pd.read_csv(
        ED_PATH,
        dtype={
            "location": str,
        },
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
            f"No ED data for "
            f"{LOCATION}."
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
                "value":
                    RAW_ED_COLUMN
            }
        )
        .sort_values(
            "date"
        )
        .drop_duplicates(
            subset=[
                "date"
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    if (
        df[
            RAW_ED_COLUMN
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "ED series contains "
            "missing values."
        )

    return df


def build_model_data():

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

    merged = (
        merged
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )

    # Leakage-safe ED feature:
    # target week t receives ED(t - 4).
    merged[
        ED_LAG_COLUMN
    ] = (
        merged[
            RAW_ED_COLUMN
        ]
        .shift(
            ED_LAG_WEEKS
        )
    )

    merged = merged.dropna(
        subset=[
            ED_LAG_COLUMN
        ]
    ).copy()

    merged = merged[
        merged["date"]
        <= TEST_END
    ].copy()

    model_df = pd.DataFrame(
        {
            "unique_id":
                LOCATION,

            "ds":
                merged[
                    "date"
                ],

            "y":
                merged[
                    TARGET_COLUMN
                ]
                .astype(float),

            ED_LAG_COLUMN:
                merged[
                    ED_LAG_COLUMN
                ]
                .astype(float),
        }
    )

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

    return (
        model_df,
        train_df,
        test_df,
    )


# ============================================================
# Model
# ============================================================

def build_informer(
    params,
    alias,
):

    kwargs = {
        "h":
            HORIZON,

        "input_size":
            params[
                "input_size"
            ],

        "loss":
            MAE(),

        "valid_loss":
            MAE(),

        "futr_exog_list":
            [
                ED_LAG_COLUMN
            ],

        "hidden_size":
            params[
                "hidden_size"
            ],

        "n_head":
            params[
                "n_head"
            ],

        "encoder_layers":
            params[
                "encoder_layers"
            ],

        "decoder_layers":
            params[
                "decoder_layers"
            ],

        "dropout":
            params[
                "dropout"
            ],

        "learning_rate":
            params[
                "learning_rate"
            ],

        "max_steps":
            params[
                "max_steps"
            ],

        "batch_size":
            params[
                "batch_size"
            ],

        "random_seed":
            params[
                "random_seed"
            ],

        "scaler_type":
            params[
                "scaler_type"
            ],

        "alias":
            alias,
    }

    # Agent search includes this parameter.
    # Manual configuration intentionally leaves it absent
    # so NeuralForecast uses the same behavior as the
    # existing manually configured Informer experiment.
    if (
        "conv_hidden_size"
        in params
    ):

        kwargs[
            "conv_hidden_size"
        ] = params[
            "conv_hidden_size"
        ]

    return Informer(
        **kwargs
    )


# ============================================================
# Validation
# ============================================================

def validation_metrics(
    train_df,
    params,
):

    alias = "InformerValidation"

    model = build_informer(
        params=params,
        alias=alias,
    )

    nf = NeuralForecast(
        models=[
            model
        ],
        freq="W-SAT",
    )

    cv = nf.cross_validation(
        df=train_df,

        n_windows=None,

        test_size=(
            VALIDATION_WEEKS
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

    horizon_rows = []

    for (
        horizon,
        group,
    ) in cv.groupby(
        "horizon"
    ):

        actual = (
            group[
                "y"
            ]
            .astype(float)
        )

        prediction = (
            group[
                alias
            ]
            .astype(float)
        )

        error = (
            prediction
            - actual
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    error ** 2
                )
            )
        )

        horizon_rows.append(
            {
                "horizon":
                    int(horizon),

                "rmse":
                    rmse,
            }
        )

    metrics = pd.DataFrame(
        horizon_rows
    )

    # Give every forecast horizon equal weight.
    mean_horizon_rmse = float(
        metrics[
            "rmse"
        ]
        .mean()
    )

    return (
        mean_horizon_rmse,
        metrics,
    )


# ============================================================
# Optuna
# ============================================================

def tune_agentic_informer(
    train_df,
):

    trial_records = []

    def objective(
        trial,
    ):

        params = informer_config(
            trial
        )

        print(
            "\n"
            + "-" * 60
        )

        print(
            f"Trial "
            f"{trial.number + 1}/"
            f"{NUM_TRIALS}"
        )

        print(
            json.dumps(
                params,
                indent=2,
            )
        )

        try:

            (
                score,
                horizon_metrics,
            ) = validation_metrics(
                train_df=train_df,
                params=params,
            )

            record = {
                "trial":
                    trial.number,

                "validation_mean_rmse":
                    score,

                "status":
                    "complete",

                "params_json":
                    json.dumps(
                        params,
                        sort_keys=True,
                    ),
            }

            for row in (
                horizon_metrics
                .itertuples(
                    index=False
                )
            ):

                record[
                    f"rmse_h{row.horizon}"
                ] = row.rmse

            trial_records.append(
                record
            )

            pd.DataFrame(
                trial_records
            ).to_csv(
                TRIALS_PATH,
                index=False,
            )

            print(
                "Validation mean "
                f"RMSE: {score:.4f}"
            )

            return score

        except Exception as exc:

            print(
                "Trial failed:",
                repr(exc),
            )

            trial_records.append(
                {
                    "trial":
                        trial.number,

                    "validation_mean_rmse":
                        np.nan,

                    "status":
                        "failed",

                    "error":
                        repr(exc),

                    "params_json":
                        json.dumps(
                            params,
                            sort_keys=True,
                        ),
                }
            )

            pd.DataFrame(
                trial_records
            ).to_csv(
                TRIALS_PATH,
                index=False,
            )

            return float(
                "inf"
            )

    sampler = (
        optuna.samplers.TPESampler(
            seed=OPTUNA_SEED
        )
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
    )

    study.optimize(
        objective,
        n_trials=NUM_TRIALS,
        show_progress_bar=False,
    )

    if not np.isfinite(
        study.best_value
    ):

        raise RuntimeError(
            "All Optuna trials failed."
        )

    best_params = (
        study.best_trial.params.copy()
    )

    # Fixed parameters do not appear in
    # study.best_trial.params.
    best_params[
        "scaler_type"
    ] = "standard"

    return (
        best_params,
        float(
            study.best_value
        ),
        study,
    )


# ============================================================
# Held-out evaluation
# ============================================================

def evaluate_config(
    model_df,
    test_df,
    params,
    alias,
):

    model = build_informer(
        params=params,
        alias=alias,
    )

    nf = NeuralForecast(
        models=[
            model
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

    rows = []

    for (
        horizon,
        group,
    ) in cv.groupby(
        "horizon"
    ):

        actual = (
            group[
                "y"
            ]
            .astype(float)
        )

        prediction = (
            group[
                alias
            ]
            .astype(float)
        )

        error = (
            prediction
            - actual
        )

        mae = float(
            np.mean(
                np.abs(
                    error
                )
            )
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    error ** 2
                )
            )
        )

        rows.append(
            {
                "horizon":
                    int(horizon),

                "mae":
                    mae,

                "rmse":
                    rmse,
            }
        )

    metrics = (
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

    return (
        cv,
        metrics,
    )


# ============================================================
# Comparison
# ============================================================

def build_comparison(
    manual_metrics,
    agent_metrics,
):

    comparison = (
        manual_metrics.merge(
            agent_metrics,
            on="horizon",
            suffixes=(
                "_manual",
                "_agent",
            ),
        )
    )

    comparison[
        "rmse_improvement_pct"
    ] = (
        (
            comparison[
                "rmse_manual"
            ]
            - comparison[
                "rmse_agent"
            ]
        )
        / comparison[
            "rmse_manual"
        ]
        * 100.0
    )

    comparison[
        "mae_improvement_pct"
    ] = (
        (
            comparison[
                "mae_manual"
            ]
            - comparison[
                "mae_agent"
            ]
        )
        / comparison[
            "mae_manual"
        ]
        * 100.0
    )

    return comparison


# ============================================================
# Main
# ============================================================

def main():

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    optuna.logging.set_verbosity(
        optuna.logging.WARNING
    )

    (
        model_df,
        train_df,
        test_df,
    ) = build_model_data()

    print(
        "\nAgentic Informer "
        "Optimization"
    )

    print(
        "=========================="
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
        f"Validation weeks: "
        f"{VALIDATION_WEEKS}"
    )

    print(
        f"Optuna trials: "
        f"{NUM_TRIALS}"
    )

    print(
        "Objective: mean validation "
        "RMSE across horizons 1-4"
    )

    print(
        "\nStarting agent-generated "
        "Informer search..."
    )

    (
        best_params,
        best_validation_rmse,
        study,
    ) = tune_agentic_informer(
        train_df=train_df
    )

    print(
        "\nBest agentic configuration"
    )

    print(
        "=========================="
    )

    print(
        json.dumps(
            best_params,
            indent=2,
        )
    )

    print(
        "\nBest validation mean RMSE: "
        f"{best_validation_rmse:.4f}"
    )

    best_output = {
        "validation_mean_rmse":
            best_validation_rmse,

        "config":
            best_params,

        "num_trials":
            NUM_TRIALS,

        "validation_weeks":
            VALIDATION_WEEKS,

        "objective":
            (
                "mean RMSE across "
                "forecast horizons 1-4"
            ),
    }

    BEST_CONFIG_PATH.write_text(
        json.dumps(
            best_output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nEvaluating manual Informer "
        "on held-out period..."
    )

    (
        manual_forecasts,
        manual_metrics,
    ) = evaluate_config(
        model_df=model_df,
        test_df=test_df,
        params=MANUAL_CONFIG,
        alias="ManualInformer",
    )

    print(
        "Evaluating agentic Informer "
        "on held-out period..."
    )

    (
        agent_forecasts,
        agent_metrics,
    ) = evaluate_config(
        model_df=model_df,
        test_df=test_df,
        params=best_params,
        alias="AgentInformer",
    )

    comparison = build_comparison(
        manual_metrics=manual_metrics,
        agent_metrics=agent_metrics,
    )

    print(
        "\nManual vs Agentic Informer"
    )

    print(
        "==========================="
    )

    print(
        comparison.to_string(
            index=False,
            formatters={
                "mae_manual":
                    "{:.4f}".format,

                "rmse_manual":
                    "{:.4f}".format,

                "mae_agent":
                    "{:.4f}".format,

                "rmse_agent":
                    "{:.4f}".format,

                "mae_improvement_pct":
                    "{:+.1f}%".format,

                "rmse_improvement_pct":
                    "{:+.1f}%".format,
            },
        )
    )

    manual_mean_rmse = float(
        manual_metrics[
            "rmse"
        ].mean()
    )

    agent_mean_rmse = float(
        agent_metrics[
            "rmse"
        ].mean()
    )

    overall_improvement = (
        (
            manual_mean_rmse
            - agent_mean_rmse
        )
        / manual_mean_rmse
        * 100.0
    )

    print(
        "\nAverage across horizons"
    )

    print(
        "======================="
    )

    print(
        "Manual mean RMSE: "
        f"{manual_mean_rmse:.4f}"
    )

    print(
        "Agent mean RMSE:  "
        f"{agent_mean_rmse:.4f}"
    )

    print(
        "Agent improvement: "
        f"{overall_improvement:+.1f}%"
    )

    comparison.to_csv(
        COMPARISON_PATH,
        index=False,
    )

    agent_metrics.to_csv(
        METRICS_PATH,
        index=False,
    )

    forecast_output = (
        manual_forecasts[
            [
                "unique_id",
                "ds",
                "cutoff",
                "y",
                "horizon",
                "ManualInformer",
            ]
        ]
        .merge(
            agent_forecasts[
                [
                    "unique_id",
                    "ds",
                    "cutoff",
                    "AgentInformer",
                ]
            ],
            on=[
                "unique_id",
                "ds",
                "cutoff",
            ],
            how="inner",
            validate="one_to_one",
        )
    )

    forecast_output.to_csv(
        FORECAST_PATH,
        index=False,
    )

    print(
        f"\nSaved: "
        f"{BEST_CONFIG_PATH}"
    )

    print(
        f"Saved: "
        f"{TRIALS_PATH}"
    )

    print(
        f"Saved: "
        f"{METRICS_PATH}"
    )

    print(
        f"Saved: "
        f"{FORECAST_PATH}"
    )

    print(
        f"Saved: "
        f"{COMPARISON_PATH}"
    )


if __name__ == "__main__":
    main()