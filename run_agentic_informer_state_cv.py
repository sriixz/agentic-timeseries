import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import Informer


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
    "forecast_results/state_cv"
)

FOLDS_PATH = (
    RESULT_DIR
    / "agentic_informer_state_cv_folds.csv"
)

TRIALS_PATH = (
    RESULT_DIR
    / "agentic_informer_state_cv_trials.csv"
)

TRIAL_STATE_METRICS_PATH = (
    RESULT_DIR
    / "agentic_informer_state_cv_trial_state_metrics.csv"
)

BEST_CONFIG_PATH = (
    RESULT_DIR
    / "agentic_informer_state_cv_best_config.json"
)

BEST_STATE_METRICS_PATH = (
    RESULT_DIR
    / "agentic_informer_state_cv_best_state_metrics.csv"
)

MANUAL_STATE_METRICS_PATH = (
    RESULT_DIR
    / "agentic_informer_state_cv_manual_state_metrics.csv"
)

COMPARISON_PATH = (
    RESULT_DIR
    / "agentic_informer_state_cv_manual_comparison.csv"
)


# ============================================================
# Experiment
# ============================================================

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

NUM_FOLDS = 5

NUM_TRIALS = 8

STATE_SPLIT_SEED = 42

OPTUNA_SEED = 2

MODEL_SEED = 4


# ============================================================
# Manual Informer configuration
# ============================================================

MANUAL_CONFIG = {
    "input_size": 12,
    "hidden_size": 64,
    "conv_hidden_size": 64,
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
# State-CV search space
#
# Same general architecture space used in the first agentic
# Informer search, except random seed is FIXED rather than
# tuned. Seed is not treated as a hyperparameter.
# ============================================================

def state_cv_config(trial):

    return {
        "input_size":
            trial.suggest_categorical(
                "input_size",
                [8, 12, 16, 24],
            ),

        "hidden_size":
            trial.suggest_categorical(
                "hidden_size",
                [32, 64, 128],
            ),

        "conv_hidden_size":
            trial.suggest_categorical(
                "conv_hidden_size",
                [32, 64, 128],
            ),

        "n_head":
            trial.suggest_categorical(
                "n_head",
                [2, 4, 8],
            ),

        "encoder_layers":
            trial.suggest_int(
                "encoder_layers",
                1,
                3,
            ),

        "decoder_layers":
            trial.suggest_int(
                "decoder_layers",
                1,
                2,
            ),

        "dropout":
            trial.suggest_float(
                "dropout",
                0.0,
                0.2,
            ),

        "learning_rate":
            trial.suggest_float(
                "learning_rate",
                5e-5,
                5e-3,
                log=True,
            ),

        "max_steps":
            trial.suggest_categorical(
                "max_steps",
                [500, 1000],
            ),

        "batch_size":
            trial.suggest_categorical(
                "batch_size",
                [16, 32],
            ),

        "random_seed":
            MODEL_SEED,

        "scaler_type":
            "standard",
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

    df["location_name"] = (
        df["location_name"]
        .astype(str)
        .str.strip()
    )

    # State-level experiment:
    # remove national US aggregate.
    df = df[
        df["location"] != "US"
    ].copy()

    return (
        df[
            [
                "date",
                "location",
                "location_name",
                TARGET_COLUMN,
            ]
        ]
        .dropna(
            subset=[
                "date",
                "location",
                "location_name",
                TARGET_COLUMN,
            ]
        )
        .sort_values(
            [
                "location",
                "date",
            ]
        )
        .drop_duplicates(
            subset=[
                "location",
                "date",
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
        df["location"] != "US"
    ].copy()

    df = (
        df[
            [
                "date",
                "location",
                "value",
            ]
        ]
        .rename(
            columns={
                "value":
                    RAW_ED_COLUMN
            }
        )
        .dropna(
            subset=[
                "date",
                "location",
                RAW_ED_COLUMN,
            ]
        )
        .sort_values(
            [
                "location",
                "date",
            ]
        )
        .drop_duplicates(
            subset=[
                "location",
                "date",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    return df


def build_all_state_data():

    hospital_df = (
        load_hospital_data()
    )

    ed_df = (
        load_ed_data()
    )

    merged = hospital_df.merge(
        ed_df,
        on=[
            "location",
            "date",
        ],
        how="inner",
        validate="one_to_one",
    )

    merged = (
        merged
        .sort_values(
            [
                "location",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # Leakage-safe ED feature.
    #
    # Within each state:
    #
    # target week t receives ED(t - 4).
    merged[
        ED_LAG_COLUMN
    ] = (
        merged.groupby(
            "location"
        )[
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
                merged["location"],

            "location_name":
                merged[
                    "location_name"
                ],

            "ds":
                merged["date"],

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

    model_df = (
        model_df
        .sort_values(
            [
                "unique_id",
                "ds",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return model_df


def find_eligible_states(
    model_df,
):

    records = []

    for (
        state_code,
        group,
    ) in model_df.groupby(
        "unique_id"
    ):

        group = group.sort_values(
            "ds"
        )

        train = group[
            group["ds"]
            <= TRAIN_END
        ]

        test = group[
            (
                group["ds"]
                >= TEST_START
            )
            & (
                group["ds"]
                <= TEST_END
            )
        ]

        state_name = (
            group[
                "location_name"
            ]
            .iloc[0]
        )

        # Need enough pre-test data for the largest candidate
        # input window plus adequate test-period observations.
        eligible = (
            len(train) >= 40
            and len(test) >= HORIZON
        )

        records.append(
            {
                "state_code":
                    state_code,

                "state_name":
                    state_name,

                "train_observations":
                    len(train),

                "test_observations":
                    len(test),

                "eligible":
                    eligible,
            }
        )

    eligibility = pd.DataFrame(
        records
    )

    eligible = (
        eligibility[
            eligibility[
                "eligible"
            ]
        ]
        .sort_values(
            "state_code"
        )
        .reset_index(
            drop=True
        )
    )

    if eligible.empty:

        raise ValueError(
            "No eligible states found."
        )

    return (
        eligibility,
        eligible,
    )


# ============================================================
# State folds
# ============================================================

def build_state_folds(
    eligible_states,
):

    state_codes = (
        eligible_states[
            "state_code"
        ]
        .astype(str)
        .tolist()
    )

    rng = np.random.default_rng(
        STATE_SPLIT_SEED
    )

    shuffled = np.array(
        state_codes,
        dtype=object,
    )

    rng.shuffle(
        shuffled
    )

    # Five approximately equal folds.
    #
    # With 51 jurisdictions this produces folds of
    # approximately 10-11 states.
    split_arrays = np.array_split(
        shuffled,
        NUM_FOLDS,
    )

    rows = []

    state_name_lookup = dict(
        zip(
            eligible_states[
                "state_code"
            ],
            eligible_states[
                "state_name"
            ],
        )
    )

    for (
        fold_index,
        fold_states,
    ) in enumerate(
        split_arrays,
        start=1,
    ):

        for state_code in fold_states:

            rows.append(
                {
                    "fold":
                        fold_index,

                    "state_code":
                        str(
                            state_code
                        ),

                    "state_name":
                        state_name_lookup[
                            str(
                                state_code
                            )
                        ],
                }
            )

    folds = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "fold",
                "state_code",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return folds


# ============================================================
# Model
# ============================================================

def build_informer(
    params,
    alias,
):

    return Informer(
        h=HORIZON,

        input_size=
            params[
                "input_size"
            ],

        loss=MAE(),

        valid_loss=MAE(),

        futr_exog_list=[
            ED_LAG_COLUMN
        ],

        hidden_size=
            params[
                "hidden_size"
            ],

        conv_hidden_size=
            params[
                "conv_hidden_size"
            ],

        n_head=
            params[
                "n_head"
            ],

        encoder_layers=
            params[
                "encoder_layers"
            ],

        decoder_layers=
            params[
                "decoder_layers"
            ],

        dropout=
            params[
                "dropout"
            ],

        learning_rate=
            params[
                "learning_rate"
            ],

        max_steps=
            params[
                "max_steps"
            ],

        batch_size=
            params[
                "batch_size"
            ],

        random_seed=
            params[
                "random_seed"
            ],

        scaler_type=
            params[
                "scaler_type"
            ],

        alias=alias,
    )


# ============================================================
# Single-state active-season evaluation
# ============================================================

def evaluate_state(
    model_df,
    state_code,
    params,
    alias,
):

    state_df = (
        model_df[
            model_df[
                "unique_id"
            ]
            == state_code
        ]
        .copy()
        .sort_values(
            "ds"
        )
        .reset_index(
            drop=True
        )
    )

    if state_df.empty:

        raise ValueError(
            f"No data for "
            f"{state_code}."
        )

    state_name = (
        state_df[
            "location_name"
        ]
        .iloc[0]
    )

    nf_df = state_df[
        [
            "unique_id",
            "ds",
            "y",
            ED_LAG_COLUMN,
        ]
    ].copy()

    test_df = nf_df[
        (
            nf_df["ds"]
            >= TEST_START
        )
        & (
            nf_df["ds"]
            <= TEST_END
        )
    ].copy()

    if test_df.empty:

        raise ValueError(
            f"No test data for "
            f"{state_name}."
        )

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

    # The model is fit using observations before the first
    # active-season cutoff and then kept fixed across the
    # rolling 2025-26 forecasts.
    cv = nf.cross_validation(
        df=nf_df,

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

    if cv.empty:

        raise ValueError(
            f"No active-season forecasts "
            f"for {state_name}."
        )

    horizon_rows = []

    for (
        horizon,
        group,
    ) in cv.groupby(
        "horizon"
    ):

        actual = (
            group["y"]
            .astype(float)
        )

        prediction = (
            group[alias]
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

        horizon_rows.append(
            {
                "state_code":
                    state_code,

                "state_name":
                    state_name,

                "horizon":
                    int(
                        horizon
                    ),

                "mae":
                    mae,

                "rmse":
                    rmse,
            }
        )

    metrics = pd.DataFrame(
        horizon_rows
    )

    if len(metrics) != HORIZON:

        raise ValueError(
            f"{state_name} produced "
            f"{len(metrics)} horizons "
            f"instead of {HORIZON}."
        )

    return metrics


# ============================================================
# Evaluate one configuration across state folds
# ============================================================

def evaluate_config_across_folds(
    model_df,
    folds,
    params,
    alias_prefix,
    trial_number=None,
    max_states=None,
):

    rows = []

    fold_scores = []

    for fold in sorted(
        folds["fold"].unique()
    ):

        fold_states = (
            folds[
                folds["fold"]
                == fold
            ][
                "state_code"
            ]
            .astype(str)
            .tolist()
        )

        if max_states is not None:

            fold_states = (
                fold_states[
                    :max_states
                ]
            )

        print(
            f"\nFold {fold}/{NUM_FOLDS}: "
            f"{len(fold_states)} states"
        )

        current_fold_rows = []

        for (
            state_index,
            state_code,
        ) in enumerate(
            fold_states,
            start=1,
        ):

            state_name = (
                folds[
                    folds[
                        "state_code"
                    ]
                    == state_code
                ][
                    "state_name"
                ]
                .iloc[0]
            )

            print(
                f"  [{state_index}/"
                f"{len(fold_states)}] "
                f"{state_name}"
            )

            alias = (
                f"{alias_prefix}_"
                f"F{fold}_"
                f"{state_code}"
            )

            try:

                metrics = (
                    evaluate_state(
                        model_df=
                            model_df,

                        state_code=
                            state_code,

                        params=
                            params,

                        alias=
                            alias,
                    )
                )

                metrics[
                    "fold"
                ] = fold

                metrics[
                    "status"
                ] = "complete"

                metrics[
                    "trial"
                ] = trial_number

                current_fold_rows.append(
                    metrics
                )

                rows.append(
                    metrics
                )

            except Exception as exc:

                print(
                    "    FAILED:",
                    repr(exc),
                )

                failure = pd.DataFrame(
                    [
                        {
                            "state_code":
                                state_code,

                            "state_name":
                                state_name,

                            "horizon":
                                np.nan,

                            "mae":
                                np.nan,

                            "rmse":
                                np.nan,

                            "fold":
                                fold,

                            "status":
                                "failed",

                            "trial":
                                trial_number,

                            "error":
                                repr(exc),
                        }
                    ]
                )

                rows.append(
                    failure
                )

        complete_fold = [
            frame
            for frame
            in current_fold_rows
            if not frame.empty
        ]

        if not complete_fold:

            raise RuntimeError(
                f"No successful states "
                f"in fold {fold}."
            )

        fold_df = pd.concat(
            complete_fold,
            ignore_index=True,
        )

        fold_score = float(
            fold_df[
                "rmse"
            ].mean()
        )

        fold_scores.append(
            {
                "fold":
                    fold,

                "mean_rmse":
                    fold_score,

                "successful_states":
                    int(
                        fold_df[
                            "state_code"
                        ]
                        .nunique()
                    ),
            }
        )

        print(
            f"Fold {fold} mean RMSE: "
            f"{fold_score:.4f}"
        )

    all_metrics = pd.concat(
        rows,
        ignore_index=True,
    )

    complete = all_metrics[
        all_metrics[
            "status"
        ]
        == "complete"
    ].copy()

    if complete.empty:

        raise RuntimeError(
            "Configuration produced "
            "no successful state results."
        )

    fold_metrics = pd.DataFrame(
        fold_scores
    )

    # Equal weight to each state-horizon observation.
    objective = float(
        complete[
            "rmse"
        ].mean()
    )

    return (
        objective,
        all_metrics,
        fold_metrics,
    )


# ============================================================
# Optuna
# ============================================================

def tune_agentic_informer(
    model_df,
    folds,
    num_trials,
    max_states=None,
):

    trial_records = []

    all_state_records = []

    effective_params = {}

    def objective(
        trial,
    ):

        params = state_cv_config(
            trial
        )

        effective_params[
            trial.number
        ] = params.copy()

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"TRIAL "
            f"{trial.number + 1}/"
            f"{num_trials}"
        )

        print(
            "=" * 70
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
                state_metrics,
                fold_metrics,
            ) = (
                evaluate_config_across_folds(
                    model_df=
                        model_df,

                    folds=
                        folds,

                    params=
                        params,

                    alias_prefix=
                        f"AgentTrial"
                        f"{trial.number}",

                    trial_number=
                        trial.number,

                    max_states=
                        max_states,
                )
            )

            all_state_records.append(
                state_metrics
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
                fold_metrics
                .itertuples(
                    index=False
                )
            ):

                record[
                    f"fold_"
                    f"{row.fold}_rmse"
                ] = (
                    row.mean_rmse
                )

            trial_records.append(
                record
            )

            pd.DataFrame(
                trial_records
            ).to_csv(
                TRIALS_PATH,
                index=False,
            )

            pd.concat(
                all_state_records,
                ignore_index=True,
            ).to_csv(
                TRIAL_STATE_METRICS_PATH,
                index=False,
            )

            print(
                "\nTrial state-CV "
                f"mean RMSE: "
                f"{score:.4f}"
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
        n_trials=num_trials,
        show_progress_bar=False,
    )

    if not np.isfinite(
        study.best_value
    ):

        raise RuntimeError(
            "All Optuna trials failed."
        )

    best_trial_number = (
        study.best_trial.number
    )

    best_params = (
        effective_params[
            best_trial_number
        ].copy()
    )

    return (
        best_params,
        float(
            study.best_value
        ),
        best_trial_number,
        study,
    )


# ============================================================
# Manual comparison
# ============================================================

def summarize_state_metrics(
    metrics,
    prefix,
):

    complete = metrics[
        metrics[
            "status"
        ]
        == "complete"
    ].copy()

    state_summary = (
        complete
        .groupby(
            [
                "state_code",
                "state_name",
            ],
            as_index=False,
        )
        .agg(
            mean_rmse=(
                "rmse",
                "mean",
            ),

            mean_mae=(
                "mae",
                "mean",
            ),
        )
        .rename(
            columns={
                "mean_rmse":
                    f"mean_rmse_"
                    f"{prefix}",

                "mean_mae":
                    f"mean_mae_"
                    f"{prefix}",
            }
        )
    )

    return state_summary


def build_comparison(
    manual_metrics,
    agent_metrics,
):

    manual = (
        summarize_state_metrics(
            metrics=
                manual_metrics,

            prefix=
                "manual",
        )
    )

    agent = (
        summarize_state_metrics(
            metrics=
                agent_metrics,

            prefix=
                "agent",
        )
    )

    comparison = manual.merge(
        agent,
        on=[
            "state_code",
            "state_name",
        ],
        how="inner",
        validate="one_to_one",
    )

    comparison[
        "rmse_improvement_pct"
    ] = (
        (
            comparison[
                "mean_rmse_manual"
            ]
            - comparison[
                "mean_rmse_agent"
            ]
        )
        / comparison[
            "mean_rmse_manual"
        ]
        * 100.0
    )

    comparison[
        "agent_better"
    ] = (
        comparison[
            "mean_rmse_agent"
        ]
        < comparison[
            "mean_rmse_manual"
        ]
    )

    return (
        comparison
        .sort_values(
            "rmse_improvement_pct",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Main
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "State-based cross-validation "
            "for agentic Informer tuning."
        )
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=NUM_TRIALS,
        help=(
            "Number of Optuna trials. "
            "Default: 8."
        ),
    )

    parser.add_argument(
        "--max-states-per-fold",
        type=int,
        default=None,
        help=(
            "Optional smoke-test limit. "
            "Example: 1 evaluates one state "
            "per fold."
        ),
    )

    parser.add_argument(
        "--skip-manual",
        action="store_true",
        help=(
            "Skip the manual Informer "
            "comparison."
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    optuna.logging.set_verbosity(
        optuna.logging.WARNING
    )

    model_df = (
        build_all_state_data()
    )

    (
        eligibility,
        eligible_states,
    ) = find_eligible_states(
        model_df
    )

    folds = build_state_folds(
        eligible_states
    )

    folds.to_csv(
        FOLDS_PATH,
        index=False,
    )

    print(
        "\nAgentic Informer "
        "State Cross-Validation"
    )

    print(
        "================================"
    )

    print(
        f"Eligible jurisdictions: "
        f"{len(eligible_states)}"
    )

    print(
        f"State folds: "
        f"{NUM_FOLDS}"
    )

    print(
        f"Optuna trials: "
        f"{args.trials}"
    )

    print(
        f"Active-season evaluation: "
        f"{TEST_START.date()} "
        f"through "
        f"{TEST_END.date()}"
    )

    print(
        "\nFold sizes:"
    )

    print(
        folds.groupby(
            "fold"
        )[
            "state_code"
        ]
        .nunique()
        .to_string()
    )

    print(
        "\nObjective:"
    )

    print(
        "mean active-season RMSE "
        "across validation "
        "state-horizon observations"
    )

    print(
        "\nImportant:"
    )

    print(
        "These five state folds are "
        "cross-validation folds."
    )

    print(
        "They are NOT treated as an "
        "untouched final test set."
    )

    (
        best_params,
        best_cv_rmse,
        best_trial_number,
        study,
    ) = tune_agentic_informer(
        model_df=
            model_df,

        folds=
            folds,

        num_trials=
            args.trials,

        max_states=
            args.max_states_per_fold,
    )

    print(
        "\nBest configuration"
    )

    print(
        "=================="
    )

    print(
        json.dumps(
            best_params,
            indent=2,
        )
    )

    print(
        "\nBest state-CV "
        f"mean RMSE: "
        f"{best_cv_rmse:.4f}"
    )

    print(
        f"Best trial: "
        f"{best_trial_number}"
    )

    best_output = {
        "best_trial":
            best_trial_number,

        "state_cv_mean_rmse":
            best_cv_rmse,

        "num_trials":
            args.trials,

        "num_folds":
            NUM_FOLDS,

        "state_split_seed":
            STATE_SPLIT_SEED,

        "model_seed":
            MODEL_SEED,

        "train_end":
            str(
                TRAIN_END.date()
            ),

        "active_season_start":
            str(
                TEST_START.date()
            ),

        "active_season_end":
            str(
                TEST_END.date()
            ),

        "config":
            best_params,

        "interpretation":
            (
                "State-fold cross-validation "
                "for hyperparameter selection; "
                "not an untouched final test."
            ),
    }

    BEST_CONFIG_PATH.write_text(
        json.dumps(
            best_output,
            indent=2,
        ),
        encoding="utf-8",
    )

    # The best trial was already evaluated across all folds.
    all_trial_metrics = pd.read_csv(
        TRIAL_STATE_METRICS_PATH,
        dtype={
            "state_code": str,
        },
    )

    all_trial_metrics[
        "state_code"
    ] = (
        all_trial_metrics[
            "state_code"
        ]
        .str.zfill(2)
    )

    best_state_metrics = (
        all_trial_metrics[
            all_trial_metrics[
                "trial"
            ]
            == best_trial_number
        ]
        .copy()
    )

    best_state_metrics.to_csv(
        BEST_STATE_METRICS_PATH,
        index=False,
    )

    if not args.skip_manual:

        print(
            "\nEvaluating manual Informer "
            "across the same state folds..."
        )

        (
            manual_score,
            manual_metrics,
            manual_fold_metrics,
        ) = (
            evaluate_config_across_folds(
                model_df=
                    model_df,

                folds=
                    folds,

                params=
                    MANUAL_CONFIG,

                alias_prefix=
                    "ManualInformer",

                trial_number=
                    None,

                max_states=
                    args.max_states_per_fold,
            )
        )

        manual_metrics.to_csv(
            MANUAL_STATE_METRICS_PATH,
            index=False,
        )

        comparison = build_comparison(
            manual_metrics=
                manual_metrics,

            agent_metrics=
                best_state_metrics,
        )

        comparison.to_csv(
            COMPARISON_PATH,
            index=False,
        )

        agent_mean = float(
            best_state_metrics[
                best_state_metrics[
                    "status"
                ]
                == "complete"
            ][
                "rmse"
            ]
            .mean()
        )

        manual_mean = float(
            manual_metrics[
                manual_metrics[
                    "status"
                ]
                == "complete"
            ][
                "rmse"
            ]
            .mean()
        )

        improvement = (
            (
                manual_mean
                - agent_mean
            )
            / manual_mean
            * 100.0
        )

        improved_states = int(
            comparison[
                "agent_better"
            ]
            .sum()
        )

        total_states = int(
            len(
                comparison
            )
        )

        print(
            "\nManual vs Agentic "
            "State-CV Summary"
        )

        print(
            "==========================="
        )

        print(
            f"Manual mean RMSE: "
            f"{manual_mean:.4f}"
        )

        print(
            f"Agent mean RMSE:  "
            f"{agent_mean:.4f}"
        )

        print(
            f"Agent improvement: "
            f"{improvement:+.1f}%"
        )

        print(
            f"Agent lower RMSE: "
            f"{improved_states}/"
            f"{total_states} states"
        )

    print(
        "\nSaved:"
    )

    for path in [
        FOLDS_PATH,
        TRIALS_PATH,
        TRIAL_STATE_METRICS_PATH,
        BEST_CONFIG_PATH,
        BEST_STATE_METRICS_PATH,
    ]:

        print(
            f"  {path}"
        )

    if not args.skip_manual:

        print(
            f"  "
            f"{MANUAL_STATE_METRICS_PATH}"
        )

        print(
            f"  "
            f"{COMPARISON_PATH}"
        )


if __name__ == "__main__":
    main()