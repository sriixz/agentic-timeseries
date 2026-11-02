from pathlib import Path
import json

import numpy as np
import optuna
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

OLD_SUMMARY_PATH = Path(
    "forecast_results/allstate_lstm_ed_summary.csv"
)

RESULT_DIR = Path("forecast_results")


SUMMARY_PATH = (
    RESULT_DIR
    / "representative_lstm_ed_optuna_summary.csv"
)

BEST_CONFIG_PATH = (
    RESULT_DIR
    / "representative_lstm_ed_optuna_configs.csv"
)


REPRESENTATIVE_STATES = [
    "Idaho",
    "Louisiana",
    "Utah",
    "Virginia",
    "Pennsylvania",
    "South Dakota",
    "Florida",
    "Tennessee",
]


HORIZON = 4

TRAIN_END = pd.Timestamp(
    "2025-09-30"
)

TEST_START = pd.Timestamp(
    "2025-10-01"
)

TEST_END = pd.Timestamp(
    "2026-05-31"
)


NUM_TRIALS = 5
VALIDATION_WEEKS = 12

MAX_STEPS = 500
SEED = 2

ED_COLUMN = "ed_visits_prop"


def load_data():
    hospital = pd.read_csv(
        HOSPITAL_PATH,
        dtype={"location": str},
    )

    ed = pd.read_csv(
        ED_PATH,
        dtype={"location": str},
    )

    hospital["date"] = pd.to_datetime(
        hospital["date"]
    )

    ed["date"] = pd.to_datetime(
        ed["date"]
    )

    hospital["location"] = (
        hospital["location"]
        .astype(str)
        .str.strip()
    )

    ed["location"] = (
        ed["location"]
        .astype(str)
        .str.strip()
    )

    ed = ed.rename(
        columns={
            "value": ED_COLUMN
        }
    )

    return hospital, ed


def build_state_data(
    hospital,
    ed,
    state_name,
):
    h = hospital[
        hospital["location_name"]
        == state_name
    ][
        [
            "date",
            "location",
            "location_name",
            "weekly_rate",
        ]
    ].copy()

    if h.empty:
        raise ValueError(
            f"No hospitalization data "
            f"for {state_name}"
        )

    location = str(
        h["location"].iloc[0]
    )

    e = ed[
        ed["location"] == location
    ][
        [
            "date",
            "location",
            ED_COLUMN,
        ]
    ].copy()

    merged = h.merge(
        e,
        on=[
            "date",
            "location",
        ],
        how="inner",
        validate="one_to_one",
    )

    merged = (
        merged
        .dropna(
            subset=[
                "weekly_rate",
                ED_COLUMN,
            ]
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    merged = merged[
        merged["date"] <= TEST_END
    ].copy()

    model_df = pd.DataFrame(
        {
            "unique_id":
                location,

            "ds":
                merged["date"],

            "y":
                merged[
                    "weekly_rate"
                ].astype(float),

            ED_COLUMN:
                merged[
                    ED_COLUMN
                ].astype(float),
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

    if len(train_df) < 60:
        raise ValueError(
            f"Insufficient training data "
            f"for {state_name}"
        )

    if len(test_df) < HORIZON:
        raise ValueError(
            f"Insufficient test data "
            f"for {state_name}"
        )

    return (
        location,
        model_df,
        train_df,
        test_df,
    )


def suggest_params(trial):
    input_size = (
        trial.suggest_categorical(
            "input_size",
            [
                8,
                12,
                16,
                24,
            ],
        )
    )

    return {
        "input_size":
            input_size,

        "encoder_hidden_size":
            trial.suggest_categorical(
                "encoder_hidden_size",
                [
                    16,
                    32,
                    64,
                ],
            ),

        "encoder_n_layers":
            trial.suggest_int(
                "encoder_n_layers",
                1,
                3,
            ),

        "context_size":
            trial.suggest_categorical(
                "context_size",
                [
                    5,
                    10,
                ],
            ),

        "decoder_hidden_size":
            trial.suggest_categorical(
                "decoder_hidden_size",
                [
                    32,
                    64,
                ],
            ),

        "learning_rate":
            trial.suggest_float(
                "learning_rate",
                1e-4,
                5e-3,
                log=True,
            ),

        "batch_size":
            trial.suggest_categorical(
                "batch_size",
                [
                    16,
                    32,
                ],
            ),
    }


def build_model(
    params,
    use_ed,
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

        "encoder_hidden_size":
            params[
                "encoder_hidden_size"
            ],

        "encoder_n_layers":
            params[
                "encoder_n_layers"
            ],

        "context_size":
            params[
                "context_size"
            ],

        "decoder_hidden_size":
            params[
                "decoder_hidden_size"
            ],

        "learning_rate":
            params[
                "learning_rate"
            ],

        "batch_size":
            params[
                "batch_size"
            ],

        "max_steps":
            MAX_STEPS,

        "scaler_type":
            "standard",

        "random_seed":
            SEED,

        "alias":
            alias,
    }

    if use_ed:
        kwargs[
            "hist_exog_list"
        ] = [
            ED_COLUMN
        ]

    return LSTM(
        **kwargs
    )


def validation_score(
    train_df,
    params,
    use_ed,
):
    alias = (
        "ValidationED"
        if use_ed
        else "ValidationBaseline"
    )

    model = build_model(
        params=params,
        use_ed=use_ed,
        alias=alias,
    )

    nf = NeuralForecast(
        models=[model],
        freq="W-SAT",
    )

    cv = nf.cross_validation(
        df=train_df,
        n_windows=None,
        test_size=VALIDATION_WEEKS,
        step_size=1,
        refit=False,
    )

    prediction = (
        cv[alias]
        .astype(float)
    )

    actual = (
        cv["y"]
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

    return rmse


def tune_variant(
    train_df,
    use_ed,
):
    def objective(trial):
        params = (
            suggest_params(
                trial
            )
        )

        try:
            score = (
                validation_score(
                    train_df=train_df,
                    params=params,
                    use_ed=use_ed,
                )
            )

            return score

        except Exception as exc:
            print(
                "Trial failed:",
                exc,
            )

            return float("inf")

    sampler = (
        optuna.samplers.TPESampler(
            seed=SEED
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

    return (
        study.best_params,
        float(
            study.best_value
        ),
    )


def evaluate_best(
    model_df,
    test_df,
    params,
    use_ed,
):
    alias = (
        "TunedED"
        if use_ed
        else "TunedBaseline"
    )

    model = build_model(
        params=params,
        use_ed=use_ed,
        alias=alias,
    )

    nf = NeuralForecast(
        models=[model],
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

    for horizon, group in (
        cv.groupby(
            "horizon"
        )
    ):
        actual = (
            group["y"]
            .astype(float)
        )

        pred = (
            group[alias]
            .astype(float)
        )

        error = (
            pred
            - actual
        )

        mae = float(
            np.mean(
                np.abs(error)
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

    return pd.DataFrame(
        rows
    )


def load_old_results():
    if not OLD_SUMMARY_PATH.exists():
        return pd.DataFrame()

    old = pd.read_csv(
        OLD_SUMMARY_PATH
    )

    return old[
        old["horizon"] == 4
    ][
        [
            "location_name",
            "baseline_rmse",
            "ed_rmse",
            "rmse_improvement_pct",
        ]
    ].rename(
        columns={
            "baseline_rmse":
                "old_baseline_rmse",

            "ed_rmse":
                "old_ed_rmse",

            "rmse_improvement_pct":
                "old_ed_improvement_pct",
        }
    )


def main():
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    optuna.logging.set_verbosity(
        optuna.logging.WARNING
    )

    hospital, ed = (
        load_data()
    )

    old_results = (
        load_old_results()
    )

    summary_rows = []
    config_rows = []

    total_states = len(
        REPRESENTATIVE_STATES
    )

    print(
        "\nRepresentative-State "
        "LSTM + ED Optuna Retuning"
    )

    print(
        "=================================="
    )

    print(
        f"States: {total_states}"
    )

    print(
        f"Trials per variant: "
        f"{NUM_TRIALS}"
    )

    print(
        "Observation windows searched: "
        "8, 12, 16, 24 weeks"
    )

    for state_index, state_name in (
        enumerate(
            REPRESENTATIVE_STATES,
            start=1,
        )
    ):
        print(
            "\n"
            + "=" * 60
        )

        print(
            f"[{state_index}/"
            f"{total_states}] "
            f"{state_name}"
        )

        print(
            "=" * 60
        )

        (
            location,
            model_df,
            train_df,
            test_df,
        ) = build_state_data(
            hospital,
            ed,
            state_name,
        )

        print(
            f"Training observations: "
            f"{len(train_df)}"
        )

        print(
            f"Test observations: "
            f"{len(test_df)}"
        )

        print(
            "\nTuning baseline..."
        )

        (
            baseline_params,
            baseline_val,
        ) = tune_variant(
            train_df=train_df,
            use_ed=False,
        )

        print(
            "Best baseline validation "
            f"RMSE: {baseline_val:.4f}"
        )

        print(
            "Best baseline config:"
        )

        print(
            baseline_params
        )

        print(
            "\nTuning ED model..."
        )

        (
            ed_params,
            ed_val,
        ) = tune_variant(
            train_df=train_df,
            use_ed=True,
        )

        print(
            "Best ED validation "
            f"RMSE: {ed_val:.4f}"
        )

        print(
            "Best ED config:"
        )

        print(
            ed_params
        )

        print(
            "\nEvaluating held-out "
            "2025-26 period..."
        )

        baseline_eval = (
            evaluate_best(
                model_df=model_df,
                test_df=test_df,
                params=(
                    baseline_params
                ),
                use_ed=False,
            )
        )

        ed_eval = (
            evaluate_best(
                model_df=model_df,
                test_df=test_df,
                params=ed_params,
                use_ed=True,
            )
        )

        combined = (
            baseline_eval.merge(
                ed_eval,
                on="horizon",
                suffixes=(
                    "_baseline",
                    "_ed",
                ),
            )
        )

        for row in (
            combined.itertuples(
                index=False
            )
        ):
            improvement = (
                (
                    row.rmse_baseline
                    - row.rmse_ed
                )
                / row.rmse_baseline
                * 100.0
            )

            summary_rows.append(
                {
                    "location":
                        location,

                    "location_name":
                        state_name,

                    "horizon":
                        row.horizon,

                    "tuned_baseline_mae":
                        row.mae_baseline,

                    "tuned_ed_mae":
                        row.mae_ed,

                    "tuned_baseline_rmse":
                        row.rmse_baseline,

                    "tuned_ed_rmse":
                        row.rmse_ed,

                    "tuned_ed_improvement_pct":
                        improvement,
                }
            )

        config_rows.append(
            {
                "location":
                    location,

                "location_name":
                    state_name,

                "baseline_validation_rmse":
                    baseline_val,

                "ed_validation_rmse":
                    ed_val,

                "baseline_input_size":
                    baseline_params[
                        "input_size"
                    ],

                "ed_input_size":
                    ed_params[
                        "input_size"
                    ],

                "baseline_config_json":
                    json.dumps(
                        baseline_params,
                        sort_keys=True,
                    ),

                "ed_config_json":
                    json.dumps(
                        ed_params,
                        sort_keys=True,
                    ),
            }
        )

        current_summary = (
            pd.DataFrame(
                summary_rows
            )
        )

        current_configs = (
            pd.DataFrame(
                config_rows
            )
        )

        current_summary.to_csv(
            SUMMARY_PATH,
            index=False,
        )

        current_configs.to_csv(
            BEST_CONFIG_PATH,
            index=False,
        )

        four_week = (
            current_summary[
                (
                    current_summary[
                        "location_name"
                    ]
                    == state_name
                )
                & (
                    current_summary[
                        "horizon"
                    ]
                    == 4
                )
            ]
            .iloc[0]
        )

        print(
            "\nTuned 4-week RMSE:"
        )

        print(
            f"Baseline "
            f"{four_week['tuned_baseline_rmse']:.4f}"
            " -> ED "
            f"{four_week['tuned_ed_rmse']:.4f}"
            " | "
            f"{four_week['tuned_ed_improvement_pct']:+.1f}%"
        )

    summary = pd.DataFrame(
        summary_rows
    )

    configs = pd.DataFrame(
        config_rows
    )

    # Add old untuned 4-week results.
    four_week = summary[
        summary["horizon"] == 4
    ].copy()

    if not old_results.empty:
        four_week = (
            four_week.merge(
                old_results,
                on="location_name",
                how="left",
            )
        )

    print(
        "\n\nFinal Tuned 4-Week Comparison"
    )

    print(
        "=============================="
    )

    columns = [
        "location_name",
        "tuned_baseline_rmse",
        "tuned_ed_rmse",
        "tuned_ed_improvement_pct",
    ]

    if (
        "old_ed_improvement_pct"
        in four_week.columns
    ):
        columns.append(
            "old_ed_improvement_pct"
        )

    print(
        four_week[
            columns
        ]
        .sort_values(
            "tuned_ed_improvement_pct",
            ascending=False,
        )
        .to_string(
            index=False,
            formatters={
                "tuned_baseline_rmse":
                    "{:.3f}".format,

                "tuned_ed_rmse":
                    "{:.3f}".format,

                "tuned_ed_improvement_pct":
                    "{:+.1f}".format,

                "old_ed_improvement_pct":
                    "{:+.1f}".format,
            },
        )
    )

    print(
        "\nSelected observation windows"
    )

    print(
        "============================"
    )

    print(
        configs[
            [
                "location_name",
                "baseline_input_size",
                "ed_input_size",
                "baseline_validation_rmse",
                "ed_validation_rmse",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {SUMMARY_PATH}"
    )

    print(
        f"Saved: {BEST_CONFIG_PATH}"
    )


if __name__ == "__main__":
    main()