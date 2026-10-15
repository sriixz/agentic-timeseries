import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI()


FORECAST_PATH = Path(
    "forecast_results/autolstm_fixed_test_forecasts.csv"
)

METRICS_PATH = Path(
    "forecast_results/autolstm_fixed_metrics_by_horizon.csv"
)

BEST_CONFIG_PATH = Path(
    "forecast_results/autolstm_best_config.json"
)

EXPERIMENT_PATH = Path(
    "configs/generated_autolstm_experiment.json"
)

OUTPUT_PATH = Path(
    "forecast_results/forecast_feedback.json"
)


ALLOWED_CHANGE_TYPES = {
    "lookback_window",
    "model_complexity",
    "learning_rate",
    "training_steps",
    "batch_size",
    "loss_function",
    "nonnegative_handling",
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


def load_forecasts() -> pd.DataFrame:
    if not FORECAST_PATH.exists():
        raise FileNotFoundError(
            f"Forecast file not found: {FORECAST_PATH}"
        )

    df = pd.read_csv(
        FORECAST_PATH,
        parse_dates=["ds", "cutoff"],
    )

    required = {
        "ds",
        "y",
        "forecast",
        "horizon",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Forecast file is missing required columns: "
            f"{sorted(missing)}"
        )

    return df


def load_metrics() -> pd.DataFrame:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {METRICS_PATH}"
        )

    df = pd.read_csv(METRICS_PATH)

    required = {
        "horizon",
        "forecast_count",
        "mae",
        "rmse",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Metrics file is missing required columns: "
            f"{sorted(missing)}"
        )

    return df


def build_horizon_diagnostics(
    forecasts: pd.DataFrame,
    metrics: pd.DataFrame,
) -> list[dict]:
    diagnostics = []

    for horizon in sorted(
        forecasts["horizon"].unique()
    ):
        subset = forecasts[
            forecasts["horizon"] == horizon
        ].copy()

        metric_row = metrics[
            metrics["horizon"] == horizon
        ]

        if metric_row.empty:
            raise ValueError(
                f"No metrics found for horizon {horizon}."
            )

        metric_row = metric_row.iloc[0]

        errors = (
            subset["forecast"] - subset["y"]
        )

        bias = float(errors.mean())

        negative_count = int(
            (subset["forecast"] < 0).sum()
        )

        actual_peak_row = subset.loc[
            subset["y"].idxmax()
        ]

        forecast_peak_row = subset.loc[
            subset["forecast"].idxmax()
        ]

        actual_peak_value = float(
            actual_peak_row["y"]
        )

        forecast_peak_value = float(
            forecast_peak_row["forecast"]
        )

        peak_magnitude_error = (
            forecast_peak_value
            - actual_peak_value
        )

        peak_timing_error_days = int(
            (
                forecast_peak_row["ds"]
                - actual_peak_row["ds"]
            ).days
        )

        peak_timing_error_weeks = (
            peak_timing_error_days / 7.0
        )

        diagnostics.append(
            {
                "horizon": int(horizon),
                "forecast_count": int(
                    metric_row["forecast_count"]
                ),
                "mae": float(
                    metric_row["mae"]
                ),
                "rmse": float(
                    metric_row["rmse"]
                ),
                "bias": bias,
                "negative_forecast_count": (
                    negative_count
                ),
                "actual_peak_date": (
                    actual_peak_row["ds"]
                    .date()
                    .isoformat()
                ),
                "forecast_peak_date": (
                    forecast_peak_row["ds"]
                    .date()
                    .isoformat()
                ),
                "actual_peak_value": (
                    actual_peak_value
                ),
                "forecast_peak_value": (
                    forecast_peak_value
                ),
                "peak_magnitude_error": (
                    peak_magnitude_error
                ),
                "peak_timing_error_weeks": (
                    peak_timing_error_weeks
                ),
            }
        )

    return diagnostics


def build_overall_diagnostics(
    horizon_diagnostics: list[dict],
) -> dict:
    first = horizon_diagnostics[0]
    last = horizon_diagnostics[-1]

    total_negative = sum(
        item["negative_forecast_count"]
        for item in horizon_diagnostics
    )

    return {
        "best_horizon_by_mae": min(
            horizon_diagnostics,
            key=lambda x: x["mae"],
        )["horizon"],
        "worst_horizon_by_mae": max(
            horizon_diagnostics,
            key=lambda x: x["mae"],
        )["horizon"],
        "mae_change_first_to_last": (
            last["mae"] - first["mae"]
        ),
        "rmse_change_first_to_last": (
            last["rmse"] - first["rmse"]
        ),
        "total_negative_predictions": (
            total_negative
        ),
        "largest_peak_overprediction": max(
            item["peak_magnitude_error"]
            for item in horizon_diagnostics
        ),
        "largest_absolute_peak_timing_error_weeks":
            max(
                abs(
                    item[
                        "peak_timing_error_weeks"
                    ]
                )
                for item
                in horizon_diagnostics
            ),
    }


def build_prompt(
    experiment: dict,
    best_config: dict,
    horizon_diagnostics: list[dict],
    overall_diagnostics: dict,
) -> str:
    return f"""
You are reviewing the output of a NeuralForecast AutoLSTM
experiment for U.S. influenza hospitalization-rate forecasting.

Your role is to act as a forecast-refinement agent.

The forecasting strategy must remain RECURSIVE for this
iteration. Do not recommend switching to direct multi-horizon
forecasting yet.

The next experiment should make only a small number of
well-motivated changes. Do not redesign the entire system.

Current experiment:

{json.dumps(experiment, indent=2)}

Current winning AutoLSTM configuration:

{json.dumps(best_config, indent=2)}

Deterministically calculated held-out diagnostics by horizon:

{json.dumps(horizon_diagnostics, indent=2)}

Overall deterministic diagnostics:

{json.dumps(overall_diagnostics, indent=2)}

Return valid JSON only using exactly this structure:

{{
  "overall_assessment": "short paragraph",
  "main_problems": [
    {{
      "problem": "short name",
      "evidence": "specific evidence from the diagnostics",
      "severity": "low|medium|high"
    }}
  ],
  "ranked_changes": [
    {{
      "rank": 1,
      "change_type": "one allowed type",
      "recommendation": "specific suggested change",
      "rationale": "why this could address the observed error"
    }}
  ],
  "recommended_first_experiment": {{
    "changes_to_apply": [
      "change 1",
      "change 2"
    ],
    "reason": "why these should be tested first"
  }}
}}

Allowed change_type values:

{sorted(ALLOWED_CHANGE_TYPES)}

Rules:

1. Use only the supplied experiment results as evidence.
2. Do not claim a change will improve performance;
   describe it as a hypothesis to test.
3. Rank no more than five changes.
4. Recommend only two changes for the first refinement
   experiment.
5. Keep recursive forecasting.
6. Do not recommend new exogenous data or new model families
   in this iteration.
7. Do not recommend ensembles yet.
8. Prefer changes that can be tested within the existing
   LSTM / AutoLSTM framework.
9. Pay particular attention to:
   - degradation with forecast horizon
   - peak magnitude error
   - peak timing error
   - systematic forecast bias
   - negative predictions
10. Return JSON only.
""".strip()


def validate_feedback(feedback: dict) -> None:
    required_top_level = {
        "overall_assessment",
        "main_problems",
        "ranked_changes",
        "recommended_first_experiment",
    }

    if set(feedback) != required_top_level:
        raise ValueError(
            "Feedback JSON has invalid top-level keys."
        )

    if not isinstance(
        feedback["overall_assessment"],
        str,
    ):
        raise ValueError(
            "overall_assessment must be a string."
        )

    problems = feedback["main_problems"]

    if not isinstance(problems, list):
        raise ValueError(
            "main_problems must be a list."
        )

    for problem in problems:
        required = {
            "problem",
            "evidence",
            "severity",
        }

        if set(problem) != required:
            raise ValueError(
                "Invalid main_problems entry."
            )

        if problem["severity"] not in {
            "low",
            "medium",
            "high",
        }:
            raise ValueError(
                "Invalid severity value."
            )

    changes = feedback["ranked_changes"]

    if not isinstance(changes, list):
        raise ValueError(
            "ranked_changes must be a list."
        )

    if len(changes) > 5:
        raise ValueError(
            "At most five ranked changes are allowed."
        )

    for change in changes:
        required = {
            "rank",
            "change_type",
            "recommendation",
            "rationale",
        }

        if set(change) != required:
            raise ValueError(
                "Invalid ranked_changes entry."
            )

        if (
            change["change_type"]
            not in ALLOWED_CHANGE_TYPES
        ):
            raise ValueError(
                "Unsupported change_type: "
                f"{change['change_type']}"
            )

    first_experiment = feedback[
        "recommended_first_experiment"
    ]

    required = {
        "changes_to_apply",
        "reason",
    }

    if set(first_experiment) != required:
        raise ValueError(
            "Invalid recommended_first_experiment."
        )

    changes_to_apply = first_experiment[
        "changes_to_apply"
    ]

    if not isinstance(
        changes_to_apply,
        list,
    ):
        raise ValueError(
            "changes_to_apply must be a list."
        )

    if len(changes_to_apply) > 2:
        raise ValueError(
            "The first experiment may contain "
            "at most two changes."
        )


def run_feedback_agent(
    experiment: dict,
    best_config: dict,
    horizon_diagnostics: list[dict],
    overall_diagnostics: dict,
) -> dict:
    response = client.responses.create(
        model="gpt-5.4-mini",
        input=build_prompt(
            experiment=experiment,
            best_config=best_config,
            horizon_diagnostics=(
                horizon_diagnostics
            ),
            overall_diagnostics=(
                overall_diagnostics
            ),
        ),
    )

    raw_text = response.output_text.strip()

    try:
        feedback = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Feedback agent did not return "
            "valid JSON.\n"
            f"Raw output:\n{raw_text}"
        ) from exc

    validate_feedback(feedback)

    return feedback


def save_feedback(
    experiment: dict,
    best_config: dict,
    horizon_diagnostics: list[dict],
    overall_diagnostics: dict,
    feedback: dict,
) -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "experiment": experiment,
        "best_config": best_config,
        "horizon_diagnostics": (
            horizon_diagnostics
        ),
        "overall_diagnostics": (
            overall_diagnostics
        ),
        "agent_feedback": feedback,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print(f"\nSaved: {OUTPUT_PATH}")


def print_diagnostics(
    horizon_diagnostics: list[dict],
) -> None:
    print("\nDeterministic Forecast Diagnostics")
    print("==================================")

    for item in horizon_diagnostics:
        print(
            f"{item['horizon']}-week: "
            f"MAE={item['mae']:.4f}, "
            f"RMSE={item['rmse']:.4f}, "
            f"bias={item['bias']:.4f}, "
            f"peak error="
            f"{item['peak_magnitude_error']:.4f}, "
            f"peak timing="
            f"{item['peak_timing_error_weeks']:+.1f} "
            f"weeks, "
            f"negative forecasts="
            f"{item['negative_forecast_count']}"
        )


def print_feedback(feedback: dict) -> None:
    print("\nAgent Assessment")
    print("================")
    print(
        feedback["overall_assessment"]
    )

    print("\nMain Problems")
    print("=============")

    for problem in feedback["main_problems"]:
        print(
            f"- [{problem['severity'].upper()}] "
            f"{problem['problem']}: "
            f"{problem['evidence']}"
        )

    print("\nRanked Suggested Changes")
    print("========================")

    for change in feedback["ranked_changes"]:
        print(
            f"{change['rank']}. "
            f"{change['change_type']}: "
            f"{change['recommendation']}"
        )
        print(
            f"   Reason: "
            f"{change['rationale']}"
        )

    first = feedback[
        "recommended_first_experiment"
    ]

    print("\nRecommended First Refinement")
    print("============================")

    for change in first["changes_to_apply"]:
        print(f"- {change}")

    print(
        f"\nReason: {first['reason']}"
    )


def main() -> None:
    forecasts = load_forecasts()
    metrics = load_metrics()

    experiment = load_json(
        EXPERIMENT_PATH
    )

    best_config_output = load_json(
        BEST_CONFIG_PATH
    )

    if "config" not in best_config_output:
        raise ValueError(
            "autolstm_best_config.json does not "
            "contain a 'config' field."
        )

    best_config = best_config_output["config"]

    horizon_diagnostics = (
        build_horizon_diagnostics(
            forecasts=forecasts,
            metrics=metrics,
        )
    )

    overall_diagnostics = (
        build_overall_diagnostics(
            horizon_diagnostics
        )
    )

    print_diagnostics(
        horizon_diagnostics
    )

    feedback = run_feedback_agent(
        experiment=experiment,
        best_config=best_config,
        horizon_diagnostics=(
            horizon_diagnostics
        ),
        overall_diagnostics=(
            overall_diagnostics
        ),
    )

    print_feedback(feedback)

    save_feedback(
        experiment=experiment,
        best_config=best_config,
        horizon_diagnostics=(
            horizon_diagnostics
        ),
        overall_diagnostics=(
            overall_diagnostics
        ),
        feedback=feedback,
    )


if __name__ == "__main__":
    main()