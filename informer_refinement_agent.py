import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from informer_config_agent import (
    DEFAULT_EXPERIMENT,
    DEFAULT_SEARCH_SPACE,
    validate_search_spec,
    search_expression,
)


load_dotenv()

client = OpenAI()


TRIALS_PATH = Path(
    "forecast_results/agentic_informer_trials.csv"
)

BEST_CONFIG_PATH = Path(
    "forecast_results/agentic_informer_best_config.json"
)

OUTPUT_CONFIG_PATH = Path(
    "configs/generated_informer_refined_config.py"
)

OUTPUT_METADATA_PATH = Path(
    "configs/generated_informer_refinement.json"
)


def load_inputs():

    if not TRIALS_PATH.exists():
        raise FileNotFoundError(
            f"Missing: {TRIALS_PATH}"
        )

    if not BEST_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing: {BEST_CONFIG_PATH}"
        )

    trials = pd.read_csv(
        TRIALS_PATH
    )

    with BEST_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        best_output = json.load(file)

    complete = trials[
        trials["status"] == "complete"
    ].copy()

    if complete.empty:
        raise ValueError(
            "No completed trials found."
        )

    trial_summary = []

    for row in complete.itertuples():

        params = json.loads(
            row.params_json
        )

        trial_summary.append(
            {
                "trial":
                    int(row.trial),

                "validation_mean_rmse":
                    float(
                        row.validation_mean_rmse
                    ),

                "rmse_h1":
                    float(row.rmse_h1),

                "rmse_h2":
                    float(row.rmse_h2),

                "rmse_h3":
                    float(row.rmse_h3),

                "rmse_h4":
                    float(row.rmse_h4),

                "params":
                    params,
            }
        )

    return (
        trial_summary,
        best_output,
    )


def build_prompt(
    trial_summary,
    best_output,
):

    best_config = best_output[
        "config"
    ]

    return f"""
You are refining a NeuralForecast Informer hyperparameter
search for influenza hospitalization forecasting.

This is the SECOND search iteration.

IMPORTANT SCIENTIFIC RULE:

You are allowed to use ONLY the supplied validation results.

You have NOT been given held-out 2025-26 test performance.
Do not make assumptions about held-out performance.

The forecasting architecture must remain Informer.

The model uses a leakage-safe four-week-lagged ED-visits
feature as future exogenous information.

FIRST-ROUND WINNER:

{json.dumps(best_config, indent=2)}

FIRST-ROUND BEST VALIDATION MEAN RMSE:

{best_output["validation_mean_rmse"]}

ALL COMPLETED FIRST-ROUND VALIDATION TRIALS:

{json.dumps(trial_summary, indent=2)}

ORIGINAL SEARCH SPACE:

{json.dumps(DEFAULT_SEARCH_SPACE, indent=2)}

Your task is to propose ONE refined Optuna search space.

The goal is not to redesign the model. Narrow or redirect
the search using the observed validation evidence.

Return valid JSON only with exactly this structure:

{{
  "assessment": "short evidence-based assessment",
  "refinement_reason": "why this second search is useful",
  "search_space": {{
    ...
  }}
}}

The search_space must contain exactly these keys:

{list(DEFAULT_SEARCH_SPACE.keys())}

Allowed specification formats:

Fixed:
{{
  "method": "fixed",
  "value": ...
}}

Categorical:
{{
  "method": "suggest_categorical",
  "values": [...]
}}

Integer:
{{
  "method": "suggest_int",
  "low": integer,
  "high": integer
}}

Float:
{{
  "method": "suggest_float",
  "low": number,
  "high": number,
  "log": true or false
}}

STRICT RULES:

1. Keep the model as Informer.
2. Keep ED enabled.
3. Do not introduce new hyperparameters.
4. Do not remove hyperparameters.
5. scaler_type must remain fixed to "standard".
6. The current winning value must remain reachable for
   every hyperparameter.
7. Prefer a narrower search than round 1 when evidence
   supports narrowing.
8. Do not claim the refinement will improve test performance.
9. Base every refinement only on the supplied validation
   results.
10. Keep the search computationally reasonable for another
    eight GPU trials.
11. Return JSON only.
""".strip()


def validate_refinement(
    refinement,
    current_best,
):

    if set(refinement) != {
        "assessment",
        "refinement_reason",
        "search_space",
    }:
        raise ValueError(
            "Invalid top-level refinement keys."
        )

    search_space = refinement[
        "search_space"
    ]

    if set(search_space) != set(
        DEFAULT_SEARCH_SPACE
    ):
        raise ValueError(
            "Refined search-space keys "
            "do not match the original."
        )

    for name, spec in search_space.items():
        validate_search_spec(
            name,
            spec,
        )

    scaler = search_space[
        "scaler_type"
    ]

    if scaler != {
        "method": "fixed",
        "value": "standard",
    }:
        raise ValueError(
            "scaler_type must remain "
            "fixed to standard."
        )

    # Ensure the first-round winner remains reachable.
    for name, best_value in current_best.items():

        if name not in search_space:
            continue

        spec = search_space[name]

        method = spec["method"]

        if method == "fixed":

            if spec["value"] != best_value:
                raise ValueError(
                    f"{name}: fixed refinement "
                    "excludes current winner."
                )

        elif method == "suggest_categorical":

            if best_value not in spec["values"]:
                raise ValueError(
                    f"{name}: categorical refinement "
                    "excludes current winner."
                )

        elif method == "suggest_int":

            if not (
                spec["low"]
                <= best_value
                <= spec["high"]
            ):
                raise ValueError(
                    f"{name}: integer refinement "
                    "excludes current winner."
                )

        elif method == "suggest_float":

            if not (
                spec["low"]
                <= best_value
                <= spec["high"]
            ):
                raise ValueError(
                    f"{name}: float refinement "
                    "excludes current winner."
                )


def generate_refinement(
    trial_summary,
    best_output,
):

    response = client.responses.create(
        model="gpt-5.4-mini",
        input=build_prompt(
            trial_summary,
            best_output,
        ),
    )

    raw_text = (
        response.output_text.strip()
    )

    try:
        refinement = json.loads(
            raw_text
        )

    except json.JSONDecodeError as exc:

        raise ValueError(
            "Refinement agent returned "
            "invalid JSON.\n"
            f"{raw_text}"
        ) from exc

    validate_refinement(
        refinement,
        best_output["config"],
    )

    return refinement


def render_config(
    refinement,
):

    lines = [
        '"""Agent-refined Informer Optuna configuration.',
        "",
        "Generated by informer_refinement_agent.py.",
        "",
        "Uses validation evidence only.",
        '"""',
        "",
        "",
        "def refined_informer_config(trial):",
        "    return {",
    ]

    for (
        parameter_name,
        spec,
    ) in refinement[
        "search_space"
    ].items():

        expression = search_expression(
            parameter_name,
            spec,
        )

        lines.append(
            f'        "{parameter_name}": '
            f'{expression},'
        )

    lines.extend(
        [
            "    }",
            "",
        ]
    )

    return "\n".join(
        lines
    )


def main():

    (
        trial_summary,
        best_output,
    ) = load_inputs()

    refinement = generate_refinement(
        trial_summary,
        best_output,
    )

    OUTPUT_CONFIG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_CONFIG_PATH.write_text(
        render_config(
            refinement
        ),
        encoding="utf-8",
    )

    metadata = {
        "experiment":
            DEFAULT_EXPERIMENT,

        "first_round_best":
            best_output,

        "agent_refinement":
            refinement,
    }

    OUTPUT_METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nAgent assessment"
    )

    print(
        "================"
    )

    print(
        refinement[
            "assessment"
        ]
    )

    print(
        "\nRefinement reason"
    )

    print(
        "================="
    )

    print(
        refinement[
            "refinement_reason"
        ]
    )

    print(
        "\nRefined search space"
    )

    print(
        "===================="
    )

    print(
        json.dumps(
            refinement[
                "search_space"
            ],
            indent=2,
        )
    )

    print(
        f"\nSaved: "
        f"{OUTPUT_CONFIG_PATH}"
    )

    print(
        f"Saved: "
        f"{OUTPUT_METADATA_PATH}"
    )


if __name__ == "__main__":
    main()