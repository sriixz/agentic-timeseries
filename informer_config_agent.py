import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI()


OUTPUT_PATH = Path(
    "configs/generated_informer_config.py"
)

METADATA_PATH = Path(
    "configs/generated_informer_experiment.json"
)


DEFAULT_EXPERIMENT = {
    "model": "Informer",
    "location": "US",
    "target": "weekly_rate",
    "train_end": "2025-09-30",
    "test_start": "2025-10-01",
    "test_end": "2026-05-31",
    "horizon": 4,
    "validation_size": 12,
    "num_samples": 8,
    "frequency": "W-SAT",
    "use_ed": True,
}


DEFAULT_SEARCH_SPACE = {
    "input_size": {
        "method": "suggest_categorical",
        "values": [8, 12, 16, 24],
    },

    "hidden_size": {
        "method": "suggest_categorical",
        "values": [32, 64, 128],
    },

    "conv_hidden_size": {
        "method": "suggest_categorical",
        "values": [32, 64, 128],
    },

    "n_head": {
        "method": "suggest_categorical",
        "values": [2, 4, 8],
    },

    "encoder_layers": {
        "method": "suggest_int",
        "low": 1,
        "high": 3,
    },

    "decoder_layers": {
        "method": "suggest_int",
        "low": 1,
        "high": 2,
    },

    "dropout": {
        "method": "suggest_float",
        "low": 0.0,
        "high": 0.2,
        "log": False,
    },

    "learning_rate": {
        "method": "suggest_float",
        "low": 0.00005,
        "high": 0.005,
        "log": True,
    },

    "max_steps": {
        "method": "suggest_categorical",
        "values": [500, 1000],
    },

    "batch_size": {
        "method": "suggest_categorical",
        "values": [16, 32],
    },

    "random_seed": {
        "method": "suggest_int",
        "low": 1,
        "high": 10,
    },

    "scaler_type": {
        "method": "fixed",
        "value": "standard",
    },
}


ALLOWED_METHODS = {
    "fixed",
    "suggest_categorical",
    "suggest_int",
    "suggest_float",
}


def build_prompt(
    experiment_request: str,
) -> str:

    return f"""
You are generating an Optuna search configuration for a
NeuralForecast Informer model used for influenza
hospitalization forecasting.

The forecasting model must remain Informer.

The final configuration will become a Python callable:

def informer_config(trial):
    return {{
        "input_size": ...,
        "hidden_size": ...,
        ...
    }}

Return valid JSON only.

The JSON must contain exactly these top-level keys:

{{
  "experiment": {{...}},
  "search_space": {{...}}
}}

Use this experiment configuration unless explicitly
overridden:

{json.dumps(DEFAULT_EXPERIMENT, indent=2)}

Use this default search space unless explicitly overridden:

{json.dumps(DEFAULT_SEARCH_SPACE, indent=2)}

Allowed search methods:

- fixed
- suggest_categorical
- suggest_int
- suggest_float

Rules:

1. The model must remain "Informer".
2. Keep every parameter in DEFAULT_SEARCH_SPACE.
3. Do not add new parameters.
4. Do not remove parameters.
5. Do not change the forecasting target.
6. Keep ED enabled unless explicitly instructed otherwise.
7. Prefer modest search spaces that can be evaluated with
   a limited number of GPU trials.
8. Do not claim a configuration will improve performance.
9. Return JSON only.

User experiment request:

{experiment_request}
""".strip()


def validate_search_spec(
    name: str,
    spec: dict,
) -> None:

    if not isinstance(
        spec,
        dict,
    ):
        raise ValueError(
            f"{name} must be an object."
        )

    method = spec.get(
        "method"
    )

    if method not in ALLOWED_METHODS:
        raise ValueError(
            f"{name}: unsupported "
            f"method {method!r}"
        )

    if method == "fixed":

        if set(spec) != {
            "method",
            "value",
        }:
            raise ValueError(
                f"{name}: invalid fixed spec."
            )

        return

    if method == "suggest_categorical":

        if set(spec) != {
            "method",
            "values",
        }:
            raise ValueError(
                f"{name}: invalid "
                "categorical spec."
            )

        values = spec["values"]

        if (
            not isinstance(
                values,
                list,
            )
            or not values
        ):
            raise ValueError(
                f"{name}: values must "
                "be a non-empty list."
            )

        return

    if method == "suggest_int":

        if set(spec) != {
            "method",
            "low",
            "high",
        }:
            raise ValueError(
                f"{name}: invalid int spec."
            )

        if not isinstance(
            spec["low"],
            int,
        ):
            raise ValueError(
                f"{name}: low must be int."
            )

        if not isinstance(
            spec["high"],
            int,
        ):
            raise ValueError(
                f"{name}: high must be int."
            )

        if (
            spec["low"]
            > spec["high"]
        ):
            raise ValueError(
                f"{name}: low > high."
            )

        return

    if method == "suggest_float":

        if set(spec) != {
            "method",
            "low",
            "high",
            "log",
        }:
            raise ValueError(
                f"{name}: invalid float spec."
            )

        low = spec["low"]
        high = spec["high"]

        if not isinstance(
            low,
            (int, float),
        ):
            raise ValueError(
                f"{name}: low must "
                "be numeric."
            )

        if not isinstance(
            high,
            (int, float),
        ):
            raise ValueError(
                f"{name}: high must "
                "be numeric."
            )

        if low > high:
            raise ValueError(
                f"{name}: low > high."
            )

        if not isinstance(
            spec["log"],
            bool,
        ):
            raise ValueError(
                f"{name}: log must "
                "be boolean."
            )

        if (
            spec["log"]
            and (
                low <= 0
                or high <= 0
            )
        ):
            raise ValueError(
                f"{name}: log bounds "
                "must be positive."
            )


def validate_generated_config(
    config: dict,
) -> None:

    if set(config) != {
        "experiment",
        "search_space",
    }:
        raise ValueError(
            "Top-level keys must be "
            "experiment and search_space."
        )

    experiment = config[
        "experiment"
    ]

    if set(experiment) != set(
        DEFAULT_EXPERIMENT
    ):
        raise ValueError(
            "Experiment keys do not "
            "match expected schema."
        )

    if (
        experiment["model"]
        != "Informer"
    ):
        raise ValueError(
            "Model must remain Informer."
        )

    if (
        not isinstance(
            experiment["horizon"],
            int,
        )
        or experiment["horizon"]
        <= 0
    ):
        raise ValueError(
            "horizon must be positive."
        )

    if (
        not isinstance(
            experiment[
                "num_samples"
            ],
            int,
        )
        or experiment[
            "num_samples"
        ] <= 0
    ):
        raise ValueError(
            "num_samples must be positive."
        )

    search_space = config[
        "search_space"
    ]

    if set(search_space) != set(
        DEFAULT_SEARCH_SPACE
    ):
        raise ValueError(
            "Search-space keys do not "
            "match expected parameters."
        )

    for (
        name,
        spec,
    ) in search_space.items():

        validate_search_spec(
            name,
            spec,
        )


def generate_config(
    experiment_request: str,
) -> dict:

    response = client.responses.create(
        model="gpt-5.4-mini",
        input=build_prompt(
            experiment_request
        ),
    )

    raw_text = (
        response.output_text.strip()
    )

    try:

        config = json.loads(
            raw_text
        )

    except json.JSONDecodeError as exc:

        raise ValueError(
            "Informer config agent "
            "returned invalid JSON.\n"
            f"{raw_text}"
        ) from exc

    validate_generated_config(
        config
    )

    return config


def search_expression(
    parameter_name: str,
    spec: dict,
) -> str:

    method = spec[
        "method"
    ]

    if method == "fixed":

        return repr(
            spec["value"]
        )

    if (
        method
        == "suggest_categorical"
    ):

        return (
            f'trial.suggest_categorical('
            f'"{parameter_name}", '
            f'{repr(spec["values"])})'
        )

    if method == "suggest_int":

        return (
            f'trial.suggest_int('
            f'"{parameter_name}", '
            f'{spec["low"]}, '
            f'{spec["high"]})'
        )

    if method == "suggest_float":

        return (
            f'trial.suggest_float('
            f'"{parameter_name}", '
            f'{repr(spec["low"])}, '
            f'{repr(spec["high"])}, '
            f'log={spec["log"]})'
        )

    raise ValueError(
        f"Unsupported method: "
        f"{method}"
    )


def render_python_config(
    config: dict,
) -> str:

    lines = [
        '"""Auto-generated Informer '
        'Optuna configuration.',
        "",
        "Generated by "
        "informer_config_agent.py.",
        '"""',
        "",
        "",
        "def informer_config(trial):",
        "    return {",
    ]

    for (
        parameter_name,
        spec,
    ) in config[
        "search_space"
    ].items():

        expression = search_expression(
            parameter_name,
            spec,
        )

        lines.append(
            f'        '
            f'"{parameter_name}": '
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


def save_config(
    config: dict,
) -> Path:

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    source = render_python_config(
        config
    )

    OUTPUT_PATH.write_text(
        source,
        encoding="utf-8",
    )

    METADATA_PATH.write_text(
        json.dumps(
            config[
                "experiment"
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Saved Informer config: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Saved experiment metadata: "
        f"{METADATA_PATH}"
    )

    return OUTPUT_PATH


def main():

    experiment_request = """
Use US FluSight weekly hospitalization-rate data.

Use hospitalization data through September 2025 for
training.

Evaluate October 2025 through May 2026.

Forecast 1 through 4 weeks ahead.

Use the leakage-safe four-week-lagged ED visit feature.

Tune the Informer architecture and optimization
configuration.

Keep the search computationally reasonable for GPU
execution.

Use eight Optuna trials.
""".strip()

    config = generate_config(
        experiment_request
    )

    save_config(
        config
    )

    print(
        "\nGenerated experiment:"
    )

    print(
        json.dumps(
            config["experiment"],
            indent=2,
        )
    )

    print(
        "\nGenerated search space:"
    )

    print(
        json.dumps(
            config["search_space"],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()