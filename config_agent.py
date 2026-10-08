import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI()

OUTPUT_PATH = Path("configs/generated_autolstm_config.py")
METADATA_PATH = Path("configs/generated_autolstm_experiment.json")


DEFAULT_EXPERIMENT = {
    "model": "LSTM",
    "location": "US",
    "target": "weekly_rate",
    "train_end": "2025-09-30",
    "test_start": "2025-10-01",
    "test_end": "2026-05-31",
    "horizon": 4,
    "validation_size": 16,
    "num_samples": 5,
    "frequency": "W-SAT",
}


DEFAULT_SEARCH_SPACE = {
    "input_size": {
        "method": "suggest_categorical",
        "values": [8, 16, 32, 64, 128],
    },
    "inference_input_size": {
        "method": "fixed",
        "value": -1,
    },
    "encoder_hidden_size": {
        "method": "suggest_categorical",
        "values": [16, 32, 64],
    },
    "encoder_n_layers": {
        "method": "suggest_int",
        "low": 1,
        "high": 3,
    },
    "context_size": {
        "method": "suggest_categorical",
        "values": [5, 10],
    },
    "decoder_hidden_size": {
        "method": "suggest_categorical",
        "values": [16, 32, 64],
    },
    "learning_rate": {
        "method": "suggest_float",
        "low": 0.0001,
        "high": 0.01,
        "log": True,
    },
    "max_steps": {
        "method": "suggest_categorical",
        "values": [300, 500, 1000],
    },
    "batch_size": {
        "method": "suggest_categorical",
        "values": [16, 32],
    },
    "random_seed": {
        "method": "suggest_int",
        "low": 1,
        "high": 20,
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


def build_prompt(experiment_request: str) -> str:
    return f"""
You are generating an AutoLSTM experiment configuration for NeuralForecast
using the Optuna backend.

The final search configuration will become a Python callable of the form:

def autolstm_config(trial):
    return {{
        "input_size": trial.suggest_categorical(...),
        ...
    }}

Return valid JSON only.

The JSON must have exactly these top-level keys:

{{
  "experiment": {{...}},
  "search_space": {{...}}
}}

Use this default experiment configuration unless the user explicitly overrides
a field:

{json.dumps(DEFAULT_EXPERIMENT, indent=2)}

Use this default search space unless the user explicitly overrides a parameter:

{json.dumps(DEFAULT_SEARCH_SPACE, indent=2)}

Allowed search-space methods are:

- "fixed"
- "suggest_categorical"
- "suggest_int"
- "suggest_float"

Schemas:

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

Rules:

1. The model must remain "LSTM".
2. Do not invent a search range for unspecified parameters.
3. For unspecified search parameters, copy the default exactly.
4. Keep every parameter present in DEFAULT_SEARCH_SPACE.
5. The search-space keys must be exactly:
   {list(DEFAULT_SEARCH_SPACE.keys())}
6. Do not add explanatory prose.
7. Return JSON only.

User experiment request:

{experiment_request}
""".strip()


def validate_experiment(experiment: dict) -> None:
    required_keys = set(DEFAULT_EXPERIMENT)
    actual_keys = set(experiment)

    if actual_keys != required_keys:
        missing = required_keys - actual_keys
        extra = actual_keys - required_keys
        raise ValueError(
            f"Invalid experiment keys. Missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    if experiment["model"] != "LSTM":
        raise ValueError("Only model='LSTM' is currently supported.")

    if not isinstance(experiment["horizon"], int) or experiment["horizon"] <= 0:
        raise ValueError("horizon must be a positive integer.")

    if (
        not isinstance(experiment["validation_size"], int)
        or experiment["validation_size"] <= 0
    ):
        raise ValueError("validation_size must be a positive integer.")

    if (
        not isinstance(experiment["num_samples"], int)
        or experiment["num_samples"] <= 0
    ):
        raise ValueError("num_samples must be a positive integer.")


def validate_search_spec(name: str, spec: dict) -> None:
    if not isinstance(spec, dict):
        raise ValueError(f"{name} must be an object.")

    method = spec.get("method")

    if method not in ALLOWED_METHODS:
        raise ValueError(
            f"{name}: unsupported method {method!r}. "
            f"Allowed={sorted(ALLOWED_METHODS)}"
        )

    if method == "fixed":
        if set(spec) != {"method", "value"}:
            raise ValueError(
                f"{name}: fixed spec must contain only method and value."
            )
        return

    if method == "suggest_categorical":
        if set(spec) != {"method", "values"}:
            raise ValueError(
                f"{name}: categorical spec must contain only method and values."
            )

        values = spec["values"]

        if not isinstance(values, list) or not values:
            raise ValueError(f"{name}: values must be a non-empty list.")

        return

    if method == "suggest_int":
        allowed = {"method", "low", "high"}
        if set(spec) != allowed:
            raise ValueError(
                f"{name}: suggest_int spec must contain "
                f"method, low, and high."
            )

        low = spec["low"]
        high = spec["high"]

        if not isinstance(low, int) or not isinstance(high, int):
            raise ValueError(f"{name}: integer bounds must be integers.")

        if low > high:
            raise ValueError(f"{name}: low must be <= high.")

        return

    if method == "suggest_float":
        allowed = {"method", "low", "high", "log"}
        if set(spec) != allowed:
            raise ValueError(
                f"{name}: suggest_float spec must contain "
                f"method, low, high, and log."
            )

        low = spec["low"]
        high = spec["high"]
        log = spec["log"]

        if not isinstance(low, (int, float)):
            raise ValueError(f"{name}: low must be numeric.")

        if not isinstance(high, (int, float)):
            raise ValueError(f"{name}: high must be numeric.")

        if low > high:
            raise ValueError(f"{name}: low must be <= high.")

        if not isinstance(log, bool):
            raise ValueError(f"{name}: log must be boolean.")

        if log and (low <= 0 or high <= 0):
            raise ValueError(
                f"{name}: log-scale float bounds must both be positive."
            )


def validate_generated_config(config: dict) -> None:
    if set(config) != {"experiment", "search_space"}:
        raise ValueError(
            "Top-level keys must be exactly: experiment, search_space."
        )

    validate_experiment(config["experiment"])

    search_space = config["search_space"]

    expected_parameters = set(DEFAULT_SEARCH_SPACE)
    actual_parameters = set(search_space)

    if actual_parameters != expected_parameters:
        missing = expected_parameters - actual_parameters
        extra = actual_parameters - expected_parameters

        raise ValueError(
            f"Invalid search-space keys. Missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    for name, spec in search_space.items():
        validate_search_spec(name, spec)


def generate_config(experiment_request: str) -> dict:
    response = client.responses.create(
        model="gpt-5.4-mini",
        input=build_prompt(experiment_request),
    )

    raw_text = response.output_text.strip()

    try:
        config = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Config agent did not return valid JSON.\n"
            f"Raw output:\n{raw_text}"
        ) from exc

    validate_generated_config(config)

    return config


def python_literal(value) -> str:
    return repr(value)


def search_expression(parameter_name: str, spec: dict) -> str:
    method = spec["method"]

    if method == "fixed":
        return python_literal(spec["value"])

    if method == "suggest_categorical":
        return (
            f'trial.suggest_categorical('
            f'"{parameter_name}", '
            f'{python_literal(spec["values"])})'
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
            f'{python_literal(spec["low"])}, '
            f'{python_literal(spec["high"])}, '
            f'log={spec["log"]})'
        )

    raise ValueError(f"Unsupported method: {method}")


def render_python_config(config: dict) -> str:
    search_space = config["search_space"]

    lines = [
        '"""Auto-generated AutoLSTM Optuna configuration.',
        "",
        "Generated by config_agent.py.",
        "Do not edit manually unless you intentionally want to override",
        "the agent-generated search space.",
        '"""',
        "",
        "",
        "def autolstm_config(trial):",
        "    return {",
    ]

    for parameter_name, spec in search_space.items():
        expression = search_expression(parameter_name, spec)

        lines.append(
            f'        "{parameter_name}": {expression},'
        )

    lines.extend(
        [
            "    }",
            "",
        ]
    )

    return "\n".join(lines)


def save_config(config: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    python_source = render_python_config(config)

    OUTPUT_PATH.write_text(
        python_source,
        encoding="utf-8",
    )

    METADATA_PATH.write_text(
        json.dumps(config["experiment"], indent=2),
        encoding="utf-8",
    )

    print(f"Saved AutoLSTM config callable: {OUTPUT_PATH}")
    print(f"Saved experiment metadata: {METADATA_PATH}")


def main() -> None:
    experiment_request = """
Use US FluSight weekly hospitalization-rate data.

Train through September 2025.

Use October 2025 through May 2026 as the held-out test period.

Forecast 1 through 4 weeks ahead.

For the observation window, sweep powers of two:
8, 16, 32, 64, and 128 weeks.

Use encoder hidden sizes 16, 32, and 64.

Use decoder hidden sizes 16, 32, and 64.

Search encoder layers from 1 through 3.

Search learning rate logarithmically from 0.0001 through 0.01.

Search maximum training steps over 300, 500, and 1000.

Search batch sizes 16 and 32.

Search random seeds from 1 through 20.

Use standard scaling.

Run 5 Optuna trials.
""".strip()

    config = generate_config(experiment_request)

    save_config(config)

    print("\nGenerated experiment:")
    print(json.dumps(config["experiment"], indent=2))

    print("\nGenerated search space:")
    print(json.dumps(config["search_space"], indent=2))


if __name__ == "__main__":
    main()