import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

openai_client = OpenAI()

OUTPUT_PATH = Path(
    "configs/generated_autolstm_config.json"
)


DEFAULT_CONFIG = {
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
        "type": "categorical",
        "values": [8, 16, 32, 64, 128],
    },
    "inference_input_size": {
        "type": "fixed",
        "value": -1,
    },
    "encoder_hidden_size": {
        "type": "categorical",
        "values": [16, 32, 64],
    },
    "encoder_n_layers": {
        "type": "int",
        "low": 1,
        "high": 3,
    },
    "context_size": {
        "type": "categorical",
        "values": [5, 10],
    },
    "decoder_hidden_size": {
        "type": "categorical",
        "values": [16, 32, 64],
    },
    "learning_rate": {
        "type": "float",
        "low": 0.0001,
        "high": 0.01,
        "log": True,
    },
    "max_steps": {
        "type": "categorical",
        "values": [300, 500, 1000],
    },
    "batch_size": {
        "type": "categorical",
        "values": [16, 32],
    },
    "random_seed": {
        "type": "int",
        "low": 1,
        "high": 20,
    },
    "scaler_type": {
        "type": "fixed",
        "value": "standard",
    },
}


ALLOWED_PARAMETER_TYPES = {
    "fixed",
    "categorical",
    "int",
    "float",
}


REQUIRED_SEARCH_PARAMETERS = set(
    DEFAULT_SEARCH_SPACE.keys()
)


def parse_json_response(response_text):
    """
    Parse a model response that should contain only JSON.
    """

    cleaned = response_text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.split(
            "\n",
            1,
        )[1]

    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit(
            "```",
            1,
        )[0]

    cleaned = cleaned.strip()

    try:
        return json.loads(
            cleaned
        )

    except json.JSONDecodeError as error:
        raise ValueError(
            "Model returned invalid JSON.\n\n"
            f"Raw response:\n{response_text}"
        ) from error


def build_config_prompt(
    experiment_request,
):
    """
    Build the prompt used to generate the AutoLSTM
    experiment configuration.
    """

    defaults_json = json.dumps(
        DEFAULT_CONFIG,
        indent=2,
    )

    search_defaults_json = json.dumps(
        DEFAULT_SEARCH_SPACE,
        indent=2,
    )

    prompt = f"""
You are a configuration-generation agent for a
time-series forecasting workflow.

Your job is NOT to train the model and NOT to analyze
forecast results.

Your only task is to convert a natural-language
experiment request into a valid JSON configuration
for a NeuralForecast AutoLSTM model using Optuna.

USER EXPERIMENT REQUEST:

{experiment_request}

DEFAULT EXPERIMENT VALUES:

{defaults_json}

DEFAULT SEARCH SPACE:

{search_defaults_json}

If the user does not specify one of the experiment
fields, use the corresponding default value exactly.

If the user does not explicitly specify a search-space
parameter, copy its default specification exactly.

Do not invent a new range for an unspecified parameter.

The output configuration must contain exactly these
top-level fields:

- model
- location
- target
- train_end
- test_start
- test_end
- horizon
- validation_size
- num_samples
- frequency
- search_space

SEARCH SPACE RULES:

Every search-space parameter must use one of these
four representations.

1. Fixed value:

{{
  "type": "fixed",
  "value": <value>
}}

2. Categorical values:

{{
  "type": "categorical",
  "values": [<value1>, <value2>, ...]
}}

3. Integer range:

{{
  "type": "int",
  "low": <integer>,
  "high": <integer>
}}

Optional:
- "step"
- "log"

4. Float range:

{{
  "type": "float",
  "low": <number>,
  "high": <number>
}}

Optional:
- "step"
- "log"

The search_space MUST contain all of these parameters:

- input_size
- inference_input_size
- encoder_hidden_size
- encoder_n_layers
- context_size
- decoder_hidden_size
- learning_rate
- max_steps
- batch_size
- random_seed
- scaler_type

GUIDANCE:

- model must currently be "LSTM".
- location should be "US" unless otherwise specified.
- target should be "weekly_rate" unless otherwise specified.
- use powers of two for input_size when the user asks
  for an exponential or power-of-two observation-window
  search.
- learning_rate should normally use a logarithmic float
  search when the user asks for logarithmic tuning.
- inference_input_size should default to -1 unless the
  user explicitly overrides it.
- scaler_type should default to "standard" unless the
  user explicitly overrides it.
- encoder_n_layers should normally use an integer range.
- random_seed should normally use an integer range.
- Do not invent unsupported parameter names.
- Do not add comments.
- Do not include Markdown.
- Do not explain your answer.
- Return ONLY valid JSON.

A valid example shape is:

{{
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
  "search_space": {{
    "input_size": {{
      "type": "categorical",
      "values": [8, 16, 32, 64, 128]
    }},
    "inference_input_size": {{
      "type": "fixed",
      "value": -1
    }},
    "encoder_hidden_size": {{
      "type": "categorical",
      "values": [16, 32, 64]
    }},
    "encoder_n_layers": {{
      "type": "int",
      "low": 1,
      "high": 3
    }},
    "context_size": {{
      "type": "categorical",
      "values": [5, 10]
    }},
    "decoder_hidden_size": {{
      "type": "categorical",
      "values": [16, 32, 64]
    }},
    "learning_rate": {{
      "type": "float",
      "low": 0.0001,
      "high": 0.01,
      "log": true
    }},
    "max_steps": {{
      "type": "categorical",
      "values": [300, 500, 1000]
    }},
    "batch_size": {{
      "type": "categorical",
      "values": [16, 32]
    }},
    "random_seed": {{
      "type": "int",
      "low": 1,
      "high": 20
    }},
    "scaler_type": {{
      "type": "fixed",
      "value": "standard"
    }}
  }}
}}
"""

    return prompt


def validate_parameter_spec(
    parameter_name,
    specification,
):
    """
    Deterministically validate one generated search-space
    parameter.
    """

    if not isinstance(
        specification,
        dict,
    ):
        raise ValueError(
            f"{parameter_name} must be an object."
        )

    parameter_type = specification.get(
        "type"
    )

    if parameter_type not in ALLOWED_PARAMETER_TYPES:
        raise ValueError(
            f"{parameter_name} has unsupported type: "
            f"{parameter_type}"
        )

    if parameter_type == "fixed":
        if "value" not in specification:
            raise ValueError(
                f"{parameter_name} fixed specification "
                "must contain 'value'."
            )

    elif parameter_type == "categorical":
        values = specification.get(
            "values"
        )

        if (
            not isinstance(values, list)
            or len(values) == 0
        ):
            raise ValueError(
                f"{parameter_name} categorical "
                "specification must contain a "
                "non-empty 'values' list."
            )

    elif parameter_type in {
        "int",
        "float",
    }:
        if (
            "low" not in specification
            or "high" not in specification
        ):
            raise ValueError(
                f"{parameter_name} {parameter_type} "
                "specification must contain "
                "'low' and 'high'."
            )

        if (
            specification["low"]
            >= specification["high"]
        ):
            raise ValueError(
                f"{parameter_name} must have low < high."
            )

        if (
            specification.get("log", False)
            and "step" in specification
        ):
            raise ValueError(
                f"{parameter_name} cannot use both "
                "'log' and 'step'."
            )


def validate_generated_config(config):
    """
    Validate the LLM-generated experiment configuration
    before writing it to disk.
    """

    required_top_level = {
        "model",
        "location",
        "target",
        "train_end",
        "test_start",
        "test_end",
        "horizon",
        "validation_size",
        "num_samples",
        "frequency",
        "search_space",
    }

    missing_fields = (
        required_top_level
        - set(config.keys())
    )

    if missing_fields:
        raise ValueError(
            "Generated config is missing fields: "
            f"{sorted(missing_fields)}"
        )

    extra_fields = (
        set(config.keys())
        - required_top_level
    )

    if extra_fields:
        raise ValueError(
            "Generated config contains unexpected "
            f"fields: {sorted(extra_fields)}"
        )

    if config["model"] != "LSTM":
        raise ValueError(
            "Generated config must use model='LSTM'."
        )

    search_space = config[
        "search_space"
    ]

    if not isinstance(
        search_space,
        dict,
    ):
        raise ValueError(
            "search_space must be an object."
        )

    missing_parameters = (
        REQUIRED_SEARCH_PARAMETERS
        - set(search_space.keys())
    )

    if missing_parameters:
        raise ValueError(
            "Generated search space is missing: "
            f"{sorted(missing_parameters)}"
        )

    extra_parameters = (
        set(search_space.keys())
        - REQUIRED_SEARCH_PARAMETERS
    )

    if extra_parameters:
        raise ValueError(
            "Generated search space contains "
            "unsupported parameters: "
            f"{sorted(extra_parameters)}"
        )

    for (
        parameter_name,
        specification,
    ) in search_space.items():
        validate_parameter_spec(
            parameter_name,
            specification,
        )

    return config


def generate_config(
    experiment_request,
):
    """
    Ask GPT to convert the natural-language experiment
    specification into structured JSON.
    """

    prompt = build_config_prompt(
        experiment_request
    )

    try:
        response = (
            openai_client.responses.create(
                model="gpt-5.4-mini",
                input=prompt,
            )
        )

        config = parse_json_response(
            response.output_text
        )

        return validate_generated_config(
            config
        )

    except Exception as error:
        raise RuntimeError(
            "AutoLSTM config agent failed: "
            f"{error}"
        ) from error


def save_config(
    config,
    output_path=OUTPUT_PATH,
):
    """
    Save a validated generated configuration to JSON.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=2,
        )

        file.write(
            "\n"
        )

    return output_path


def main():
    experiment_request = """
Train an LSTM forecasting model using US FluSight
weekly hospitalization-rate data.

Use data through September 2025 for training and use
October 2025 through May 2026 as the held-out test
period.

Forecast 1 to 4 weeks ahead.

For the observation window, search powers of two from
8 through 128.

Search encoder and decoder hidden sizes among
16, 32, and 64.

Search between 1 and 3 encoder layers.

Search learning rates between 0.0001 and 0.01 on a
logarithmic scale.

Search training lengths of 300, 500, and 1000 steps.

Use batch sizes 16 and 32.

Search random seeds from 1 through 20.

Use standard scaling.

Use five Optuna trials for this initial experiment.
"""

    print(
        "--- CONFIG GENERATION AGENT ---"
    )

    print(
        "\nExperiment request:"
    )

    print(
        experiment_request.strip()
    )

    generated_config = generate_config(
        experiment_request
    )

    output_path = save_config(
        generated_config
    )

    print(
        "\n--- GENERATED CONFIG ---"
    )

    print(
        json.dumps(
            generated_config,
            indent=2,
        )
    )

    print(
        "\nSaved configuration:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()