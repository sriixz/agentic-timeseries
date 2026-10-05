from config_agent import generate_config, save_config
from autolstm_forecast import main as run_forecast


EXPERIMENT_REQUEST = """
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


def main():
    print(
        "--- AGENTIC LSTM WORKFLOW ---"
    )

    print(
        "\nGenerating experiment configuration..."
    )

    config = generate_config(
        EXPERIMENT_REQUEST
    )

    config_path = save_config(
        config
    )

    print(
        f"\nGenerated config: {config_path}"
    )

    print(
        "\nStarting AutoLSTM training..."
    )

    run_forecast()


if __name__ == "__main__":
    main()