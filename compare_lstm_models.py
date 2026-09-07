from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("forecast_results")
PLOTS_DIR = Path("plots")

MANUAL_METRICS_PATH = (
    RESULTS_DIR / "lstm_fixed_metrics_by_horizon.csv"
)

AUTOLSTM_METRICS_PATH = (
    RESULTS_DIR / "autolstm_fixed_metrics_by_horizon.csv"
)

COMPARISON_CSV_PATH = (
    RESULTS_DIR / "lstm_vs_autolstm_comparison.csv"
)

MAE_PLOT_PATH = (
    PLOTS_DIR / "lstm_vs_autolstm_mae.png"
)

RMSE_PLOT_PATH = (
    PLOTS_DIR / "lstm_vs_autolstm_rmse.png"
)


def load_metrics():
    """
    Load the manual LSTM and AutoLSTM fixed-weight
    evaluation metrics.
    """

    manual_df = pd.read_csv(
        MANUAL_METRICS_PATH
    )

    auto_df = pd.read_csv(
        AUTOLSTM_METRICS_PATH
    )

    return manual_df, auto_df


def build_comparison(
    manual_df,
    auto_df,
):
    """
    Merge both model results by forecast horizon and
    calculate percent change from manual LSTM to AutoLSTM.

    Positive improvement means AutoLSTM reduced error.
    Negative improvement means AutoLSTM performed worse.
    """

    manual = manual_df.rename(
        columns={
            "forecast_count":
                "manual_forecast_count",
            "mae":
                "manual_mae",
            "rmse":
                "manual_rmse",
        }
    )

    auto = auto_df.rename(
        columns={
            "forecast_count":
                "autolstm_forecast_count",
            "mae":
                "autolstm_mae",
            "rmse":
                "autolstm_rmse",
        }
    )

    comparison = manual.merge(
        auto,
        on="horizon_weeks",
        how="inner",
    )

    comparison[
        "mae_improvement_percent"
    ] = (
        (
            comparison["manual_mae"]
            - comparison["autolstm_mae"]
        )
        / comparison["manual_mae"]
        * 100
    )

    comparison[
        "rmse_improvement_percent"
    ] = (
        (
            comparison["manual_rmse"]
            - comparison["autolstm_rmse"]
        )
        / comparison["manual_rmse"]
        * 100
    )

    return comparison


def print_comparison(comparison):
    """
    Print a readable horizon-by-horizon comparison.
    """

    print(
        "\n--- MANUAL LSTM VS AUTOLSTM ---"
    )

    display_columns = [
        "horizon_weeks",
        "manual_mae",
        "autolstm_mae",
        "mae_improvement_percent",
        "manual_rmse",
        "autolstm_rmse",
        "rmse_improvement_percent",
    ]

    print(
        comparison[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        "\n--- INTERPRETATION ---"
    )

    for _, row in comparison.iterrows():
        horizon = int(
            row["horizon_weeks"]
        )

        mae_change = (
            row["mae_improvement_percent"]
        )

        rmse_change = (
            row["rmse_improvement_percent"]
        )

        if mae_change >= 0:
            mae_text = (
                f"MAE improved by {mae_change:.2f}%"
            )
        else:
            mae_text = (
                f"MAE worsened by "
                f"{abs(mae_change):.2f}%"
            )

        if rmse_change >= 0:
            rmse_text = (
                f"RMSE improved by {rmse_change:.2f}%"
            )
        else:
            rmse_text = (
                f"RMSE worsened by "
                f"{abs(rmse_change):.2f}%"
            )

        print(
            f"{horizon}-week horizon: "
            f"{mae_text}; {rmse_text}"
        )


def plot_mae(comparison):
    """
    Compare MAE across forecast horizons.
    """

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        comparison["horizon_weeks"],
        comparison["manual_mae"],
        marker="o",
        label="Manual LSTM",
    )

    plt.plot(
        comparison["horizon_weeks"],
        comparison["autolstm_mae"],
        marker="o",
        label="AutoLSTM",
    )

    plt.xticks(
        comparison["horizon_weeks"]
    )

    plt.xlabel(
        "Forecast horizon (weeks)"
    )

    plt.ylabel(
        "MAE"
    )

    plt.title(
        "Manual LSTM vs AutoLSTM: MAE by Horizon"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        MAE_PLOT_PATH,
        dpi=150,
    )

    plt.close()

    return MAE_PLOT_PATH


def plot_rmse(comparison):
    """
    Compare RMSE across forecast horizons.
    """

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        comparison["horizon_weeks"],
        comparison["manual_rmse"],
        marker="o",
        label="Manual LSTM",
    )

    plt.plot(
        comparison["horizon_weeks"],
        comparison["autolstm_rmse"],
        marker="o",
        label="AutoLSTM",
    )

    plt.xticks(
        comparison["horizon_weeks"]
    )

    plt.xlabel(
        "Forecast horizon (weeks)"
    )

    plt.ylabel(
        "RMSE"
    )

    plt.title(
        "Manual LSTM vs AutoLSTM: RMSE by Horizon"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RMSE_PLOT_PATH,
        dpi=150,
    )

    plt.close()

    return RMSE_PLOT_PATH


def save_comparison(comparison):
    """
    Save the combined metrics table.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        COMPARISON_CSV_PATH,
        index=False,
    )

    return COMPARISON_CSV_PATH


def main():
    print(
        "--- LOADING MODEL METRICS ---"
    )

    manual_df, auto_df = load_metrics()

    print(
        f"Manual LSTM rows: {len(manual_df)}"
    )

    print(
        f"AutoLSTM rows: {len(auto_df)}"
    )

    comparison = build_comparison(
        manual_df,
        auto_df,
    )

    print_comparison(
        comparison
    )

    comparison_path = save_comparison(
        comparison
    )

    mae_plot = plot_mae(
        comparison
    )

    rmse_plot = plot_rmse(
        comparison
    )

    print(
        "\n--- SAVED OUTPUTS ---"
    )

    print(
        f"comparison: {comparison_path}"
    )

    print(
        f"mae_plot: {mae_plot}"
    )

    print(
        f"rmse_plot: {rmse_plot}"
    )


if __name__ == "__main__":
    main()