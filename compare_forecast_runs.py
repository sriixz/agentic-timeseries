import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASELINE_FORECAST_PATH = Path(
    "forecast_results/autolstm_fixed_test_forecasts.csv"
)

REFINED_FORECAST_PATH = Path(
    "forecast_results/refined_autolstm_test_forecasts.csv"
)

OUTPUT_DIR = Path("forecast_results")
PLOT_DIR = Path("plots")

COMPARISON_CSV_PATH = (
    OUTPUT_DIR
    / "baseline_vs_refined_comparison.csv"
)

SUMMARY_JSON_PATH = (
    OUTPUT_DIR
    / "baseline_vs_refined_summary.json"
)

PLOT_PATH = (
    PLOT_DIR
    / "baseline_vs_refined_comparison.png"
)


def load_baseline_forecasts() -> pd.DataFrame:
    if not BASELINE_FORECAST_PATH.exists():
        raise FileNotFoundError(
            f"Baseline forecast file not found: "
            f"{BASELINE_FORECAST_PATH}"
        )

    df = pd.read_csv(
        BASELINE_FORECAST_PATH,
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
            "Baseline forecast file missing columns: "
            f"{sorted(missing)}"
        )

    return df


def load_refined_forecasts() -> pd.DataFrame:
    if not REFINED_FORECAST_PATH.exists():
        raise FileNotFoundError(
            f"Refined forecast file not found: "
            f"{REFINED_FORECAST_PATH}"
        )

    df = pd.read_csv(
        REFINED_FORECAST_PATH,
        parse_dates=["ds", "cutoff"],
    )

    required = {
        "ds",
        "y",
        "raw_forecast",
        "forecast",
        "horizon",
        "was_clipped",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Refined forecast file missing columns: "
            f"{sorted(missing)}"
        )

    return df


def calculate_horizon_metrics(
    forecasts: pd.DataFrame,
    forecast_column: str,
) -> pd.DataFrame:
    rows = []

    for horizon, group in forecasts.groupby("horizon"):
        group = group.copy()

        error = (
            group[forecast_column]
            - group["y"]
        )

        mae = float(
            error.abs().mean()
        )

        rmse = float(
            error.pow(2).mean() ** 0.5
        )

        bias = float(
            error.mean()
        )

        actual_peak_row = group.loc[
            group["y"].idxmax()
        ]

        forecast_peak_row = group.loc[
            group[forecast_column].idxmax()
        ]

        actual_peak_value = float(
            actual_peak_row["y"]
        )

        forecast_peak_value = float(
            forecast_peak_row[
                forecast_column
            ]
        )

        peak_magnitude_error = (
            forecast_peak_value
            - actual_peak_value
        )

        peak_timing_error_weeks = (
            (
                forecast_peak_row["ds"]
                - actual_peak_row["ds"]
            ).days
            / 7.0
        )

        negative_count = int(
            (
                group[forecast_column] < 0
            ).sum()
        )

        rows.append(
            {
                "horizon": int(horizon),
                "forecast_count": len(group),
                "mae": mae,
                "rmse": rmse,
                "bias": bias,
                "peak_magnitude_error": (
                    peak_magnitude_error
                ),
                "peak_timing_error_weeks": (
                    peak_timing_error_weeks
                ),
                "negative_forecast_count": (
                    negative_count
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("horizon")
        .reset_index(drop=True)
    )


def percent_improvement(
    baseline: float,
    refined: float,
) -> float:
    if baseline == 0:
        return 0.0

    return (
        (baseline - refined)
        / abs(baseline)
        * 100.0
    )


def build_comparison(
    baseline_metrics: pd.DataFrame,
    refined_raw_metrics: pd.DataFrame,
    refined_corrected_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for horizon in baseline_metrics["horizon"]:
        baseline = baseline_metrics[
            baseline_metrics["horizon"]
            == horizon
        ].iloc[0]

        refined_raw = refined_raw_metrics[
            refined_raw_metrics["horizon"]
            == horizon
        ].iloc[0]

        refined = refined_corrected_metrics[
            refined_corrected_metrics["horizon"]
            == horizon
        ].iloc[0]

        baseline_peak_abs = abs(
            baseline["peak_magnitude_error"]
        )

        refined_peak_abs = abs(
            refined["peak_magnitude_error"]
        )

        baseline_timing_abs = abs(
            baseline[
                "peak_timing_error_weeks"
            ]
        )

        refined_timing_abs = abs(
            refined[
                "peak_timing_error_weeks"
            ]
        )

        rows.append(
            {
                "horizon": int(horizon),

                "baseline_mae": (
                    baseline["mae"]
                ),
                "refined_mae": (
                    refined["mae"]
                ),
                "mae_improvement_pct": (
                    percent_improvement(
                        baseline["mae"],
                        refined["mae"],
                    )
                ),

                "baseline_rmse": (
                    baseline["rmse"]
                ),
                "refined_rmse": (
                    refined["rmse"]
                ),
                "rmse_improvement_pct": (
                    percent_improvement(
                        baseline["rmse"],
                        refined["rmse"],
                    )
                ),

                "baseline_bias": (
                    baseline["bias"]
                ),
                "refined_bias": (
                    refined["bias"]
                ),

                "baseline_peak_magnitude_error":
                    baseline[
                        "peak_magnitude_error"
                    ],

                "refined_peak_magnitude_error":
                    refined[
                        "peak_magnitude_error"
                    ],

                "peak_magnitude_abs_improvement_pct":
                    percent_improvement(
                        baseline_peak_abs,
                        refined_peak_abs,
                    ),

                "baseline_peak_timing_error_weeks":
                    baseline[
                        "peak_timing_error_weeks"
                    ],

                "refined_peak_timing_error_weeks":
                    refined[
                        "peak_timing_error_weeks"
                    ],

                "peak_timing_abs_change_weeks": (
                    refined_timing_abs
                    - baseline_timing_abs
                ),

                "baseline_negative_predictions":
                    int(
                        baseline[
                            "negative_forecast_count"
                        ]
                    ),

                "refined_raw_negative_predictions":
                    int(
                        refined_raw[
                            "negative_forecast_count"
                        ]
                    ),

                "refined_corrected_negative_predictions":
                    int(
                        refined[
                            "negative_forecast_count"
                        ]
                    ),
            }
        )

    return pd.DataFrame(rows)


def build_summary(
    comparison: pd.DataFrame,
) -> dict:
    mae_improved = int(
        (
            comparison["refined_mae"]
            < comparison["baseline_mae"]
        ).sum()
    )

    rmse_improved = int(
        (
            comparison["refined_rmse"]
            < comparison["baseline_rmse"]
        ).sum()
    )

    peak_magnitude_improved = int(
        (
            comparison[
                "refined_peak_magnitude_error"
            ].abs()
            <
            comparison[
                "baseline_peak_magnitude_error"
            ].abs()
        ).sum()
    )

    timing_improved = int(
        (
            comparison[
                "refined_peak_timing_error_weeks"
            ].abs()
            <
            comparison[
                "baseline_peak_timing_error_weeks"
            ].abs()
        ).sum()
    )

    timing_worsened = int(
        (
            comparison[
                "refined_peak_timing_error_weeks"
            ].abs()
            >
            comparison[
                "baseline_peak_timing_error_weeks"
            ].abs()
        ).sum()
    )

    baseline_negatives = int(
        comparison[
            "baseline_negative_predictions"
        ].sum()
    )

    refined_raw_negatives = int(
        comparison[
            "refined_raw_negative_predictions"
        ].sum()
    )

    refined_corrected_negatives = int(
        comparison[
            "refined_corrected_negative_predictions"
        ].sum()
    )

    return {
        "horizons_compared": int(
            len(comparison)
        ),
        "mae_improved_horizons": (
            mae_improved
        ),
        "rmse_improved_horizons": (
            rmse_improved
        ),
        "peak_magnitude_improved_horizons": (
            peak_magnitude_improved
        ),
        "peak_timing_improved_horizons": (
            timing_improved
        ),
        "peak_timing_worsened_horizons": (
            timing_worsened
        ),
        "baseline_total_negative_predictions":
            baseline_negatives,
        "refined_raw_total_negative_predictions":
            refined_raw_negatives,
        "refined_corrected_total_negative_predictions":
            refined_corrected_negatives,
        "average_mae_improvement_pct": float(
            comparison[
                "mae_improvement_pct"
            ].mean()
        ),
        "average_rmse_improvement_pct": float(
            comparison[
                "rmse_improvement_pct"
            ].mean()
        ),
        "average_peak_magnitude_improvement_pct":
            float(
                comparison[
                    "peak_magnitude_abs_improvement_pct"
                ].mean()
            ),
    }


def save_results(
    comparison: pd.DataFrame,
    summary: dict,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        COMPARISON_CSV_PATH,
        index=False,
    )

    with SUMMARY_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print(
        f"\nSaved: {COMPARISON_CSV_PATH}"
    )
    print(
        f"Saved: {SUMMARY_JSON_PATH}"
    )


def plot_comparison(
    comparison: pd.DataFrame,
) -> None:
    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    horizons = comparison[
        "horizon"
    ].tolist()

    x = list(range(len(horizons)))

    width = 0.35

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 10),
    )

    axes[0].bar(
        [
            value - width / 2
            for value in x
        ],
        comparison["baseline_mae"],
        width=width,
        label="Baseline",
    )

    axes[0].bar(
        [
            value + width / 2
            for value in x
        ],
        comparison["refined_mae"],
        width=width,
        label="Refined",
    )

    axes[0].set_xticks(
        x,
        [
            f"{h}-week"
            for h in horizons
        ],
    )

    axes[0].set_ylabel("MAE")

    axes[0].set_title(
        "Held-Out MAE: Baseline vs Refined AutoLSTM"
    )

    axes[0].legend()
    axes[0].grid(
        axis="y",
        alpha=0.25,
    )

    axes[1].bar(
        [
            value - width / 2
            for value in x
        ],
        comparison[
            "baseline_peak_magnitude_error"
        ].abs(),
        width=width,
        label="Baseline",
    )

    axes[1].bar(
        [
            value + width / 2
            for value in x
        ],
        comparison[
            "refined_peak_magnitude_error"
        ].abs(),
        width=width,
        label="Refined",
    )

    axes[1].set_xticks(
        x,
        [
            f"{h}-week"
            for h in horizons
        ],
    )

    axes[1].set_ylabel(
        "Absolute peak magnitude error"
    )

    axes[1].set_title(
        "Peak Magnitude Error: "
        "Baseline vs Refined AutoLSTM"
    )

    axes[1].legend()
    axes[1].grid(
        axis="y",
        alpha=0.25,
    )

    fig.suptitle(
        "Agentic Forecast Refinement Comparison",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        PLOT_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {PLOT_PATH}"
    )


def print_comparison(
    comparison: pd.DataFrame,
    summary: dict,
) -> None:
    print(
        "\nBaseline vs Refined Forecast Comparison"
    )
    print(
        "======================================="
    )

    for row in comparison.itertuples():
        print(
            f"\n{row.horizon}-week horizon"
        )

        print(
            f"  MAE: "
            f"{row.baseline_mae:.4f} "
            f"-> {row.refined_mae:.4f} "
            f"("
            f"{row.mae_improvement_pct:+.1f}% "
            f"improvement)"
        )

        print(
            f"  RMSE: "
            f"{row.baseline_rmse:.4f} "
            f"-> {row.refined_rmse:.4f} "
            f"("
            f"{row.rmse_improvement_pct:+.1f}% "
            f"improvement)"
        )

        print(
            f"  Bias: "
            f"{row.baseline_bias:+.4f} "
            f"-> {row.refined_bias:+.4f}"
        )

        print(
            f"  Peak magnitude error: "
            f"{row.baseline_peak_magnitude_error:+.4f} "
            f"-> "
            f"{row.refined_peak_magnitude_error:+.4f}"
        )

        print(
            f"  Peak timing error: "
            f"{row.baseline_peak_timing_error_weeks:+.1f} "
            f"-> "
            f"{row.refined_peak_timing_error_weeks:+.1f} "
            f"weeks"
        )

        print(
            f"  Negative forecasts: "
            f"baseline="
            f"{row.baseline_negative_predictions}, "
            f"refined raw="
            f"{row.refined_raw_negative_predictions}, "
            f"refined corrected="
            f"{row.refined_corrected_negative_predictions}"
        )

    print("\nOverall Summary")
    print("===============")

    print(
        f"MAE improved on "
        f"{summary['mae_improved_horizons']}/"
        f"{summary['horizons_compared']} horizons."
    )

    print(
        f"RMSE improved on "
        f"{summary['rmse_improved_horizons']}/"
        f"{summary['horizons_compared']} horizons."
    )

    print(
        f"Peak magnitude error improved on "
        f"{summary['peak_magnitude_improved_horizons']}/"
        f"{summary['horizons_compared']} horizons."
    )

    print(
        f"Peak timing improved on "
        f"{summary['peak_timing_improved_horizons']}/"
        f"{summary['horizons_compared']} horizons "
        f"and worsened on "
        f"{summary['peak_timing_worsened_horizons']}/"
        f"{summary['horizons_compared']}."
    )

    print(
        f"Average MAE improvement: "
        f"{summary['average_mae_improvement_pct']:.1f}%"
    )

    print(
        f"Average RMSE improvement: "
        f"{summary['average_rmse_improvement_pct']:.1f}%"
    )

    print(
        f"Average peak magnitude improvement: "
        f"{summary['average_peak_magnitude_improvement_pct']:.1f}%"
    )

    print(
        f"Negative predictions: "
        f"baseline="
        f"{summary['baseline_total_negative_predictions']}, "
        f"refined raw="
        f"{summary['refined_raw_total_negative_predictions']}, "
        f"refined corrected="
        f"{summary['refined_corrected_total_negative_predictions']}"
    )


def main() -> None:
    baseline_forecasts = (
        load_baseline_forecasts()
    )

    refined_forecasts = (
        load_refined_forecasts()
    )

    baseline_metrics = (
        calculate_horizon_metrics(
            forecasts=baseline_forecasts,
            forecast_column="forecast",
        )
    )

    refined_raw_metrics = (
        calculate_horizon_metrics(
            forecasts=refined_forecasts,
            forecast_column="raw_forecast",
        )
    )

    refined_corrected_metrics = (
        calculate_horizon_metrics(
            forecasts=refined_forecasts,
            forecast_column="forecast",
        )
    )

    comparison = build_comparison(
        baseline_metrics=baseline_metrics,
        refined_raw_metrics=refined_raw_metrics,
        refined_corrected_metrics=(
            refined_corrected_metrics
        ),
    )

    summary = build_summary(
        comparison
    )

    save_results(
        comparison=comparison,
        summary=summary,
    )

    plot_comparison(
        comparison
    )

    print_comparison(
        comparison=comparison,
        summary=summary,
    )


if __name__ == "__main__":
    main()