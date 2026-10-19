from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULT_DIR = Path("forecast_results")
PLOT_DIR = Path("plots")

BASELINE_FORECASTS = (
    RESULT_DIR
    / "peak_window_baseline_forecasts.csv"
)

POINT_FORECASTS = (
    RESULT_DIR
    / "peak_window_point_forecasts.csv"
)

WINDOW_FORECASTS = (
    RESULT_DIR
    / "peak_window_next4w_forecasts.csv"
)

OUTPUT_CSV = (
    RESULT_DIR
    / "peak_feature_variant_comparison.csv"
)

OUTPUT_PLOT = (
    PLOT_DIR
    / "peak_feature_variant_comparison.png"
)


def load_forecasts(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required forecast file not found: {path}"
        )

    df = pd.read_csv(
        path,
        parse_dates=[
            "ds",
            "cutoff",
        ],
    )

    required = {
        "ds",
        "y",
        "raw_forecast",
        "forecast",
        "horizon",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{path} missing columns: "
            f"{sorted(missing)}"
        )

    return df


def calculate_metrics(
    forecasts: pd.DataFrame,
    forecast_column: str,
) -> pd.DataFrame:
    rows = []

    for horizon, group in (
        forecasts.groupby(
            "horizon"
        )
    ):
        error = (
            group[
                forecast_column
            ]
            - group["y"]
        )

        mae = float(
            error.abs().mean()
        )

        rmse = float(
            error.pow(2)
            .mean()
            ** 0.5
        )

        bias = float(
            error.mean()
        )

        actual_peak = (
            group.loc[
                group[
                    "y"
                ].idxmax()
            ]
        )

        forecast_peak = (
            group.loc[
                group[
                    forecast_column
                ].idxmax()
            ]
        )

        peak_magnitude_error = (
            float(
                forecast_peak[
                    forecast_column
                ]
            )
            - float(
                actual_peak["y"]
            )
        )

        peak_timing_error_weeks = (
            (
                forecast_peak["ds"]
                - actual_peak["ds"]
            ).days
            / 7.0
        )

        negative_count = int(
            (
                group[
                    forecast_column
                ]
                < 0
            ).sum()
        )

        rows.append(
            {
                "horizon": int(
                    horizon
                ),
                "mae": mae,
                "rmse": rmse,
                "bias": bias,
                "peak_magnitude_error":
                    peak_magnitude_error,
                "peak_timing_error_weeks":
                    peak_timing_error_weeks,
                "negative_forecast_count":
                    negative_count,
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "horizon"
        )
        .reset_index(
            drop=True
        )
    )


def add_variant_name(
    metrics: pd.DataFrame,
    variant: str,
) -> pd.DataFrame:
    result = (
        metrics.copy()
    )

    result[
        "variant"
    ] = variant

    return result


def build_all_metrics(
    baseline_forecasts: pd.DataFrame,
    point_forecasts: pd.DataFrame,
    window_forecasts: pd.DataFrame,
) -> pd.DataFrame:
    baseline_metrics = (
        calculate_metrics(
            forecasts=(
                baseline_forecasts
            ),
            forecast_column=(
                "raw_forecast"
            ),
        )
    )

    point_metrics = (
        calculate_metrics(
            forecasts=(
                point_forecasts
            ),
            forecast_column=(
                "raw_forecast"
            ),
        )
    )

    window_raw_metrics = (
        calculate_metrics(
            forecasts=(
                window_forecasts
            ),
            forecast_column=(
                "raw_forecast"
            ),
        )
    )

    window_clipped_metrics = (
        calculate_metrics(
            forecasts=(
                window_forecasts
            ),
            forecast_column=(
                "forecast"
            ),
        )
    )

    combined = pd.concat(
        [
            add_variant_name(
                baseline_metrics,
                "baseline",
            ),
            add_variant_name(
                point_metrics,
                "exact_4w_feature",
            ),
            add_variant_name(
                window_raw_metrics,
                "next4w_raw",
            ),
            add_variant_name(
                window_clipped_metrics,
                "next4w_clipped",
            ),
        ],
        ignore_index=True,
    )

    return combined[
        [
            "variant",
            "horizon",
            "mae",
            "rmse",
            "bias",
            "peak_magnitude_error",
            "peak_timing_error_weeks",
            "negative_forecast_count",
        ]
    ]


def print_comparison(
    metrics: pd.DataFrame,
) -> None:
    print(
        "\nPeak Feature Variant Comparison"
    )
    print(
        "==============================="
    )

    for horizon in sorted(
        metrics[
            "horizon"
        ].unique()
    ):
        print(
            f"\n{horizon}-week horizon"
        )

        subset = metrics[
            metrics[
                "horizon"
            ]
            == horizon
        ]

        for row in (
            subset.itertuples()
        ):
            print(
                f"\n  {row.variant}"
            )

            print(
                f"    MAE: "
                f"{row.mae:.4f}"
            )

            print(
                f"    RMSE: "
                f"{row.rmse:.4f}"
            )

            print(
                f"    Bias: "
                f"{row.bias:+.4f}"
            )

            print(
                "    Peak magnitude error: "
                f"{row.peak_magnitude_error:+.4f}"
            )

            print(
                "    Peak timing error: "
                f"{row.peak_timing_error_weeks:+.1f} "
                "weeks"
            )

            print(
                "    Negative forecasts: "
                f"{row.negative_forecast_count}"
            )


def print_summary(
    metrics: pd.DataFrame,
) -> None:
    print(
        "\nSummary"
    )
    print(
        "======="
    )

    for variant in [
        "baseline",
        "exact_4w_feature",
        "next4w_raw",
        "next4w_clipped",
    ]:
        subset = metrics[
            metrics[
                "variant"
            ]
            == variant
        ]

        print(
            f"\n{variant}"
        )

        print(
            "  Average MAE: "
            f"{subset['mae'].mean():.4f}"
        )

        print(
            "  Average RMSE: "
            f"{subset['rmse'].mean():.4f}"
        )

        print(
            "  Average absolute peak "
            "magnitude error: "
            f"{subset['peak_magnitude_error'].abs().mean():.4f}"
        )

        print(
            "  Total negative forecasts: "
            f"{int(subset['negative_forecast_count'].sum())}"
        )


def save_results(
    metrics: pd.DataFrame,
) -> None:
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(
        f"\nSaved: {OUTPUT_CSV}"
    )


def plot_comparison(
    metrics: pd.DataFrame,
) -> None:
    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    horizons = sorted(
        metrics[
            "horizon"
        ].unique()
    )

    variants = [
        "baseline",
        "exact_4w_feature",
        "next4w_raw",
        "next4w_clipped",
    ]

    labels = {
        "baseline":
            "Baseline",
        "exact_4w_feature":
            "Exact 4-week",
        "next4w_raw":
            "Next-4-week raw",
        "next4w_clipped":
            "Next-4-week clipped",
    }

    x = list(
        range(
            len(horizons)
        )
    )

    width = 0.18

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(
            11,
            10,
        ),
    )

    offsets = [
        -1.5,
        -0.5,
        0.5,
        1.5,
    ]

    for offset, variant in zip(
        offsets,
        variants,
    ):
        subset = (
            metrics[
                metrics[
                    "variant"
                ]
                == variant
            ]
            .sort_values(
                "horizon"
            )
        )

        axes[0].bar(
            [
                value
                + offset
                * width
                for value
                in x
            ],
            subset["rmse"],
            width=width,
            label=labels[
                variant
            ],
        )

        axes[1].bar(
            [
                value
                + offset
                * width
                for value
                in x
            ],
            subset[
                "peak_magnitude_error"
            ].abs(),
            width=width,
            label=labels[
                variant
            ],
        )

    axes[0].set_xticks(
        x,
        [
            f"{h}-week"
            for h
            in horizons
        ],
    )

    axes[0].set_ylabel(
        "RMSE"
    )

    axes[0].set_title(
        "RMSE by Forecast Horizon"
    )

    axes[0].legend()

    axes[0].grid(
        axis="y",
        alpha=0.25,
    )

    axes[1].set_xticks(
        x,
        [
            f"{h}-week"
            for h
            in horizons
        ],
    )

    axes[1].set_ylabel(
        "Absolute peak magnitude error"
    )

    axes[1].set_title(
        "Peak Magnitude Error "
        "by Forecast Horizon"
    )

    axes[1].legend()

    axes[1].grid(
        axis="y",
        alpha=0.25,
    )

    fig.suptitle(
        "Peak-Timing Feature Variants"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_PLOT,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {OUTPUT_PLOT}"
    )


def main():
    baseline_forecasts = (
        load_forecasts(
            BASELINE_FORECASTS
        )
    )

    point_forecasts = (
        load_forecasts(
            POINT_FORECASTS
        )
    )

    window_forecasts = (
        load_forecasts(
            WINDOW_FORECASTS
        )
    )

    metrics = (
        build_all_metrics(
            baseline_forecasts=(
                baseline_forecasts
            ),
            point_forecasts=(
                point_forecasts
            ),
            window_forecasts=(
                window_forecasts
            ),
        )
    )

    print_comparison(
        metrics
    )

    print_summary(
        metrics
    )

    save_results(
        metrics
    )

    plot_comparison(
        metrics
    )


if __name__ == "__main__":
    main()