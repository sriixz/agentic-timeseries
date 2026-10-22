from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULT_DIR = Path("forecast_results")
PLOT_DIR = Path("plots")


LSTM_SUMMARY_PATH = (
    RESULT_DIR
    / "multilocation_peak_feature_4w_summary.csv"
)

INFORMER_SUMMARY_PATH = (
    RESULT_DIR
    / "multilocation_informer_4w_summary.csv"
)

OUTPUT_CSV = (
    RESULT_DIR
    / "lstm_vs_informer_peak_feature_4w.csv"
)

OUTPUT_PLOT = (
    PLOT_DIR
    / "lstm_vs_informer_peak_feature_4w.png"
)


LOCATION_ORDER = [
    "US",
    "CA",
    "FL",
    "NY",
    "TX",
]


def load_summary(
    path: Path,
    model_name: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required summary file not found: {path}"
        )

    df = pd.read_csv(
        path
    )

    required = {
        "state_abbr",
        "baseline_rmse",
        "feature_rmse",
        "baseline_peak_magnitude_error",
        "feature_peak_magnitude_error",
        "baseline_peak_timing_error_weeks",
        "feature_peak_timing_error_weeks",
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

    df = df[
        df[
            "state_abbr"
        ].isin(
            LOCATION_ORDER
        )
    ].copy()

    df[
        "state_abbr"
    ] = pd.Categorical(
        df[
            "state_abbr"
        ],
        categories=LOCATION_ORDER,
        ordered=True,
    )

    df = (
        df
        .sort_values(
            "state_abbr"
        )
        .reset_index(
            drop=True
        )
    )

    rename_map = {
        "baseline_rmse":
            f"{model_name}_baseline_rmse",

        "feature_rmse":
            f"{model_name}_feature_rmse",

        "baseline_peak_magnitude_error":
            (
                f"{model_name}_baseline_"
                "peak_magnitude_error"
            ),

        "feature_peak_magnitude_error":
            (
                f"{model_name}_feature_"
                "peak_magnitude_error"
            ),

        "baseline_peak_timing_error_weeks":
            (
                f"{model_name}_baseline_"
                "peak_timing_error_weeks"
            ),

        "feature_peak_timing_error_weeks":
            (
                f"{model_name}_feature_"
                "peak_timing_error_weeks"
            ),
    }

    return df.rename(
        columns=rename_map
    )


def build_comparison() -> pd.DataFrame:
    lstm = load_summary(
        LSTM_SUMMARY_PATH,
        "lstm",
    )

    informer = load_summary(
        INFORMER_SUMMARY_PATH,
        "informer",
    )

    comparison = lstm.merge(
        informer,
        on="state_abbr",
        how="inner",
        suffixes=(
            "_lstm",
            "_informer",
        ),
    )

    columns = [
        "state_abbr",

        "lstm_baseline_rmse",
        "lstm_feature_rmse",

        "informer_baseline_rmse",
        "informer_feature_rmse",

        "lstm_baseline_peak_magnitude_error",
        "lstm_feature_peak_magnitude_error",

        "informer_baseline_peak_magnitude_error",
        "informer_feature_peak_magnitude_error",

        "lstm_baseline_peak_timing_error_weeks",
        "lstm_feature_peak_timing_error_weeks",

        "informer_baseline_peak_timing_error_weeks",
        "informer_feature_peak_timing_error_weeks",
    ]

    comparison = comparison[
        columns
    ].copy()

    comparison[
        "lstm_rmse_improvement_pct"
    ] = (
        (
            comparison[
                "lstm_baseline_rmse"
            ]
            - comparison[
                "lstm_feature_rmse"
            ]
        )
        / comparison[
            "lstm_baseline_rmse"
        ]
        * 100
    )

    comparison[
        "informer_rmse_improvement_pct"
    ] = (
        (
            comparison[
                "informer_baseline_rmse"
            ]
            - comparison[
                "informer_feature_rmse"
            ]
        )
        / comparison[
            "informer_baseline_rmse"
        ]
        * 100
    )

    comparison[
        "lstm_peak_error_improvement"
    ] = (
        comparison[
            "lstm_baseline_peak_magnitude_error"
        ].abs()
        - comparison[
            "lstm_feature_peak_magnitude_error"
        ].abs()
    )

    comparison[
        "informer_peak_error_improvement"
    ] = (
        comparison[
            "informer_baseline_peak_magnitude_error"
        ].abs()
        - comparison[
            "informer_feature_peak_magnitude_error"
        ].abs()
    )

    return comparison


def print_summary(
    comparison: pd.DataFrame,
) -> None:
    print(
        "\nLSTM vs Informer Peak-Feature Comparison"
    )

    print(
        "======================================="
    )

    for row in (
        comparison.itertuples()
    ):
        print(
            f"\n{row.state_abbr}"
        )

        print(
            "  LSTM RMSE: "
            f"{row.lstm_baseline_rmse:.3f} "
            "-> "
            f"{row.lstm_feature_rmse:.3f} "
            f"({row.lstm_rmse_improvement_pct:+.1f}%)"
        )

        print(
            "  Informer RMSE: "
            f"{row.informer_baseline_rmse:.3f} "
            "-> "
            f"{row.informer_feature_rmse:.3f} "
            f"({row.informer_rmse_improvement_pct:+.1f}%)"
        )

        print(
            "  LSTM peak error: "
            f"{row.lstm_baseline_peak_magnitude_error:+.3f} "
            "-> "
            f"{row.lstm_feature_peak_magnitude_error:+.3f}"
        )

        print(
            "  Informer peak error: "
            f"{row.informer_baseline_peak_magnitude_error:+.3f} "
            "-> "
            f"{row.informer_feature_peak_magnitude_error:+.3f}"
        )

        print(
            "  LSTM timing: "
            f"{row.lstm_baseline_peak_timing_error_weeks:+.1f} "
            "-> "
            f"{row.lstm_feature_peak_timing_error_weeks:+.1f}"
        )

        print(
            "  Informer timing: "
            f"{row.informer_baseline_peak_timing_error_weeks:+.1f} "
            "-> "
            f"{row.informer_feature_peak_timing_error_weeks:+.1f}"
        )

    lstm_improved = int(
        (
            comparison[
                "lstm_feature_rmse"
            ]
            < comparison[
                "lstm_baseline_rmse"
            ]
        ).sum()
    )

    informer_improved = int(
        (
            comparison[
                "informer_feature_rmse"
            ]
            < comparison[
                "informer_baseline_rmse"
            ]
        ).sum()
    )

    print(
        "\nOverall"
    )

    print(
        "======="
    )

    print(
        "LSTM feature improved RMSE in "
        f"{lstm_improved}/"
        f"{len(comparison)} locations."
    )

    print(
        "Informer feature improved RMSE in "
        f"{informer_improved}/"
        f"{len(comparison)} locations."
    )

    print(
        "Average LSTM RMSE improvement: "
        f"{comparison['lstm_rmse_improvement_pct'].mean():+.1f}%"
    )

    print(
        "Average Informer RMSE improvement: "
        f"{comparison['informer_rmse_improvement_pct'].mean():+.1f}%"
    )


def plot_comparison(
    comparison: pd.DataFrame,
) -> None:
    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    locations = (
        comparison[
            "state_abbr"
        ]
        .astype(str)
        .tolist()
    )

    x = list(
        range(
            len(locations)
        )
    )

    width = 0.18

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(
            12,
            10,
        ),
    )

    offsets = [
        -1.5,
        -0.5,
        0.5,
        1.5,
    ]

    rmse_series = [
        (
            comparison[
                "lstm_baseline_rmse"
            ],
            "LSTM baseline",
        ),
        (
            comparison[
                "lstm_feature_rmse"
            ],
            "LSTM + peak feature",
        ),
        (
            comparison[
                "informer_baseline_rmse"
            ],
            "Informer baseline",
        ),
        (
            comparison[
                "informer_feature_rmse"
            ],
            "Informer + peak feature",
        ),
    ]

    for offset, (
        values,
        label,
    ) in zip(
        offsets,
        rmse_series,
    ):
        axes[0].bar(
            [
                value
                + offset
                * width
                for value
                in x
            ],
            values,
            width=width,
            label=label,
        )

    axes[0].set_xticks(
        x,
        locations,
    )

    axes[0].set_ylabel(
        "4-week RMSE"
    )

    axes[0].set_title(
        "4-Week RMSE Across Locations"
    )

    axes[0].legend()

    axes[0].grid(
        axis="y",
        alpha=0.25,
    )

    peak_series = [
        (
            comparison[
                "lstm_baseline_peak_magnitude_error"
            ].abs(),
            "LSTM baseline",
        ),
        (
            comparison[
                "lstm_feature_peak_magnitude_error"
            ].abs(),
            "LSTM + peak feature",
        ),
        (
            comparison[
                "informer_baseline_peak_magnitude_error"
            ].abs(),
            "Informer baseline",
        ),
        (
            comparison[
                "informer_feature_peak_magnitude_error"
            ].abs(),
            "Informer + peak feature",
        ),
    ]

    for offset, (
        values,
        label,
    ) in zip(
        offsets,
        peak_series,
    ):
        axes[1].bar(
            [
                value
                + offset
                * width
                for value
                in x
            ],
            values,
            width=width,
            label=label,
        )

    axes[1].set_xticks(
        x,
        locations,
    )

    axes[1].set_ylabel(
        "Absolute peak magnitude error"
    )

    axes[1].set_title(
        "Peak Magnitude Error Across Locations"
    )

    axes[1].legend()

    axes[1].grid(
        axis="y",
        alpha=0.25,
    )

    fig.suptitle(
        "LSTM vs Informer: "
        "Effect of Peak-Probability Feature"
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
        f"\nSaved: {OUTPUT_PLOT}"
    )


def main():
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison = (
        build_comparison()
    )

    comparison.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print_summary(
        comparison
    )

    plot_comparison(
        comparison
    )

    print(
        f"Saved: {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()