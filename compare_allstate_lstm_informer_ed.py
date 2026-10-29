from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LSTM_PATH = Path(
    "forecast_results/allstate_lstm_ed_summary.csv"
)

INFORMER_PATH = Path(
    "forecast_results/allstate_informer_ed_summary.csv"
)

RESULT_DIR = Path("forecast_results")
PLOT_DIR = Path("plots")


OUTPUT_CSV = (
    RESULT_DIR
    / "allstate_lstm_vs_informer_ed_comparison.csv"
)

OUTPUT_PLOT = (
    PLOT_DIR
    / "allstate_lstm_vs_informer_ed_scatter.png"
)


def load_four_week(
    path,
    prefix,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    df = pd.read_csv(
        path,
        dtype={
            "location": str,
        },
    )

    df = df[
        df["horizon"] == 4
    ].copy()

    keep = [
        "location",
        "location_name",
        "baseline_rmse",
        "ed_rmse",
        "rmse_improvement_pct",
    ]

    df = df[
        keep
    ].copy()

    df = df.rename(
        columns={
            "baseline_rmse":
                f"{prefix}_baseline_rmse",

            "ed_rmse":
                f"{prefix}_ed_rmse",

            "rmse_improvement_pct":
                f"{prefix}_improvement_pct",
        }
    )

    return df


def classify_quadrant(
    row,
):
    lstm = row[
        "lstm_improvement_pct"
    ]

    informer = row[
        "informer_improvement_pct"
    ]

    if lstm > 0 and informer > 0:
        return "Both improve"

    if lstm > 0 and informer <= 0:
        return "LSTM only"

    if lstm <= 0 and informer > 0:
        return "Informer only"

    return "Neither improves"


def main():
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    lstm = load_four_week(
        LSTM_PATH,
        "lstm",
    )

    informer = load_four_week(
        INFORMER_PATH,
        "informer",
    )

    merged = lstm.merge(
        informer,
        on=[
            "location",
            "location_name",
        ],
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError(
            "No overlapping states found."
        )

    merged[
        "quadrant"
    ] = merged.apply(
        classify_quadrant,
        axis=1,
    )

    merged[
        "architecture_gap_pct"
    ] = (
        merged[
            "informer_improvement_pct"
        ]
        - merged[
            "lstm_improvement_pct"
        ]
    )

    merged[
        "best_architecture"
    ] = np.where(
        merged[
            "informer_improvement_pct"
        ]
        > merged[
            "lstm_improvement_pct"
        ],
        "Informer",
        "LSTM",
    )

    merged.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(
        "\nLSTM vs Informer ED Comparison"
    )

    print(
        "=============================="
    )

    print(
        f"Overlapping states: "
        f"{len(merged)}"
    )

    counts = (
        merged[
            "quadrant"
        ]
        .value_counts()
    )

    for label in [
        "Both improve",
        "LSTM only",
        "Informer only",
        "Neither improves",
    ]:
        print(
            f"{label}: "
            f"{int(counts.get(label, 0))}"
        )

    pearson = (
        merged[
            "lstm_improvement_pct"
        ]
        .corr(
            merged[
                "informer_improvement_pct"
            ]
        )
    )

    spearman = (
        merged[
            "lstm_improvement_pct"
        ]
        .rank()
        .corr(
            merged[
                "informer_improvement_pct"
            ]
            .rank()
        )
    )

    print(
        "\nCorrelation between "
        "architecture-specific improvements"
    )

    print(
        f"Pearson r:  "
        f"{pearson:+.3f}"
    )

    print(
        f"Spearman ρ: "
        f"{spearman:+.3f}"
    )

    print(
        "\nLargest Informer advantages"
    )

    print(
        merged[
            [
                "location_name",
                "lstm_improvement_pct",
                "informer_improvement_pct",
                "architecture_gap_pct",
                "quadrant",
            ]
        ]
        .sort_values(
            "architecture_gap_pct",
            ascending=False,
        )
        .head(10)
        .to_string(
            index=False
        )
    )

    print(
        "\nLargest LSTM advantages"
    )

    print(
        merged[
            [
                "location_name",
                "lstm_improvement_pct",
                "informer_improvement_pct",
                "architecture_gap_pct",
                "quadrant",
            ]
        ]
        .sort_values(
            "architecture_gap_pct",
            ascending=True,
        )
        .head(10)
        .to_string(
            index=False
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            9,
        )
    )

    ax.scatter(
        merged[
            "lstm_improvement_pct"
        ],
        merged[
            "informer_improvement_pct"
        ],
        s=60,
        alpha=0.8,
    )

    ax.axvline(
        0,
        linewidth=1,
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    all_values = pd.concat(
        [
            merged[
                "lstm_improvement_pct"
            ],
            merged[
                "informer_improvement_pct"
            ],
        ]
    )

    lower = float(
        all_values.min()
    )

    upper = float(
        all_values.max()
    )

    margin = (
        max(
            abs(lower),
            abs(upper),
        )
        * 0.08
    )

    diagonal_min = (
        lower - margin
    )

    diagonal_max = (
        upper + margin
    )

    ax.plot(
        [
            diagonal_min,
            diagonal_max,
        ],
        [
            diagonal_min,
            diagonal_max,
        ],
        linestyle="--",
        linewidth=1.25,
        label="Equal improvement",
    )

    # Label the largest architecture
    # disagreements plus strong outliers.
    informer_adv = (
        merged.nlargest(
            4,
            "architecture_gap_pct",
        )
    )

    lstm_adv = (
        merged.nsmallest(
            4,
            "architecture_gap_pct",
        )
    )

    extreme_bad = (
        merged.nsmallest(
            2,
            "informer_improvement_pct",
        )
    )

    selected = pd.concat(
        [
            informer_adv,
            lstm_adv,
            extreme_bad,
        ]
    ).drop_duplicates(
        subset=[
            "location"
        ]
    )

    for row in (
        selected.itertuples()
    ):
        ax.annotate(
            row.location_name,
            (
                row.lstm_improvement_pct,
                row.informer_improvement_pct,
            ),
            xytext=(
                5,
                5,
            ),
            textcoords=(
                "offset points"
            ),
            fontsize=8,
        )

    ax.set_xlabel(
        "LSTM 4-week RMSE improvement "
        "from ED feature (%)"
    )

    ax.set_ylabel(
        "Informer 4-week RMSE improvement "
        "from ED feature (%)"
    )

    ax.set_title(
        "State-Level ED-Feature Benefit: "
        "LSTM vs Informer"
    )

    ax.grid(
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_PLOT,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"\nSaved: {OUTPUT_CSV}"
    )

    print(
        f"Saved: {OUTPUT_PLOT}"
    )


if __name__ == "__main__":
    main()