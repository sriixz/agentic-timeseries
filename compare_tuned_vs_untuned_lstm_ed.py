from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OLD_PATH = Path(
    "forecast_results/allstate_lstm_ed_summary.csv"
)

TUNED_PATH = Path(
    "forecast_results/representative_lstm_ed_optuna_summary.csv"
)

DIAGNOSTIC_PATH = Path(
    "forecast_results/state_ed_diagnostic_summary.csv"
)

OUTPUT_CSV = Path(
    "forecast_results/representative_lstm_ed_tuned_vs_untuned.csv"
)

UPDATED_DIAGNOSTIC = Path(
    "forecast_results/state_ed_diagnostic_summary_with_tuning.csv"
)

OUTPUT_PLOT = Path(
    "plots/representative_lstm_ed_tuned_vs_untuned.png"
)


def main():
    old = pd.read_csv(
        OLD_PATH,
        dtype={"location": str},
    )

    tuned = pd.read_csv(
        TUNED_PATH,
        dtype={"location": str},
    )

    old_4w = old[
        old["horizon"] == 4
    ][
        [
            "location",
            "location_name",
            "baseline_rmse",
            "ed_rmse",
            "rmse_improvement_pct",
        ]
    ].rename(
        columns={
            "baseline_rmse":
                "untuned_baseline_rmse",

            "ed_rmse":
                "untuned_ed_rmse",

            "rmse_improvement_pct":
                "untuned_ed_improvement_pct",
        }
    )

    tuned_4w = tuned[
        tuned["horizon"] == 4
    ][
        [
            "location",
            "location_name",
            "tuned_baseline_rmse",
            "tuned_ed_rmse",
            "tuned_ed_improvement_pct",
        ]
    ].copy()

    comparison = old_4w.merge(
        tuned_4w,
        on=[
            "location",
            "location_name",
        ],
        how="inner",
        validate="one_to_one",
    )

    comparison[
        "tuning_change_pct_points"
    ] = (
        comparison[
            "tuned_ed_improvement_pct"
        ]
        - comparison[
            "untuned_ed_improvement_pct"
        ]
    )

    comparison[
        "untuned_behavior"
    ] = comparison[
        "untuned_ed_improvement_pct"
    ].apply(
        lambda x:
        "ED helps"
        if x > 0
        else "ED hurts"
    )

    comparison[
        "tuned_behavior"
    ] = comparison[
        "tuned_ed_improvement_pct"
    ].apply(
        lambda x:
        "ED helps"
        if x > 0
        else "ED hurts"
    )

    comparison[
        "behavior_flip"
    ] = (
        comparison[
            "untuned_behavior"
        ]
        != comparison[
            "tuned_behavior"
        ]
    )

    comparison = (
        comparison
        .sort_values(
            "tuned_ed_improvement_pct",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PLOT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(
        "\nTuned vs Untuned LSTM + ED"
    )

    print(
        "=========================="
    )

    print(
        comparison[
            [
                "location_name",
                "untuned_ed_improvement_pct",
                "tuned_ed_improvement_pct",
                "tuning_change_pct_points",
                "behavior_flip",
            ]
        ]
        .to_string(
            index=False,
            formatters={
                "untuned_ed_improvement_pct":
                    "{:+.1f}".format,

                "tuned_ed_improvement_pct":
                    "{:+.1f}".format,

                "tuning_change_pct_points":
                    "{:+.1f}".format,
            },
        )
    )

    flips = comparison[
        comparison[
            "behavior_flip"
        ]
    ].copy()

    print(
        "\nBehavior flips after tuning"
    )

    print(
        "---------------------------"
    )

    if flips.empty:
        print(
            "None"
        )
    else:
        print(
            flips[
                [
                    "location_name",
                    "untuned_ed_improvement_pct",
                    "tuned_ed_improvement_pct",
                ]
            ]
            .to_string(
                index=False,
                formatters={
                    "untuned_ed_improvement_pct":
                        "{:+.1f}".format,

                    "tuned_ed_improvement_pct":
                        "{:+.1f}".format,
                },
            )
        )

    print(
        "\nSummary"
    )

    print(
        "-------"
    )

    print(
        f"Representative states: "
        f"{len(comparison)}"
    )

    print(
        f"Untuned ED helped: "
        f"{int((comparison['untuned_ed_improvement_pct'] > 0).sum())}/"
        f"{len(comparison)}"
    )

    print(
        f"Tuned ED helped: "
        f"{int((comparison['tuned_ed_improvement_pct'] > 0).sum())}/"
        f"{len(comparison)}"
    )

    print(
        f"Behavior flips: "
        f"{int(comparison['behavior_flip'].sum())}"
    )

    print(
        "Median untuned ED improvement: "
        f"{comparison['untuned_ed_improvement_pct'].median():+.1f}%"
    )

    print(
        "Median tuned ED improvement: "
        f"{comparison['tuned_ed_improvement_pct'].median():+.1f}%"
    )

    # ----------------------------------
    # Plot
    # ----------------------------------

    plot_df = comparison.sort_values(
        "untuned_ed_improvement_pct"
    ).copy()

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    y = range(
        len(plot_df)
    )

    ax.scatter(
        plot_df[
            "untuned_ed_improvement_pct"
        ],
        y,
        s=80,
        label="Untuned",
    )

    ax.scatter(
        plot_df[
            "tuned_ed_improvement_pct"
        ],
        y,
        s=80,
        label="Tuned",
    )

    for i, row in enumerate(
        plot_df.itertuples()
    ):
        ax.plot(
            [
                row.untuned_ed_improvement_pct,
                row.tuned_ed_improvement_pct,
            ],
            [
                i,
                i,
            ],
            linewidth=1.5,
            alpha=0.6,
        )

    ax.axvline(
        0,
        linewidth=1,
    )

    ax.set_yticks(
        list(y)
    )

    ax.set_yticklabels(
        plot_df[
            "location_name"
        ]
    )

    ax.set_xlabel(
        "4-week RMSE improvement "
        "from ED feature (%)"
    )

    ax.set_ylabel(
        "State"
    )

    ax.set_title(
        "Effect of LSTM Retuning on "
        "ED-Feature Benefit"
    )

    ax.grid(
        axis="x",
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

    # ----------------------------------
    # Update diagnostic summary
    # ----------------------------------

    if DIAGNOSTIC_PATH.exists():
        diagnostic = pd.read_csv(
            DIAGNOSTIC_PATH,
            dtype={"location": str},
        )

        tuning_cols = comparison[
            [
                "location",
                "location_name",
                "tuned_baseline_rmse",
                "tuned_ed_rmse",
                "tuned_ed_improvement_pct",
                "tuning_change_pct_points",
                "behavior_flip",
            ]
        ].copy()

        updated = diagnostic.merge(
            tuning_cols,
            on=[
                "location",
                "location_name",
            ],
            how="left",
            validate="one_to_one",
        )

        updated.to_csv(
            UPDATED_DIAGNOSTIC,
            index=False,
        )

        print(
            f"\nSaved: {UPDATED_DIAGNOSTIC}"
        )

    print(
        f"Saved: {OUTPUT_CSV}"
    )

    print(
        f"Saved: {OUTPUT_PLOT}"
    )


if __name__ == "__main__":
    main()