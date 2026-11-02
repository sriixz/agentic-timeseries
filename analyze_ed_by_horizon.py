from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SUMMARY_PATH = Path(
    "forecast_results/allstate_lstm_ed_summary.csv"
)

OUTPUT_CSV = Path(
    "forecast_results/allstate_lstm_ed_horizon_summary.csv"
)

OUTPUT_PLOT = Path(
    "plots/allstate_lstm_ed_horizon_summary.png"
)


def main():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing: {SUMMARY_PATH}"
        )

    df = pd.read_csv(
        SUMMARY_PATH
    )

    required = {
        "location",
        "location_name",
        "horizon",
        "baseline_rmse",
        "ed_rmse",
        "rmse_improvement_pct",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    rows = []

    for horizon, group in df.groupby(
        "horizon"
    ):
        improved = int(
            (
                group[
                    "rmse_improvement_pct"
                ]
                > 0
            ).sum()
        )

        total = len(group)

        rows.append(
            {
                "horizon":
                    int(horizon),

                "states":
                    total,

                "states_improved":
                    improved,

                "pct_states_improved":
                    100.0
                    * improved
                    / total,

                "mean_rmse_improvement_pct":
                    group[
                        "rmse_improvement_pct"
                    ].mean(),

                "median_rmse_improvement_pct":
                    group[
                        "rmse_improvement_pct"
                    ].median(),

                "mean_baseline_rmse":
                    group[
                        "baseline_rmse"
                    ].mean(),

                "mean_ed_rmse":
                    group[
                        "ed_rmse"
                    ].mean(),
            }
        )

    summary = (
        pd.DataFrame(rows)
        .sort_values("horizon")
        .reset_index(drop=True)
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PLOT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(
        "\nAll-State LSTM + ED "
        "Horizon Summary"
    )

    print(
        "=============================="
    )

    print(
        summary.to_string(
            index=False,
            formatters={
                "pct_states_improved":
                    "{:.1f}".format,

                "mean_rmse_improvement_pct":
                    "{:+.1f}".format,

                "median_rmse_improvement_pct":
                    "{:+.1f}".format,

                "mean_baseline_rmse":
                    "{:.3f}".format,

                "mean_ed_rmse":
                    "{:.3f}".format,
            },
        )
    )

    print(
        "\nCompact view"
    )

    print(
        "============"
    )

    for row in summary.itertuples():
        print(
            f"{row.horizon}-week: "
            f"{row.states_improved}/"
            f"{row.states} states improved | "
            f"mean "
            f"{row.mean_rmse_improvement_pct:+.1f}% | "
            f"median "
            f"{row.median_rmse_improvement_pct:+.1f}%"
        )

    # Plot mean and median improvement
    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        summary["horizon"],
        summary[
            "mean_rmse_improvement_pct"
        ],
        marker="o",
        linewidth=2,
        label="Mean",
    )

    ax.plot(
        summary["horizon"],
        summary[
            "median_rmse_improvement_pct"
        ],
        marker="o",
        linewidth=2,
        label="Median",
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_xticks(
        summary["horizon"]
    )

    ax.set_xlabel(
        "Forecast horizon (weeks ahead)"
    )

    ax.set_ylabel(
        "RMSE improvement from ED feature (%)"
    )

    ax.set_title(
        "LSTM + ED Benefit by Forecast Horizon"
    )

    ax.grid(
        alpha=0.25
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