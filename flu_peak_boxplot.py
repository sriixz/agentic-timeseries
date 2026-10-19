from pathlib import Path

import matplotlib.pyplot as plt

from flu_peak_statistics import (
    build_state_peak_table,
    load_data,
)


PLOT_DIR = Path("plots")

OUTPUT_PATH = (
    PLOT_DIR
    / "state_peak_timing_boxplot.png"
)


def build_plot_data(peaks):
    """
    Sort states by median seasonal peak position.

    Each state's values are the observed peak positions
    across available seasons.
    """
    grouped = []

    for state_abbr, group in peaks.groupby("state_abbr"):
        positions = (
            group["peak_position"]
            .astype(float)
            .sort_values()
            .tolist()
        )

        median_position = float(
            group["peak_position"]
            .astype(float)
            .median()
        )

        grouped.append(
            {
                "state_abbr": state_abbr,
                "positions": positions,
                "median_position": median_position,
            }
        )

    # Sort from earliest typical peak to latest typical peak.
    grouped.sort(
        key=lambda item: (
            item["median_position"],
            item["state_abbr"],
        )
    )

    return grouped


def plot_peak_timing_boxplots(peaks):
    plot_data = build_plot_data(peaks)

    state_labels = [
        item["state_abbr"]
        for item in plot_data
    ]

    values = [
        item["positions"]
        for item in plot_data
    ]

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(14, 15)
    )

    ax.boxplot(
        values,
        orientation="horizontal",
        tick_labels=state_labels,
        showfliers=False,
        widths=0.55,
    )

    # Overlay actual seasonal peak observations.
    # Keep these smaller and slightly transparent so the
    # boxplots remain the main visual element.
    for index, item in enumerate(
        plot_data,
        start=1,
    ):
        ax.scatter(
            item["positions"],
            [index] * len(item["positions"]),
            s=16,
            alpha=0.65,
            zorder=3,
        )

    # Seasonal epiweek positions:
    # EW45 -> 45
    # EW49 -> 49
    # EW52 -> 52
    # EW01 -> 54
    # EW05 -> 58
    # EW09 -> 62
    # EW12 -> 65
    tick_positions = [
        45,
        49,
        52,
        54,
        58,
        62,
        65,
    ]

    tick_labels = [
        "EW45",
        "EW49",
        "EW52",
        "EW01",
        "EW05",
        "EW09",
        "EW12",
    ]

    ax.set_xticks(
        tick_positions,
        tick_labels,
    )

    ax.set_xlim(
        44,
        66,
    )

    ax.set_xlabel(
        "Typical influenza hospitalization peak timing "
        "(epidemiological week)"
    )

    ax.set_ylabel("State")

    ax.set_title(
        "State Influenza Hospitalization Peak Timing "
        "and Cross-Season Variability"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    # Put earliest-peaking states at the top.
    ax.invert_yaxis()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {OUTPUT_PATH}"
    )


def main():
    df = load_data()

    peaks = build_state_peak_table(df)

    print(
        f"Peak observations: {len(peaks)}"
    )

    print(
        f"States/DC represented: "
        f"{peaks['state_abbr'].nunique()}"
    )

    plot_peak_timing_boxplots(peaks)


if __name__ == "__main__":
    main()