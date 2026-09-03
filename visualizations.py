from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tools import (
    prepare_flu_data,
    summarize_cross_season_peak_timing,
    summarize_state_peaks,
)


PLOTS_DIR = Path("plots")


def ensure_plots_dir():
    """
    Create the plots directory if it does not already exist.
    """

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def plot_national_weekly_rate():
    """
    Plot the US national weekly influenza hospitalization rate
    across the full available dataset.
    """

    ensure_plots_dir()

    df = prepare_flu_data()

    national_df = (
        df[df["location_name"] == "US"]
        .copy()
        .sort_values("date")
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        national_df["date"],
        national_df["weekly_rate"],
        linewidth=2,
    )

    ax.set_title(
        "US Weekly Influenza Hospitalization Rate"
    )

    ax.set_xlabel("Date")

    ax.set_ylabel(
        "Weekly hospitalization rate"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    output_path = (
        PLOTS_DIR
        / "national_weekly_hospitalization_rate.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path


def plot_cross_season_timing_groups():
    """
    Plot early / typical / late jurisdiction counts
    for each available influenza season.

    Partial seasons are marked with an asterisk.
    """

    ensure_plots_dir()

    summaries = (
        summarize_cross_season_peak_timing()
    )

    rows = []

    for summary in summaries:
        season_label = summary["season"]

        if summary["is_partial"]:
            season_label += " *"

        rows.append(
            {
                "season": season_label,
                "early": summary["early_count"],
                "typical": summary["typical_count"],
                "late": summary["late_count"],
            }
        )

    timing_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.bar(
        timing_df["season"],
        timing_df["early"],
        label="Early",
    )

    ax.bar(
        timing_df["season"],
        timing_df["typical"],
        bottom=timing_df["early"],
        label="Typical",
    )

    ax.bar(
        timing_df["season"],
        timing_df["late"],
        bottom=(
            timing_df["early"]
            + timing_df["typical"]
        ),
        label="Late",
    )

    ax.set_title(
        "Jurisdiction Peak Timing by Flu Season"
    )

    ax.set_xlabel("Flu season")

    ax.set_ylabel(
        "Number of jurisdictions"
    )

    ax.legend()

    ax.text(
        0.01,
        -0.16,
        "* Partial season",
        transform=ax.transAxes,
        fontsize=9,
    )

    fig.tight_layout()

    output_path = (
        PLOTS_DIR
        / "cross_season_peak_timing_groups.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path


def plot_season_peak_timing(season):
    """
    Plot each jurisdiction's hospitalization peak date
    for a selected influenza season.

    A vertical line marks the dominant jurisdictional
    peak date for the season.
    """

    ensure_plots_dir()

    state_peaks = summarize_state_peaks(
        season
    )

    timing_summary = (
        summarize_cross_season_peak_timing()
    )

    selected_summary = None

    for summary in timing_summary:
        if summary["season"] == season:
            selected_summary = summary
            break

    if selected_summary is None:
        raise ValueError(
            f"Season not found: {season}"
        )

    dominant_peak_date = pd.to_datetime(
        selected_summary[
            "dominant_peak_date"
        ]
    )

    peak_df = pd.DataFrame(
        state_peaks
    )

    peak_df["peak_date"] = pd.to_datetime(
        peak_df["peak_date"]
    )

    peak_df = peak_df.sort_values(
        [
            "peak_date",
            "location_name",
        ]
    )

    fig, ax = plt.subplots(
        figsize=(12, 12)
    )

    ax.scatter(
        peak_df["peak_date"],
        peak_df["location_name"],
        s=35,
    )

    ax.axvline(
        dominant_peak_date,
        linestyle="--",
        linewidth=2,
        label=(
            "Dominant peak: "
            f"{dominant_peak_date.date()}"
        ),
    )

    ax.set_title(
        f"Jurisdiction Peak Timing — {season}"
    )

    ax.set_xlabel("Peak date")

    ax.set_ylabel("Jurisdiction")

    ax.grid(
        True,
        axis="x",
        alpha=0.3,
    )

    ax.legend()

    fig.autofmt_xdate()

    fig.tight_layout()

    safe_season = season.replace(
        "-",
        "_",
    )

    output_path = (
        PLOTS_DIR
        / (
            "jurisdiction_peak_timing_"
            f"{safe_season}.png"
        )
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path


def generate_visual_summary(
    season="2024-2025",
):
    """
    Generate the current FluSight visual summary.

    Returns a dictionary containing the paths
    of all generated plots.
    """

    print(
        "\n--- GENERATING VISUAL SUMMARY ---"
    )

    national_plot = (
        plot_national_weekly_rate()
    )

    timing_groups_plot = (
        plot_cross_season_timing_groups()
    )

    season_plot = (
        plot_season_peak_timing(
            season
        )
    )

    generated = {
        "national_weekly_rate": str(
            national_plot
        ),
        "cross_season_timing_groups": str(
            timing_groups_plot
        ),
        "season_peak_timing": str(
            season_plot
        ),
        "selected_season": season,
    }

    print(
        f"Saved: {national_plot}"
    )

    print(
        f"Saved: {timing_groups_plot}"
    )

    print(
        f"Saved: {season_plot}"
    )

    return generated


if __name__ == "__main__":
    generate_visual_summary()