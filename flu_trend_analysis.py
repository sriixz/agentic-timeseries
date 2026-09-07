from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tools import prepare_flu_data


PLOTS_DIR = Path("plots")
RESULTS_DIR = Path("analysis_results")

CURRENT_SEASON = "2025-2026"

PAST_COMPLETE_SEASONS = [
    "2022-2023",
    "2023-2024",
    "2024-2025",
]


def prepare_us_data():
    """
    Load the FluSight dataset and keep only
    the US national observations.

    Hospitalization count ('value') is the
    primary analysis variable.
    """

    df = prepare_flu_data()

    us_df = (
        df[df["location_name"] == "US"]
        .copy()
        .sort_values("date")
    )

    us_df = us_df.dropna(
        subset=["value"]
    )

    return us_df


def summarize_season(season_df, season):
    """
    Produce deterministic count-based statistics
    for one flu season.
    """

    season_df = (
        season_df
        .sort_values("date")
        .copy()
    )

    if season_df.empty:
        return None

    peak_row = season_df.loc[
        season_df["value"].idxmax()
    ]

    first_row = season_df.iloc[0]
    latest_row = season_df.iloc[-1]

    latest_value = float(
        latest_row["value"]
    )

    previous_value = None
    weekly_change = None
    weekly_percent_change = None

    if len(season_df) >= 2:
        previous_row = season_df.iloc[-2]

        previous_value = float(
            previous_row["value"]
        )

        weekly_change = (
            latest_value - previous_value
        )

        if previous_value != 0:
            weekly_percent_change = (
                weekly_change
                / previous_value
                * 100
            )

    peak_value = float(
        peak_row["value"]
    )

    peak_date = pd.Timestamp(
        peak_row["date"]
    )

    latest_date = pd.Timestamp(
        latest_row["date"]
    )

    weeks_from_peak = int(
        (
            latest_date - peak_date
        ).days / 7
    )

    if latest_date < peak_date:
        phase = "pre-peak"

    elif latest_date == peak_date:
        phase = "at peak"

    else:
        phase = "post-peak"

    return {
        "season": season,
        "start_date": first_row[
            "date"
        ].date(),
        "end_date": latest_date.date(),
        "observation_count": len(
            season_df
        ),
        "latest_value": round(
            latest_value,
            2,
        ),
        "previous_value": (
            round(
                previous_value,
                2,
            )
            if previous_value is not None
            else None
        ),
        "weekly_change": (
            round(
                weekly_change,
                2,
            )
            if weekly_change is not None
            else None
        ),
        "weekly_percent_change": (
            round(
                weekly_percent_change,
                2,
            )
            if weekly_percent_change
            is not None
            else None
        ),
        "peak_date": peak_date.date(),
        "peak_value": round(
            peak_value,
            2,
        ),
        "phase": phase,
        "weeks_from_peak": weeks_from_peak,
    }


def build_season_summary_table(
    us_df,
):
    """
    Summarize the current season and the three
    recent complete historical seasons.
    """

    seasons = (
        PAST_COMPLETE_SEASONS
        + [CURRENT_SEASON]
    )

    summaries = []

    for season in seasons:
        season_df = us_df[
            us_df["season"] == season
        ]

        summary = summarize_season(
            season_df,
            season,
        )

        if summary is not None:
            summaries.append(
                summary
            )

    return pd.DataFrame(
        summaries
    )


def build_current_season_trend(
    us_df,
):
    """
    Produce a concise deterministic description
    of the current season.
    """

    current_df = (
        us_df[
            us_df["season"]
            == CURRENT_SEASON
        ]
        .sort_values("date")
        .copy()
    )

    summary = summarize_season(
        current_df,
        CURRENT_SEASON,
    )

    if summary is None:
        return None

    peak_value = summary[
        "peak_value"
    ]

    latest_value = summary[
        "latest_value"
    ]

    decline_from_peak = (
        latest_value - peak_value
    )

    decline_percent = None

    if peak_value != 0:
        decline_percent = (
            decline_from_peak
            / peak_value
            * 100
        )

    summary[
        "change_from_peak"
    ] = round(
        decline_from_peak,
        2,
    )

    summary[
        "percent_change_from_peak"
    ] = round(
        decline_percent,
        2,
    )

    return summary


def compare_peak_values(
    summary_df,
):
    """
    Compare the current-season national peak
    hospitalization count with historical peaks.
    """

    current_row = summary_df[
        summary_df["season"]
        == CURRENT_SEASON
    ].iloc[0]

    current_peak = float(
        current_row["peak_value"]
    )

    comparisons = []

    for _, row in summary_df.iterrows():
        season = row["season"]

        if season == CURRENT_SEASON:
            continue

        past_peak = float(
            row["peak_value"]
        )

        difference = (
            current_peak - past_peak
        )

        percent_difference = None

        if past_peak != 0:
            percent_difference = (
                difference
                / past_peak
                * 100
            )

        comparisons.append(
            {
                "current_season":
                    CURRENT_SEASON,
                "comparison_season":
                    season,
                "current_peak_value":
                    current_peak,
                "past_peak_value":
                    past_peak,
                "difference":
                    round(
                        difference,
                        2,
                    ),
                "percent_difference":
                    round(
                        percent_difference,
                        2,
                    ),
            }
        )

    return pd.DataFrame(
        comparisons
    )


def plot_season_counts(
    us_df,
):
    """
    Plot national hospitalization counts for the
    recent seasons on a common seasonal-week axis.
    """

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    seasons = (
        PAST_COMPLETE_SEASONS
        + [CURRENT_SEASON]
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    for season in seasons:
        season_df = (
            us_df[
                us_df["season"]
                == season
            ]
            .sort_values("date")
            .copy()
        )

        if season_df.empty:
            continue

        season_df[
            "season_week"
        ] = range(
            1,
            len(season_df) + 1,
        )

        ax.plot(
            season_df[
                "season_week"
            ],
            season_df["value"],
            label=season,
            linewidth=2,
        )

    ax.set_title(
        "US Influenza Hospital Admissions "
        "by Flu Season"
    )

    ax.set_xlabel(
        "Week of flu season"
    )

    ax.set_ylabel(
        "Weekly hospital admissions"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    output_path = (
        PLOTS_DIR
        / "us_hospitalization_counts_by_season.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path


def save_results(
    summary_df,
    comparisons_df,
):
    """
    Save deterministic analysis outputs.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        RESULTS_DIR
        / "us_season_trend_summary.csv"
    )

    comparison_path = (
        RESULTS_DIR
        / "us_current_vs_past_peaks.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    comparisons_df.to_csv(
        comparison_path,
        index=False,
    )

    return (
        summary_path,
        comparison_path,
    )


def main():
    print(
        "\n--- LOADING US HOSPITALIZATION DATA ---"
    )

    us_df = prepare_us_data()

    print(
        f"US observations: {len(us_df)}"
    )

    print(
        f"Date range: "
        f"{us_df['date'].min().date()} "
        f"to "
        f"{us_df['date'].max().date()}"
    )

    print(
        "\nPrimary trend variable: "
        "weekly hospitalization count (value)"
    )

    print(
        "Secondary normalized variable: "
        "weekly_rate"
    )

    summary_df = (
        build_season_summary_table(
            us_df
        )
    )

    print(
        "\n--- CURRENT AND PAST "
        "SEASON TRENDS ---"
    )

    display_columns = [
        "season",
        "start_date",
        "end_date",
        "latest_value",
        "peak_date",
        "peak_value",
        "phase",
        "weeks_from_peak",
    ]

    print(
        summary_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    current_summary = (
        build_current_season_trend(
            us_df
        )
    )

    print(
        "\n--- CURRENT SEASON DETAIL ---"
    )

    for key, value in (
        current_summary.items()
    ):
        print(
            f"{key}: {value}"
        )

    comparisons_df = (
        compare_peak_values(
            summary_df
        )
    )

    print(
        "\n--- CURRENT PEAK VS "
        "PAST SEASONS ---"
    )

    print(
        comparisons_df.to_string(
            index=False
        )
    )

    plot_path = (
        plot_season_counts(
            us_df
        )
    )

    (
        summary_path,
        comparison_path,
    ) = save_results(
        summary_df,
        comparisons_df,
    )

    print(
        "\n--- SAVED OUTPUTS ---"
    )

    print(
        f"Season summary: "
        f"{summary_path}"
    )

    print(
        f"Peak comparisons: "
        f"{comparison_path}"
    )

    print(
        f"Trend plot: "
        f"{plot_path}"
    )


if __name__ == "__main__":
    main()