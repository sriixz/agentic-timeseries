from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from epiweeks import Week


DATA_PATH = Path("data/target-hospital-admissions.csv")
RESULT_DIR = Path("analysis_results")
PLOT_DIR = Path("plots")

SEASONS = [
    "2022-2023",
    "2023-2024",
    "2024-2025",
    "2025-2026",
]


FIPS_TO_STATE = {
    "01": "AL",
    "02": "AK",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "15": "HI",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
}


def season_label(date: pd.Timestamp) -> str:
    """
    Flu season runs from August through July.
    """
    if date.month >= 8:
        start_year = date.year
    else:
        start_year = date.year - 1

    return f"{start_year}-{start_year + 1}"


def seasonal_epiweek_position(date: pd.Timestamp) -> int:
    """
    Convert epiweek into a continuous within-season position.

    Example:
        EW45 -> 45
        EW52 -> 52
        EW01 -> 54
        EW02 -> 55

    This prevents the calendar-year boundary from making
    EW01 appear earlier than EW52.
    """
    week = Week.fromdate(date.date()).week

    if week >= 31:
        return week

    return week + 53


def position_to_epiweek(position: float) -> str:
    """
    Convert a continuous seasonal epiweek position back to
    a readable epidemiological week label.
    """
    rounded = int(round(position))

    if rounded <= 53:
        week = rounded
    else:
        week = rounded - 53

    return f"EW{week:02d}"


def full_epiweek_label(date: pd.Timestamp) -> str:
    week = Week.fromdate(date.date())
    return f"{week.year}-EW{week.week:02d}"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find FluSight data at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required = {
        "date",
        "location",
        "location_name",
        "weekly_rate",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset missing required columns: {sorted(missing)}"
        )

    df["date"] = pd.to_datetime(df["date"])

    df["location"] = (
        df["location"]
        .astype(str)
        .str.strip()
        .str.zfill(2)
    )

    df["season"] = df["date"].apply(season_label)

    return df


def build_state_peak_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find each state's maximum weekly hospitalization rate
    within each season.
    """
    states = df[
        df["location"].isin(FIPS_TO_STATE)
        & df["season"].isin(SEASONS)
    ].copy()

    states = states.dropna(
        subset=["weekly_rate", "date"]
    )

    peak_indices = (
        states.groupby(["season", "location"])["weekly_rate"]
        .idxmax()
    )

    peaks = states.loc[peak_indices].copy()

    peaks["state_abbr"] = peaks["location"].map(FIPS_TO_STATE)

    peaks["peak_epiweek"] = peaks["date"].apply(
        full_epiweek_label
    )

    peaks["peak_position"] = peaks["date"].apply(
        seasonal_epiweek_position
    )

    return peaks[
        [
            "season",
            "location",
            "state_abbr",
            "location_name",
            "date",
            "peak_epiweek",
            "peak_position",
            "weekly_rate",
        ]
    ].sort_values(
        ["state_abbr", "season"]
    )


def median_absolute_deviation(values: pd.Series) -> float:
    """
    Median absolute deviation measured in weeks.
    """
    median = values.median()

    return float(
        np.median(
            np.abs(values.to_numpy(dtype=float) - median)
        )
    )


def build_state_statistics(
    peaks: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        location,
        state_abbr,
        location_name,
    ), group in peaks.groupby(
        ["location", "state_abbr", "location_name"]
    ):
        positions = group["peak_position"].astype(float)

        median_position = float(positions.median())

        q1 = float(positions.quantile(0.25))
        q3 = float(positions.quantile(0.75))

        iqr = q3 - q1

        mad = median_absolute_deviation(positions)

        minimum = float(positions.min())
        maximum = float(positions.max())

        full_range = maximum - minimum

        rows.append(
            {
                "location": location,
                "state_abbr": state_abbr,
                "location_name": location_name,
                "seasons_observed": len(group),
                "median_peak_position": median_position,
                "median_peak_epiweek": position_to_epiweek(
                    median_position
                ),
                "q1_peak_epiweek": position_to_epiweek(q1),
                "q3_peak_epiweek": position_to_epiweek(q3),
                "iqr_weeks": iqr,
                "mad_weeks": mad,
                "range_weeks": full_range,
                "earliest_peak_epiweek": position_to_epiweek(
                    minimum
                ),
                "latest_peak_epiweek": position_to_epiweek(
                    maximum
                ),
            }
        )

    stats = pd.DataFrame(rows)

    stats = stats.sort_values(
        ["iqr_weeks", "mad_weeks", "state_abbr"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    return stats


def add_variability_group(
    stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a simple relative stability label.

    This is descriptive only. It is based on the distribution
    of state IQR values in this dataset, not on an external
    epidemiological threshold.
    """
    stats = stats.copy()

    low_cutoff = stats["iqr_weeks"].quantile(1 / 3)
    high_cutoff = stats["iqr_weeks"].quantile(2 / 3)

    def classify(iqr: float) -> str:
        if iqr <= low_cutoff:
            return "lower variability"

        if iqr >= high_cutoff:
            return "higher variability"

        return "moderate variability"

    stats["variability_group"] = stats["iqr_weeks"].apply(
        classify
    )

    return stats


def print_summary(stats: pd.DataFrame) -> None:
    print("\nState Peak Timing Statistics")
    print("============================")
    print(
        f"States/DC represented: {len(stats)}"
    )

    print("\nMost stable peak timing by IQR")
    print("------------------------------")

    stable = stats.nsmallest(
        10,
        ["iqr_weeks", "mad_weeks"],
    )

    for row in stable.itertuples():
        print(
            f"{row.state_abbr}: "
            f"median={row.median_peak_epiweek}, "
            f"IQR={row.iqr_weeks:.2f} weeks, "
            f"MAD={row.mad_weeks:.2f} weeks"
        )

    print("\nMost variable peak timing by IQR")
    print("--------------------------------")

    variable = stats.nlargest(
        10,
        ["iqr_weeks", "mad_weeks"],
    )

    for row in variable.itertuples():
        print(
            f"{row.state_abbr}: "
            f"median={row.median_peak_epiweek}, "
            f"IQR={row.iqr_weeks:.2f} weeks, "
            f"MAD={row.mad_weeks:.2f} weeks"
        )

    wyoming = stats[
        stats["state_abbr"] == "WY"
    ]

    if not wyoming.empty:
        row = wyoming.iloc[0]

        print("\nWyoming")
        print("-------")
        print(
            f"Median peak: {row['median_peak_epiweek']}"
        )
        print(
            f"IQR: {row['iqr_weeks']:.2f} weeks"
        )
        print(
            f"MAD: {row['mad_weeks']:.2f} weeks"
        )
        print(
            f"Observed range: "
            f"{row['earliest_peak_epiweek']} "
            f"to {row['latest_peak_epiweek']}"
        )


def save_results(
    peaks: pd.DataFrame,
    stats: pd.DataFrame,
) -> None:
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    peaks_path = (
        RESULT_DIR
        / "state_peak_epiweeks_by_season.csv"
    )

    stats_path = (
        RESULT_DIR
        / "state_peak_timing_statistics.csv"
    )

    peaks.to_csv(
        peaks_path,
        index=False,
    )

    stats.to_csv(
        stats_path,
        index=False,
    )

    print(f"\nSaved: {peaks_path}")
    print(f"Saved: {stats_path}")


def plot_state_variability(
    stats: pd.DataFrame,
) -> None:
    """
    Plot state IQR values from lowest to highest variability.
    """
    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_df = stats.sort_values(
        "iqr_weeks",
        ascending=True,
    )

    plt.figure(
        figsize=(12, 11)
    )

    plt.barh(
        plot_df["state_abbr"],
        plot_df["iqr_weeks"],
    )

    plt.xlabel(
        "Peak epiweek variability (IQR, weeks)"
    )

    plt.ylabel("State")

    plt.title(
        "Cross-Season Variability in State Influenza "
        "Hospitalization Peak Timing"
    )

    plt.grid(
        axis="x",
        alpha=0.25,
    )

    plt.tight_layout()

    output_path = (
        PLOT_DIR
        / "state_peak_epiweek_variability.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


def main() -> None:
    df = load_data()

    peaks = build_state_peak_table(df)

    stats = build_state_statistics(peaks)

    stats = add_variability_group(stats)

    print_summary(stats)

    save_results(
        peaks=peaks,
        stats=stats,
    )

    plot_state_variability(stats)


if __name__ == "__main__":
    main()