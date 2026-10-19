from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from epiweeks import Week


DATA_PATH = Path("data/target-hospital-admissions.csv")

RESULT_DIR = Path("analysis_results")
PLOT_DIR = Path("plots")

OUTPUT_CSV = (
    RESULT_DIR
    / "us_peak_probability_by_epiweek.csv"
)

OUTPUT_PLOT = (
    PLOT_DIR
    / "us_peak_probability_shifted_4w.png"
)

SEASONS = [
    "2022-2023",
    "2023-2024",
    "2024-2025",
    "2025-2026",
]

# Controls how broadly each observed historical peak
# contributes probability to nearby epiweeks.
#
# 1.5 means an observed peak influences nearby weeks,
# but remains concentrated around its observed timing.
BANDWIDTH_WEEKS = 1.5

FORECAST_SHIFT_WEEKS = 4


def season_label(date: pd.Timestamp) -> str:
    """
    Flu season runs from August through July.
    """
    if date.month >= 8:
        start_year = date.year
    else:
        start_year = date.year - 1

    return f"{start_year}-{start_year + 1}"


def seasonal_epiweek_position(
    date: pd.Timestamp,
) -> int:
    """
    Convert calendar epiweek into a continuous
    seasonal epiweek position.

    Examples:
        EW45 -> 45
        EW52 -> 52
        EW53 -> 53
        EW01 -> 54
        EW02 -> 55

    This preserves chronology across the
    calendar-year boundary.
    """
    epiweek = Week.fromdate(
        date.date()
    ).week

    if epiweek >= 31:
        return epiweek

    return epiweek + 53


def position_to_epiweek(
    position: int,
) -> str:
    """
    Convert continuous seasonal position back
    to an epiweek label.
    """
    position = int(position)

    if position <= 53:
        week = position
    else:
        week = position - 53

    return f"EW{week:02d}"


def full_epiweek_label(
    date: pd.Timestamp,
) -> str:
    week = Week.fromdate(
        date.date()
    )

    return (
        f"{week.year}-EW{week.week:02d}"
    )


def load_national_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required = {
        "date",
        "location",
        "weekly_rate",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Dataset missing required columns: "
            f"{sorted(missing)}"
        )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df["location"] = (
        df["location"]
        .astype(str)
        .str.strip()
    )

    national = df[
        df["location"] == "US"
    ].copy()

    if national.empty:
        raise ValueError(
            "No national rows with location='US' "
            "were found."
        )

    national["season"] = (
        national["date"]
        .apply(season_label)
    )

    national = national[
        national["season"].isin(
            SEASONS
        )
    ].copy()

    national = national.dropna(
        subset=[
            "weekly_rate",
            "date",
        ]
    )

    national = national.sort_values(
        "date"
    )

    return national


def find_national_peaks(
    national: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find the maximum national weekly hospitalization
    rate in each season.
    """
    peak_indices = (
        national.groupby("season")[
            "weekly_rate"
        ]
        .idxmax()
    )

    peaks = (
        national.loc[peak_indices]
        .copy()
        .sort_values("season")
    )

    peaks["peak_epiweek"] = (
        peaks["date"]
        .apply(full_epiweek_label)
    )

    peaks["peak_position"] = (
        peaks["date"]
        .apply(
            seasonal_epiweek_position
        )
    )

    return peaks[
        [
            "season",
            "date",
            "peak_epiweek",
            "peak_position",
            "weekly_rate",
        ]
    ].reset_index(
        drop=True
    )


def gaussian_kernel(
    x: np.ndarray,
    center: float,
    bandwidth: float,
) -> np.ndarray:
    """
    Gaussian kernel centered at one observed
    historical peak week.
    """
    return np.exp(
        -0.5
        * (
            (x - center)
            / bandwidth
        ) ** 2
    )


def build_probability_curve(
    peaks: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct a smoothed probability distribution
    over seasonal epiweeks.

    Every observed historical peak contributes a
    Gaussian kernel.

    The final curve is normalized so the probabilities
    across the season sum to 1.
    """

    # Full seasonal timeline:
    #
    # 31 ... 53 corresponds roughly Aug-Dec
    # 54 ... 83 corresponds EW01-EW30
    positions = np.arange(
        31,
        84,
        dtype=int,
    )

    density = np.zeros(
        len(positions),
        dtype=float,
    )

    for peak_position in peaks[
        "peak_position"
    ]:
        density += gaussian_kernel(
            x=positions,
            center=float(
                peak_position
            ),
            bandwidth=BANDWIDTH_WEEKS,
        )

    total = density.sum()

    if total <= 0:
        raise ValueError(
            "Peak probability curve has "
            "zero total density."
        )

    probability = (
        density / total
    )

    result = pd.DataFrame(
        {
            "seasonal_position":
                positions,
            "epiweek": [
                position_to_epiweek(
                    position
                )
                for position
                in positions
            ],
            "peak_probability":
                probability,
        }
    )

    return result


def add_shifted_feature(
    probability_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Shift the peak-probability curve FOUR WEEKS EARLIER.

    If the historical probability of a peak is high
    at week t, the forecasting feature should become
    high at week t - 4.

    Equivalently:

        shifted_feature(t)
            = peak_probability(t + 4)

    This is the feature Adiga sketched: information
    about a likely future peak is exposed to the model
    four weeks before that peak would normally occur.
    """
    result = (
        probability_df
        .copy()
    )

    probability_lookup = dict(
        zip(
            result[
                "seasonal_position"
            ],
            result[
                "peak_probability"
            ],
        )
    )

    shifted_values = []

    for position in result[
        "seasonal_position"
    ]:
        future_position = (
            position
            + FORECAST_SHIFT_WEEKS
        )

        shifted_probability = (
            probability_lookup.get(
                future_position,
                0.0,
            )
        )

        shifted_values.append(
            shifted_probability
        )

    result[
        "peak_probability_shifted_4w"
    ] = shifted_values

    return result


def print_peak_summary(
    peaks: pd.DataFrame,
    probability_df: pd.DataFrame,
) -> None:
    print(
        "\nObserved National Peak Weeks"
    )
    print(
        "============================"
    )

    for row in peaks.itertuples():
        print(
            f"{row.season}: "
            f"{row.peak_epiweek}, "
            f"rate={row.weekly_rate:.4f}"
        )

    original_peak = (
        probability_df.loc[
            probability_df[
                "peak_probability"
            ].idxmax()
        ]
    )

    shifted_peak = (
        probability_df.loc[
            probability_df[
                "peak_probability_shifted_4w"
            ].idxmax()
        ]
    )

    print(
        "\nPeak Probability Curve"
    )
    print(
        "======================"
    )

    print(
        "Highest original peak probability: "
        f"{original_peak['epiweek']}"
    )

    print(
        "Highest shifted 4-week feature: "
        f"{shifted_peak['epiweek']}"
    )

    print(
        "\nProbability sum: "
        f"{probability_df['peak_probability'].sum():.6f}"
    )


def save_results(
    probability_df: pd.DataFrame,
) -> None:
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    probability_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(
        f"\nSaved: {OUTPUT_CSV}"
    )


def plot_probability_curves(
    probability_df: pd.DataFrame,
    peaks: pd.DataFrame,
) -> None:
    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    ax.plot(
        probability_df[
            "seasonal_position"
        ],
        probability_df[
            "peak_probability"
        ],
        marker="o",
        markersize=3,
        linewidth=2,
        label=(
            "Historical peak probability"
        ),
    )

    ax.plot(
        probability_df[
            "seasonal_position"
        ],
        probability_df[
            "peak_probability_shifted_4w"
        ],
        marker="o",
        markersize=3,
        linewidth=2,
        label=(
            "4-week-ahead shifted feature"
        ),
    )

    # Show the actual historical national peak
    # observations at the bottom of the figure.
    for row in peaks.itertuples():
        ax.axvline(
            row.peak_position,
            alpha=0.15,
            linewidth=1,
        )

    tick_positions = [
        40,
        44,
        48,
        52,
        53,
        54,
        58,
        62,
        66,
        70,
    ]

    tick_labels = [
        position_to_epiweek(
            position
        )
        for position
        in tick_positions
    ]

    ax.set_xticks(
        tick_positions,
        tick_labels,
    )

    # Focus the visual on the epidemic peak window,
    # while the CSV still contains the full season.
    ax.set_xlim(
        40,
        70,
    )

    ax.set_xlabel(
        "Epidemiological week"
    )

    ax.set_ylabel(
        "Probability of seasonal peak"
    )

    ax.set_title(
        "U.S. Influenza Peak Probability "
        "and 4-Week-Ahead Shifted Feature"
    )

    ax.legend()

    ax.grid(
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_PLOT,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {OUTPUT_PLOT}"
    )


def main():
    national = (
        load_national_data()
    )

    peaks = (
        find_national_peaks(
            national
        )
    )

    probability_df = (
        build_probability_curve(
            peaks
        )
    )

    probability_df = (
        add_shifted_feature(
            probability_df
        )
    )

    print_peak_summary(
        peaks=peaks,
        probability_df=probability_df,
    )

    save_results(
        probability_df
    )

    plot_probability_curves(
        probability_df=probability_df,
        peaks=peaks,
    )


if __name__ == "__main__":
    main()