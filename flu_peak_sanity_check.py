from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from epiweeks import Week


DATA_PATH = Path("data/target-hospital-admissions.csv")
PLOT_DIR = Path("plots")

STATES_TO_CHECK = {
    "ID": "Idaho",
    "MI": "Michigan",
    "WY": "Wyoming",
    "PA": "Pennsylvania",
}

STATE_FIPS = {
    "ID": "16",
    "MI": "26",
    "WY": "56",
    "PA": "42",
}

SEASONS = [
    "2022-2023",
    "2023-2024",
    "2024-2025",
    "2025-2026",
]


def season_label(date: pd.Timestamp) -> str:
    if date.month >= 8:
        start_year = date.year
    else:
        start_year = date.year - 1

    return f"{start_year}-{start_year + 1}"


def epiweek_label(date_value) -> str:
    timestamp = pd.Timestamp(date_value)
    week = Week.fromdate(timestamp.date())

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


def plot_state(df: pd.DataFrame, state_abbr: str) -> None:
    fips = STATE_FIPS[state_abbr]
    state_name = STATES_TO_CHECK[state_abbr]

    subset = df[
        (df["location"] == fips)
        & (df["season"].isin(SEASONS))
    ].copy()

    if subset.empty:
        print(f"No data found for {state_name}")
        return

    plt.figure(figsize=(13, 7))

    for season in SEASONS:
        season_df = subset[
            subset["season"] == season
        ].sort_values("date")

        if season_df.empty:
            continue

        plt.plot(
            season_df["date"],
            season_df["weekly_rate"],
            marker="o",
            markersize=3,
            label=season,
        )

        peak_row = season_df.loc[
            season_df["weekly_rate"].idxmax()
        ]

        plt.scatter(
            peak_row["date"],
            peak_row["weekly_rate"],
            s=90,
            zorder=5,
        )

        plt.annotate(
            epiweek_label(peak_row["date"]),
            (
                peak_row["date"],
                peak_row["weekly_rate"],
            ),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    plt.title(
        f"{state_name} Influenza Hospitalization Rate by Season"
    )
    plt.xlabel("Date")
    plt.ylabel("Weekly hospitalization rate")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    output_path = (
        PLOT_DIR
        / f"peak_sanity_{state_abbr.lower()}.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


def main() -> None:
    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    for state_abbr in STATES_TO_CHECK:
        plot_state(
            df=df,
            state_abbr=state_abbr,
        )


if __name__ == "__main__":
    main()