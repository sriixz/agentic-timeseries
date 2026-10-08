from pathlib import Path

import pandas as pd
import plotly.express as px
from epiweeks import Week


DATA_PATH = Path("data/target-hospital-admissions.csv")
PLOT_DIR = Path("plots")
RESULT_DIR = Path("analysis_results")

SEASONS_TO_PLOT = [
    "2022-2023",
    "2023-2024",
    "2024-2025",
    "2025-2026",
]


# CDC / Census state FIPS -> USPS abbreviation.
# Includes the 50 states and Washington, D.C.
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
    Convert a date into an August-to-July influenza season label.

    Example:
        2025-10-04 -> 2025-2026
        2026-01-03 -> 2025-2026
    """
    if date.month >= 8:
        start_year = date.year
    else:
        start_year = date.year - 1

    return f"{start_year}-{start_year + 1}"


def seasonal_epiweek_position(date: pd.Timestamp) -> int:
    """
    Convert an epidemiological week into a continuous seasonal position.

    Flu seasons begin around late summer / early fall, so ordinary epiweek
    numbers wrap from EW52/EW53 back to EW01. To preserve chronological order:

        EW31 -> 31
        EW52 -> 52
        EW01 -> 54
        EW02 -> 55
        ...
        EW30 -> 83

    This lets all seasons use the same chronological color scale.
    """
    epiweek = Week.fromdate(date.date()).week

    if epiweek >= 31:
        return epiweek

    return epiweek + 53


def epiweek_label(date: pd.Timestamp) -> str:
    week = Week.fromdate(date.date())
    return f"{week.year}-EW{week.week:02d}"


def colorbar_label(position: int) -> str:
    """
    Convert continuous seasonal epiweek position back into an EW label.
    """
    if position <= 53:
        week = position
    else:
        week = position - 53

    return f"EW{week:02d}"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find FluSight data at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {
        "date",
        "location",
        "location_name",
        "value",
        "weekly_rate",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    df["date"] = pd.to_datetime(df["date"])

    # Keep location codes as strings and preserve leading zeros.
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
    Find each state's peak weekly hospitalization rate within each season.

    weekly_rate is used for jurisdiction-level analysis because it provides
    a normalized state-level measure. The date of the peak is the main
    quantity used in the choropleth.
    """
    states = df[df["location"].isin(FIPS_TO_STATE)].copy()

    states = states.dropna(
        subset=["weekly_rate", "date"]
    )

    peak_indices = (
        states.groupby(["season", "location"])["weekly_rate"]
        .idxmax()
    )

    peaks = states.loc[peak_indices].copy()

    peaks["state_abbr"] = peaks["location"].map(FIPS_TO_STATE)
    peaks["peak_epiweek"] = peaks["date"].apply(epiweek_label)
    peaks["seasonal_epiweek_position"] = peaks["date"].apply(
        seasonal_epiweek_position
    )

    peaks = peaks[
        [
            "season",
            "location",
            "state_abbr",
            "location_name",
            "date",
            "peak_epiweek",
            "seasonal_epiweek_position",
            "weekly_rate",
            "value",
        ]
    ].sort_values(
        ["season", "seasonal_epiweek_position", "state_abbr"]
    )

    return peaks


def create_choropleth(
    season_peaks: pd.DataFrame,
    season: str,
    color_min: int,
    color_max: int,
) -> None:
    """
    Create one U.S. choropleth for a single influenza season.

    All seasons use the same color range so colors are comparable across maps.
    """
    fig = px.choropleth(
        season_peaks,
        locations="state_abbr",
        locationmode="USA-states",
        color="seasonal_epiweek_position",
        scope="usa",
        hover_name="location_name",
        hover_data={
            "state_abbr": False,
            "seasonal_epiweek_position": False,
            "peak_epiweek": True,
            "date": "|%Y-%m-%d",
            "weekly_rate": ":.3f",
            "value": ":,.0f",
        },
        color_continuous_scale="Viridis",
        range_color=(color_min, color_max),
        labels={
            "peak_epiweek": "Peak epiweek",
            "date": "Peak date",
            "weekly_rate": "Peak weekly rate",
            "value": "Peak admissions",
            "seasonal_epiweek_position": "Peak timing",
        },
        title=f"State Influenza Hospitalization Peak Timing — {season}",
    )

    # Fixed tick positions make the maps directly comparable.
    tick_positions = [
    position
    for position in [45, 49, 52, 54, 58, 62, 65]
    if color_min <= position <= color_max
    ]

    fig.update_coloraxes(
        colorbar=dict(
            title="Peak epiweek",
            tickvals=tick_positions,
            ticktext=[
                colorbar_label(position)
                for position in tick_positions
            ],
        )
    )

    fig.update_layout(
        title_x=0.5,
        margin=dict(l=20, r=20, t=70, b=20),
    )

    html_path = (
        PLOT_DIR
        / f"state_peak_epiweek_{season.replace('-', '_')}.html"
    )

    png_path = (
        PLOT_DIR
        / f"state_peak_epiweek_{season.replace('-', '_')}.png"
    )

    fig.write_html(html_path)

    try:
        fig.write_image(
            png_path,
            width=1200,
            height=750,
            scale=2,
        )
        print(f"Saved: {png_path}")
    except Exception as exc:
        print(
            "PNG export failed, but the interactive HTML map was saved."
        )
        print(f"Reason: {exc}")

    print(f"Saved: {html_path}")


def save_peak_table(peaks: pd.DataFrame) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULT_DIR / "state_peak_epiweeks_by_season.csv"

    peaks.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")


def print_season_summary(
    peaks: pd.DataFrame,
    season: str,
) -> None:
    subset = peaks[peaks["season"] == season]

    if subset.empty:
        print(f"\n{season}: no state data found")
        return

    earliest = subset.loc[
        subset["seasonal_epiweek_position"].idxmin()
    ]

    latest = subset.loc[
        subset["seasonal_epiweek_position"].idxmax()
    ]

    print(f"\n{season}")
    print("-" * len(season))
    print(f"States/DC represented: {len(subset)}")
    print(
        "Earliest peak: "
        f"{earliest['location_name']} "
        f"({earliest['peak_epiweek']})"
    )
    print(
        "Latest peak: "
        f"{latest['location_name']} "
        f"({latest['peak_epiweek']})"
    )


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    peaks = build_state_peak_table(df)

    available = set(peaks["season"])

    selected_seasons = [
        season
        for season in SEASONS_TO_PLOT
        if season in available
    ]

    if not selected_seasons:
        raise ValueError(
            "None of the requested seasons were found in the dataset."
        )

    selected = peaks[
        peaks["season"].isin(selected_seasons)
    ].copy()

    # Use ONE global color scale across all maps.
    # This is critical for meaningful cross-season visual comparison.
    color_min = int(
        selected["seasonal_epiweek_position"].min()
    )
    color_max = int(
        selected["seasonal_epiweek_position"].max()
    )

    print("\nState Peak Epiweek Choropleths")
    print("==============================")
    print(
        f"Shared seasonal color range: "
        f"{colorbar_label(color_min)} "
        f"through {colorbar_label(color_max)}"
    )

    save_peak_table(peaks)

    for season in selected_seasons:
        season_peaks = peaks[
            peaks["season"] == season
        ].copy()

        print_season_summary(peaks, season)

        create_choropleth(
            season_peaks=season_peaks,
            season=season,
            color_min=color_min,
            color_max=color_max,
        )


if __name__ == "__main__":
    main()