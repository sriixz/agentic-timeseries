from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from epiweeks import Week


HOSPITAL_PATH = Path(
    "data/target-hospital-admissions.csv"
)

ED_SUMMARY_PATH = Path(
    "forecast_results/allstate_lstm_ed_summary.csv"
)

RESULT_DIR = Path("forecast_results")
PLOT_DIR = Path("plots")


TARGET_COLUMN = "weekly_rate"

# Use complete pre-test seasons only.
HISTORICAL_SEASONS = [
    "2022-23",
    "2023-24",
    "2024-25",
]

TEST_SEASON = "2025-26"


def seasonal_epiweek_position(
    date_value,
) -> int:
    timestamp = pd.Timestamp(
        date_value
    )

    week = Week.fromdate(
        timestamp.date()
    ).week

    # Handle the Dec/Jan wrap cleanly.
    if week >= 31:
        return week

    return week + 53


def flu_season_label(
    date_value,
) -> str:
    timestamp = pd.Timestamp(
        date_value
    )

    year = timestamp.year

    # Flu season runs across calendar years.
    # July-Dec belongs to season starting
    # in that calendar year.
    if timestamp.month >= 7:
        start_year = year
    else:
        start_year = year - 1

    end_year = (
        start_year + 1
    )

    return (
        f"{start_year}-"
        f"{str(end_year)[-2:]}"
    )


def load_hospital_data() -> pd.DataFrame:
    if not HOSPITAL_PATH.exists():
        raise FileNotFoundError(
            f"Missing hospitalization data: "
            f"{HOSPITAL_PATH}"
        )

    df = pd.read_csv(
        HOSPITAL_PATH,
        dtype={
            "location": str,
        },
    )

    required = {
        "date",
        "location",
        "location_name",
        TARGET_COLUMN,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Hospitalization data missing "
            f"columns: {sorted(missing)}"
        )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df["location"] = (
        df["location"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["location"] != "US"
    ].copy()

    df = df.dropna(
        subset=[
            "date",
            TARGET_COLUMN,
        ]
    )

    df["season"] = (
        df["date"]
        .apply(
            flu_season_label
        )
    )

    df[
        "seasonal_position"
    ] = (
        df["date"]
        .apply(
            seasonal_epiweek_position
        )
    )

    return df


def load_ed_summary() -> pd.DataFrame:
    if not ED_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing ED summary: "
            f"{ED_SUMMARY_PATH}"
        )

    df = pd.read_csv(
        ED_SUMMARY_PATH,
        dtype={
            "location": str,
        },
    )

    required = {
        "location",
        "location_name",
        "horizon",
        "baseline_rmse",
        "ed_rmse",
        "rmse_improvement_pct",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "ED summary missing columns: "
            f"{sorted(missing)}"
        )

    df = df[
        df["horizon"] == 4
    ].copy()

    if df.empty:
        raise ValueError(
            "No 4-week ED results found."
        )

    return df


def find_season_peaks(
    hospital_df: pd.DataFrame,
) -> pd.DataFrame:
    scoped = hospital_df[
        hospital_df["season"].isin(
            HISTORICAL_SEASONS
            + [TEST_SEASON]
        )
    ].copy()

    if scoped.empty:
        raise ValueError(
            "No requested seasons found."
        )

    peak_rows = []

    for (
        location,
        location_name,
        season
    ), group in scoped.groupby(
        [
            "location",
            "location_name",
            "season",
        ]
    ):
        group = (
            group
            .sort_values("date")
            .copy()
        )

        peak_index = (
            group[
                TARGET_COLUMN
            ]
            .idxmax()
        )

        peak = group.loc[
            peak_index
        ]

        peak_rows.append(
            {
                "location":
                    location,

                "location_name":
                    location_name,

                "season":
                    season,

                "peak_date":
                    peak["date"],

                "peak_rate":
                    float(
                        peak[
                            TARGET_COLUMN
                        ]
                    ),

                "peak_position":
                    int(
                        peak[
                            "seasonal_position"
                        ]
                    ),
            }
        )

    return pd.DataFrame(
        peak_rows
    )


def median_absolute_deviation(
    values: pd.Series,
) -> float:
    median = float(
        values.median()
    )

    return float(
        (
            values
            - median
        )
        .abs()
        .median()
    )


def summarize_peak_behavior(
    peak_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        location,
        location_name
    ), group in peak_df.groupby(
        [
            "location",
            "location_name",
        ]
    ):
        historical = group[
            group["season"].isin(
                HISTORICAL_SEASONS
            )
        ].copy()

        test = group[
            group["season"]
            == TEST_SEASON
        ].copy()

        if len(historical) < 2:
            continue

        if len(test) != 1:
            continue

        historical_positions = (
            historical[
                "peak_position"
            ]
            .astype(float)
        )

        q1 = float(
            historical_positions
            .quantile(0.25)
        )

        q3 = float(
            historical_positions
            .quantile(0.75)
        )

        historical_median = float(
            historical_positions
            .median()
        )

        historical_iqr = (
            q3 - q1
        )

        historical_mad = (
            median_absolute_deviation(
                historical_positions
            )
        )

        historical_range = float(
            historical_positions.max()
            - historical_positions.min()
        )

        test_peak_position = float(
            test.iloc[0][
                "peak_position"
            ]
        )

        test_peak_date = (
            test.iloc[0][
                "peak_date"
            ]
        )

        test_peak_rate = float(
            test.iloc[0][
                "peak_rate"
            ]
        )

        test_deviation = abs(
            test_peak_position
            - historical_median
        )

        rows.append(
            {
                "location":
                    location,

                "location_name":
                    location_name,

                "historical_peak_count":
                    len(
                        historical_positions
                    ),

                "historical_peak_median":
                    historical_median,

                "historical_peak_iqr":
                    historical_iqr,

                "historical_peak_mad":
                    historical_mad,

                "historical_peak_range":
                    historical_range,

                "test_peak_position":
                    test_peak_position,

                "test_peak_date":
                    test_peak_date,

                "test_peak_rate":
                    test_peak_rate,

                "test_peak_deviation_weeks":
                    test_deviation,
            }
        )

    return pd.DataFrame(
        rows
    )


def pearson_corr(
    x: pd.Series,
    y: pd.Series,
) -> float:
    valid = (
        x.notna()
        & y.notna()
    )

    if valid.sum() < 2:
        return float("nan")

    return float(
        x[valid]
        .corr(
            y[valid]
        )
    )


def spearman_corr(
    x: pd.Series,
    y: pd.Series,
) -> float:
    valid = (
        x.notna()
        & y.notna()
    )

    if valid.sum() < 2:
        return float("nan")

    x_rank = (
        x[valid]
        .rank(
            method="average"
        )
    )

    y_rank = (
        y[valid]
        .rank(
            method="average"
        )
    )

    return float(
        x_rank.corr(
            y_rank
        )
    )


def print_correlation(
    df: pd.DataFrame,
    feature_column: str,
    label: str,
) -> None:
    x = df[
        feature_column
    ]

    y = df[
        "rmse_improvement_pct"
    ]

    pearson = (
        pearson_corr(
            x,
            y,
        )
    )

    spearman = (
        spearman_corr(
            x,
            y,
        )
    )

    print(
        f"\n{label}"
    )

    print(
        f"  Pearson r:  "
        f"{pearson:+.3f}"
    )

    print(
        f"  Spearman ρ: "
        f"{spearman:+.3f}"
    )


def annotate_extremes(
    ax,
    df: pd.DataFrame,
    x_column: str,
) -> None:
    # Label several strongest and weakest
    # ED-feature outcomes without cluttering
    # every state.
    best = (
        df.nlargest(
            3,
            "rmse_improvement_pct",
        )
    )

    worst = (
        df.nsmallest(
            3,
            "rmse_improvement_pct",
        )
    )

    selected = pd.concat(
        [
            best,
            worst,
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
                getattr(
                    row,
                    x_column
                ),
                row.rmse_improvement_pct,
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


def add_trend_line(
    ax,
    df: pd.DataFrame,
    x_column: str,
) -> None:
    scoped = df[
        [
            x_column,
            "rmse_improvement_pct",
        ]
    ].dropna()

    if len(scoped) < 2:
        return

    x = scoped[
        x_column
    ].to_numpy(
        dtype=float
    )

    y = scoped[
        "rmse_improvement_pct"
    ].to_numpy(
        dtype=float
    )

    if np.allclose(
        x,
        x[0]
    ):
        return

    slope, intercept = (
        np.polyfit(
            x,
            y,
            1,
        )
    )

    x_line = np.linspace(
        x.min(),
        x.max(),
        100,
    )

    y_line = (
        slope
        * x_line
        + intercept
    )

    ax.plot(
        x_line,
        y_line,
        linestyle="--",
        linewidth=1.5,
        label="Linear trend",
    )


def plot_relationship(
    df: pd.DataFrame,
    x_column: str,
    x_label: str,
    title: str,
    output_name: str,
) -> None:
    scoped = df[
        [
            "location",
            "location_name",
            x_column,
            "rmse_improvement_pct",
        ]
    ].dropna().copy()

    fig, ax = plt.subplots(
        figsize=(
            10,
            7,
        )
    )

    ax.scatter(
        scoped[
            x_column
        ],
        scoped[
            "rmse_improvement_pct"
        ],
        s=55,
        alpha=0.8,
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    add_trend_line(
        ax,
        scoped,
        x_column,
    )

    annotate_extremes(
        ax,
        scoped,
        x_column,
    )

    ax.set_xlabel(
        x_label
    )

    ax.set_ylabel(
        "4-week RMSE improvement "
        "from ED feature (%)"
    )

    ax.set_title(
        title
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    path = (
        PLOT_DIR
        / output_name
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {path}"
    )


def main():
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    hospital_df = (
        load_hospital_data()
    )

    ed_summary = (
        load_ed_summary()
    )

    peak_df = (
        find_season_peaks(
            hospital_df
        )
    )

    peak_summary = (
        summarize_peak_behavior(
            peak_df
        )
    )

    analysis = (
        ed_summary.merge(
            peak_summary,
            on=[
                "location",
                "location_name",
            ],
            how="inner",
            validate="one_to_one",
        )
    )

    analysis = (
        analysis
        .sort_values(
            "rmse_improvement_pct",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    median_improvement = float(
        analysis[
            "rmse_improvement_pct"
        ].median()
    )

    mean_improvement = float(
        analysis[
            "rmse_improvement_pct"
        ].mean()
    )

    improved_count = int(
        (
            analysis[
                "rmse_improvement_pct"
            ]
            > 0
        ).sum()
    )

    print(
        "\nED Feature State Analysis"
    )

    print(
        "========================="
    )

    print(
        f"States analyzed: "
        f"{len(analysis)}"
    )

    print(
        f"States improved: "
        f"{improved_count}/"
        f"{len(analysis)}"
    )

    print(
        "Mean 4-week RMSE improvement: "
        f"{mean_improvement:+.1f}%"
    )

    print(
        "Median 4-week RMSE improvement: "
        f"{median_improvement:+.1f}%"
    )

    print_correlation(
        analysis,
        "historical_peak_iqr",
        "RMSE improvement vs "
        "historical peak IQR",
    )

    print_correlation(
        analysis,
        "historical_peak_mad",
        "RMSE improvement vs "
        "historical peak MAD",
    )

    print_correlation(
        analysis,
        "historical_peak_range",
        "RMSE improvement vs "
        "historical peak range",
    )

    print_correlation(
        analysis,
        "test_peak_deviation_weeks",
        "RMSE improvement vs "
        "2025-26 peak atypicality",
    )

    print(
        "\nMost atypical 2025-26 peaks"
    )

    print(
        analysis[
            [
                "location_name",
                "historical_peak_median",
                "test_peak_position",
                "test_peak_deviation_weeks",
                "rmse_improvement_pct",
            ]
        ]
        .sort_values(
            "test_peak_deviation_weeks",
            ascending=False,
        )
        .head(10)
        .to_string(
            index=False
        )
    )

    output_path = (
        RESULT_DIR
        / "allstate_ed_peak_variability_analysis.csv"
    )

    analysis.to_csv(
        output_path,
        index=False,
    )

    plot_relationship(
        analysis,
        x_column=(
            "historical_peak_iqr"
        ),
        x_label=(
            "Historical peak timing IQR "
            "(weeks)"
        ),
        title=(
            "ED-Feature Benefit vs "
            "Historical Peak-Timing Variability"
        ),
        output_name=(
            "ed_rmse_vs_peak_variability.png"
        ),
    )

    plot_relationship(
        analysis,
        x_column=(
            "test_peak_deviation_weeks"
        ),
        x_label=(
            "2025-26 peak deviation from "
            "historical median (weeks)"
        ),
        title=(
            "ED-Feature Benefit vs "
            "2025-26 Peak Atypicality"
        ),
        output_name=(
            "ed_rmse_vs_test_peak_atypicality.png"
        ),
    )

    print(
        f"\nSaved: {output_path}"
    )


if __name__ == "__main__":
    main()