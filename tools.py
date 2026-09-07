from collections import Counter
from pathlib import Path

import pandas as pd
import yfinance as yf


FLU_DATA_PATH = Path("data") / "target-hospital-admissions.csv"


# -------------------------------------------------------------------
# STOCK DATA TOOL
# -------------------------------------------------------------------

def fetch_stock_data(symbol, period="1mo"):
    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(period=period)

        if history.empty:
            raise ValueError(
                f"No stock data returned for {symbol} with period {period}."
            )

        observations = []

        for date, row in history.iterrows():
            observations.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "close": round(float(row["Close"]), 2),
                }
            )

        return {
            "symbol": symbol,
            "period": period,
            "observations": observations,
        }

    except Exception as error:
        raise RuntimeError(
            f"Failed to fetch stock data for {symbol}: {error}"
        ) from error


# -------------------------------------------------------------------
# FLUSIGHT DATA LOADING
# -------------------------------------------------------------------

def load_flu_data():
    try:
        if not FLU_DATA_PATH.exists():
            raise FileNotFoundError(
                f"FluSight data file not found at: {FLU_DATA_PATH}"
            )

        data = pd.read_csv(FLU_DATA_PATH)

        required_columns = {
            "date",
            "location",
            "location_name",
            "value",
            "weekly_rate",
        }

        missing_columns = required_columns - set(data.columns)

        if missing_columns:
            raise ValueError(
                f"Dataset is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        data["date"] = pd.to_datetime(data["date"])

        data["value"] = pd.to_numeric(
            data["value"],
            errors="coerce",
        )

        data["weekly_rate"] = pd.to_numeric(
            data["weekly_rate"],
            errors="coerce",
        )

        data = data.sort_values(
            ["date", "location_name"]
        ).reset_index(drop=True)

        return data

    except Exception as error:
        raise RuntimeError(
            f"Failed to load FluSight data: {error}"
        ) from error


def assign_flu_season(date):
    if date.month >= 8:
        start_year = date.year
    else:
        start_year = date.year - 1

    return f"{start_year}-{start_year + 1}"


def prepare_flu_data():
    data = load_flu_data().copy()

    data["season"] = data[
        "date"
    ].apply(assign_flu_season)

    return data


# -------------------------------------------------------------------
# DATASET SUMMARY
# -------------------------------------------------------------------

def summarize_flu_dataset():
    data = prepare_flu_data()

    national_data = data[
        data["location"] == "US"
    ]

    state_data = data[
        data["location"] != "US"
    ]

    season_counts = (
        national_data
        .groupby("season")
        .size()
        .to_dict()
    )

    season_completeness = {}

    for season, count in season_counts.items():
        season_completeness[season] = {
            "observation_count": int(count),
            "is_partial": bool(count < 50),
        }

    return {
        "date_range": {
            "start": data[
                "date"
            ].min().strftime("%Y-%m-%d"),
            "end": data[
                "date"
            ].max().strftime("%Y-%m-%d"),
        },
        "row_count": len(data),
        "location_count": data[
            "location_name"
        ].nunique(),
        "state_level_location_count": (
            state_data[
                "location_name"
            ].nunique()
        ),
        "seasons": sorted(
            data[
                "season"
            ].unique().tolist()
        ),
        "season_count": data[
            "season"
        ].nunique(),
        "national_observation_count": len(
            national_data
        ),
        "season_completeness": (
            season_completeness
        ),
        "missing_values": {
            "value": int(
                data[
                    "value"
                ].isna().sum()
            ),
            "weekly_rate": int(
                data[
                    "weekly_rate"
                ].isna().sum()
            ),
        },
    }


# -------------------------------------------------------------------
# NATIONAL SEASONAL PEAKS
# -------------------------------------------------------------------

def summarize_national_seasons():
    """
    Summarize the US national peak for every season.

    Hospitalization count ('value') determines the peak.
    weekly_rate is included as a secondary normalized measure.
    """

    data = prepare_flu_data()

    national_data = data[
        data["location"] == "US"
    ].dropna(
        subset=["value"]
    )

    results = []

    for season, season_data in national_data.groupby(
        "season"
    ):
        if season_data.empty:
            continue

        peak_row = season_data.loc[
            season_data[
                "value"
            ].idxmax()
        ]

        results.append(
            {
                "season": season,
                "peak_date": (
                    peak_row[
                        "date"
                    ].strftime("%Y-%m-%d")
                ),
                "peak_hospital_admissions": (
                    int(
                        peak_row[
                            "value"
                        ]
                    )
                ),
                "peak_weekly_rate": (
                    round(
                        float(
                            peak_row[
                                "weekly_rate"
                            ]
                        ),
                        4,
                    )
                    if pd.notna(
                        peak_row[
                            "weekly_rate"
                        ]
                    )
                    else None
                ),
                "observation_count": len(
                    season_data
                ),
            }
        )

    return results


# -------------------------------------------------------------------
# NATIONAL COUNT-BASED TREND ANALYSIS
# -------------------------------------------------------------------

def summarize_national_season_trends():
    """
    Produce count-based US national trend summaries
    for every available flu season.

    The primary variable is 'value':
    weekly hospital admissions.

    weekly_rate remains available as a secondary
    normalized measure.
    """

    data = prepare_flu_data()

    national_data = (
        data[
            data["location"] == "US"
        ]
        .dropna(
            subset=["value"]
        )
        .sort_values("date")
    )

    dataset_summary = summarize_flu_dataset()

    results = []

    for season, season_data in national_data.groupby(
        "season"
    ):
        season_data = (
            season_data
            .sort_values("date")
            .copy()
        )

        if season_data.empty:
            continue

        first_row = season_data.iloc[0]
        latest_row = season_data.iloc[-1]

        peak_row = season_data.loc[
            season_data[
                "value"
            ].idxmax()
        ]

        latest_value = float(
            latest_row[
                "value"
            ]
        )

        previous_value = None
        weekly_change = None
        weekly_percent_change = None

        if len(season_data) >= 2:
            previous_row = (
                season_data.iloc[-2]
            )

            previous_value = float(
                previous_row[
                    "value"
                ]
            )

            weekly_change = (
                latest_value
                - previous_value
            )

            if previous_value != 0:
                weekly_percent_change = (
                    weekly_change
                    / previous_value
                    * 100
                )

        peak_value = float(
            peak_row[
                "value"
            ]
        )

        peak_date = pd.Timestamp(
            peak_row[
                "date"
            ]
        )

        latest_date = pd.Timestamp(
            latest_row[
                "date"
            ]
        )

        weeks_from_peak = int(
            (
                latest_date
                - peak_date
            ).days
            / 7
        )

        if latest_date < peak_date:
            phase = "pre-peak"

        elif latest_date == peak_date:
            phase = "at_peak"

        else:
            phase = "post-peak"

        change_from_peak = (
            latest_value
            - peak_value
        )

        percent_change_from_peak = None

        if peak_value != 0:
            percent_change_from_peak = (
                change_from_peak
                / peak_value
                * 100
            )

        season_metadata = (
            dataset_summary[
                "season_completeness"
            ].get(
                season,
                {},
            )
        )

        results.append(
            {
                "season": season,
                "is_partial": (
                    season_metadata.get(
                        "is_partial",
                        False,
                    )
                ),
                "start_date": (
                    first_row[
                        "date"
                    ].strftime("%Y-%m-%d")
                ),
                "end_date": (
                    latest_date.strftime(
                        "%Y-%m-%d"
                    )
                ),
                "observation_count": len(
                    season_data
                ),
                "latest_hospital_admissions": (
                    int(
                        latest_value
                    )
                ),
                "previous_week_hospital_admissions": (
                    int(
                        previous_value
                    )
                    if previous_value
                    is not None
                    else None
                ),
                "weekly_change_in_admissions": (
                    int(
                        weekly_change
                    )
                    if weekly_change
                    is not None
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
                "peak_date": (
                    peak_date.strftime(
                        "%Y-%m-%d"
                    )
                ),
                "peak_hospital_admissions": (
                    int(
                        peak_value
                    )
                ),
                "peak_weekly_rate": (
                    round(
                        float(
                            peak_row[
                                "weekly_rate"
                            ]
                        ),
                        4,
                    )
                    if pd.notna(
                        peak_row[
                            "weekly_rate"
                        ]
                    )
                    else None
                ),
                "season_phase": phase,
                "weeks_from_peak": (
                    weeks_from_peak
                ),
                "change_from_peak": (
                    int(
                        change_from_peak
                    )
                ),
                "percent_change_from_peak": (
                    round(
                        percent_change_from_peak,
                        2,
                    )
                    if percent_change_from_peak
                    is not None
                    else None
                ),
            }
        )

    results.sort(
        key=lambda item: item[
            "season"
        ]
    )

    return results


def summarize_current_season_trend():
    """
    Return the latest available flu season's
    national count-based trend summary.
    """

    season_trends = (
        summarize_national_season_trends()
    )

    if not season_trends:
        return None

    return season_trends[-1]


def compare_current_peak_to_past_seasons():
    """
    Compare the current national hospitalization-count
    peak against every prior season.

    Positive percent_difference means the current
    season peak was higher than that historical season.
    Negative means it was lower.
    """

    season_trends = (
        summarize_national_season_trends()
    )

    if len(season_trends) < 2:
        return []

    current = season_trends[-1]

    current_season = current[
        "season"
    ]

    current_peak = current[
        "peak_hospital_admissions"
    ]

    comparisons = []

    for past in season_trends[:-1]:
        past_peak = past[
            "peak_hospital_admissions"
        ]

        difference = (
            current_peak
            - past_peak
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
                "current_season": (
                    current_season
                ),
                "comparison_season": (
                    past[
                        "season"
                    ]
                ),
                "comparison_season_is_partial": (
                    past[
                        "is_partial"
                    ]
                ),
                "current_peak_hospital_admissions": (
                    current_peak
                ),
                "past_peak_hospital_admissions": (
                    past_peak
                ),
                "difference_in_peak_admissions": (
                    difference
                ),
                "percent_difference": (
                    round(
                        percent_difference,
                        2,
                    )
                    if percent_difference
                    is not None
                    else None
                ),
            }
        )

    return comparisons


# -------------------------------------------------------------------
# JURISDICTION PEAKS
# -------------------------------------------------------------------

def summarize_state_peaks(season):
    """
    Jurisdiction-level peak timing continues to use
    weekly_rate so states/jurisdictions of different
    population sizes remain comparable.

    Raw peak value is retained alongside the rate.
    """

    data = prepare_flu_data()

    state_data = data[
        (
            data["location"]
            != "US"
        )
        & (
            data["season"]
            == season
        )
    ].dropna(
        subset=["weekly_rate"]
    )

    summaries = []

    for (
        location_name,
        location_data,
    ) in state_data.groupby(
        "location_name"
    ):
        if location_data.empty:
            continue

        peak_row = location_data.loc[
            location_data[
                "weekly_rate"
            ].idxmax()
        ]

        summaries.append(
            {
                "location_name": (
                    location_name
                ),
                "peak_date": (
                    peak_row[
                        "date"
                    ].strftime(
                        "%Y-%m-%d"
                    )
                ),
                "peak_value": (
                    int(
                        peak_row[
                            "value"
                        ]
                    )
                    if pd.notna(
                        peak_row[
                            "value"
                        ]
                    )
                    else None
                ),
                "peak_weekly_rate": (
                    round(
                        float(
                            peak_row[
                                "weekly_rate"
                            ]
                        ),
                        4,
                    )
                ),
            }
        )

    summaries.sort(
        key=lambda item: item[
            "location_name"
        ]
    )

    return summaries


# -------------------------------------------------------------------
# PEAK TIMING
# -------------------------------------------------------------------

def summarize_peak_timing(season):
    state_peaks = summarize_state_peaks(
        season
    )

    peak_dates = [
        item[
            "peak_date"
        ]
        for item in state_peaks
    ]

    counts = Counter(
        peak_dates
    )

    distribution = [
        {
            "peak_date": date,
            "jurisdiction_count": (
                count
            ),
        }
        for date, count
        in sorted(
            counts.items()
        )
    ]

    if not distribution:
        return {
            "season": season,
            "dominant_peak_date": None,
            "dominant_peak_count": 0,
            "distribution": [],
        }

    (
        dominant_date,
        dominant_count,
    ) = max(
        counts.items(),
        key=lambda item: item[1],
    )

    return {
        "season": season,
        "dominant_peak_date": (
            dominant_date
        ),
        "dominant_peak_count": (
            dominant_count
        ),
        "distribution": (
            distribution
        ),
    }


def classify_peak_timing(
    season,
    tolerance_weeks=1,
):
    state_peaks = (
        summarize_state_peaks(
            season
        )
    )

    timing_summary = (
        summarize_peak_timing(
            season
        )
    )

    dominant_date_string = (
        timing_summary[
            "dominant_peak_date"
        ]
    )

    if dominant_date_string is None:
        return {
            "season": season,
            "dominant_peak_date": None,
            "tolerance_weeks": (
                tolerance_weeks
            ),
            "early_count": 0,
            "typical_count": 0,
            "late_count": 0,
            "early": [],
            "typical": [],
            "late": [],
        }

    dominant_date = pd.Timestamp(
        dominant_date_string
    )

    tolerance_days = (
        tolerance_weeks
        * 7
    )

    early = []
    typical = []
    late = []

    for item in state_peaks:
        peak_date = pd.Timestamp(
            item[
                "peak_date"
            ]
        )

        difference_days = (
            peak_date
            - dominant_date
        ).days

        entry = {
            "location_name": (
                item[
                    "location_name"
                ]
            ),
            "peak_date": (
                item[
                    "peak_date"
                ]
            ),
            "difference_days": (
                difference_days
            ),
            "peak_value": (
                item[
                    "peak_value"
                ]
            ),
            "peak_weekly_rate": (
                item[
                    "peak_weekly_rate"
                ]
            ),
        }

        if (
            difference_days
            < -tolerance_days
        ):
            early.append(entry)

        elif (
            difference_days
            > tolerance_days
        ):
            late.append(entry)

        else:
            typical.append(entry)

    return {
        "season": season,
        "dominant_peak_date": (
            dominant_date_string
        ),
        "tolerance_weeks": (
            tolerance_weeks
        ),
        "early_count": len(
            early
        ),
        "typical_count": len(
            typical
        ),
        "late_count": len(
            late
        ),
        "early": early,
        "typical": typical,
        "late": late,
    }


# -------------------------------------------------------------------
# RATE STATISTICS
# -------------------------------------------------------------------

def summarize_peak_rate_statistics(
    season,
):
    state_peaks = (
        summarize_state_peaks(
            season
        )
    )

    rates = [
        item[
            "peak_weekly_rate"
        ]
        for item in state_peaks
        if item[
            "peak_weekly_rate"
        ] is not None
    ]

    if not rates:
        return {
            "season": season,
            "jurisdiction_count": 0,
            "minimum_peak_weekly_rate": None,
            "maximum_peak_weekly_rate": None,
            "mean_peak_weekly_rate": None,
            "median_peak_weekly_rate": None,
        }

    rate_series = pd.Series(
        rates
    )

    return {
        "season": season,
        "jurisdiction_count": len(
            rates
        ),
        "minimum_peak_weekly_rate": (
            round(
                float(
                    rate_series.min()
                ),
                4,
            )
        ),
        "maximum_peak_weekly_rate": (
            round(
                float(
                    rate_series.max()
                ),
                4,
            )
        ),
        "mean_peak_weekly_rate": (
            round(
                float(
                    rate_series.mean()
                ),
                4,
            )
        ),
        "median_peak_weekly_rate": (
            round(
                float(
                    rate_series.median()
                ),
                4,
            )
        ),
    }


def summarize_timing_group_rates(
    season,
):
    groups = classify_peak_timing(
        season
    )

    result = {
        "season": season,
    }

    for group_name in [
        "early",
        "typical",
        "late",
    ]:
        entries = groups[
            group_name
        ]

        rates = [
            item[
                "peak_weekly_rate"
            ]
            for item in entries
            if item[
                "peak_weekly_rate"
            ] is not None
        ]

        if not rates:
            result[
                group_name
            ] = {
                "count": 0,
                "minimum_peak_weekly_rate": None,
                "maximum_peak_weekly_rate": None,
                "mean_peak_weekly_rate": None,
                "median_peak_weekly_rate": None,
            }

            continue

        rate_series = pd.Series(
            rates
        )

        result[
            group_name
        ] = {
            "count": len(
                rates
            ),
            "minimum_peak_weekly_rate": (
                round(
                    float(
                        rate_series.min()
                    ),
                    4,
                )
            ),
            "maximum_peak_weekly_rate": (
                round(
                    float(
                        rate_series.max()
                    ),
                    4,
                )
            ),
            "mean_peak_weekly_rate": (
                round(
                    float(
                        rate_series.mean()
                    ),
                    4,
                )
            ),
            "median_peak_weekly_rate": (
                round(
                    float(
                        rate_series.median()
                    ),
                    4,
                )
            ),
        }

    return result


def summarize_peak_rate_extremes(
    season,
    top_n=5,
):
    state_peaks = (
        summarize_state_peaks(
            season
        )
    )

    valid = [
        item
        for item in state_peaks
        if item[
            "peak_weekly_rate"
        ] is not None
    ]

    descending = sorted(
        valid,
        key=lambda item: (
            item[
                "peak_weekly_rate"
            ]
        ),
        reverse=True,
    )

    ascending = sorted(
        valid,
        key=lambda item: (
            item[
                "peak_weekly_rate"
            ]
        ),
    )

    return {
        "season": season,
        "highest_peak_rates": (
            descending[
                :top_n
            ]
        ),
        "lowest_peak_rates": (
            ascending[
                :top_n
            ]
        ),
    }


# -------------------------------------------------------------------
# CROSS-SEASON SUMMARY
# -------------------------------------------------------------------

def summarize_cross_season_peak_timing():
    dataset_summary = (
        summarize_flu_dataset()
    )

    results = []

    for season in (
        dataset_summary[
            "seasons"
        ]
    ):
        peak_timing = (
            summarize_peak_timing(
                season
            )
        )

        groups = (
            classify_peak_timing(
                season
            )
        )

        results.append(
            {
                "season": season,
                "is_partial": (
                    dataset_summary[
                        "season_completeness"
                    ][season][
                        "is_partial"
                    ]
                ),
                "dominant_peak_date": (
                    peak_timing[
                        "dominant_peak_date"
                    ]
                ),
                "dominant_peak_count": (
                    peak_timing[
                        "dominant_peak_count"
                    ]
                ),
                "peak_date_distribution": (
                    peak_timing[
                        "distribution"
                    ]
                ),
                "early_count": (
                    groups[
                        "early_count"
                    ]
                ),
                "typical_count": (
                    groups[
                        "typical_count"
                    ]
                ),
                "late_count": (
                    groups[
                        "late_count"
                    ]
                ),
                "early_peakers": [
                    item[
                        "location_name"
                    ]
                    for item
                    in groups[
                        "early"
                    ]
                ],
                "late_peakers": [
                    item[
                        "location_name"
                    ]
                    for item
                    in groups[
                        "late"
                    ]
                ],
            }
        )

    return results


# -------------------------------------------------------------------
# DETAILED SINGLE-SEASON SUMMARY
# -------------------------------------------------------------------

def build_detailed_season_summary(
    season,
):
    national_trends = (
        summarize_national_season_trends()
    )

    national_trend = next(
        (
            item
            for item in national_trends
            if item[
                "season"
            ] == season
        ),
        None,
    )

    return {
        "season": season,

        # National raw hospitalization-count trend
        "national_hospitalization_trend": (
            national_trend
        ),

        # Jurisdiction-level normalized-rate analysis
        "jurisdiction_count": len(
            summarize_state_peaks(
                season
            )
        ),
        "peak_timing": (
            summarize_peak_timing(
                season
            )
        ),
        "timing_groups": (
            classify_peak_timing(
                season
            )
        ),
        "peak_rate_statistics": (
            summarize_peak_rate_statistics(
                season
            )
        ),
        "timing_group_rate_statistics": (
            summarize_timing_group_rates(
                season
            )
        ),
        "peak_rate_extremes": (
            summarize_peak_rate_extremes(
                season
            )
        ),
        "state_peaks": (
            summarize_state_peaks(
                season
            )
        ),
    }


# -------------------------------------------------------------------
# SCOPE-AWARE FIRST-PASS SUMMARIES
# -------------------------------------------------------------------

def build_latest_season_summary():
    dataset_summary = (
        summarize_flu_dataset()
    )

    latest_season = (
        dataset_summary[
            "seasons"
        ][-1]
    )

    return {
        "analysis_scope": (
            "latest_season"
        ),
        "dataset_summary": (
            dataset_summary
        ),
        "latest_season": (
            latest_season
        ),

        # Primary national count analysis
        "current_season_trend": (
            summarize_current_season_trend()
        ),

        # Historical context for current peak
        "current_peak_vs_past_seasons": (
            compare_current_peak_to_past_seasons()
        ),

        # Full season/jurisdiction detail
        "latest_season_detail": (
            build_detailed_season_summary(
                latest_season
            )
        ),
    }


def build_cross_season_summary():
    dataset_summary = (
        summarize_flu_dataset()
    )

    return {
        "analysis_scope": (
            "cross_season"
        ),
        "dataset_summary": (
            dataset_summary
        ),

        # National raw-count trend summaries
        "national_season_trends": (
            summarize_national_season_trends()
        ),

        # Direct current-vs-history count comparison
        "current_peak_vs_past_seasons": (
            compare_current_peak_to_past_seasons()
        ),

        # Existing national peak summary
        "national_seasonal_peaks": (
            summarize_national_seasons()
        ),

        # Existing jurisdiction timing analysis
        "cross_season_peak_timing": (
            summarize_cross_season_peak_timing()
        ),
    }


def build_all_seasons_summary():
    dataset_summary = (
        summarize_flu_dataset()
    )

    detailed_seasons = []

    for season in (
        dataset_summary[
            "seasons"
        ]
    ):
        detailed_seasons.append(
            build_detailed_season_summary(
                season
            )
        )

    return {
        "analysis_scope": (
            "all_seasons"
        ),
        "dataset_summary": (
            dataset_summary
        ),

        # National hospitalization-count trends
        "national_season_trends": (
            summarize_national_season_trends()
        ),

        # Current season compared with historical peaks
        "current_peak_vs_past_seasons": (
            compare_current_peak_to_past_seasons()
        ),

        "national_seasonal_peaks": (
            summarize_national_seasons()
        ),

        "season_details": (
            detailed_seasons
        ),
    }


def build_flu_summary_for_scope(
    analysis_scope,
):
    """
    Build only the data required by the
    planner's selected scope.
    """

    if (
        analysis_scope
        == "latest_season"
    ):
        return (
            build_latest_season_summary()
        )

    if (
        analysis_scope
        == "cross_season"
    ):
        return (
            build_cross_season_summary()
        )

    if (
        analysis_scope
        == "all_seasons"
    ):
        return (
            build_all_seasons_summary()
        )

    raise ValueError(
        f"Unsupported FluSight analysis scope: "
        f"{analysis_scope}"
    )


# -------------------------------------------------------------------
# BACKWARD-COMPATIBLE SUMMARY
# -------------------------------------------------------------------

def build_flu_analysis_summary():
    """
    Retained so test_flu.py and older code
    still work.
    """

    return (
        build_cross_season_summary()
    )