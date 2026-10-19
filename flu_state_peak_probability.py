from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from epiweeks import Week

from flu_peak_statistics import (
    FIPS_TO_STATE,
    build_state_peak_table,
    load_data,
)


RESULT_DIR = Path("analysis_results")
PLOT_DIR = Path("plots")

CURVE_OUTPUT_PATH = (
    RESULT_DIR
    / "state_peak_probability_by_epiweek.csv"
)

FEATURE_OUTPUT_PATH = (
    RESULT_DIR
    / "state_weekly_peak_probability_features.csv"
)

BANDWIDTH_WEEKS = 1.5
FORECAST_SHIFT_WEEKS = 4

SANITY_STATES = [
    "ID",
    "MI",
    "WY",
]


def seasonal_epiweek_position(
    date: pd.Timestamp,
) -> int:
    """
    Convert epidemiological week to a continuous
    seasonal position.

    Examples:
        EW45 -> 45
        EW52 -> 52
        EW53 -> 53
        EW01 -> 54
        EW02 -> 55

    This preserves chronology across the
    calendar-year boundary.
    """
    week = Week.fromdate(
        date.date()
    ).week

    if week >= 31:
        return week

    return week + 53


def position_to_epiweek(
    position: int,
) -> str:
    """
    Convert seasonal position back into an
    epidemiological week label.
    """
    position = int(position)

    if position <= 53:
        week = position
    else:
        week = position - 53

    return f"EW{week:02d}"


def gaussian_kernel(
    x: np.ndarray,
    center: float,
    bandwidth: float,
) -> np.ndarray:
    """
    Gaussian kernel centered at one observed
    seasonal peak week.
    """
    return np.exp(
        -0.5
        * (
            (x - center)
            / bandwidth
        ) ** 2
    )


def build_state_probability_curve(
    state_peaks: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one state's smoothed probability distribution
    over seasonal epiweeks.

    Each observed seasonal peak contributes one
    Gaussian kernel.

    The resulting distribution is normalized so that
    peak_probability sums to 1 for that state.
    """
    positions = np.arange(
        31,
        84,
        dtype=int,
    )

    density = np.zeros(
        len(positions),
        dtype=float,
    )

    for peak_position in state_peaks[
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
            "State peak-probability density "
            "has zero total mass."
        )

    probability = (
        density / total
    )

    result = pd.DataFrame(
        {
            "seasonal_position": positions,
            "epiweek": [
                position_to_epiweek(
                    position
                )
                for position
                in positions
            ],
            "peak_probability": probability,
        }
    )

    return result


def add_shifted_feature(
    probability_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Shift the peak-probability distribution four
    weeks earlier.

    If peak probability is high at time t + 4,
    then the shifted feature is high at time t.

    Therefore:

        shifted_feature(t)
            = peak_probability(t + 4)

    This exposes likely future peak timing to the
    forecasting model four weeks in advance.
    """
    result = (
        probability_df.copy()
    )

    lookup = dict(
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
            int(position)
            + FORECAST_SHIFT_WEEKS
        )

        shifted_values.append(
            float(
                lookup.get(
                    future_position,
                    0.0,
                )
            )
        )

    result[
        "peak_probability_shifted_4w"
    ] = shifted_values

    return result


def build_all_state_curves(
    peaks: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one peak-probability curve for every
    state/DC.
    """
    all_curves = []

    for state_abbr, group in peaks.groupby(
        "state_abbr"
    ):
        group = (
            group.copy()
            .sort_values("season")
        )

        probability_df = (
            build_state_probability_curve(
                group
            )
        )

        probability_df = (
            add_shifted_feature(
                probability_df
            )
        )

        probability_df[
            "state_abbr"
        ] = state_abbr

        probability_df[
            "location"
        ] = group[
            "location"
        ].iloc[0]

        probability_df[
            "location_name"
        ] = group[
            "location_name"
        ].iloc[0]

        probability_df[
            "seasons_observed"
        ] = len(group)

        probability_df[
            "observed_peak_positions"
        ] = ",".join(
            str(int(value))
            for value
            in group[
                "peak_position"
            ].tolist()
        )

        probability_df[
            "observed_peak_epiweeks"
        ] = ",".join(
            group[
                "peak_epiweek"
            ].astype(str)
        )

        all_curves.append(
            probability_df
        )

    curves = pd.concat(
        all_curves,
        ignore_index=True,
    )

    return curves[
        [
            "location",
            "state_abbr",
            "location_name",
            "seasons_observed",
            "seasonal_position",
            "epiweek",
            "peak_probability",
            "peak_probability_shifted_4w",
            "observed_peak_positions",
            "observed_peak_epiweeks",
        ]
    ]


def build_weekly_feature_table(
    df: pd.DataFrame,
    curves: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach each state's seasonal peak-probability
    features to every weekly observation.

    The same state-specific seasonal prior is repeated
    across seasons.

    This is intentionally the exploratory retrospective
    version discussed with Adiga.
    """
    states = df[
        df["location"].isin(
            FIPS_TO_STATE
        )
    ].copy()

    states["seasonal_position"] = (
        states["date"]
        .apply(
            seasonal_epiweek_position
        )
    )

    feature_columns = curves[
        [
            "location",
            "seasonal_position",
            "peak_probability",
            "peak_probability_shifted_4w",
        ]
    ].copy()

    weekly = states.merge(
        feature_columns,
        on=[
            "location",
            "seasonal_position",
        ],
        how="left",
        validate="many_to_one",
    )

    missing_original = int(
        weekly[
            "peak_probability"
        ].isna().sum()
    )

    missing_shifted = int(
        weekly[
            "peak_probability_shifted_4w"
        ].isna().sum()
    )

    if (
        missing_original > 0
        or missing_shifted > 0
    ):
        raise ValueError(
            "Missing probability features after merge: "
            f"original={missing_original}, "
            f"shifted={missing_shifted}"
        )

    return weekly


def print_state_summary(
    peaks: pd.DataFrame,
    curves: pd.DataFrame,
) -> None:
    print(
        "\nState Peak Probability Summary"
    )
    print(
        "=============================="
    )

    print(
        "States/DC represented: "
        f"{curves['state_abbr'].nunique()}"
    )

    print(
        "Historical peak observations: "
        f"{len(peaks)}"
    )

    for state_abbr in SANITY_STATES:
        state_peaks = peaks[
            peaks[
                "state_abbr"
            ] == state_abbr
        ].sort_values(
            "season"
        )

        state_curve = curves[
            curves[
                "state_abbr"
            ] == state_abbr
        ]

        if (
            state_peaks.empty
            or state_curve.empty
        ):
            print(
                f"\n{state_abbr}: "
                "no data available"
            )
            continue

        original_peak = (
            state_curve.loc[
                state_curve[
                    "peak_probability"
                ].idxmax()
            ]
        )

        shifted_peak = (
            state_curve.loc[
                state_curve[
                    "peak_probability_shifted_4w"
                ].idxmax()
            ]
        )

        print(
            f"\n{state_abbr}"
        )
        print(
            "-" * len(state_abbr)
        )

        print(
            "Observed peaks:"
        )

        for row in state_peaks.itertuples():
            print(
                f"  {row.season}: "
                f"{row.peak_epiweek}"
            )

        print(
            "Highest original probability: "
            f"{original_peak['epiweek']}"
        )

        print(
            "Highest shifted feature: "
            f"{shifted_peak['epiweek']}"
        )

        print(
            "Probability sum: "
            f"{state_curve['peak_probability'].sum():.6f}"
        )


def plot_state_curve(
    state_abbr: str,
    peaks: pd.DataFrame,
    curves: pd.DataFrame,
) -> None:
    state_curve = curves[
        curves[
            "state_abbr"
        ] == state_abbr
    ].copy()

    state_peaks = peaks[
        peaks[
            "state_abbr"
        ] == state_abbr
    ].copy()

    if (
        state_curve.empty
        or state_peaks.empty
    ):
        print(
            f"No plot generated for {state_abbr}"
        )
        return

    state_name = (
        state_curve[
            "location_name"
        ].iloc[0]
    )

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    ax.plot(
        state_curve[
            "seasonal_position"
        ],
        state_curve[
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
        state_curve[
            "seasonal_position"
        ],
        state_curve[
            "peak_probability_shifted_4w"
        ],
        marker="o",
        markersize=3,
        linewidth=2,
        label=(
            "4-week-ahead shifted feature"
        ),
    )

    for row in state_peaks.itertuples():
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

    ax.set_xticks(
        tick_positions,
        [
            position_to_epiweek(
                position
            )
            for position
            in tick_positions
        ],
    )

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
        f"{state_name}: Peak Probability "
        "and 4-Week-Ahead Shifted Feature"
    )

    ax.legend()

    ax.grid(
        alpha=0.25,
    )

    fig.tight_layout()

    output_path = (
        PLOT_DIR
        / (
            "state_peak_probability_"
            f"{state_abbr.lower()}.png"
        )
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


def save_results(
    curves: pd.DataFrame,
    weekly_features: pd.DataFrame,
) -> None:
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    curves.to_csv(
        CURVE_OUTPUT_PATH,
        index=False,
    )

    weekly_features.to_csv(
        FEATURE_OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved: {CURVE_OUTPUT_PATH}"
    )

    print(
        f"Saved: {FEATURE_OUTPUT_PATH}"
    )


def main():
    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    peaks = build_state_peak_table(
        df
    )

    curves = build_all_state_curves(
        peaks
    )

    weekly_features = (
        build_weekly_feature_table(
            df=df,
            curves=curves,
        )
    )

    print_state_summary(
        peaks=peaks,
        curves=curves,
    )

    save_results(
        curves=curves,
        weekly_features=weekly_features,
    )

    for state_abbr in SANITY_STATES:
        plot_state_curve(
            state_abbr=state_abbr,
            peaks=peaks,
            curves=curves,
        )


if __name__ == "__main__":
    main()