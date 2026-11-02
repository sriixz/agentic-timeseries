from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HOSPITAL_PATH = Path(
    "data/target-hospital-admissions.csv"
)

ED_PATH = Path(
    "data/target-ed-visits-prop.csv"
)

LSTM_SUMMARY_PATH = Path(
    "forecast_results/allstate_lstm_ed_summary.csv"
)

OUTPUT_CSV = Path(
    "forecast_results/allstate_ed_hospital_leadlag.csv"
)

OVERLAY_PLOT = Path(
    "plots/ed_hospital_overlay_examples.png"
)

SCATTER_PLOT = Path(
    "plots/ed_lead_vs_lstm_improvement.png"
)

TEST_START = pd.Timestamp("2025-10-01")
TEST_END = pd.Timestamp("2026-05-31")

REPRESENTATIVE_STATES = [
    "Idaho",
    "Louisiana",
    "New York",
    "Virginia",
    "South Dakota",
    "Florida",
]


def zscore(series):
    series = series.astype(float)

    std = series.std()

    if pd.isna(std) or std == 0:
        return pd.Series(
            np.zeros(len(series)),
            index=series.index,
        )

    return (
        series - series.mean()
    ) / std


def load_data():
    hospital = pd.read_csv(
        HOSPITAL_PATH,
        dtype={
            "location": str,
        },
    )

    ed = pd.read_csv(
        ED_PATH,
        dtype={
            "location": str,
        },
    )

    hospital["date"] = pd.to_datetime(
        hospital["date"]
    )

    ed["date"] = pd.to_datetime(
        ed["date"]
    )

    hospital["location"] = (
        hospital["location"]
        .astype(str)
        .str.strip()
    )

    ed["location"] = (
        ed["location"]
        .astype(str)
        .str.strip()
    )

    ed = ed.rename(
        columns={
            "value":
                "ed_visits_prop"
        }
    )

    return hospital, ed


def build_state_series(
    hospital,
    ed,
    location,
):
    h = hospital[
        hospital["location"] == location
    ][
        [
            "date",
            "location",
            "location_name",
            "weekly_rate",
        ]
    ].copy()

    e = ed[
        ed["location"] == location
    ][
        [
            "date",
            "location",
            "ed_visits_prop",
        ]
    ].copy()

    merged = h.merge(
        e,
        on=[
            "date",
            "location",
        ],
        how="inner",
        validate="one_to_one",
    )

    merged = merged[
        (
            merged["date"]
            >= TEST_START
        )
        & (
            merged["date"]
            <= TEST_END
        )
    ].copy()

    merged = (
        merged
        .dropna(
            subset=[
                "weekly_rate",
                "ed_visits_prop",
            ]
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    return merged


def peak_lead_weeks(df):
    if df.empty:
        return None

    hospital_idx = (
        df["weekly_rate"]
        .idxmax()
    )

    ed_idx = (
        df["ed_visits_prop"]
        .idxmax()
    )

    hospital_peak_date = (
        df.loc[
            hospital_idx,
            "date",
        ]
    )

    ed_peak_date = (
        df.loc[
            ed_idx,
            "date",
        ]
    )

    lead_weeks = (
        hospital_peak_date
        - ed_peak_date
    ).days / 7.0

    return {
        "hospital_peak_date":
            hospital_peak_date,

        "ed_peak_date":
            ed_peak_date,

        "ed_lead_weeks":
            lead_weeks,
    }


def main():
    hospital, ed = load_data()

    lstm_summary = pd.read_csv(
        LSTM_SUMMARY_PATH,
        dtype={
            "location": str,
        },
    )

    lstm_4w = lstm_summary[
        lstm_summary["horizon"] == 4
    ][
        [
            "location",
            "location_name",
            "rmse_improvement_pct",
        ]
    ].copy()

    rows = []
    overlay_data = {}

    for row in (
        lstm_4w
        .itertuples(
            index=False
        )
    ):
        state_df = build_state_series(
            hospital,
            ed,
            row.location,
        )

        if state_df.empty:
            continue

        peak_info = peak_lead_weeks(
            state_df
        )

        if peak_info is None:
            continue

        rows.append(
            {
                "location":
                    row.location,

                "location_name":
                    row.location_name,

                "hospital_peak_date":
                    peak_info[
                        "hospital_peak_date"
                    ],

                "ed_peak_date":
                    peak_info[
                        "ed_peak_date"
                    ],

                "ed_lead_weeks":
                    peak_info[
                        "ed_lead_weeks"
                    ],

                "lstm_4w_rmse_improvement_pct":
                    row.rmse_improvement_pct,
            }
        )

        if (
            row.location_name
            in REPRESENTATIVE_STATES
        ):
            overlay_data[
                row.location_name
            ] = state_df

    leadlag = pd.DataFrame(
        rows
    )

    leadlag = (
        leadlag
        .sort_values(
            "ed_lead_weeks",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OVERLAY_PLOT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    leadlag.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(
        "\nED vs Hospitalization Lead/Lag"
    )

    print(
        "=============================="
    )

    print(
        f"States analyzed: "
        f"{len(leadlag)}"
    )

    median_lead = (
        leadlag[
            "ed_lead_weeks"
        ]
        .median()
    )

    mean_lead = (
        leadlag[
            "ed_lead_weeks"
        ]
        .mean()
    )

    print(
        f"Median ED lead: "
        f"{median_lead:+.1f} weeks"
    )

    print(
        f"Mean ED lead: "
        f"{mean_lead:+.1f} weeks"
    )

    print()

    exact_counts = (
        leadlag[
            "ed_lead_weeks"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "Lead/lag distribution"
    )

    print(
        "---------------------"
    )

    for lead, count in (
        exact_counts.items()
    ):
        if lead > 0:
            label = (
                f"ED leads by "
                f"{lead:.0f} weeks"
            )
        elif lead < 0:
            label = (
                f"ED lags by "
                f"{abs(lead):.0f} weeks"
            )
        else:
            label = (
                "Same peak week"
            )

        print(
            f"{label}: "
            f"{count}"
        )

    one_two_week = int(
        (
            leadlag[
                "ed_lead_weeks"
            ]
            .isin(
                [
                    1.0,
                    2.0,
                ]
            )
        ).sum()
    )

    positive_lead = int(
        (
            leadlag[
                "ed_lead_weeks"
            ]
            > 0
        ).sum()
    )

    simultaneous = int(
        (
            leadlag[
                "ed_lead_weeks"
            ]
            == 0
        ).sum()
    )

    lagging = int(
        (
            leadlag[
                "ed_lead_weeks"
            ]
            < 0
        ).sum()
    )

    total = len(
        leadlag
    )

    print(
        "\nSummary"
    )

    print(
        "-------"
    )

    print(
        f"ED leads hospitalization: "
        f"{positive_lead}/{total}"
    )

    print(
        f"ED leads by exactly 1-2 weeks: "
        f"{one_two_week}/{total}"
    )

    print(
        f"Same peak week: "
        f"{simultaneous}/{total}"
    )

    print(
        f"ED lags hospitalization: "
        f"{lagging}/{total}"
    )

    valid = leadlag[
        [
            "ed_lead_weeks",
            "lstm_4w_rmse_improvement_pct",
        ]
    ].dropna()

    if len(valid) >= 3:
        rho, pvalue = spearmanr(
            valid[
                "ed_lead_weeks"
            ],
            valid[
                "lstm_4w_rmse_improvement_pct"
            ],
        )

        print(
            "\nLead vs LSTM 4-week "
            "ED improvement"
        )

        print(
            "---------------------"
        )

        print(
            f"Spearman rho: "
            f"{rho:+.3f}"
        )

        print(
            f"p-value: "
            f"{pvalue:.4f}"
        )

    print(
        "\nLargest ED leads"
    )

    print(
        leadlag[
            [
                "location_name",
                "ed_lead_weeks",
                "lstm_4w_rmse_improvement_pct",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print(
        "\nLargest ED lags"
    )

    print(
        leadlag[
            [
                "location_name",
                "ed_lead_weeks",
                "lstm_4w_rmse_improvement_pct",
            ]
        ]
        .tail(10)
        .sort_values(
            "ed_lead_weeks"
        )
        .to_string(
            index=False
        )
    )

    # -----------------------------
    # Overlay examples
    # -----------------------------

    available_states = [
        state
        for state
        in REPRESENTATIVE_STATES
        if state in overlay_data
    ]

    n = len(
        available_states
    )

    if n > 0:
        rows_plot = int(
            np.ceil(
                n / 2
            )
        )

        fig, axes = plt.subplots(
            rows_plot,
            2,
            figsize=(
                14,
                4 * rows_plot,
            ),
            squeeze=False,
        )

        axes = axes.flatten()

        for ax, state in zip(
            axes,
            available_states,
        ):
            df = (
                overlay_data[
                    state
                ]
                .copy()
            )

            df[
                "hospital_z"
            ] = zscore(
                df[
                    "weekly_rate"
                ]
            )

            df[
                "ed_z"
            ] = zscore(
                df[
                    "ed_visits_prop"
                ]
            )

            ax.plot(
                df["date"],
                df["hospital_z"],
                marker="o",
                label="Hospitalizations",
            )

            ax.plot(
                df["date"],
                df["ed_z"],
                marker="o",
                label="ED visits",
            )

            hospital_peak = (
                df.loc[
                    df[
                        "weekly_rate"
                    ]
                    .idxmax(),
                    "date",
                ]
            )

            ed_peak = (
                df.loc[
                    df[
                        "ed_visits_prop"
                    ]
                    .idxmax(),
                    "date",
                ]
            )

            lead = (
                hospital_peak
                - ed_peak
            ).days / 7.0

            ax.axvline(
                hospital_peak,
                linestyle="--",
                linewidth=1,
            )

            ax.set_title(
                f"{state} — "
                f"ED lead: "
                f"{lead:+.0f} weeks"
            )

            ax.set_ylabel(
                "Standardized value (z-score)"
            )

            ax.grid(
                alpha=0.25
            )

        for ax in axes[n:]:
            ax.axis("off")

        handles, labels = (
            axes[0]
            .get_legend_handles_labels()
        )

        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
        )

        fig.suptitle(
            "2025-26 ED Visits vs "
            "Hospitalization Timing",
            y=0.995,
        )

        fig.tight_layout(
            rect=[
                0,
                0,
                1,
                0.97,
            ]
        )

        fig.savefig(
            OVERLAY_PLOT,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)

    # -----------------------------
    # Lead vs improvement scatter
    # -----------------------------

    fig, ax = plt.subplots(
        figsize=(
            9,
            7,
        )
    )

    ax.scatter(
        leadlag[
            "ed_lead_weeks"
        ],
        leadlag[
            "lstm_4w_rmse_improvement_pct"
        ],
        s=55,
        alpha=0.8,
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.axvline(
        0,
        linewidth=1,
    )

    ax.set_xlabel(
        "ED peak lead over "
        "hospitalization peak (weeks)"
    )

    ax.set_ylabel(
        "LSTM 4-week RMSE improvement "
        "from ED feature (%)"
    )

    ax.set_title(
        "Does ED Lead Time Predict "
        "Forecast Improvement?"
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        SCATTER_PLOT,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"\nSaved: {OUTPUT_CSV}"
    )

    print(
        f"Saved: {OVERLAY_PLOT}"
    )

    print(
        f"Saved: {SCATTER_PLOT}"
    )


if __name__ == "__main__":
    main()