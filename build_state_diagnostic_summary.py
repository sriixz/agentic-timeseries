from pathlib import Path

import pandas as pd


LSTM_SUMMARY = Path(
    "forecast_results/allstate_lstm_ed_summary.csv"
)

INFORMER_SUMMARY = Path(
    "forecast_results/allstate_informer_ed_summary.csv"
)

LEADLAG = Path(
    "forecast_results/allstate_ed_hospital_leadlag.csv"
)

PHASE_DETAIL = Path(
    "forecast_results/allstate_lstm_ed_phase_detail.csv"
)

OUTPUT = Path(
    "forecast_results/state_ed_diagnostic_summary.csv"
)


def load_four_week(path, prefix):
    df = pd.read_csv(
        path,
        dtype={"location": str},
    )

    df = df[
        df["horizon"] == 4
    ].copy()

    return df[
        [
            "location",
            "location_name",
            "baseline_rmse",
            "ed_rmse",
            "rmse_improvement_pct",
        ]
    ].rename(
        columns={
            "baseline_rmse":
                f"{prefix}_baseline_rmse",

            "ed_rmse":
                f"{prefix}_ed_rmse",

            "rmse_improvement_pct":
                f"{prefix}_4w_ed_improvement_pct",
        }
    )


def pivot_horizons():
    df = pd.read_csv(
        LSTM_SUMMARY,
        dtype={"location": str},
    )

    pivot = df.pivot(
        index=[
            "location",
            "location_name",
        ],
        columns="horizon",
        values="rmse_improvement_pct",
    )

    pivot = pivot.rename(
        columns={
            1: "lstm_1w_ed_improvement_pct",
            2: "lstm_2w_ed_improvement_pct",
            3: "lstm_3w_ed_improvement_pct",
            4: "lstm_4w_ed_improvement_pct",
        }
    )

    return pivot.reset_index()


def pivot_phases():
    df = pd.read_csv(
        PHASE_DETAIL,
        dtype={"location": str},
    )

    df = df[
        df["horizon"] == 4
    ].copy()

    pivot = df.pivot(
        index=[
            "location",
            "location_name",
        ],
        columns="phase",
        values="rmse_improvement_pct",
    )

    pivot = pivot.rename(
        columns={
            "Growth":
                "lstm_4w_growth_improvement_pct",

            "Peak":
                "lstm_4w_peak_improvement_pct",

            "Decline":
                "lstm_4w_decline_improvement_pct",
        }
    )

    return pivot.reset_index()


def classify_lead(row):
    lead = row["ed_lead_weeks"]

    if pd.isna(lead):
        return "Unknown"

    if lead >= 2:
        return "ED leads >=2 weeks"

    if lead == 1:
        return "ED leads 1 week"

    if lead == 0:
        return "Same peak week"

    return "ED lags hospitalization"


def classify_feature_behavior(row):
    lstm = row[
        "lstm_4w_ed_improvement_pct"
    ]

    informer = row[
        "informer_4w_ed_improvement_pct"
    ]

    if (
        pd.isna(lstm)
        or pd.isna(informer)
    ):
        return "Incomplete"

    if lstm > 0 and informer > 0:
        return "ED robust"

    if lstm > 0 and informer <= 0:
        return "LSTM-sensitive"

    if lstm <= 0 and informer > 0:
        return "Informer-sensitive"

    return "ED harmful"


def classify_phase_pattern(row):
    growth = row.get(
        "lstm_4w_growth_improvement_pct"
    )

    peak = row.get(
        "lstm_4w_peak_improvement_pct"
    )

    decline = row.get(
        "lstm_4w_decline_improvement_pct"
    )

    values = {
        "Growth": growth,
        "Peak": peak,
        "Decline": decline,
    }

    valid = {
        key: value
        for key, value in values.items()
        if pd.notna(value)
    }

    if not valid:
        return "Unknown"

    best_phase = max(
        valid,
        key=valid.get,
    )

    positive = [
        key
        for key, value in valid.items()
        if value > 0
    ]

    if len(positive) == 3:
        return (
            f"Helps all phases; "
            f"strongest {best_phase}"
        )

    if len(positive) == 0:
        return "Hurts all phases"

    return (
        "Helps "
        + "/".join(positive)
        + f"; strongest {best_phase}"
    )


def main():
    horizon = pivot_horizons()

    lstm_4w = load_four_week(
        LSTM_SUMMARY,
        "lstm",
    )

    informer_4w = load_four_week(
        INFORMER_SUMMARY,
        "informer",
    )

    phase = pivot_phases()

    leadlag = pd.read_csv(
        LEADLAG,
        dtype={"location": str},
    )

    leadlag = leadlag[
        [
            "location",
            "location_name",
            "hospital_peak_date",
            "ed_peak_date",
            "ed_lead_weeks",
        ]
    ].copy()

    # Start with LSTM horizon results.
    merged = horizon.merge(
        informer_4w[
            [
                "location",
                "location_name",
                "informer_baseline_rmse",
                "informer_ed_rmse",
                "informer_4w_ed_improvement_pct",
            ]
        ],
        on=[
            "location",
            "location_name",
        ],
        how="left",
    )

    # Add exact LSTM 4-week RMSE values.
    merged = merged.merge(
        lstm_4w[
            [
                "location",
                "location_name",
                "lstm_baseline_rmse",
                "lstm_ed_rmse",
            ]
        ],
        on=[
            "location",
            "location_name",
        ],
        how="left",
    )

    # Add phase information.
    merged = merged.merge(
        phase,
        on=[
            "location",
            "location_name",
        ],
        how="left",
    )

    # Add lead/lag information.
    merged = merged.merge(
        leadlag,
        on=[
            "location",
            "location_name",
        ],
        how="left",
    )

    merged[
        "lead_category"
    ] = merged.apply(
        classify_lead,
        axis=1,
    )

    merged[
        "feature_behavior"
    ] = merged.apply(
        classify_feature_behavior,
        axis=1,
    )

    merged[
        "phase_pattern"
    ] = merged.apply(
        classify_phase_pattern,
        axis=1,
    )

    merged[
        "architecture_gap_pct"
    ] = (
        merged[
            "informer_4w_ed_improvement_pct"
        ]
        - merged[
            "lstm_4w_ed_improvement_pct"
        ]
    )

    merged[
        "lstm_short_horizon_mean_pct"
    ] = merged[
        [
            "lstm_1w_ed_improvement_pct",
            "lstm_2w_ed_improvement_pct",
        ]
    ].mean(
        axis=1
    )

    merged[
        "lstm_long_horizon_mean_pct"
    ] = merged[
        [
            "lstm_3w_ed_improvement_pct",
            "lstm_4w_ed_improvement_pct",
        ]
    ].mean(
        axis=1
    )

    merged[
        "horizon_pattern"
    ] = merged.apply(
        lambda row:
        (
            "Short-horizon stronger"
            if (
                row[
                    "lstm_short_horizon_mean_pct"
                ]
                >
                row[
                    "lstm_long_horizon_mean_pct"
                ]
            )
            else "Long-horizon stronger"
        ),
        axis=1,
    )

    merged = (
        merged
        .sort_values(
            "location_name"
        )
        .reset_index(
            drop=True
        )
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        OUTPUT,
        index=False,
    )

    print(
        "\nState ED Diagnostic Summary"
    )

    print(
        "==========================="
    )

    print(
        f"States: {len(merged)}"
    )

    print(
        "\nFeature behavior"
    )

    print(
        merged[
            "feature_behavior"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nLead categories"
    )

    print(
        merged[
            "lead_category"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nHorizon pattern"
    )

    print(
        merged[
            "horizon_pattern"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nPhase pattern"
    )

    print(
        merged[
            "phase_pattern"
        ]
        .value_counts()
        .head(12)
        .to_string()
    )

    print(
        "\nMost architecture-sensitive states"
    )

    print(
        merged.assign(
            abs_gap=(
                merged[
                    "architecture_gap_pct"
                ]
                .abs()
            )
        )
        .sort_values(
            "abs_gap",
            ascending=False,
        )
        [
            [
                "location_name",
                "feature_behavior",
                "ed_lead_weeks",
                "lstm_1w_ed_improvement_pct",
                "lstm_4w_ed_improvement_pct",
                "informer_4w_ed_improvement_pct",
                "phase_pattern",
            ]
        ]
        .head(12)
        .to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()