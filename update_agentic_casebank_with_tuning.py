from pathlib import Path

import pandas as pd


CASEBANK_PATH = Path(
    "forecast_results/agentic_feature_casebank.csv"
)

TUNING_PATH = Path(
    "forecast_results/representative_lstm_ed_tuned_vs_untuned.csv"
)

OUTPUT_PATH = Path(
    "forecast_results/agentic_feature_casebank_with_tuning.csv"
)


def tuned_interpretation(row):
    if pd.isna(
        row.get(
            "tuned_ed_improvement_pct"
        )
    ):
        return "Not retuned"

    tuned = row[
        "tuned_ed_improvement_pct"
    ]

    untuned = row[
        "untuned_ed_improvement_pct"
    ]

    if untuned <= 0 and tuned > 0:
        return (
            "ED failure reversed after tuning"
        )

    if untuned > 0 and tuned > 0:
        return (
            "ED benefit persists after tuning"
        )

    if untuned <= 0 and tuned <= 0:
        return (
            "ED remains non-beneficial after tuning"
        )

    return (
        "ED benefit disappeared after tuning"
    )


def main():
    casebank = pd.read_csv(
        CASEBANK_PATH,
        dtype={"location": str},
    )

    tuning = pd.read_csv(
        TUNING_PATH,
        dtype={"location": str},
    )

    tuning = tuning[
        [
            "location",
            "location_name",
            "untuned_ed_improvement_pct",
            "tuned_baseline_rmse",
            "tuned_ed_rmse",
            "tuned_ed_improvement_pct",
            "tuning_change_pct_points",
            "behavior_flip",
        ]
    ].copy()

    updated = casebank.merge(
        tuning,
        on=[
            "location",
            "location_name",
        ],
        how="left",
        validate="one_to_one",
    )

    updated[
        "tuning_interpretation"
    ] = updated.apply(
        tuned_interpretation,
        axis=1,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    updated.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    tuned = updated[
        updated[
            "tuned_ed_improvement_pct"
        ].notna()
    ].copy()

    print(
        "\nCase Bank Updated With Tuning"
    )

    print(
        "============================="
    )

    print(
        f"Total states: {len(updated)}"
    )

    print(
        f"Retuned representative states: "
        f"{len(tuned)}"
    )

    print(
        "\nTuning interpretations"
    )

    print(
        tuned[
            "tuning_interpretation"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nRepresentative-state results"
    )

    print(
        tuned[
            [
                "location_name",
                "untuned_ed_improvement_pct",
                "tuned_ed_improvement_pct",
                "behavior_flip",
                "tuning_interpretation",
            ]
        ]
        .sort_values(
            "tuned_ed_improvement_pct",
            ascending=False,
        )
        .to_string(
            index=False,
            formatters={
                "untuned_ed_improvement_pct":
                    "{:+.1f}".format,

                "tuned_ed_improvement_pct":
                    "{:+.1f}".format,
            },
        )
    )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()