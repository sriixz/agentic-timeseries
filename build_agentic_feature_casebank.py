from pathlib import Path

import numpy as np
import pandas as pd


LSTM_PATH = Path(
    "forecast_results/allstate_lstm_ed_summary.csv"
)

INFORMER_PATH = Path(
    "forecast_results/allstate_informer_ed_summary.csv"
)

CONTEXT_PATH = Path(
    "forecast_results/allstate_ed_peak_variability_analysis.csv"
)

OUTPUT_PATH = Path(
    "forecast_results/agentic_feature_casebank.csv"
)

RECOMMENDATION_PATH = Path(
    "forecast_results/agentic_feature_recommendations.csv"
)


def load_model_results(
    path,
    prefix,
):
    df = pd.read_csv(
        path,
        dtype={
            "location": str,
        },
    )

    df = df[
        df["horizon"] == 4
    ].copy()

    keep = [
        "location",
        "location_name",
        "baseline_rmse",
        "ed_rmse",
        "rmse_improvement_pct",
    ]

    df = df[
        keep
    ].copy()

    df = df.rename(
        columns={
            "baseline_rmse":
                f"{prefix}_baseline_rmse",

            "ed_rmse":
                f"{prefix}_ed_rmse",

            "rmse_improvement_pct":
                f"{prefix}_ed_improvement_pct",
        }
    )

    return df


def classify_behavior(
    row,
):
    lstm = row[
        "lstm_ed_improvement_pct"
    ]

    informer = row[
        "informer_ed_improvement_pct"
    ]

    if lstm > 0 and informer > 0:
        return "ED robust"

    if lstm > 0 and informer <= 0:
        return "LSTM-sensitive"

    if lstm <= 0 and informer > 0:
        return "Informer-sensitive"

    return "ED harmful"


def choose_configuration(
    row,
):
    candidates = {
        "LSTM baseline":
            row[
                "lstm_baseline_rmse"
            ],

        "LSTM + ED":
            row[
                "lstm_ed_rmse"
            ],

        "Informer baseline":
            row[
                "informer_baseline_rmse"
            ],

        "Informer + ED":
            row[
                "informer_ed_rmse"
            ],
    }

    valid = {
        key: value
        for key, value
        in candidates.items()
        if pd.notna(value)
    }

    if not valid:
        return pd.Series(
            {
                "recommended_configuration":
                    np.nan,

                "recommended_model":
                    np.nan,

                "recommended_feature":
                    np.nan,

                "recommended_rmse":
                    np.nan,
            }
        )

    best = min(
        valid,
        key=valid.get,
    )

    if best.startswith(
        "LSTM"
    ):
        model = "LSTM"
    else:
        model = "Informer"

    if "+ ED" in best:
        feature = "Historical/lagged ED"
    else:
        feature = "None"

    return pd.Series(
        {
            "recommended_configuration":
                best,

            "recommended_model":
                model,

            "recommended_feature":
                feature,

            "recommended_rmse":
                valid[best],
        }
    )


def build_reason(
    row,
):
    behavior = row[
        "feature_behavior"
    ]

    lstm = row[
        "lstm_ed_improvement_pct"
    ]

    informer = row[
        "informer_ed_improvement_pct"
    ]

    recommended = row[
        "recommended_configuration"
    ]

    if behavior == "ED robust":
        prefix = (
            "ED improves both architectures."
        )

    elif behavior == "LSTM-sensitive":
        prefix = (
            "ED improves LSTM but hurts "
            "or does not improve Informer."
        )

    elif behavior == "Informer-sensitive":
        prefix = (
            "ED improves Informer but hurts "
            "or does not improve LSTM."
        )

    else:
        prefix = (
            "ED does not improve either "
            "architecture."
        )

    return (
        f"{prefix} "
        f"LSTM ED change={lstm:+.1f}%; "
        f"Informer ED change={informer:+.1f}%. "
        f"Lowest observed RMSE: {recommended}."
    )


def main():
    lstm = load_model_results(
        LSTM_PATH,
        "lstm",
    )

    informer = load_model_results(
        INFORMER_PATH,
        "informer",
    )

    context = pd.read_csv(
        CONTEXT_PATH,
        dtype={
            "location": str,
        },
    )

    context_columns = [
        "location",
        "location_name",
        "historical_peak_iqr",
        "historical_peak_mad",
        "historical_peak_range",
        "test_peak_deviation_weeks",
        "test_peak_date",
        "test_peak_rate",
    ]

    context = context[
        [
            column
            for column
            in context_columns
            if column in context.columns
        ]
    ].copy()

    merged = lstm.merge(
        informer,
        on=[
            "location",
            "location_name",
        ],
        how="inner",
        validate="one_to_one",
    )

    merged = merged.merge(
        context,
        on=[
            "location",
            "location_name",
        ],
        how="left",
        validate="one_to_one",
    )

    merged[
        "feature_behavior"
    ] = merged.apply(
        classify_behavior,
        axis=1,
    )

    recommendations = merged.apply(
        choose_configuration,
        axis=1,
    )

    merged = pd.concat(
        [
            merged,
            recommendations,
        ],
        axis=1,
    )

    merged[
        "architecture_gap_pct"
    ] = (
        merged[
            "informer_ed_improvement_pct"
        ]
        - merged[
            "lstm_ed_improvement_pct"
        ]
    )

    merged[
        "recommendation_reason"
    ] = merged.apply(
        build_reason,
        axis=1,
    )

    merged = (
        merged
        .sort_values(
            [
                "feature_behavior",
                "location_name",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    merged.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    recommendation_columns = [
        "location",
        "location_name",
        "feature_behavior",
        "historical_peak_iqr",
        "test_peak_deviation_weeks",
        "lstm_ed_improvement_pct",
        "informer_ed_improvement_pct",
        "recommended_configuration",
        "recommended_rmse",
        "recommendation_reason",
    ]

    recommendations_out = merged[
        recommendation_columns
    ].copy()

    recommendations_out.to_csv(
        RECOMMENDATION_PATH,
        index=False,
    )

    print(
        "\nAgentic Feature Case Bank"
    )

    print(
        "========================="
    )

    print(
        f"States included: "
        f"{len(merged)}"
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
        "\nRecommended configurations"
    )

    print(
        merged[
            "recommended_configuration"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nMost architecture-sensitive states"
    )

    architecture_sensitive = (
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
        .head(10)
    )

    print(
        architecture_sensitive[
            [
                "location_name",
                "feature_behavior",
                "lstm_ed_improvement_pct",
                "informer_ed_improvement_pct",
                "architecture_gap_pct",
                "recommended_configuration",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )

    print(
        f"Saved: {RECOMMENDATION_PATH}"
    )


if __name__ == "__main__":
    main()