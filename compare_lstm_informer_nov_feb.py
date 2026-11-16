from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LSTM_FORECASTS = Path(
    "forecast_results/allstate_lstm_ed_forecasts.csv"
)

INFORMER_FORECASTS = Path(
    "forecast_results/allstate_informer_ed_forecasts.csv"
)

OUTPUT_CSV = Path(
    "forecast_results/lstm_vs_informer_nov_feb.csv"
)

SUMMARY_CSV = Path(
    "forecast_results/lstm_vs_informer_nov_feb_horizon_summary.csv"
)

OUTPUT_PLOT = Path(
    "plots/lstm_vs_informer_nov_feb_horizons.png"
)

START_DATE = pd.Timestamp("2025-11-01")
END_DATE = pd.Timestamp("2026-02-28")


def rmse(y_true, y_pred):
    error = (
        y_pred.astype(float)
        - y_true.astype(float)
    )

    return float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )


def mae(y_true, y_pred):
    return float(
        np.mean(
            np.abs(
                y_pred.astype(float)
                - y_true.astype(float)
            )
        )
    )


def detect_cols(df, model_name):
    excluded = {
        "unique_id",
        "location",
        "location_name",
        "ds",
        "cutoff",
        "y",
        "horizon",
    }

    numeric = [
        col
        for col in df.columns
        if col not in excluded
        and pd.api.types.is_numeric_dtype(
            df[col]
        )
    ]

    ed_candidates = [
        col
        for col in numeric
        if "ed" in col.lower()
    ]

    baseline_candidates = [
        col
        for col in numeric
        if "ed" not in col.lower()
    ]

    if len(ed_candidates) != 1:
        raise ValueError(
            f"Could not uniquely detect "
            f"{model_name} ED column: "
            f"{ed_candidates}"
        )

    if len(baseline_candidates) != 1:
        raise ValueError(
            f"Could not uniquely detect "
            f"{model_name} baseline column: "
            f"{baseline_candidates}"
        )

    return (
        baseline_candidates[0],
        ed_candidates[0],
    )


def prepare(df):
    df = df.copy()

    df["ds"] = pd.to_datetime(
        df["ds"]
    )

    if "cutoff" in df.columns:
        df["cutoff"] = pd.to_datetime(
            df["cutoff"]
        )

    if "location" not in df.columns:
        if "unique_id" not in df.columns:
            raise ValueError(
                "Missing location / unique_id"
            )

        df["location"] = (
            df["unique_id"]
            .astype(str)
        )

    df["location"] = (
        df["location"]
        .astype(str)
    )

    if "location_name" not in df.columns:
        df["location_name"] = (
            df["location"]
        )

    df = df[
        (
            df["ds"]
            >= START_DATE
        )
        & (
            df["ds"]
            <= END_DATE
        )
    ].copy()

    return df


def summarize_model(
    df,
    model_name,
):
    baseline_col, ed_col = detect_cols(
        df,
        model_name,
    )

    rows = []

    for (
        location,
        location_name,
        horizon,
    ), group in df.groupby(
        [
            "location",
            "location_name",
            "horizon",
        ]
    ):

        group = (
            group
            .dropna(
                subset=[
                    "y",
                    baseline_col,
                    ed_col,
                ]
            )
            .sort_values(
                "ds"
            )
            .drop_duplicates(
                subset=[
                    "ds"
                ],
                keep="last",
            )
        )

        if group.empty:
            continue

        baseline_rmse = rmse(
            group["y"],
            group[baseline_col],
        )

        ed_rmse = rmse(
            group["y"],
            group[ed_col],
        )

        baseline_mae = mae(
            group["y"],
            group[baseline_col],
        )

        ed_mae = mae(
            group["y"],
            group[ed_col],
        )

        improvement = (
            (
                baseline_rmse
                - ed_rmse
            )
            / baseline_rmse
            * 100.0
            if baseline_rmse > 0
            else np.nan
        )

        rows.append(
            {
                "location":
                    location,

                "location_name":
                    location_name,

                "horizon":
                    int(horizon),

                f"{model_name}_baseline_rmse":
                    baseline_rmse,

                f"{model_name}_ed_rmse":
                    ed_rmse,

                f"{model_name}_ed_improvement_pct":
                    improvement,

                f"{model_name}_baseline_mae":
                    baseline_mae,

                f"{model_name}_ed_mae":
                    ed_mae,

                f"{model_name}_n":
                    len(group),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_horizon_summary(
    merged,
):
    rows = []

    for horizon in sorted(
        merged["horizon"].unique()
    ):

        scoped = merged[
            merged["horizon"]
            == horizon
        ].copy()

        scoped[
            "informer_advantage_pct"
        ] = (
            (
                scoped[
                    "lstm_ed_rmse"
                ]
                - scoped[
                    "informer_ed_rmse"
                ]
            )
            / scoped[
                "lstm_ed_rmse"
            ]
            * 100.0
        )

        rows.append(
            {
                "horizon":
                    int(horizon),

                "states":
                    len(scoped),

                "lstm_ed_mean_rmse":
                    float(
                        scoped[
                            "lstm_ed_rmse"
                        ].mean()
                    ),

                "informer_ed_mean_rmse":
                    float(
                        scoped[
                            "informer_ed_rmse"
                        ].mean()
                    ),

                "informer_lower_count":
                    int(
                        (
                            scoped[
                                "informer_ed_rmse"
                            ]
                            < scoped[
                                "lstm_ed_rmse"
                            ]
                        ).sum()
                    ),

                "median_informer_advantage_pct":
                    float(
                        scoped[
                            "informer_advantage_pct"
                        ].median()
                    ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    average_row = {
        "horizon":
            "Average",

        "states":
            int(
                round(
                    summary[
                        "states"
                    ].mean()
                )
            ),

        "lstm_ed_mean_rmse":
            float(
                summary[
                    "lstm_ed_mean_rmse"
                ].mean()
            ),

        "informer_ed_mean_rmse":
            float(
                summary[
                    "informer_ed_mean_rmse"
                ].mean()
            ),

        "informer_lower_count":
            float(
                summary[
                    "informer_lower_count"
                ].mean()
            ),

        "median_informer_advantage_pct":
            float(
                summary[
                    "median_informer_advantage_pct"
                ].mean()
            ),
    }

    summary_out = pd.concat(
        [
            summary,
            pd.DataFrame(
                [average_row]
            ),
        ],
        ignore_index=True,
    )

    return summary_out


def make_plot(
    summary,
):
    plot_df = summary[
        summary[
            "horizon"
        ]
        != "Average"
    ].copy()

    x = np.arange(
        len(plot_df)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    ax.bar(
        x - width / 2,
        plot_df[
            "lstm_ed_mean_rmse"
        ],
        width,
        label="LSTM + ED",
    )

    ax.bar(
        x + width / 2,
        plot_df[
            "informer_ed_mean_rmse"
        ],
        width,
        label="Informer + ED",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            f"{int(h)}-week"
            for h in plot_df[
                "horizon"
            ]
        ]
    )

    ax.set_ylabel(
        "Mean RMSE across states"
    )

    ax.set_xlabel(
        "Forecast horizon"
    )

    ax.set_title(
        "Active-Season Forecast Error: "
        "LSTM + ED vs Informer + ED\n"
        "November 2025 through February 2026"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend()

    for index, row in enumerate(
        plot_df.itertuples(
            index=False
        )
    ):

        ax.text(
            index,
            max(
                row.lstm_ed_mean_rmse,
                row.informer_ed_mean_rmse,
            )
            + 0.25,
            (
                f"Informer lower: "
                f"{int(row.informer_lower_count)}"
                f"/{int(row.states)}"
            ),
            ha="center",
            fontsize=9,
        )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_PLOT,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def main():

    if not LSTM_FORECASTS.exists():
        raise FileNotFoundError(
            f"Missing: "
            f"{LSTM_FORECASTS}"
        )

    if not INFORMER_FORECASTS.exists():
        raise FileNotFoundError(
            f"Missing: "
            f"{INFORMER_FORECASTS}"
        )

    lstm = prepare(
        pd.read_csv(
            LSTM_FORECASTS,
            dtype={
                "location": str,
                "unique_id": str,
            },
        )
    )

    informer = prepare(
        pd.read_csv(
            INFORMER_FORECASTS,
            dtype={
                "location": str,
                "unique_id": str,
            },
        )
    )

    lstm_summary = summarize_model(
        lstm,
        "lstm",
    )

    informer_summary = summarize_model(
        informer,
        "informer",
    )

    merged = lstm_summary.merge(
        informer_summary,
        on=[
            "location",
            "location_name",
            "horizon",
        ],
        how="inner",
        validate="one_to_one",
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PLOT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    horizon_summary = (
        build_horizon_summary(
            merged
        )
    )

    horizon_summary.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    print(
        "\nLSTM vs Informer - "
        "Nov 2025 through Feb 2026"
    )

    print(
        "======================================"
    )

    print(
        horizon_summary.to_string(
            index=False,
            formatters={
                "lstm_ed_mean_rmse":
                    "{:.3f}".format,

                "informer_ed_mean_rmse":
                    "{:.3f}".format,

                "informer_lower_count":
                    (
                        lambda value:
                        (
                            f"{value:.1f}"
                            if isinstance(
                                value,
                                float,
                            )
                            and not value.is_integer()
                            else f"{int(value)}"
                        )
                    ),

                "median_informer_advantage_pct":
                    "{:+.1f}%".format,
            },
        )
    )

    make_plot(
        horizon_summary
    )

    print(
        f"\nSaved: "
        f"{OUTPUT_CSV}"
    )

    print(
        f"Saved: "
        f"{SUMMARY_CSV}"
    )

    print(
        f"Saved: "
        f"{OUTPUT_PLOT}"
    )


if __name__ == "__main__":
    main()