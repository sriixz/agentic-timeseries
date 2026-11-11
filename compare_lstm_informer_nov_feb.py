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

OUTPUT_PLOT = Path(
    "plots/lstm_vs_informer_nov_feb_4w.png"
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

    print(
        f"\n{model_name} numeric candidates:"
    )

    print(
        numeric
    )

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

    baseline_col = (
        baseline_candidates[0]
    )

    ed_col = (
        ed_candidates[0]
    )

    print(
        f"{model_name} baseline: "
        f"{baseline_col}"
    )

    print(
        f"{model_name} + ED: "
        f"{ed_col}"
    )

    return baseline_col, ed_col


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
            df["ds"] >= START_DATE
        )
        & (
            df["ds"] <= END_DATE
        )
    ].copy()

    return df


def summarize_model(
    df,
    model_name,
):
    baseline_col, ed_col = (
        detect_cols(
            df,
            model_name,
        )
    )

    rows = []

    for (
        location,
        location_name,
        horizon
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
            .sort_values("ds")
            .drop_duplicates(
                subset=["ds"],
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


def main():
    if not LSTM_FORECASTS.exists():
        raise FileNotFoundError(
            f"Missing: {LSTM_FORECASTS}"
        )

    if not INFORMER_FORECASTS.exists():
        raise FileNotFoundError(
            f"Missing: {INFORMER_FORECASTS}"
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

    lstm_summary = (
        summarize_model(
            lstm,
            "lstm",
        )
    )

    informer_summary = (
        summarize_model(
            informer,
            "informer",
        )
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

    print(
        "\nLSTM vs Informer — "
        "Nov 2025 through Feb 2026"
    )

    print(
        "================================="
    )

    for horizon in sorted(
        merged["horizon"]
        .unique()
    ):
        scoped = merged[
            merged["horizon"]
            == horizon
        ]

        print(
            f"\n{horizon}-week horizon"
        )

        print(
            "-" * 40
        )

        print(
            "LSTM baseline RMSE: "
            f"{scoped['lstm_baseline_rmse'].mean():.3f}"
        )

        print(
            "LSTM + ED RMSE: "
            f"{scoped['lstm_ed_rmse'].mean():.3f}"
        )

        print(
            "LSTM median ED improvement: "
            f"{scoped['lstm_ed_improvement_pct'].median():+.1f}%"
        )

        print(
            "Informer baseline RMSE: "
            f"{scoped['informer_baseline_rmse'].mean():.3f}"
        )

        print(
            "Informer + ED RMSE: "
            f"{scoped['informer_ed_rmse'].mean():.3f}"
        )

        print(
            "Informer median ED improvement: "
            f"{scoped['informer_ed_improvement_pct'].median():+.1f}%"
        )

        print(
            "Informer + ED lower RMSE than "
            "LSTM + ED in "
            f"{int((scoped['informer_ed_rmse'] < scoped['lstm_ed_rmse']).sum())}"
            f"/{len(scoped)} states"
        )

    four_week = merged[
        merged["horizon"] == 4
    ].copy()

    four_week[
        "informer_minus_lstm_ed_rmse"
    ] = (
        four_week[
            "informer_ed_rmse"
        ]
        - four_week[
            "lstm_ed_rmse"
        ]
    )

    four_week[
        "informer_advantage_pct"
    ] = (
        (
            four_week[
                "lstm_ed_rmse"
            ]
            - four_week[
                "informer_ed_rmse"
            ]
        )
        / four_week[
            "lstm_ed_rmse"
        ]
        * 100.0
    )

    print(
        "\n4-week Nov–Feb comparison"
    )

    print(
        "========================="
    )

    print(
        "States compared: "
        f"{len(four_week)}"
    )

    print(
        "Informer + ED lower RMSE: "
        f"{int((four_week['informer_ed_rmse'] < four_week['lstm_ed_rmse']).sum())}"
        f"/{len(four_week)}"
    )

    print(
        "Median Informer advantage "
        "over LSTM + ED: "
        f"{four_week['informer_advantage_pct'].median():+.1f}%"
    )

    print(
        "\nLargest Informer advantages"
    )

    print(
        four_week[
            [
                "location_name",
                "lstm_ed_rmse",
                "informer_ed_rmse",
                "informer_advantage_pct",
            ]
        ]
        .sort_values(
            "informer_advantage_pct",
            ascending=False,
        )
        .head(10)
        .to_string(
            index=False,
            formatters={
                "lstm_ed_rmse":
                    "{:.3f}".format,

                "informer_ed_rmse":
                    "{:.3f}".format,

                "informer_advantage_pct":
                    "{:+.1f}".format,
            },
        )
    )

    print(
        "\nLargest LSTM advantages"
    )

    print(
        four_week[
            [
                "location_name",
                "lstm_ed_rmse",
                "informer_ed_rmse",
                "informer_advantage_pct",
            ]
        ]
        .sort_values(
            "informer_advantage_pct"
        )
        .head(10)
        .to_string(
            index=False,
            formatters={
                "lstm_ed_rmse":
                    "{:.3f}".format,

                "informer_ed_rmse":
                    "{:.3f}".format,

                "informer_advantage_pct":
                    "{:+.1f}".format,
            },
        )
    )

    # --------------------------------
    # 4-week state scatter
    # --------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    ax.scatter(
        four_week[
            "lstm_ed_rmse"
        ],
        four_week[
            "informer_ed_rmse"
        ],
        s=60,
        alpha=0.8,
    )

    max_val = max(
        four_week[
            "lstm_ed_rmse"
        ].max(),
        four_week[
            "informer_ed_rmse"
        ].max(),
    )

    ax.plot(
        [
            0,
            max_val,
        ],
        [
            0,
            max_val,
        ],
        linestyle="--",
        linewidth=1.5,
        label="Equal RMSE",
    )

    ax.set_xlabel(
        "LSTM + ED RMSE "
        "(Nov 2025–Feb 2026)"
    )

    ax.set_ylabel(
        "Informer + ED RMSE "
        "(Nov 2025–Feb 2026)"
    )

    ax.set_title(
        "Active-Season 4-Week Forecast Error: "
        "LSTM vs Informer"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_PLOT,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"\nSaved: {OUTPUT_CSV}"
    )

    print(
        f"Saved: {OUTPUT_PLOT}"
    )


if __name__ == "__main__":
    main()