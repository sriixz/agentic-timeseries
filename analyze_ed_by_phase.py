from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FORECAST_PATH = Path(
    "forecast_results/allstate_lstm_ed_forecasts.csv"
)

OUTPUT_DETAIL = Path(
    "forecast_results/allstate_lstm_ed_phase_detail.csv"
)

OUTPUT_SUMMARY = Path(
    "forecast_results/allstate_lstm_ed_phase_summary.csv"
)

OUTPUT_PLOT = Path(
    "plots/allstate_lstm_ed_phase_summary.png"
)

PEAK_HALF_WINDOW_WEEKS = 2


def detect_prediction_columns(df):
    """
    Find baseline and ED prediction columns robustly.

    Excludes known metadata/actual columns, then looks
    for columns containing ED vs non-ED model names.
    """
    excluded = {
        "unique_id",
        "ds",
        "cutoff",
        "y",
        "horizon",
        "location",
        "location_name",
    }

    candidates = [
        c for c in df.columns
        if c not in excluded
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    print(
        "Numeric prediction candidates:",
        candidates
    )

    ed_candidates = [
        c for c in candidates
        if "ed" in c.lower()
    ]

    baseline_candidates = [
        c for c in candidates
        if "ed" not in c.lower()
        and (
            "baseline" in c.lower()
            or "lstm" in c.lower()
        )
    ]

    if len(ed_candidates) == 1:
        ed_col = ed_candidates[0]
    else:
        # Common names from our experiments.
        preferred_ed = [
            "LSTMED",
            "LSTM_ED",
            "LSTM+ED",
            "ED",
        ]

        ed_col = next(
            (
                c for c in preferred_ed
                if c in candidates
            ),
            None,
        )

    if len(baseline_candidates) == 1:
        baseline_col = baseline_candidates[0]
    else:
        preferred_baseline = [
            "LSTMBaseline",
            "LSTM_Baseline",
            "Baseline",
            "LSTM",
        ]

        baseline_col = next(
            (
                c for c in preferred_baseline
                if c in candidates
            ),
            None,
        )

    if baseline_col is None or ed_col is None:
        raise ValueError(
            "Could not uniquely detect baseline/ED "
            "prediction columns.\n"
            f"Candidates: {candidates}\n"
            f"Baseline candidates: {baseline_candidates}\n"
            f"ED candidates: {ed_candidates}"
        )

    print(
        f"Using baseline column: {baseline_col}"
    )

    print(
        f"Using ED column: {ed_col}"
    )

    return baseline_col, ed_col


def assign_phase(
    date,
    peak_date,
):
    peak_start = (
        peak_date
        - pd.Timedelta(
            weeks=PEAK_HALF_WINDOW_WEEKS
        )
    )

    peak_end = (
        peak_date
        + pd.Timedelta(
            weeks=PEAK_HALF_WINDOW_WEEKS
        )
    )

    if date < peak_start:
        return "Growth"

    if date > peak_end:
        return "Decline"

    return "Peak"


def rmse(
    actual,
    prediction,
):
    error = (
        prediction.astype(float)
        - actual.astype(float)
    )

    return float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )


def mae(
    actual,
    prediction,
):
    return float(
        np.mean(
            np.abs(
                prediction.astype(float)
                - actual.astype(float)
            )
        )
    )


def main():
    if not FORECAST_PATH.exists():
        raise FileNotFoundError(
            f"Missing: {FORECAST_PATH}"
        )

    df = pd.read_csv(
        FORECAST_PATH,
        dtype={
            "location": str,
            "unique_id": str,
        },
    )

    df["ds"] = pd.to_datetime(
        df["ds"]
    )

    if "cutoff" in df.columns:
        df["cutoff"] = pd.to_datetime(
            df["cutoff"]
        )

    required = {
        "ds",
        "y",
        "horizon",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    baseline_col, ed_col = (
        detect_prediction_columns(df)
    )

    # Normalize location columns.
    if "location" not in df.columns:
        if "unique_id" not in df.columns:
            raise ValueError(
                "No location or unique_id column."
            )

        df["location"] = (
            df["unique_id"]
            .astype(str)
        )

    if "location_name" not in df.columns:
        df["location_name"] = (
            df["location"]
        )

    # -----------------------------------------
    # Determine one observed hospitalization
    # peak per state from the full test-period
    # actual trajectory.
    #
    # Using horizon 1 avoids repeated actuals
    # across horizons/windows.
    # -----------------------------------------

    peak_dates = {}

    for (
        location,
        location_name
    ), group in df.groupby(
        [
            "location",
            "location_name",
        ]
    ):
        if (
            group["horizon"] == 1
        ).any():
            actual_group = (
                group[
                    group["horizon"] == 1
                ]
                .sort_values("ds")
                .drop_duplicates(
                    subset=["ds"]
                )
            )
        else:
            actual_group = (
                group
                .sort_values("ds")
                .drop_duplicates(
                    subset=["ds"]
                )
            )

        actual_group = (
            actual_group
            .dropna(
                subset=["y"]
            )
        )

        if actual_group.empty:
            continue

        peak_idx = (
            actual_group["y"]
            .astype(float)
            .idxmax()
        )

        peak_dates[
            (
                location,
                location_name,
            )
        ] = (
            actual_group.loc[
                peak_idx,
                "ds",
            ]
        )

    # Assign epidemic phase.
    phase_values = []

    for row in df.itertuples():
        key = (
            str(row.location),
            row.location_name,
        )

        peak_date = (
            peak_dates.get(key)
        )

        if peak_date is None:
            phase_values.append(
                np.nan
            )
        else:
            phase_values.append(
                assign_phase(
                    row.ds,
                    peak_date,
                )
            )

    df["phase"] = phase_values

    df = df.dropna(
        subset=[
            "phase",
            "y",
            baseline_col,
            ed_col,
        ]
    ).copy()

    # -----------------------------------------
    # Per-state x horizon x phase metrics
    # -----------------------------------------

    detail_rows = []

    for (
        location,
        location_name,
        horizon,
        phase
    ), group in df.groupby(
        [
            "location",
            "location_name",
            "horizon",
            "phase",
        ]
    ):
        # Deduplicate target dates if needed.
        group = (
            group
            .sort_values("ds")
            .drop_duplicates(
                subset=["ds"],
                keep="last",
            )
        )

        n = len(group)

        if n == 0:
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

        if baseline_rmse > 0:
            rmse_improvement = (
                (
                    baseline_rmse
                    - ed_rmse
                )
                / baseline_rmse
                * 100.0
            )
        else:
            rmse_improvement = np.nan

        if baseline_mae > 0:
            mae_improvement = (
                (
                    baseline_mae
                    - ed_mae
                )
                / baseline_mae
                * 100.0
            )
        else:
            mae_improvement = np.nan

        detail_rows.append(
            {
                "location":
                    location,

                "location_name":
                    location_name,

                "horizon":
                    int(horizon),

                "phase":
                    phase,

                "n_forecasts":
                    n,

                "baseline_rmse":
                    baseline_rmse,

                "ed_rmse":
                    ed_rmse,

                "rmse_improvement_pct":
                    rmse_improvement,

                "baseline_mae":
                    baseline_mae,

                "ed_mae":
                    ed_mae,

                "mae_improvement_pct":
                    mae_improvement,

                "observed_peak_date":
                    peak_dates[
                        (
                            location,
                            location_name,
                        )
                    ],
            }
        )

    detail = pd.DataFrame(
        detail_rows
    )

    OUTPUT_DETAIL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PLOT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    detail.to_csv(
        OUTPUT_DETAIL,
        index=False,
    )

    # -----------------------------------------
    # Aggregate across states.
    # -----------------------------------------

    summary_rows = []

    phase_order = [
        "Growth",
        "Peak",
        "Decline",
    ]

    for (
        horizon,
        phase
    ), group in detail.groupby(
        [
            "horizon",
            "phase",
        ]
    ):
        group = group.dropna(
            subset=[
                "rmse_improvement_pct"
            ]
        )

        if group.empty:
            continue

        improved = int(
            (
                group[
                    "rmse_improvement_pct"
                ]
                > 0
            ).sum()
        )

        summary_rows.append(
            {
                "horizon":
                    int(horizon),

                "phase":
                    phase,

                "states":
                    len(group),

                "states_improved":
                    improved,

                "pct_states_improved":
                    (
                        100.0
                        * improved
                        / len(group)
                    ),

                "mean_rmse_improvement_pct":
                    group[
                        "rmse_improvement_pct"
                    ].mean(),

                "median_rmse_improvement_pct":
                    group[
                        "rmse_improvement_pct"
                    ].median(),

                "mean_mae_improvement_pct":
                    group[
                        "mae_improvement_pct"
                    ].mean(),

                "median_mae_improvement_pct":
                    group[
                        "mae_improvement_pct"
                    ].median(),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary["phase"] = pd.Categorical(
        summary["phase"],
        categories=phase_order,
        ordered=True,
    )

    summary = (
        summary
        .sort_values(
            [
                "horizon",
                "phase",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    print(
        "\nLSTM + ED Phase-Specific Analysis"
    )

    print(
        "================================"
    )

    for horizon in sorted(
        summary["horizon"]
        .unique()
    ):
        print(
            f"\n{horizon}-week horizon"
        )

        print(
            "-" * 40
        )

        scoped = summary[
            summary["horizon"]
            == horizon
        ]

        for row in scoped.itertuples():
            print(
                f"{str(row.phase):8s} | "
                f"{row.states_improved}/"
                f"{row.states} improved | "
                f"mean "
                f"{row.mean_rmse_improvement_pct:+.1f}% | "
                f"median "
                f"{row.median_rmse_improvement_pct:+.1f}%"
            )

    # -----------------------------------------
    # Main plot:
    # median improvement by phase/horizon.
    # -----------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    for phase in phase_order:
        scoped = summary[
            summary["phase"]
            == phase
        ].copy()

        ax.plot(
            scoped["horizon"],
            scoped[
                "median_rmse_improvement_pct"
            ],
            marker="o",
            linewidth=2,
            label=phase,
        )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_xticks(
        sorted(
            summary[
                "horizon"
            ].unique()
        )
    )

    ax.set_xlabel(
        "Forecast horizon (weeks ahead)"
    )

    ax.set_ylabel(
        "Median RMSE improvement "
        "from ED feature (%)"
    )

    ax.set_title(
        "Where Does Historical ED Help? "
        "Performance by Epidemic Phase"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        title="Phase"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_PLOT,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"\nSaved: {OUTPUT_DETAIL}"
    )

    print(
        f"Saved: {OUTPUT_SUMMARY}"
    )

    print(
        f"Saved: {OUTPUT_PLOT}"
    )


if __name__ == "__main__":
    main()