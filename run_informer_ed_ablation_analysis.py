from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from neuralforecast import NeuralForecast
from neuralforecast.models import Informer
from neuralforecast.losses.pytorch import MAE


# ============================================================
# Paths
# ============================================================

HOSPITAL_PATH = Path(
    "data/target-hospital-admissions.csv"
)

ED_PATH = Path(
    "data/target-ed-visits-prop.csv"
)

OUTPUT_DIR = Path(
    "forecast_results"
)

PLOT_DIR = Path(
    "plots"
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "informer_ed_ablation_summary.csv"
)

FORECAST_CSV = (
    OUTPUT_DIR
    / "informer_ed_ablation_forecasts.csv"
)

SUMMARY_PLOT = (
    PLOT_DIR
    / "informer_ed_ablation_peak_phases.png"
)


# ============================================================
# Experiment configuration
# ============================================================

STATES = [
    "Oregon",
    "Pennsylvania",
    "Virginia",
    "Utah",
    "New York",
]

HORIZON = 4
INPUT_SIZE = 12

MAX_STEPS = 1000
SEED = 2

HIDDEN_SIZE = 64
CONV_HIDDEN_SIZE = 64
N_HEAD = 4
ENCODER_LAYERS = 2
DECODER_LAYERS = 1
DROPOUT = 0.05
LEARNING_RATE = 1e-4
BATCH_SIZE = 32

ED_COLUMN = "ed_visits_prop"
ED_LAG_COLUMN = "ed_lag4"

SEASON_START = pd.Timestamp(
    "2025-10-01"
)

SEASON_END = pd.Timestamp(
    "2026-05-31"
)

PHASE_OFFSETS = {
    "Peak -4w": -4,
    "Peak -2w": -2,
    "Peak": 0,
    "Peak +2w": 2,
}


# ============================================================
# Helpers
# ============================================================

def rmse(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    return float(
        np.sqrt(
            np.mean(
                (
                    y_pred
                    - y_true
                )
                ** 2
            )
        )
    )


def mae(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    return float(
        np.mean(
            np.abs(
                y_pred
                - y_true
            )
        )
    )


# ============================================================
# Data preparation
# ============================================================

def load_data():
    hospital = pd.read_csv(
        HOSPITAL_PATH,
        dtype={
            "location": str
        },
    )

    ed = pd.read_csv(
        ED_PATH,
        dtype={
            "location": str
        },
    )

    hospital["date"] = (
        pd.to_datetime(
            hospital["date"]
        )
    )

    ed["date"] = (
        pd.to_datetime(
            ed["date"]
        )
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
                ED_COLUMN
        }
    )

    merged = hospital.merge(
        ed[
            [
                "date",
                "location",
                ED_COLUMN,
            ]
        ],
        on=[
            "date",
            "location",
        ],
        how="inner",
        validate="one_to_one",
    )

    merged = (
        merged
        .sort_values(
            [
                "location",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # Same leakage-safe ED formulation used in
    # the prior Informer experiments.
    merged[
        ED_LAG_COLUMN
    ] = (
        merged
        .groupby(
            "location"
        )[
            ED_COLUMN
        ]
        .shift(4)
    )

    merged = merged.dropna(
        subset=[
            "weekly_rate",
            ED_LAG_COLUMN,
        ]
    ).copy()

    return merged


def state_frame(
    data,
    state_name,
):
    df = data[
        data[
            "location_name"
        ]
        == state_name
    ].copy()

    if df.empty:
        raise ValueError(
            f"No data for "
            f"{state_name}"
        )

    return (
        df
        .sort_values("date")
        .reset_index(
            drop=True
        )
    )


def actual_peak_date(
    df,
):
    season = df[
        (
            df["date"]
            >= SEASON_START
        )
        & (
            df["date"]
            <= SEASON_END
        )
    ].copy()

    if season.empty:
        raise ValueError(
            "No 2025-26 season data."
        )

    idx = (
        season[
            "weekly_rate"
        ]
        .astype(float)
        .idxmax()
    )

    return season.loc[
        idx,
        "date",
    ]


def phase_cutoffs(
    df,
    peak_date,
):
    available_dates = set(
        df["date"].tolist()
    )

    result = {}

    for (
        label,
        offset,
    ) in PHASE_OFFSETS.items():

        desired = (
            peak_date
            + pd.Timedelta(
                weeks=offset
            )
        )

        if desired in available_dates:
            result[label] = desired

        else:
            nearest_idx = (
                (
                    df["date"]
                    - desired
                )
                .abs()
                .idxmin()
            )

            result[label] = (
                df.loc[
                    nearest_idx,
                    "date",
                ]
            )

    return result


# ============================================================
# Model
# ============================================================

def build_model():
    return Informer(
        h=HORIZON,
        input_size=INPUT_SIZE,

        futr_exog_list=[
            ED_LAG_COLUMN
        ],

        loss=MAE(),
        valid_loss=MAE(),

        max_steps=MAX_STEPS,

        hidden_size=(
            HIDDEN_SIZE
        ),

        conv_hidden_size=(
            CONV_HIDDEN_SIZE
        ),

        n_head=N_HEAD,

        encoder_layers=(
            ENCODER_LAYERS
        ),

        decoder_layers=(
            DECODER_LAYERS
        ),

        dropout=DROPOUT,

        learning_rate=(
            LEARNING_RATE
        ),

        batch_size=BATCH_SIZE,

        scaler_type="standard",

        random_seed=SEED,

        alias="InformerED",

        accelerator="auto",
        devices=1,
    )


# ============================================================
# One cutoff
# ============================================================

def run_cutoff(
    state_df,
    state_name,
    phase_label,
    cutoff,
):
    print(
        "\n"
        + "=" * 70
    )

    print(
        f"{state_name} | "
        f"{phase_label} | "
        f"{cutoff.date()}"
    )

    print(
        "=" * 70
    )

    location = str(
        state_df[
            "location"
        ]
        .iloc[0]
    )

    train_raw = (
        state_df[
            state_df[
                "date"
            ]
            <= cutoff
        ]
        .copy()
    )

    future_dates = [
        cutoff
        + pd.Timedelta(
            weeks=i
        )
        for i in range(
            1,
            HORIZON + 1,
        )
    ]

    future_raw = (
        state_df[
            state_df[
                "date"
            ]
            .isin(
                future_dates
            )
        ]
        .copy()
        .sort_values(
            "date"
        )
    )

    if len(
        future_raw
    ) != HORIZON:
        raise ValueError(
            f"{state_name} "
            f"{phase_label}: "
            "missing future dates"
        )

    train_df = pd.DataFrame(
        {
            "unique_id":
                location,

            "ds":
                train_raw[
                    "date"
                ],

            "y":
                train_raw[
                    "weekly_rate"
                ]
                .astype(float),

            ED_LAG_COLUMN:
                train_raw[
                    ED_LAG_COLUMN
                ]
                .astype(float),
        }
    )

    # --------------------------------
    # Real ED inputs
    # --------------------------------

    real_futr = pd.DataFrame(
        {
            "unique_id":
                location,

            "ds":
                future_raw[
                    "date"
                ],

            ED_LAG_COLUMN:
                future_raw[
                    ED_LAG_COLUMN
                ]
                .astype(float),
        }
    )

    # --------------------------------
    # Ablated ED inputs
    # --------------------------------
    #
    # Replace the 4 known future
    # lagged-ED values with the median
    # ED level observed during training.
    #
    # This preserves a plausible-scale
    # input rather than using an
    # unrealistic raw zero.
    # --------------------------------

    neutral_ed = float(
        train_raw[
            ED_LAG_COLUMN
        ]
        .median()
    )

    ablated_futr = (
        real_futr.copy()
    )

    ablated_futr[
        ED_LAG_COLUMN
    ] = neutral_ed

    actual = (
        future_raw[
            "weekly_rate"
        ]
        .astype(float)
        .to_numpy()
    )

    # --------------------------------
    # Fit ONCE
    # --------------------------------

    model = build_model()

    nf = NeuralForecast(
        models=[model],
        freq="W-SAT",
    )

    nf.fit(
        df=train_df
    )

    # --------------------------------
    # Prediction 1:
    # actual ED signal
    # --------------------------------

    real_pred = nf.predict(
        futr_df=real_futr
    )

    # --------------------------------
    # Prediction 2:
    # exact same fitted model,
    # ED replaced by neutral median
    # --------------------------------

    ablated_pred = nf.predict(
        futr_df=ablated_futr
    )

    pred_col = "InformerED"

    real_values = (
        real_pred[
            pred_col
        ]
        .astype(float)
        .to_numpy()
    )

    ablated_values = (
        ablated_pred[
            pred_col
        ]
        .astype(float)
        .to_numpy()
    )

    real_rmse = rmse(
        actual,
        real_values,
    )

    ablated_rmse = rmse(
        actual,
        ablated_values,
    )

    real_mae = mae(
        actual,
        real_values,
    )

    ablated_mae = mae(
        actual,
        ablated_values,
    )

    # Positive means using the real
    # ED signal produced LOWER error.
    rmse_benefit_pct = (
        (
            ablated_rmse
            - real_rmse
        )
        / ablated_rmse
        * 100.0
        if ablated_rmse > 0
        else np.nan
    )

    mean_abs_forecast_change = float(
        np.mean(
            np.abs(
                real_values
                - ablated_values
            )
        )
    )

    max_abs_forecast_change = float(
        np.max(
            np.abs(
                real_values
                - ablated_values
            )
        )
    )

    print(
        "Real ED RMSE:",
        f"{real_rmse:.4f}",
    )

    print(
        "Ablated ED RMSE:",
        f"{ablated_rmse:.4f}",
    )

    print(
        "ED RMSE benefit:",
        f"{rmse_benefit_pct:+.1f}%",
    )

    print(
        "Mean abs forecast change:",
        f"{mean_abs_forecast_change:.4f}",
    )

    # --------------------------------
    # Detailed forecast rows
    # --------------------------------

    detail_rows = []

    for h in range(
        1,
        HORIZON + 1,
    ):
        i = h - 1

        detail_rows.append(
            {
                "location_name":
                    state_name,

                "phase":
                    phase_label,

                "cutoff":
                    cutoff,

                "horizon":
                    h,

                "ds":
                    future_raw[
                        "date"
                    ].iloc[i],

                "actual":
                    actual[i],

                "real_ed":
                    float(
                        real_futr[
                            ED_LAG_COLUMN
                        ].iloc[i]
                    ),

                "ablated_ed":
                    neutral_ed,

                "forecast_real_ed":
                    real_values[i],

                "forecast_ablated_ed":
                    ablated_values[i],

                "forecast_change":
                    (
                        real_values[i]
                        - ablated_values[i]
                    ),

                "abs_forecast_change":
                    abs(
                        real_values[i]
                        - ablated_values[i]
                    ),
            }
        )

    summary = {
        "location_name":
            state_name,

        "phase":
            phase_label,

        "cutoff":
            cutoff,

        "neutral_ed":
            neutral_ed,

        "real_ed_rmse":
            real_rmse,

        "ablated_ed_rmse":
            ablated_rmse,

        "ed_rmse_benefit_pct":
            rmse_benefit_pct,

        "real_ed_mae":
            real_mae,

        "ablated_ed_mae":
            ablated_mae,

        "mean_abs_forecast_change":
            mean_abs_forecast_change,

        "max_abs_forecast_change":
            max_abs_forecast_change,
    }

    return (
        summary,
        detail_rows,
    )


# ============================================================
# Plot
# ============================================================

def make_summary_plot(
    summary_df,
):
    phase_order = list(
        PHASE_OFFSETS.keys()
    )

    matrix = []

    for state in STATES:

        state_rows = (
            summary_df[
                summary_df[
                    "location_name"
                ]
                == state
            ]
            .set_index(
                "phase"
            )
        )

        matrix.append(
            [
                state_rows.loc[
                    phase,
                    "ed_rmse_benefit_pct",
                ]
                for phase in phase_order
            ]
        )

    matrix = np.asarray(
        matrix,
        dtype=float,
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
    )

    ax.set_xticks(
        np.arange(
            len(
                phase_order
            )
        )
    )

    ax.set_xticklabels(
        phase_order
    )

    ax.set_yticks(
        np.arange(
            len(
                STATES
            )
        )
    )

    ax.set_yticklabels(
        STATES
    )

    ax.set_xlabel(
        "Forecast origin relative "
        "to observed hospitalization peak"
    )

    ax.set_ylabel(
        "State"
    )

    ax.set_title(
        "Informer ED Ablation: "
        "Forecast Benefit Around Peak"
    )

    # Add values to cells.
    for i in range(
        matrix.shape[0]
    ):
        for j in range(
            matrix.shape[1]
        ):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:+.1f}%",
                ha="center",
                va="center",
            )

    cbar = fig.colorbar(
        image,
        ax=ax,
    )

    cbar.set_label(
        "RMSE benefit from real ED "
        "vs median-ablated ED (%)"
    )

    fig.tight_layout()

    fig.savefig(
        SUMMARY_PLOT,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {SUMMARY_PLOT}"
    )


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nInformer ED Ablation Analysis"
    )

    print(
        "============================="
    )

    print(
        "CUDA available:",
        torch.cuda.is_available(),
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    data = load_data()

    summary_rows = []
    detail_rows = []

    total_runs = (
        len(STATES)
        * len(
            PHASE_OFFSETS
        )
    )

    run_number = 0

    for state_name in STATES:

        state_df = state_frame(
            data,
            state_name,
        )

        peak_date = actual_peak_date(
            state_df
        )

        print(
            "\n"
            + "#" * 70
        )

        print(
            state_name
        )

        print(
            "Observed peak:",
            peak_date.date(),
        )

        print(
            "#" * 70
        )

        cutoffs = phase_cutoffs(
            state_df,
            peak_date,
        )

        for (
            phase_label,
            cutoff,
        ) in cutoffs.items():

            run_number += 1

            print(
                f"\nRun "
                f"{run_number}/"
                f"{total_runs}"
            )

            (
                summary,
                details,
            ) = run_cutoff(
                state_df=state_df,
                state_name=state_name,
                phase_label=phase_label,
                cutoff=cutoff,
            )

            summary_rows.append(
                summary
            )

            detail_rows.extend(
                details
            )

            # Incremental saves in case
            # the SLURM job is interrupted.
            pd.DataFrame(
                summary_rows
            ).to_csv(
                SUMMARY_CSV,
                index=False,
            )

            pd.DataFrame(
                detail_rows
            ).to_csv(
                FORECAST_CSV,
                index=False,
            )

    summary_df = pd.DataFrame(
        summary_rows
    )

    detail_df = pd.DataFrame(
        detail_rows
    )

    summary_df.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    detail_df.to_csv(
        FORECAST_CSV,
        index=False,
    )

    print(
        "\nOverall summary"
    )

    print(
        "==============="
    )

    print(
        "Runs:",
        len(summary_df),
    )

    print(
        "Real ED improves RMSE:",
        (
            summary_df[
                "ed_rmse_benefit_pct"
            ]
            > 0
        ).sum(),
        "/",
        len(summary_df),
    )

    print(
        "Median ED RMSE benefit:",
        f"{summary_df['ed_rmse_benefit_pct'].median():+.1f}%",
    )

    print(
        "\nMedian ED benefit by phase"
    )

    print(
        summary_df.groupby(
            "phase"
        )[
            "ed_rmse_benefit_pct"
        ]
        .median()
        .reindex(
            PHASE_OFFSETS.keys()
        )
        .to_string()
    )

    print(
        "\nMean absolute forecast "
        "change by phase"
    )

    print(
        summary_df.groupby(
            "phase"
        )[
            "mean_abs_forecast_change"
        ]
        .mean()
        .reindex(
            PHASE_OFFSETS.keys()
        )
        .to_string()
    )

    make_summary_plot(
        summary_df
    )

    print(
        f"\nSaved: "
        f"{SUMMARY_CSV}"
    )

    print(
        f"Saved: "
        f"{FORECAST_CSV}"
    )

    print(
        "\nDONE"
    )


if __name__ == "__main__":
    main()