from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from neuralforecast import NeuralForecast
from neuralforecast.models import Informer
from neuralforecast.models.informer import ProbAttention
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

ATTENTION_CSV = (
    OUTPUT_DIR
    / "informer_ed_attention_by_peak_phase.csv"
)

FORECAST_CSV = (
    OUTPUT_DIR
    / "informer_ed_attention_forecasts.csv"
)

SUMMARY_PLOT = (
    PLOT_DIR
    / "informer_ed_attention_peak_phases.png"
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
# Instrumented Informer
# ============================================================

class AttentionInformer(Informer):
    """
    NeuralForecast Informer with encoder attention capture.

    This exposes attention tensors already produced internally
    by Informer's ProbSparse encoder attention.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        enabled = 0

        for module in (
            self.encoder.modules()
        ):
            if isinstance(
                module,
                ProbAttention,
            ):
                module.output_attention = True
                enabled += 1

        self.last_encoder_attentions = None

        print(
            "Enabled encoder attention "
            f"for {enabled} layer(s)"
        )

    def forward(
        self,
        windows_batch,
    ):
        insample_y = (
            windows_batch[
                "insample_y"
            ]
        )

        futr_exog = (
            windows_batch[
                "futr_exog"
            ]
        )

        if self.futr_exog_size > 0:

            x_mark_enc = (
                futr_exog[
                    :,
                    : self.input_size,
                    :,
                ]
            )

            x_mark_dec = (
                futr_exog[
                    :,
                    -(
                        self.label_len
                        + self.h
                    ) :,
                    :,
                ]
            )

        else:

            x_mark_enc = None
            x_mark_dec = None

        x_dec = torch.zeros(
            size=(
                len(insample_y),
                self.h,
                1,
            ),
            device=(
                insample_y.device
            ),
        )

        x_dec = torch.cat(
            [
                insample_y[
                    :,
                    -self.label_len :,
                    :,
                ],
                x_dec,
            ],
            dim=1,
        )

        enc_out = (
            self.enc_embedding(
                insample_y,
                x_mark_enc,
            )
        )

        enc_out, attns = (
            self.encoder(
                enc_out,
                attn_mask=None,
            )
        )

        if not self.training:

            self.last_encoder_attentions = [
                (
                    attn
                    .detach()
                    .cpu()
                    if attn is not None
                    else None
                )
                for attn in attns
            ]

        dec_out = (
            self.dec_embedding(
                x_dec,
                x_mark_dec,
            )
        )

        dec_out = (
            self.decoder(
                dec_out,
                enc_out,
                x_mask=None,
                cross_mask=None,
            )
        )

        forecast = (
            dec_out[
                :,
                -self.h :,
            ]
        )

        return forecast


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

    # Leakage-safe Informer ED feature:
    # at date t, expose ED from t-4 weeks.
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

    df = (
        df
        .sort_values("date")
        .reset_index(
            drop=True
        )
    )

    return df


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
            "No held-out season data."
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

    for label, offset in (
        PHASE_OFFSETS.items()
    ):

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
# Informer construction
# ============================================================

def build_model(
    alias,
):
    return AttentionInformer(
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

        alias=alias,

        accelerator="auto",
        devices=1,
    )


# ============================================================
# Attention processing
# ============================================================

def layer1_key_importance(
    model,
):
    attns = (
        model
        .last_encoder_attentions
    )

    if attns is None:
        raise RuntimeError(
            "No attention captured."
        )

    if len(attns) < 1:
        raise RuntimeError(
            "No encoder layers "
            "captured."
        )

    layer1 = attns[0]

    if layer1 is None:
        raise RuntimeError(
            "Layer-1 attention "
            "is None."
        )

    if layer1.ndim != 4:
        raise RuntimeError(
            "Unexpected attention "
            f"shape: "
            f"{tuple(layer1.shape)}"
        )

    # Shape:
    # batch x heads x queries x keys
    #
    # Average across:
    # batch, heads, query positions
    #
    # Leaves importance over the
    # 12 key/input positions.
    importance = (
        layer1
        .mean(
            dim=(0, 1, 2)
        )
        .numpy()
    )

    total = (
        importance.sum()
    )

    if total > 0:
        importance = (
            importance
            / total
        )

    return importance


# ============================================================
# One state / one forecast origin
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
        f"cutoff {cutoff.date()}"
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
            state_df["date"]
            <= cutoff
        ]
        .copy()
    )

    if (
        len(train_raw)
        < INPUT_SIZE + HORIZON
    ):
        raise ValueError(
            "Insufficient training "
            f"history for {state_name}"
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
            ].isin(
                future_dates
            )
        ]
        .copy()
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
                ].astype(float),

            ED_LAG_COLUMN:
                train_raw[
                    ED_LAG_COLUMN
                ].astype(float),
        }
    )

    futr_df = pd.DataFrame(
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
                ].astype(float),
        }
    )

    actual_future = (
        future_raw[
            [
                "date",
                "weekly_rate",
            ]
        ]
        .rename(
            columns={
                "date": "ds",
                "weekly_rate":
                    "actual",
            }
        )
    )

    alias = (
        "AttentionInformer"
    )

    model = build_model(
        alias=alias
    )

    nf = NeuralForecast(
        models=[model],
        freq="W-SAT",
    )

    nf.fit(
        df=train_df
    )

    fitted_model = (
        nf.models[0]
    )

    fitted_model.last_encoder_attentions = None

    forecast = nf.predict(
        futr_df=futr_df
    )

    importance = (
        layer1_key_importance(
            fitted_model
        )
    )

    if len(
        importance
    ) != INPUT_SIZE:

        raise RuntimeError(
            "Expected "
            f"{INPUT_SIZE} attention "
            "positions but got "
            f"{len(importance)}"
        )

    forecast = (
        forecast
        .merge(
            actual_future,
            on="ds",
            how="left",
        )
    )

    forecast[
        "location_name"
    ] = state_name

    forecast[
        "phase"
    ] = phase_label

    forecast[
        "cutoff"
    ] = cutoff

    return (
        importance,
        forecast,
    )


# ============================================================
# Plotting
# ============================================================

def plot_state_heatmap(
    state_name,
    state_attention,
):

    phase_order = list(
        PHASE_OFFSETS.keys()
    )

    matrix = []

    for phase in phase_order:

        scoped = (
            state_attention[
                state_attention[
                    "phase"
                ]
                == phase
            ]
        )

        if scoped.empty:
            continue

        values = [
            float(
                scoped[
                    f"lag_{lag}"
                ].iloc[0]
            )
            for lag in range(
                11,
                -1,
                -1,
            )
        ]

        matrix.append(
            values
        )

    matrix = np.array(
        matrix
    )

    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
    )

    ax.set_xticks(
        np.arange(
            INPUT_SIZE
        )
    )

    ax.set_xticklabels(
        [
            f"t-{lag}"
            if lag > 0
            else "t"
            for lag in range(
                11,
                -1,
                -1,
            )
        ]
    )

    ax.set_yticks(
        np.arange(
            len(phase_order)
        )
    )

    ax.set_yticklabels(
        phase_order
    )

    ax.set_xlabel(
        "Week in 12-week "
        "observation window"
    )

    ax.set_ylabel(
        "Forecast origin "
        "relative to observed peak"
    )

    ax.set_title(
        f"{state_name}: "
        "Informer Temporal Attention"
    )

    cbar = fig.colorbar(
        image,
        ax=ax,
    )

    cbar.set_label(
        "Mean encoder attention"
    )

    fig.tight_layout()

    output = (
        PLOT_DIR
        / (
            "informer_attention_"
            + state_name
            .lower()
            .replace(
                " ",
                "_",
            )
            + ".png"
        )
    )

    fig.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output}"
    )


def plot_summary(
    attention_df,
):

    state_order = STATES

    phase_order = list(
        PHASE_OFFSETS.keys()
    )

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(14, 13),
    )

    axes = (
        axes
        .flatten()
    )

    for ax, state_name in zip(
        axes,
        state_order,
    ):

        scoped_state = (
            attention_df[
                attention_df[
                    "location_name"
                ]
                == state_name
            ]
        )

        matrix = []

        for phase in phase_order:

            row = scoped_state[
                scoped_state[
                    "phase"
                ]
                == phase
            ]

            values = [
                float(
                    row[
                        f"lag_{lag}"
                    ]
                    .iloc[0]
                )
                for lag in range(
                    11,
                    -1,
                    -1,
                )
            ]

            matrix.append(
                values
            )

        matrix = np.array(
            matrix
        )

        image = ax.imshow(
            matrix,
            aspect="auto",
        )

        ax.set_title(
            state_name
        )

        ax.set_xticks(
            np.arange(
                INPUT_SIZE
            )
        )

        ax.set_xticklabels(
            [
                f"t-{lag}"
                if lag > 0
                else "t"
                for lag in range(
                    11,
                    -1,
                    -1,
                )
            ],
            rotation=45,
        )

        ax.set_yticks(
            np.arange(
                len(
                    phase_order
                )
            )
        )

        ax.set_yticklabels(
            phase_order
        )

    for ax in axes[
        len(state_order) :
    ]:
        ax.axis("off")

    fig.suptitle(
        "Informer First-Layer Temporal "
        "Attention Around 2025–26 Peaks",
        fontsize=16,
    )

    fig.colorbar(
        image,
        ax=list(
            axes[
                : len(
                    state_order
                )
            ]
        ),
        label=(
            "Mean encoder attention"
        ),
        shrink=0.8,
    )

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
        "\nInformer ED Temporal "
        "Attention Analysis"
    )

    print(
        "================================"
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

    attention_rows = []
    forecast_frames = []

    total_runs = (
        len(STATES)
        * len(
            PHASE_OFFSETS
        )
    )

    run_number = 0

    for state_name in STATES:

        state_df = (
            state_frame(
                data,
                state_name,
            )
        )

        peak_date = (
            actual_peak_date(
                state_df
            )
        )

        print(
            "\n"
            + "#" * 70
        )

        print(
            f"{state_name}"
        )

        print(
            "Observed 2025-26 "
            f"peak: "
            f"{peak_date.date()}"
        )

        print(
            "#" * 70
        )

        cutoffs = (
            phase_cutoffs(
                state_df,
                peak_date,
            )
        )

        for (
            phase_label,
            cutoff
        ) in cutoffs.items():

            run_number += 1

            print(
                f"\nRun "
                f"{run_number}/"
                f"{total_runs}"
            )

            (
                importance,
                forecast,
            ) = run_cutoff(
                state_df=state_df,
                state_name=state_name,
                phase_label=phase_label,
                cutoff=cutoff,
            )

            row = {
                "location_name":
                    state_name,

                "peak_date":
                    peak_date,

                "phase":
                    phase_label,

                "cutoff":
                    cutoff,
            }

            # Position 1 corresponds
            # to oldest observation
            # t-11; position 12 = t.
            for position, value in (
                enumerate(
                    importance
                )
            ):

                lag = (
                    INPUT_SIZE
                    - 1
                    - position
                )

                row[
                    f"lag_{lag}"
                ] = float(
                    value
                )

            attention_rows.append(
                row
            )

            forecast_frames.append(
                forecast
            )

            pd.DataFrame(
                attention_rows
            ).to_csv(
                ATTENTION_CSV,
                index=False,
            )

            pd.concat(
                forecast_frames,
                ignore_index=True,
            ).to_csv(
                FORECAST_CSV,
                index=False,
            )

    attention_df = pd.DataFrame(
        attention_rows
    )

    forecast_df = pd.concat(
        forecast_frames,
        ignore_index=True,
    )

    # --------------------------------
    # Diagnostics
    # --------------------------------

    lag_columns = [
        f"lag_{lag}"
        for lag in range(
            11,
            -1,
            -1,
        )
    ]

    attention_df[
        "recent_4w_attention"
    ] = attention_df[
        [
            "lag_3",
            "lag_2",
            "lag_1",
            "lag_0",
        ]
    ].sum(
        axis=1
    )

    attention_df[
        "older_8w_attention"
    ] = attention_df[
        [
            f"lag_{lag}"
            for lag in range(
                11,
                3,
                -1,
            )
        ]
    ].sum(
        axis=1
    )

    attention_df[
        "max_attention_lag"
    ] = attention_df[
        lag_columns
    ].idxmax(
        axis=1
    ).str.replace(
        "lag_",
        "",
        regex=False,
    )

    attention_df.to_csv(
        ATTENTION_CSV,
        index=False,
    )

    forecast_df.to_csv(
        FORECAST_CSV,
        index=False,
    )

    print(
        "\nAttention summary"
    )

    print(
        "================="
    )

    print(
        attention_df[
            [
                "location_name",
                "phase",
                "recent_4w_attention",
                "older_8w_attention",
                "max_attention_lag",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\nMean recent-4-week "
        "attention by phase"
    )

    print(
        attention_df.groupby(
            "phase"
        )[
            "recent_4w_attention"
        ]
        .mean()
        .reindex(
            PHASE_OFFSETS.keys()
        )
        .to_string()
    )

    for state_name in STATES:

        plot_state_heatmap(
            state_name,
            attention_df[
                attention_df[
                    "location_name"
                ]
                == state_name
            ],
        )

    plot_summary(
        attention_df
    )

    print(
        f"\nSaved: "
        f"{ATTENTION_CSV}"
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