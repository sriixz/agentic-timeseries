import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import Informer
from neuralforecast.models.informer import ProbAttention


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
    "forecast_results/weekly_attention"
)

PLOT_DIR = Path(
    "plots/weekly_attention"
)


# ============================================================
# Experiment configuration
#
# Keep same architecture as previous attention experiment
# for direct comparability.
# ============================================================

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

TRAIN_END = pd.Timestamp(
    "2025-09-30"
)

ATTENTION_START = pd.Timestamp(
    "2025-11-01"
)

ATTENTION_END = pd.Timestamp(
    "2026-02-28"
)


# ============================================================
# Instrumented Informer
# ============================================================

class AttentionInformer(Informer):

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
        insample_y = windows_batch[
            "insample_y"
        ]

        futr_exog = windows_batch[
            "futr_exog"
        ]

        if self.futr_exog_size > 0:

            x_mark_enc = futr_exog[
                :,
                : self.input_size,
                :,
            ]

            x_mark_dec = futr_exog[
                :,
                -(
                    self.label_len
                    + self.h
                ) :,
                :,
            ]

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

        enc_out = self.enc_embedding(
            insample_y,
            x_mark_enc,
        )

        enc_out, attns = self.encoder(
            enc_out,
            attn_mask=None,
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

        dec_out = self.dec_embedding(
            x_dec,
            x_mark_dec,
        )

        dec_out = self.decoder(
            dec_out,
            enc_out,
            x_mask=None,
            cross_mask=None,
        )

        forecast = dec_out[
            :,
            -self.h :,
        ]

        return forecast


# ============================================================
# Data
# ============================================================

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
            "value": ED_COLUMN
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


def eligible_states(
    data,
):

    states = []

    for state_name, group in (
        data.groupby(
            "location_name"
        )
    ):

        if state_name == "US":
            continue

        train = group[
            group["date"]
            <= TRAIN_END
        ]

        attention_period = group[
            (
                group["date"]
                >= ATTENTION_START
            )
            & (
                group["date"]
                <= ATTENTION_END
            )
        ]

        if len(train) < 60:
            continue

        if attention_period.empty:
            continue

        states.append(
            state_name
        )

    return sorted(
        states
    )


# ============================================================
# Model
# ============================================================

def build_model():

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

        alias="AttentionInformer",

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

    layer1 = attns[0]

    if layer1 is None:
        raise RuntimeError(
            "Layer-1 attention is None."
        )

    if layer1.ndim != 4:
        raise RuntimeError(
            "Unexpected attention shape: "
            f"{tuple(layer1.shape)}"
        )

    importance = (
        layer1
        .mean(
            dim=(0, 1, 2)
        )
        .numpy()
    )

    total = importance.sum()

    if total > 0:
        importance = (
            importance
            / total
        )

    if len(importance) != INPUT_SIZE:
        raise RuntimeError(
            f"Expected {INPUT_SIZE} "
            "attention positions, got "
            f"{len(importance)}"
        )

    return importance


# ============================================================
# One state
# ============================================================

def run_state(
    data,
    state_name,
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        state_name
    )

    print(
        "=" * 70
    )

    state_df = (
        data[
            data[
                "location_name"
            ]
            == state_name
        ]
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )

    location = str(
        state_df[
            "location"
        ]
        .iloc[0]
    )

    train_raw = state_df[
        state_df["date"]
        <= TRAIN_END
    ].copy()

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

    print(
        "Training observations:",
        len(train_df),
    )

    model = build_model()

    nf = NeuralForecast(
        models=[
            model
        ],
        freq="W-SAT",
    )

    # --------------------------------
    # FIT ONCE
    # --------------------------------

    nf.fit(
        df=train_df
    )

    fitted_model = nf.models[0]

    # Weekly forecast origins
    cutoffs = (
        state_df[
            (
                state_df["date"]
                >= ATTENTION_START
            )
            & (
                state_df["date"]
                <= ATTENTION_END
            )
        ][
            "date"
        ]
        .sort_values()
        .tolist()
    )

    attention_rows = []

    total = len(cutoffs)

    for index, cutoff in enumerate(
        cutoffs,
        start=1,
    ):

        print(
            f"{state_name}: "
            f"week {index}/{total} "
            f"| {cutoff.date()}"
        )

        # --------------------------------
        # Update observed history only.
        #
        # Model weights remain frozen.
        # --------------------------------

        history_raw = state_df[
            state_df["date"]
            <= cutoff
        ].copy()

        history_df = pd.DataFrame(
            {
                "unique_id":
                    location,

                "ds":
                    history_raw[
                        "date"
                    ],

                "y":
                    history_raw[
                        "weekly_rate"
                    ]
                    .astype(float),

                ED_LAG_COLUMN:
                    history_raw[
                        ED_LAG_COLUMN
                    ]
                    .astype(float),
            }
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
            .sort_values(
                "date"
            )
            .copy()
        )

        if len(
            future_raw
        ) != HORIZON:

            print(
                "Skipping cutoff due to "
                "missing future dates:",
                cutoff.date(),
            )

            continue

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
                    ]
                    .astype(float),
            }
        )

        fitted_model.last_encoder_attentions = None

        # Same fitted model.
        # New historical window supplied
        # for each weekly forecast origin.
        nf.predict(
            df=history_df,
            futr_df=futr_df,
        )

        importance = (
            layer1_key_importance(
                fitted_model
            )
        )

        row = {
            "location":
                location,

            "location_name":
                state_name,

            "cutoff":
                cutoff,
        }

        for (
            position,
            value,
        ) in enumerate(
            importance
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

        row[
            "recent_4w_attention"
        ] = float(
            sum(
                row[
                    f"lag_{lag}"
                ]
                for lag in range(
                    0,
                    4,
                )
            )
        )

        row[
            "older_8w_attention"
        ] = float(
            sum(
                row[
                    f"lag_{lag}"
                ]
                for lag in range(
                    4,
                    12,
                )
            )
        )

        lag_values = {
            lag:
                row[
                    f"lag_{lag}"
                ]
            for lag in range(
                12
            )
        }

        row[
            "max_attention_lag"
        ] = max(
            lag_values,
            key=lag_values.get,
        )

        attention_rows.append(
            row
        )

    attention_df = pd.DataFrame(
        attention_rows
    )

    if attention_df.empty:
        raise RuntimeError(
            f"No valid attention rows "
            f"for {state_name}."
        )

    safe_name = (
        state_name
        .lower()
        .replace(
            " ",
            "_",
        )
    )

    csv_path = (
        OUTPUT_DIR
        / (
            f"informer_weekly_attention_"
            f"{safe_name}.csv"
        )
    )

    attention_df.to_csv(
        csv_path,
        index=False,
    )

    plot_state_heatmap(
        state_name=state_name,
        attention_df=attention_df,
    )

    print(
        f"Saved: {csv_path}"
    )


# ============================================================
# Plotting
# ============================================================

def plot_state_heatmap(
    state_name,
    attention_df,
):

    lag_order = list(
        range(
            11,
            -1,
            -1,
        )
    )

    matrix = attention_df[
        [
            f"lag_{lag}"
            for lag in lag_order
        ]
    ].to_numpy(
        dtype=float
    )

    labels = (
        pd.to_datetime(
            attention_df[
                "cutoff"
            ]
        )
        .dt.strftime(
            "%b %d"
        )
        .tolist()
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            8,
        )
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
            (
                f"t-{lag}"
                if lag > 0
                else "t"
            )
            for lag in lag_order
        ],
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        np.arange(
            len(labels)
        )
    )

    ax.set_yticklabels(
        labels
    )

    ax.set_xlabel(
        "Week in 12-week observation window"
    )

    ax.set_ylabel(
        "Forecast origin"
    )

    ax.set_title(
        f"{state_name}: "
        "Informer First-Layer Temporal Attention\n"
        "Frozen model weights, "
        "Nov 2025 - Feb 2026"
    )

    cbar = fig.colorbar(
        image,
        ax=ax,
    )

    cbar.set_label(
        "Mean encoder attention"
    )

    fig.tight_layout()

    safe_name = (
        state_name
        .lower()
        .replace(
            " ",
            "_",
        )
    )

    output = (
        PLOT_DIR
        / (
            f"informer_weekly_attention_"
            f"{safe_name}.png"
        )
    )

    fig.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {output}"
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--state-index",
        type=int,
        default=None,
        help=(
            "Optional zero-based index "
            "into eligible state list."
        ),
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nAll-State Weekly "
        "Informer Attention"
    )

    print(
        "============================"
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

    states = eligible_states(
        data
    )

    print(
        "Eligible states:",
        len(states),
    )

    for index, state in enumerate(
        states
    ):
        print(
            f"{index}: {state}"
        )

    if args.state_index is not None:

        if (
            args.state_index < 0
            or args.state_index
            >= len(states)
        ):
            raise ValueError(
                "state-index out of range."
            )

        run_state(
            data=data,
            state_name=(
                states[
                    args.state_index
                ]
            ),
        )

    else:

        for state_name in states:

            run_state(
                data=data,
                state_name=state_name,
            )

    print(
        "\nDONE"
    )


if __name__ == "__main__":
    main()