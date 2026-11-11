import numpy as np
import pandas as pd
import torch

from neuralforecast import NeuralForecast
from neuralforecast.models import Informer
from neuralforecast.models.informer import ProbAttention
from neuralforecast.losses.pytorch import MAE


class AttentionInformer(Informer):
    """
    Informer that exposes the encoder attention tensors that
    NeuralForecast normally computes and discards.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Enable attention output for every ProbAttention module
        # in the encoder only.
        enabled = 0

        for module in self.encoder.modules():
            if isinstance(module, ProbAttention):
                module.output_attention = True
                enabled += 1

        self.last_encoder_attentions = None

        print(
            f"Enabled attention output for "
            f"{enabled} encoder ProbAttention module(s)"
        )

    def forward(self, windows_batch):
        insample_y = windows_batch["insample_y"]
        futr_exog = windows_batch["futr_exog"]

        if self.futr_exog_size > 0:
            x_mark_enc = futr_exog[
                :, : self.input_size, :
            ]

            x_mark_dec = futr_exog[
                :,
                -(self.label_len + self.h) :,
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
            device=insample_y.device,
        )

        x_dec = torch.cat(
            [
                insample_y[
                    :, -self.label_len :, :
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

        # Save CPU copies only during evaluation/prediction.
        if not self.training:
            self.last_encoder_attentions = [
                (
                    a.detach().cpu()
                    if a is not None
                    else None
                )
                for a in attns
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
            :, -self.h :
        ]

        return forecast


def main():
    print(
        "Torch:",
        torch.__version__,
    )

    print(
        "CUDA available:",
        torch.cuda.is_available(),
    )

    # Tiny synthetic weekly series with one future exogenous feature.
    n = 90

    dates = pd.date_range(
        "2024-01-06",
        periods=n,
        freq="W-SAT",
    )

    x = np.arange(n)

    df = pd.DataFrame(
        {
            "unique_id": ["test"] * n,
            "ds": dates,
            "y": (
                np.sin(x / 7.0)
                + 0.02 * x
            ),
            "ed_signal": (
                np.sin((x + 1) / 7.0)
                + 0.02 * x
            ),
        }
    )

    model = AttentionInformer(
        h=4,
        input_size=12,
        futr_exog_list=[
            "ed_signal"
        ],
        loss=MAE(),
        valid_loss=MAE(),
        max_steps=5,
        hidden_size=16,
        conv_hidden_size=16,
        n_head=2,
        encoder_layers=2,
        decoder_layers=1,
        dropout=0.05,
        learning_rate=1e-3,
        batch_size=16,
        random_seed=2,
        alias="AttentionInformer",
    )

    nf = NeuralForecast(
        models=[model],
        freq="W-SAT",
    )

    nf.fit(
        df=df.iloc[:-4].copy()
    )

    pred = nf.predict(
        futr_df=df[
            [
                "unique_id",
                "ds",
                "ed_signal",
            ]
        ]
        .iloc[-4:]
        .copy()
    )

    print(
        "\nForecast:"
    )

    print(pred)

    fitted_model = nf.models[0]

    attns = (
        fitted_model
        .last_encoder_attentions
    )

    print(
        "\nAttention capture"
    )

    print(
        "================="
    )

    if attns is None:
        raise RuntimeError(
            "No attention tensors captured."
        )

    print(
        "Encoder layers captured:",
        len(attns),
    )

    for i, attn in enumerate(
        attns,
        start=1,
    ):
        if attn is None:
            print(
                f"Layer {i}: None"
            )
            continue

        print(
            f"Layer {i}: "
            f"shape={tuple(attn.shape)}"
        )

        print(
            "  min:",
            float(attn.min()),
        )

        print(
            "  max:",
            float(attn.max()),
        )

        print(
            "  mean:",
            float(attn.mean()),
        )

        # Average over batch, heads, and query positions.
        key_importance = (
            attn
            .mean(dim=(0, 1, 2))
            .numpy()
        )

        key_importance = (
            key_importance
            / key_importance.sum()
        )

        print(
            "  mean attention by "
            "input position:"
        )

        for position, value in enumerate(
            key_importance,
            start=1,
        ):
            print(
                f"    position "
                f"{position:2d}: "
                f"{value:.4f}"
            )

    print(
        "\nSUCCESS: Informer encoder "
        "attention was captured."
    )


if __name__ == "__main__":
    main()