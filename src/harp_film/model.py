import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Configs:
    """Model hyperparameters used by HARP-FiLM."""

    seq_len: int = 24
    label_len: int = 24
    pred_len: int = 24
    d_model: int = 256
    d_ff: int = 256
    n_heads: int = 8
    e_layers: int = 4
    dropout: float = 0.15
    activation: str = "relu"
    input_feature_dim: int = 9
    k_recent: int = 8
    gain_limit: float = 0.8


class LearnableTemporalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TemporalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :].to(x.device)


class EnhancedCNN2D(nn.Module):
    """Multi-branch CNN with 3x3, 5x5, and 7x7 kernels."""

    def __init__(self, in_channels: int, out_channels: int, kernel_sizes=(3, 5, 7)):
        super().__init__()
        assert out_channels % len(kernel_sizes) == 0, "out_channels must be divisible by number of branches."
        branch_ch = out_channels // len(kernel_sizes)
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, branch_ch, k, padding=k // 2, bias=False),
                    nn.BatchNorm2d(branch_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(branch_ch, branch_ch, k, padding=k // 2, bias=False),
                    nn.BatchNorm2d(branch_ch),
                    nn.ReLU(inplace=True),
                )
                for k in kernel_sizes
            ]
        )
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        res = x if self.residual is None else self.residual(x)
        y = torch.cat([conv(x) for conv in self.convs], dim=1)
        y = torch.relu(y + res)
        return self.dropout(y)


class AuxMLPEncoder(nn.Module):
    """Encode auxiliary physical drivers and temporal encodings to latent dimension D."""

    def __init__(self, in_dim: int, d_model: int, hidden_multiplier: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden = max(d_model, int(hidden_multiplier * d_model))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x):
        return self.net(x)


class AdaptiveFusionFiLM(nn.Module):
    """Physics-FiLM modulation with bounded scale and gated injection."""

    def __init__(self, d_model: int, dropout: float = 0.15):
        super().__init__()
        self.film = nn.Linear(d_model, 2 * d_model)
        self.gate_mlp = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_tec, x_aux, return_diagnostics: bool = False):
        gamma_beta = self.film(x_aux)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma_tanh = torch.tanh(gamma)
        x_mod = x_tec * (1.0 + gamma_tanh) + beta
        gate = self.gate_mlp(torch.cat([x_tec, x_aux, x_tec * x_aux], dim=-1))
        y = x_tec + gate * (x_mod - x_tec)
        y = y + self.out(y)
        if return_diagnostics:
            diagnostics = {"gamma_tanh": gamma_tanh, "beta": beta, "gate": gate}
            return y, diagnostics
        return y


class AttnPoolTimeK(nn.Module):
    """Horizon-aware attention pooling over the most recent k hidden states."""

    def __init__(self, d_model: int, k_recent: int = 8, pred_len: int = 24):
        super().__init__()
        self.k = int(k_recent)
        self.Q = nn.Parameter(torch.randn(1, pred_len, d_model))
        self.out = nn.Linear(d_model, d_model)

    def forward(self, mem, return_attention: bool = False):
        B, L, D = mem.shape
        k = min(self.k, L)
        recent = mem[:, -k:, :]
        Q = self.Q.expand(B, -1, -1)
        attn = torch.softmax((Q @ recent.transpose(1, 2)) / (D**0.5), dim=-1)
        ctx = attn @ recent
        ctx = self.out(ctx)
        if return_attention:
            return ctx, attn
        return ctx


class ResidualHeadV(nn.Module):
    """Adaptive dual-baseline residual head.

    base_t = alpha_t * last_frame + (1 - alpha_t) * last_day
    y_t = base_t * (1 + gain_t) + delta_t + bias_t
    """

    def __init__(self, d_model: int, pred_len: int, H: int, W: int, use_gain_bias: bool = True, gain_limit: float = 0.8):
        super().__init__()
        self.pred_len = pred_len
        self.H = H
        self.W = W
        self.use_gain_bias = use_gain_bias
        self.gain_limit = float(gain_limit)
        self.step_embed = nn.Parameter(torch.randn(1, pred_len, d_model))
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, H * W))
        if self.use_gain_bias:
            self.gb = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        self.alpha_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        self.post = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.15),
        )

    def forward(self, mem_ctx_per_step, last_frame, last_day=None, return_diagnostics: bool = False):
        B, T, D = mem_ctx_per_step.shape
        assert T == self.pred_len, f"Expected pred_len={self.pred_len}, got {T}."
        z = mem_ctx_per_step + self.step_embed
        z = z + self.post(z)

        base_last = last_frame.expand(-1, T, -1, -1)
        alpha = None
        if last_day is None:
            base = base_last
        else:
            base_day = last_day.expand(-1, T, -1, -1) if last_day.shape[1] == 1 else last_day
            alpha = self.alpha_mlp(z).view(B, T, 1, 1)
            base = alpha * base_last + (1.0 - alpha) * base_day

        delta = self.mlp(z).view(B, T, self.H, self.W)
        gain = bias = None
        if self.use_gain_bias:
            gb = self.gb(z)
            gain = torch.tanh(gb[..., 0:1]) * self.gain_limit
            bias = gb[..., 1:2] * 0.10
            y = base * (1.0 + gain.view(B, T, 1, 1)) + delta + bias.view(B, T, 1, 1)
        else:
            y = base + delta

        if return_diagnostics:
            diagnostics = {"alpha": alpha, "delta": delta, "gain": gain, "bias": bias}
            return y, diagnostics
        return y


class CNNEncoder(nn.Module):
    """HARP-FiLM model."""

    def __init__(
        self,
        configs: Configs,
        cnn_channels: int,
        num_latitudes: int,
        num_longitudes: int,
        input_feature_dim: int,
        use_learnable_temporal: bool = True,
    ):
        super().__init__()
        self.num_latitudes = num_latitudes
        self.num_longitudes = num_longitudes
        self.pred_len = configs.pred_len
        self.use_phys = input_feature_dim is not None and input_feature_dim > 0

        self.cnn_stack = nn.Sequential(
            EnhancedCNN2D(in_channels=5, out_channels=cnn_channels, kernel_sizes=(3, 5, 7)),
            EnhancedCNN2D(in_channels=cnn_channels, out_channels=cnn_channels, kernel_sizes=(3, 5, 7)),
            EnhancedCNN2D(in_channels=cnn_channels, out_channels=cnn_channels, kernel_sizes=(3, 5, 7)),
        )
        self.proj = nn.Linear(cnn_channels * num_latitudes * num_longitudes, configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)

        if self.use_phys:
            self.aux_mlp = AuxMLPEncoder(input_feature_dim, configs.d_model, hidden_multiplier=2, dropout=configs.dropout)
            self.afuse = AdaptiveFusionFiLM(configs.d_model, dropout=configs.dropout)

        self.temb = (
            LearnableTemporalEncoding(configs.d_model, max_len=configs.seq_len)
            if use_learnable_temporal
            else TemporalEncoding(configs.d_model, max_len=configs.seq_len)
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=configs.d_model,
            nhead=configs.n_heads,
            dim_feedforward=configs.d_ff,
            dropout=configs.dropout,
            activation=configs.activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=configs.e_layers)
        self.norm = nn.LayerNorm(configs.d_model)
        self.time_pool = AttnPoolTimeK(configs.d_model, k_recent=configs.k_recent, pred_len=configs.pred_len)
        self.res_head = ResidualHeadV(
            configs.d_model,
            configs.pred_len,
            num_latitudes,
            num_longitudes,
            gain_limit=configs.gain_limit,
        )

    def forward(self, x_dec, x_mark_dec, return_diagnostics: bool = False):
        B, L, C, H, W = x_dec.shape
        x = self.cnn_stack(x_dec.view(B * L, C, H, W)).view(B, L, -1)
        x_tec = self.dropout(self.proj(x))
        diagnostics = {}

        if self.use_phys:
            x_aux = self.aux_mlp(x_mark_dec.float())
            if return_diagnostics:
                x_seq, film_diag = self.afuse(x_tec, x_aux, return_diagnostics=True)
                diagnostics["film"] = film_diag
            else:
                x_seq = self.afuse(x_tec, x_aux)
        else:
            x_seq = x_tec

        x_seq = self.temb(x_seq)
        mem = self.norm(self.encoder(x_seq))

        if return_diagnostics:
            ctx_per_step, attn = self.time_pool(mem, return_attention=True)
            diagnostics["temporal_attention"] = attn
        else:
            ctx_per_step = self.time_pool(mem)

        if L >= self.pred_len:
            last_day = x_dec[:, -self.pred_len :, 0, :, :]
        else:
            last_day = x_dec[:, -1:, 0, :, :].expand(-1, self.pred_len, -1, -1)
        last_frame = x_dec[:, -1:, 0, :, :]

        if return_diagnostics:
            y, head_diag = self.res_head(ctx_per_step, last_frame, last_day, return_diagnostics=True)
            diagnostics["residual_head"] = head_diag
            return y, diagnostics
        return self.res_head(ctx_per_step, last_frame, last_day)


# Backward-compatible alias used by the original single-file script.
CNN_Encoder = CNNEncoder
