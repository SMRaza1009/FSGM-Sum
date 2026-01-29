import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.positional_encoding import (
    FixedPositionalEncoding,
    LearnablePositionalEncoding,
    RelativePositionalEncoding,
    ConditionalPositionalEncoding,
)


class _FSGMAttention(nn.Module):
    """
    Factorized attention logits generator over (T x D).

    Input:  seq  [T, D]
    Output: logits [T, D]

    - Temporal logits: multi-dilation depthwise temporal mixing (cheap).
    - Feature logits : pooled channel descriptor + tiny MLP (cheap).
    - Optional FFT scalar bias to help periodic patterns (still cheap).
    """

    def __init__(
        self,
        dim: int,
        hidden: int = 64,
        channel_reduction: int = 16,
        use_fft_bias: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.use_fft_bias = bool(use_fft_bias)
        self.eps = float(eps)

        # Temporal path: project features -> lightweight temporal mixing.
        self.temporal_proj = nn.Linear(self.dim, self.hidden)

        def _dw_pw_block(dilation: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(
                    in_channels=self.hidden,
                    out_channels=self.hidden,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    groups=self.hidden,
                    bias=False,
                ),
                nn.Conv1d(self.hidden, self.hidden, kernel_size=1, bias=False),
                nn.GELU(),
                nn.BatchNorm1d(self.hidden, eps=self.eps),
            )

        self.temporal_blocks = nn.ModuleList([
            _dw_pw_block(1),
            _dw_pw_block(2),
            _dw_pw_block(4),
        ])
        self.temporal_head = nn.Conv1d(self.hidden, 1, kernel_size=1, bias=True)

        # Feature path: pooled descriptor -> lightweight channel gating.
        red = max(4, self.dim // max(1, int(channel_reduction)))
        self.channel_mlp = nn.Sequential(
            nn.Linear(self.dim * 2, red),
            nn.ReLU(inplace=True),
            nn.Linear(red, self.dim),
        )

        # Optional tiny FFT-derived scalar bias.
        if self.use_fft_bias:
            self.fft_scale = nn.Sequential(
                nn.Linear(1, 8),
                nn.ReLU(inplace=True),
                nn.Linear(8, 1),
            )

        # Stable initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2.0))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: [T, D]
        if seq.dim() != 2:
            raise ValueError(f"_FSGMAttention expects [T,D], got {tuple(seq.shape)}")
        T, D = seq.shape
        if D != self.dim:
            raise ValueError(f"Feature dim mismatch: expected {self.dim}, got {D}")

        # -------- Temporal logits --------
        h = self.temporal_proj(seq)               # [T, hidden]
        h = h.transpose(0, 1).unsqueeze(0)        # [1, hidden, T]
        mix = 0.0
        for blk in self.temporal_blocks:
            mix = mix + blk(h)
        mix = mix / float(len(self.temporal_blocks))
        temporal_logits = self.temporal_head(mix).squeeze(0).squeeze(0)  # [T]

        if self.use_fft_bias:
            s = seq.mean(dim=1)  # [T]
            mag = torch.abs(torch.fft.rfft(s, dim=0))
            stat = mag.mean().clamp(min=self.eps)
            bias = self.fft_scale(stat.view(1, 1)).view(())
            temporal_logits = temporal_logits + bias

        # -------- Feature logits --------
        mean_pool = seq.mean(dim=0)          # [D]
        max_pool = seq.max(dim=0).values     # [D]
        feat_desc = torch.cat([mean_pool, max_pool], dim=0)  # [2D]
        feature_logits = self.channel_mlp(feat_desc)         # [D]

        # -------- Combine to (T x D) logits --------
        logits = temporal_logits.unsqueeze(1) + feature_logits.unsqueeze(0)  # [T, D]
        return logits


class CSTA_FSGM(nn.Module):
    """
    Plug-and-play replacement for CSTA_* models.

    Input : x [1, 3, T, 1024]  (as produced by dataset.py)
    Output: scores [T] in [0,1]

    Keeps your existing flags: Softmax_axis, Balance, positional encodings,
    CLS token and CLS_mix, skip connections, Layernorm, key/value embeddings.
    """

    def __init__(
        self,
        model_name,
        Scale,
        Softmax_axis,
        Balance,
        Positional_encoding,
        Positional_encoding_shape,
        Positional_encoding_way,
        Dropout_on,
        Dropout_ratio,
        Classifier_on,
        CLS_on,
        CLS_mix,
        key_value_emb,
        Skip_connection,
        Layernorm,
        dim: int = 1024,
        fsgm_hidden: int = 64,
        fsgm_channel_reduction: int = 16,
        fsgm_use_fft_bias: bool = True,
    ):
        super().__init__()

        self.model_name = model_name
        self.Scale = Scale
        self.Softmax_axis = Softmax_axis
        self.Balance = Balance

        self.Positional_encoding = Positional_encoding
        self.Positional_encoding_shape = Positional_encoding_shape
        self.Positional_encoding_way = Positional_encoding_way
        self.Dropout_on = Dropout_on
        self.Dropout_ratio = Dropout_ratio

        self.Classifier_on = Classifier_on
        self.CLS_on = CLS_on
        self.CLS_mix = CLS_mix

        self.key_value_emb = key_value_emb
        self.Skip_connection = Skip_connection
        self.Layernorm = Layernorm

        self.dim = int(dim)

        # Positional encoding operator (same options as your models)
        if self.Positional_encoding is not None:
            if self.Positional_encoding == 'FPE':
                self.Positional_encoding_op = FixedPositionalEncoding(
                    Positional_encoding_shape=self.Positional_encoding_shape,
                    dim=self.dim,
                )
            elif self.Positional_encoding == 'RPE':
                self.Positional_encoding_op = RelativePositionalEncoding(
                    Positional_encoding_shape=self.Positional_encoding_shape,
                    dim=self.dim,
                )
            elif self.Positional_encoding == 'LPE':
                self.Positional_encoding_op = LearnablePositionalEncoding(
                    Positional_encoding_shape=self.Positional_encoding_shape,
                    dim=self.dim,
                )
            elif self.Positional_encoding == 'CPE':
                self.Positional_encoding_op = ConditionalPositionalEncoding(
                    Positional_encoding_shape=self.Positional_encoding_shape,
                    Positional_encoding_way=self.Positional_encoding_way,
                    dim=self.dim,
                )
            else:
                raise ValueError(f"Unknown Positional_encoding={self.Positional_encoding}")

        if self.Positional_encoding_way == 'Transformer':
            self.Positional_encoding_embedding = nn.Linear(self.dim, self.dim)
        elif self.Positional_encoding_way in ['PGL_SUM', None]:
            pass
        else:
            raise ValueError(f"Unknown Positional_encoding_way={self.Positional_encoding_way}")

        if self.Dropout_on:
            self.dropout = nn.Dropout(p=float(self.Dropout_ratio))

        # Classifier (same as your models)
        if self.Classifier_on:
            self.linear1 = nn.Sequential(
                nn.Linear(in_features=self.dim, out_features=self.dim),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.LayerNorm(normalized_shape=self.dim, eps=1e-6),
            )
            self.linear2 = nn.Sequential(
                nn.Linear(in_features=self.dim, out_features=1),
                nn.Sigmoid(),
            )
            for name, param in self.named_parameters():
                if name in ['linear1.0.weight', 'linear2.0.weight']:
                    nn.init.xavier_uniform_(param, gain=np.sqrt(2.0))
                elif name in ['linear1.0.bias', 'linear2.0.bias']:
                    nn.init.constant_(param, 0.1)
        else:
            self.gap = nn.AdaptiveAvgPool1d(1)

        # CLS token in input space (fixed 1024 like your CSTA models)
        if self.CLS_on:
            self.CLS = nn.Parameter(torch.zeros(1, 3, 1, 1024))

        # Key/Value embeddings for compatibility
        if self.key_value_emb is not None:
            kv = self.key_value_emb.lower()
            if kv == 'k':
                self.key_embedding = nn.Linear(in_features=1024, out_features=self.dim)
            elif kv == 'v':
                self.value_embedding = nn.Linear(in_features=self.dim, out_features=self.dim)
            elif ''.join(sorted(kv)) == 'kv':
                self.key_embedding = nn.Linear(in_features=1024, out_features=self.dim)
                if self.model_name == 'FSGM_Attention':
                    self.value_embedding = nn.Linear(in_features=1024, out_features=self.dim)
            else:
                raise ValueError(f"Unknown key_value_emb={self.key_value_emb}")

        # Skip-connection layer norms (keep identical behavior)
        if self.Layernorm:
            if self.Skip_connection == 'KC':
                self.layernorm1 = nn.BatchNorm2d(num_features=1)
            elif self.Skip_connection == 'CF':
                self.layernorm2 = nn.BatchNorm2d(num_features=1)
            elif self.Skip_connection == 'IF':
                self.layernorm3 = nn.BatchNorm2d(num_features=1)
            elif self.Skip_connection is None:
                pass
            else:
                raise ValueError(f"Unknown Skip_connection={self.Skip_connection}")

        # FSGM attention logits generator
        self.fsgm = _FSGMAttention(
            dim=self.dim,
            hidden=fsgm_hidden,
            channel_reduction=fsgm_channel_reduction,
            use_fft_bias=fsgm_use_fft_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, 3, T, 1024]
        n_frame = x.shape[2]

        # Linear projection if using Transformer way
        if self.Positional_encoding_way == 'Transformer':
            x = self.Positional_encoding_embedding(x)

        # Stack CLS token
        if self.CLS_on:
            x = torch.cat((self.CLS.to(x.dtype).to(x.device), x), dim=2)
            CT_adjust = nn.AdaptiveAvgPool2d((n_frame, self.dim))

        # Positional encoding (Transformer way)
        if self.Positional_encoding_way == 'Transformer':
            if self.Positional_encoding is not None:
                x = self.Positional_encoding_op(x)
            if self.Dropout_on:
                x = self.dropout(x)
        elif self.Positional_encoding_way in ['PGL_SUM', None]:
            pass
        else:
            raise

        # Key Embedding
        if self.key_value_emb is not None and self.key_value_emb.lower() in ['k', 'kv']:
            key = self.key_embedding(x)
        elif self.key_value_emb is None:
            key = x
        else:
            raise

        # ===== Replacement for CNN-as-attention =====
        key_seq = key.squeeze(0)[0]  # [T(+1), D]
        x_att = self.fsgm(key_seq)   # [T(+1), D]

        # Skip connection (KC)
        if self.Skip_connection is not None:
            if self.Skip_connection == 'KC':
                x_att = x_att + key_seq
                if self.Layernorm:
                    x_att = self.layernorm1(x_att.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            elif self.Skip_connection in ['CF', 'IF']:
                pass
            else:
                raise

        # Combine CLS token ("CNN stage")
        if self.CLS_on:
            if self.CLS_mix == 'CNN':
                x_att = CT_adjust(x_att.unsqueeze(0)).squeeze(0)
                x = CT_adjust(x.squeeze(0)).unsqueeze(0)
            elif self.CLS_mix in ['SM', 'Final']:
                pass
            else:
                raise

        # Scaling factor
        if self.Scale is not None:
            if self.Scale == 'D':
                scaling_factor = x_att.shape[1]
            elif self.Scale == 'T':
                scaling_factor = x_att.shape[0]
            elif self.Scale == 'T_D':
                scaling_factor = x_att.shape[0] * x_att.shape[1]
            else:
                raise
            x_att = x_att / (scaling_factor ** 0.5)

        # Positional encoding (PGL-SUM way)
        if self.Positional_encoding_way == 'PGL_SUM':
            if self.Positional_encoding is not None:
                x_att = self.Positional_encoding_op(x_att)
        elif self.Positional_encoding_way in ['Transformer', None]:
            pass
        else:
            raise

        # softmax_axis
        x_base = x.squeeze(0)[0]  # [T(+1), D] first channel
        if self.Softmax_axis == 'T':
            temporal_attention = F.softmax(x_att, dim=0)
        elif self.Softmax_axis == 'D':
            spatial_attention = F.softmax(x_att, dim=1)
        elif self.Softmax_axis == 'TD':
            temporal_attention = F.softmax(x_att, dim=0)
            spatial_attention = F.softmax(x_att, dim=1)
        elif self.Softmax_axis is None:
            pass
        else:
            raise

        # Combine CLS token for softmax outputs (SM)
        if self.CLS_on:
            if self.CLS_mix == 'SM':
                if self.Softmax_axis == 'T':
                    temporal_attention = CT_adjust(temporal_attention.unsqueeze(0)).squeeze(0)
                elif self.Softmax_axis == 'D':
                    spatial_attention = CT_adjust(spatial_attention.unsqueeze(0)).squeeze(0)
                elif self.Softmax_axis == 'TD':
                    temporal_attention = CT_adjust(temporal_attention.unsqueeze(0)).squeeze(0)
                    spatial_attention = CT_adjust(spatial_attention.unsqueeze(0)).squeeze(0)
                elif self.Softmax_axis is None:
                    pass
                else:
                    raise
            elif self.CLS_mix in ['CNN', 'Final']:
                pass
            else:
                raise

        # Dropout (PGL-SUM way)
        if self.Dropout_on and self.Positional_encoding_way == 'PGL_SUM':
            if self.Softmax_axis == 'T':
                temporal_attention = self.dropout(temporal_attention)
            elif self.Softmax_axis == 'D':
                spatial_attention = self.dropout(spatial_attention)
            elif self.Softmax_axis == 'TD':
                temporal_attention = self.dropout(temporal_attention)
                spatial_attention = self.dropout(spatial_attention)
            elif self.Softmax_axis is None:
                pass
            else:
                raise

        # Value Embedding
        if self.key_value_emb is not None and self.key_value_emb.lower() in ['v', 'kv']:
            if self.model_name == 'FSGM_Attention':
                x_out = self.value_embedding(x_base)
            elif self.model_name == 'FSGM':
                x_out = x_att
            else:
                raise
        elif self.key_value_emb is None:
            if self.model_name == 'FSGM':
                x_out = x_att
            elif self.model_name == 'FSGM_Attention':
                x_out = x_base
            else:
                raise
        else:
            raise

        # Combine CLS token for outputs (SM)
        if self.CLS_on and self.CLS_mix == 'SM':
            x_out = CT_adjust(x_out.unsqueeze(0)).squeeze(0)

        # Apply Attention maps
        if self.Softmax_axis == 'T':
            x_out = x_out * temporal_attention
        elif self.Softmax_axis == 'D':
            x_out = x_out * spatial_attention
        elif self.Softmax_axis == 'TD':
            T, D = x_out.shape
            adjust_frame = T / D
            adjust_dimension = D / T
            if self.Balance == 'T':
                x_out = x_out * temporal_attention * adjust_frame + x_out * spatial_attention
            elif self.Balance == 'D':
                x_out = x_out * temporal_attention + x_out * spatial_attention * adjust_dimension
            elif self.Balance == 'BD':
                if T > D:
                    x_out = x_out * temporal_attention + x_out * spatial_attention * adjust_dimension
                elif T < D:
                    x_out = x_out * temporal_attention * adjust_frame + x_out * spatial_attention
                else:
                    x_out = x_out * temporal_attention + x_out * spatial_attention
            elif self.Balance == 'BU':
                if T > D:
                    x_out = x_out * temporal_attention * adjust_frame + x_out * spatial_attention
                elif T < D:
                    x_out = x_out * temporal_attention + x_out * spatial_attention * adjust_dimension
                else:
                    x_out = x_out * temporal_attention + x_out * spatial_attention
            elif self.Balance is None:
                x_out = x_out * temporal_attention + x_out * spatial_attention
            else:
                raise
        elif self.Softmax_axis is None:
            x_out = x_out * x_att
        else:
            raise

        # Skip Connection (CF)
        if self.Skip_connection is not None:
            if self.Skip_connection == 'CF':
                if x_out.shape != x_att.shape and self.CLS_on:
                    x_att2 = CT_adjust(x_att.unsqueeze(0)).squeeze(0)
                else:
                    x_att2 = x_att
                x_out = x_out + x_att2
                if self.Layernorm:
                    x_out = self.layernorm2(x_out.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            elif self.Skip_connection in ['KC', 'IF']:
                pass
            else:
                raise

        # Skip Connection (IF)
        if self.Skip_connection is not None:
            if self.Skip_connection == 'IF':
                x_out = x_out + x_base
                if self.Layernorm:
                    x_out = self.layernorm3(x_out.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            elif self.Skip_connection in ['KC', 'CF']:
                pass
            else:
                raise

        # Combine CLS token (Final)
        if self.CLS_on:
            if self.CLS_mix == 'Final':
                x_out = CT_adjust(x_out.unsqueeze(0)).squeeze(0)
            elif self.CLS_mix in ['CNN', 'SM']:
                pass
            else:
                raise

        # Classifier
        if self.Classifier_on:
            x_out = self.linear1(x_out)
            x_out = self.linear2(x_out)
            x_out = x_out.squeeze()
        else:
            x_out = self.gap(x_out)
            x_out = x_out.squeeze()

        return x_out
