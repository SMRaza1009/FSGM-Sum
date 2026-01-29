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
    """Factorized spatiotemporal attention logit generator for (T x D)."""

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

        # Temporal branch
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

        # Channel branch
        red = max(4, self.dim // max(1, int(channel_reduction)))
        self.channel_mlp = nn.Sequential(
            nn.Linear(self.dim * 2, red),
            nn.ReLU(inplace=True),
            nn.Linear(red, self.dim),
        )

        # Optional spectrum bias (very lightweight)
        if self.use_fft_bias:
            self.fft_scale = nn.Sequential(
                nn.Linear(1, 8),
                nn.ReLU(inplace=True),
                nn.Linear(8, 1),
            )

        # Init
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
        """seq: [T, D] -> logits: [T, D]"""
        if seq.dim() != 2:
            raise ValueError(f"_FSGMAttention expects [T,D], got {tuple(seq.shape)}")
        T, D = seq.shape
        if D != self.dim:
            raise ValueError(f"Feature dim mismatch: expected {self.dim}, got {D}")

        # Temporal logits
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

        # Channel logits
        mean_pool = seq.mean(dim=0)          # [D]
        max_pool = seq.max(dim=0).values     # [D]
        feat_desc = torch.cat([mean_pool, max_pool], dim=0)  # [2D]
        feature_logits = self.channel_mlp(feat_desc)         # [D]

        return temporal_logits.unsqueeze(1) + feature_logits.unsqueeze(0)  # [T, D]


class CSTA_FSGM(nn.Module):
    """Plug-and-play model (replaces CSTA_*). Adds calibrated sigmoid (learnable temperature)."""

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
        # temperature calibration
        temp_init: float = 1.0,
        temp_min: float = 0.2,
        temp_max: float = 5.0,
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

        # --- calibrated sigmoid temperature ---
        # Unconstrained scalar -> positive temp via softplus; clamp to avoid extremes.
        self._temp_unconstrained = nn.Parameter(torch.tensor(float(temp_init)))
        self._temp_min = float(temp_min)
        self._temp_max = float(temp_max)

        # Positional encoding
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

        # Classifier
        if self.Classifier_on:
            self.linear1 = nn.Sequential(
                nn.Linear(in_features=self.dim, out_features=self.dim),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.LayerNorm(normalized_shape=self.dim, eps=1e-6),
            )
            # IMPORTANT: logits only. Sigmoid applied with temperature in forward.
            self.linear2 = nn.Linear(in_features=self.dim, out_features=1)

            nn.init.xavier_uniform_(self.linear1[0].weight, gain=np.sqrt(2.0))
            nn.init.constant_(self.linear1[0].bias, 0.1)
            nn.init.xavier_uniform_(self.linear2.weight, gain=np.sqrt(2.0))
            nn.init.constant_(self.linear2.bias, 0.1)
        else:
            self.gap = nn.AdaptiveAvgPool1d(1)

        # CLS token
        if self.CLS_on:
            self.CLS = nn.Parameter(torch.zeros(1, 3, 1, 1024))

        # Key/Value embeddings
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

        # Layer norms (same as CSTA codebase)
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

        self.fsgm = _FSGMAttention(
            dim=self.dim,
            hidden=fsgm_hidden,
            channel_reduction=fsgm_channel_reduction,
            use_fft_bias=fsgm_use_fft_bias,
        )

    def _temperature(self) -> torch.Tensor:
        t = F.softplus(self._temp_unconstrained) + self._temp_min
        return torch.clamp(t, min=self._temp_min, max=self._temp_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B=1, C=3, T, 1024] -> scores: [T]"""
        n_frame = x.shape[2]

        if self.Positional_encoding_way == 'Transformer':
            x = self.Positional_encoding_embedding(x)

        if self.CLS_on:
            x = torch.cat((self.CLS.to(x.dtype).to(x.device), x), dim=2)
            CT_adjust = nn.AdaptiveAvgPool2d((n_frame, self.dim))

        # Transformer-style PE
        if self.Positional_encoding_way == 'Transformer':
            if self.Positional_encoding is not None:
                x = self.Positional_encoding_op(x)
            if self.Dropout_on:
                x = self.dropout(x)

        # Key embedding
        if self.key_value_emb is not None and self.key_value_emb.lower() in ['k', 'kv']:
            key = self.key_embedding(x)
        elif self.key_value_emb is None:
            key = x
        else:
            raise

        # Collapse B and C (consistent with CSTA repo convention)
        key_seq = key.squeeze(0)[0]  # [T(+1), D]
        x_att = self.fsgm(key_seq)   # [T(+1), D]

        # KC skip
        if self.Skip_connection == 'KC':
            x_att = x_att + key_seq
            if self.Layernorm:
                x_att = self.layernorm1(x_att.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        # Optional CLS mixing by pooling
        if self.CLS_on and self.CLS_mix == 'CNN':
            x_att = CT_adjust(x_att.unsqueeze(0)).squeeze(0)
            x = CT_adjust(x.squeeze(0)).unsqueeze(0)

        # Scaling
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

        # PGL-SUM PE
        if self.Positional_encoding_way == 'PGL_SUM':
            if self.Positional_encoding is not None:
                x_att = self.Positional_encoding_op(x_att)

        x_base = x.squeeze(0)[0]  # [T(+1), D]

        # softmax axes
        if self.Softmax_axis == 'T':
            temporal_attention = F.softmax(x_att, dim=0)
        elif self.Softmax_axis == 'D':
            spatial_attention = F.softmax(x_att, dim=1)
        elif self.Softmax_axis == 'TD':
            temporal_attention = F.softmax(x_att, dim=0)
            spatial_attention = F.softmax(x_att, dim=1)
        elif self.Softmax_axis is None:
            temporal_attention = None
            spatial_attention = None
        else:
            raise

        if self.CLS_on and self.CLS_mix == 'SM' and self.Softmax_axis is not None:
            if self.Softmax_axis == 'T':
                temporal_attention = CT_adjust(temporal_attention.unsqueeze(0)).squeeze(0)
            elif self.Softmax_axis == 'D':
                spatial_attention = CT_adjust(spatial_attention.unsqueeze(0)).squeeze(0)
            elif self.Softmax_axis == 'TD':
                temporal_attention = CT_adjust(temporal_attention.unsqueeze(0)).squeeze(0)
                spatial_attention = CT_adjust(spatial_attention.unsqueeze(0)).squeeze(0)

        if self.Dropout_on and self.Positional_encoding_way == 'PGL_SUM' and self.Softmax_axis is not None:
            if self.Softmax_axis == 'T':
                temporal_attention = self.dropout(temporal_attention)
            elif self.Softmax_axis == 'D':
                spatial_attention = self.dropout(spatial_attention)
            elif self.Softmax_axis == 'TD':
                temporal_attention = self.dropout(temporal_attention)
                spatial_attention = self.dropout(spatial_attention)

        # Value embedding
        if self.key_value_emb is not None and self.key_value_emb.lower() in ['v', 'kv']:
            if self.model_name == 'FSGM_Attention':
                x_out = self.value_embedding(x_base)
            else:
                x_out = x_att
        else:
            x_out = x_base if self.model_name == 'FSGM_Attention' else x_att

        if self.CLS_on and self.CLS_mix == 'SM':
            x_out = CT_adjust(x_out.unsqueeze(0)).squeeze(0)

        # apply attention
        if self.Softmax_axis == 'T':
            x_out = x_out * temporal_attention
        elif self.Softmax_axis == 'D':
            x_out = x_out * spatial_attention
        elif self.Softmax_axis == 'TD':
            Tt, Dd = x_out.shape
            adjust_frame = Tt / Dd
            adjust_dimension = Dd / Tt
            if self.Balance == 'T':
                x_out = x_out * temporal_attention * adjust_frame + x_out * spatial_attention
            elif self.Balance == 'D':
                x_out = x_out * temporal_attention + x_out * spatial_attention * adjust_dimension
            else:
                x_out = x_out * temporal_attention + x_out * spatial_attention
        elif self.Softmax_axis is None:
            x_out = x_out * x_att

        # Final CLS adjust
        if self.CLS_on and self.CLS_mix == 'Final':
            x_out = CT_adjust(x_out.unsqueeze(0)).squeeze(0)

        # classifier
        if self.Classifier_on:
            x_out = self.linear1(x_out)
            logits = self.linear2(x_out).squeeze(-1)  # [T]
            x_out = torch.sigmoid(logits / self._temperature())
        else:
            x_out = self.gap(x_out).squeeze()

        # Ensure strict 1D [T]
        return x_out.view(-1)

