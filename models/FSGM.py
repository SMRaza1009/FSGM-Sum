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
    """Factorized spatio-temporal gate: [T, D] key sequence -> [T, D] gate logits.

    Temporal branch: projection -> parallel dilated depthwise-separable Conv1d
    blocks -> 1x1 head, plus an FFT spectral bias (Sec. 3.3.1-3.3.2).
    Channel branch: temporal mean/max pooling -> bottleneck MLP (Sec. 3.3.3).
    The two logit vectors are fused by an outer sum (Sec. 3.3.4).
    """

    def __init__(
        self,
        dim: int,
        hidden: int = 64,
        channel_reduction: int = 16,
        dilations=(1, 2, 4),
        fft_bias: str = 'timestep',
        branches: str = 'both',
        pooling: str = 'meanmax',
        fusion: str = 'add',
        eps: float = 1e-6,
    ):
        super().__init__()
        if fft_bias not in ('timestep', 'scalar', 'none'):
            raise ValueError(f"Unknown fft_bias={fft_bias}")
        if branches not in ('both', 'temporal', 'channel'):
            raise ValueError(f"Unknown branches={branches}")
        if pooling not in ('meanmax', 'mean', 'max'):
            raise ValueError(f"Unknown pooling={pooling}")
        if fusion not in ('add', 'mul'):
            raise ValueError(f"Unknown fusion={fusion}")

        self.dim = int(dim)
        self.hidden = int(hidden)
        self.fft_bias = fft_bias
        self.branches = branches
        self.pooling = pooling
        self.fusion = fusion
        self.eps = float(eps)
        self.use_temporal = branches in ('both', 'temporal')
        self.use_channel = branches in ('both', 'channel')

        # Temporal branch
        if self.use_temporal:
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

            self.temporal_blocks = nn.ModuleList([_dw_pw_block(int(d)) for d in dilations])
            self.temporal_head = nn.Conv1d(self.hidden, 1, kernel_size=1, bias=True)

            # Per-timestep spectral bias: g_fft = Conv1d(1->8,k=3) -> GELU -> Conv1d(8->1,k=3)
            # over frequency bins, mapped back to time with an inverse real FFT.
            if self.fft_bias == 'timestep':
                self.fft_filter = nn.Sequential(
                    nn.Conv1d(1, 8, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv1d(8, 1, kernel_size=3, padding=1),
                )
            # Ablation only: one scalar added to every timestep (cancelled by the temporal softmax).
            elif self.fft_bias == 'scalar':
                self.fft_scale = nn.Sequential(
                    nn.Linear(1, 8),
                    nn.ReLU(inplace=True),
                    nn.Linear(8, 1),
                )

        # Channel branch
        if self.use_channel:
            red = max(4, self.dim // max(1, int(channel_reduction)))
            in_dim = self.dim * 2 if self.pooling == 'meanmax' else self.dim
            self.channel_mlp = nn.Sequential(
                nn.Linear(in_dim, red),
                nn.ReLU(inplace=True),
                nn.Linear(red, self.dim),
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

    def _spectral_bias(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: [T, D] -> per-timestep bias b: [T] (Eqs. for s, m, m~, b in Sec. 3.3.2)."""
        T = seq.shape[0]
        s = seq.mean(dim=1)                                   # [T]
        mag = torch.abs(torch.fft.rfft(s, dim=0))             # [F], F = T//2 + 1
        mag_f = self.fft_filter(mag.view(1, 1, -1)).view(-1)  # [F]
        b = torch.fft.irfft(torch.complex(mag_f, torch.zeros_like(mag_f)), n=T, dim=0)  # [T]
        return b / (b.std() + self.eps)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: [T, D] -> logits: [T, D]"""
        if seq.dim() != 2:
            raise ValueError(f"_FSGMAttention expects [T,D], got {tuple(seq.shape)}")
        T, D = seq.shape
        if D != self.dim:
            raise ValueError(f"Feature dim mismatch: expected {self.dim}, got {D}")

        # Temporal logits
        if self.use_temporal:
            h = self.temporal_proj(seq)               # [T, hidden]
            h = h.transpose(0, 1).unsqueeze(0)        # [1, hidden, T]
            mix = 0.0
            for blk in self.temporal_blocks:
                mix = mix + blk(h)
            mix = mix / float(len(self.temporal_blocks))
            temporal_logits = self.temporal_head(mix).squeeze(0).squeeze(0)  # [T]

            if self.fft_bias == 'timestep':
                temporal_logits = temporal_logits + self._spectral_bias(seq)
            elif self.fft_bias == 'scalar':
                s = seq.mean(dim=1)
                mag = torch.abs(torch.fft.rfft(s, dim=0))
                stat = mag.mean().clamp(min=self.eps)
                temporal_logits = temporal_logits + self.fft_scale(stat.view(1, 1)).view(())

        # Channel logits
        if self.use_channel:
            if self.pooling == 'meanmax':
                feat_desc = torch.cat([seq.mean(dim=0), seq.max(dim=0).values], dim=0)  # [2D]
            elif self.pooling == 'mean':
                feat_desc = seq.mean(dim=0)           # [D]
            else:
                feat_desc = seq.max(dim=0).values     # [D]
            feature_logits = self.channel_mlp(feat_desc)  # [D]

        if self.branches == 'temporal':
            return temporal_logits.unsqueeze(1).expand(T, D)
        if self.branches == 'channel':
            return feature_logits.unsqueeze(0).expand(T, D)
        if self.fusion == 'mul':
            return temporal_logits.unsqueeze(1) * feature_logits.unsqueeze(0)  # [T, D]
        return temporal_logits.unsqueeze(1) + feature_logits.unsqueeze(0)      # [T, D]


class FSGMSum(nn.Module):
    """FSGM-Sum frame-importance model with a temperature-calibrated sigmoid head.

    Input: [1, 3, T, 1024] replicated frame features. Output: [T] scores in [0, 1].
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
        fsgm_dilations=(1, 2, 4),
        fsgm_fft_bias: str = 'timestep',
        fsgm_branches: str = 'both',
        fsgm_pooling: str = 'meanmax',
        fsgm_fusion: str = 'add',
        # temperature calibration
        temp_learnable: bool = True,
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
        # With temp_learnable=False the temperature is fixed at gamma=1 (ablation).
        self.temp_learnable = bool(temp_learnable)
        if self.temp_learnable:
            self._temp_unconstrained = nn.Parameter(torch.tensor(float(temp_init)))
        self._temp_min = float(temp_min)
        self._temp_max = float(temp_max)

        # fsgm_branches='none' bypasses the gate entirely (base regressor ablation)
        self.use_gate = fsgm_branches != 'none'

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

        # Normalization after the key residual
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

        if self.use_gate:
            self.fsgm = _FSGMAttention(
                dim=self.dim,
                hidden=fsgm_hidden,
                channel_reduction=fsgm_channel_reduction,
                dilations=fsgm_dilations,
                fft_bias=fsgm_fft_bias,
                branches=fsgm_branches,
                pooling=fsgm_pooling,
                fusion=fsgm_fusion,
            )

    def _temperature(self) -> torch.Tensor:
        if not self.temp_learnable:
            return torch.tensor(1.0)
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

        # The three input channels are identical replicas; use the first one
        key_seq = key.squeeze(0)[0]  # [T(+1), D]

        if not self.use_gate:
            return self._head(x.squeeze(0)[0], n_frame)

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

        return self._head(x_out, n_frame)

    def _head(self, x_out: torch.Tensor, n_frame: int) -> torch.Tensor:
        """Final CLS pooling + temperature-calibrated classifier: [T(+1), D] -> [T]."""
        if self.CLS_on and self.CLS_mix == 'Final':
            x_out = nn.AdaptiveAvgPool2d((n_frame, self.dim))(x_out.unsqueeze(0)).squeeze(0)

        # classifier
        if self.Classifier_on:
            x_out = self.linear1(x_out)
            logits = self.linear2(x_out).squeeze(-1)  # [T]
            x_out = torch.sigmoid(logits / self._temperature())
        else:
            x_out = self.gap(x_out).squeeze()

        # Ensure strict 1D [T]
        return x_out.view(-1)

