"""Thin adapters reusing modern backbones from Time-Series-Library.

We do NOT rewrite any backbone.  Each adapter builds the official TSLib
``Model(configs)`` and exposes a uniform ``forward(x, mask=None)`` where ``x``
is [B, 1000, 12] (time-major).  For backbones that already ship a
classification head (TimesNet / Informer / Nonstationary_Transformer /
PatchTST) we reuse it as-is.  For backbones that only ship a forecasting /
anomaly head (MultiPatchFormer / KAN-AD) we add a small global-pool + Linear
classification head on top of their encoder features.

TSLib models are imported lazily (inside the adapter constructor) so that a
missing optional dependency of one model does not break the others at import
time.
"""
import os
import sys

import torch
import torch.nn as nn

TSLIB_ROOT = os.environ.get("TSLIB_ROOT", "/data/code/ouyanghao/oyh/Time-Series-Library")
if TSLIB_ROOT not in sys.path:
    sys.path.append(TSLIB_ROOT)

_MODEL_MODULES = {
    "TimesNet": "models.TimesNet",
    "Informer": "models.Informer",
    "Nonstationary_Transformer": "models.Nonstationary_Transformer",
    "PatchTST": "models.PatchTST",
}


def _tslib_model(model_name):
    import importlib
    return getattr(importlib.import_module(_MODEL_MODULES[model_name]), "Model")


class _ModernAdapter(nn.Module):
    def __init__(self, configs, model_name):
        super().__init__()
        self.inner = _tslib_model(model_name)(configs)

    def forward(self, x, mask=None):
        # Modern backbones use x_mark_enc as a [B, T] padding mask in their
        # classification head (output *= x_mark_enc.unsqueeze(-1)).  Full-length
        # ECG is all-valid, so default to an all-ones mask.
        B, T, _ = x.shape
        if mask is None:
            mask = torch.ones(B, T, device=x.device)
        return self.inner(x, mask, None, None)


class TimesNetAdapter(_ModernAdapter):
    def __init__(self, configs):
        super().__init__(configs, "TimesNet")


class InformerAdapter(_ModernAdapter):
    def __init__(self, configs):
        super().__init__(configs, "Informer")


class NonstationaryTransformerAdapter(_ModernAdapter):
    def __init__(self, configs):
        super().__init__(configs, "Nonstationary_Transformer")


class PatchTSTAdapter(_ModernAdapter):
    def __init__(self, configs):
        super().__init__(configs, "PatchTST")


class MultiPatchFormerAdapter(nn.Module):
    """MultiPatchFormer is forecasting-only in TSLib; add a classification head
    on its channel-wise encoder features ``encoding_1_ch`` ([B, C, d_model]).
    The encoder path is copied verbatim from ``MultiPatchFormer.forecast``."""

    def __init__(self, configs):
        super().__init__()
        from models.MultiPatchFormer import Model as _MPF
        # forecast heads are sized by pred_len; a dummy non-zero value avoids
        # Linear(_, 0) in __init__ (those heads are unused for classification).
        if getattr(configs, "pred_len", 0) == 0:
            configs.pred_len = 96
        self.inner = _MPF(configs)
        # Original strides (7, 6) for patch scales 3/4 only align the multi-scale
        # patch counts for short sequences (<=~97); at seq_len=1000 they give
        # 141/163 patches vs 125.  Re-strided to 8 so all 4 scales yield 125
        # patches, matching the `pe` buffer and `embedding_channel`.
        d4 = configs.d_model // 4
        m = self.inner
        m.padding_patch_layer3 = nn.ReplicationPad1d((0, 16))
        m.embedding_patch_3 = nn.Conv1d(1, d4, kernel_size=24, stride=8)
        m.padding_patch_layer4 = nn.ReplicationPad1d((0, 24))
        m.embedding_patch_4 = nn.Conv1d(1, d4, kernel_size=32, stride=8)
        self.head = nn.Linear(configs.d_model, configs.num_class)

    def forward(self, x, mask=None):
        B, L, C = x.shape
        means = x.mean(1, keepdim=True).detach()
        x_enc = x - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        x_i = x_enc.permute(0, 2, 1)  # [B, C, L]

        def _embed(xi, conv):
            # [B, C, Lp] -> [B*C, 1, Lp] -> conv -> [B*C, P, d/4]
            return conv(xi.reshape(B * C, -1).unsqueeze(-1).permute(0, 2, 1)).permute(0, 2, 1)

        m = self.inner
        encoding_patch = torch.cat(
            (
                _embed(x_i, m.embedding_patch_1),
                _embed(m.padding_patch_layer2(x_i), m.embedding_patch_2),
                _embed(m.padding_patch_layer3(x_i), m.embedding_patch_3),
                _embed(m.padding_patch_layer4(x_i), m.embedding_patch_4),
            ),
            dim=-1,
        ) + m.pe  # [B*C, P, d_model]

        for i in range(m.N):
            encoding_patch = m.encoder_list[i](encoding_patch)[0]

        x_patch_c = encoding_patch.reshape(B, C, -1)          # [B, C, P*d_model]
        x_ch = m.embedding_channel(x_patch_c.permute(0, 2, 1)).transpose(1, 2)  # [B, C, d_model]
        encoding_1_ch = m.encoder_list_ch[0](x_ch)[0]         # [B, C, d_model]
        feat = encoding_1_ch.mean(dim=1)                      # [B, d_model]
        return self.head(feat)


class KANADAdapter(nn.Module):
    """KAN-AD is anomaly-detection-only in TSLib; reuse its KANADModel encoder
    as a per-channel feature extractor and add a global-pool + Linear head."""

    def __init__(self, configs):
        super().__init__()
        from models.KANAD import KANADModel
        self.enc = KANADModel(window=configs.seq_len, order=configs.d_model)
        self.head = nn.Linear(configs.enc_in, configs.num_class)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        xin = x.reshape(B * D, L)      # [B*D, L] per-channel
        out = self.enc(xin)            # [B*D, L]
        out = out.reshape(B, L, D)     # [B, L, D]
        feat = out.mean(dim=1)         # [B, D] pool over time
        return self.head(feat)
