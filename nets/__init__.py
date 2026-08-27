"""Model registry.

Uniform interface: every model is an ``nn.Module`` with
``forward(x, mask=None) -> [B, num_class]`` where ``x`` is [B, 1000, 12]
(time-major) and ``[:,0]`` is the OMI logit, ``[:,1]`` the AMI logit.

- 6 modern models are re-used from Time-Series-Library (via ``sys.path``):
  TimesNet / Informer / Nonstationary_Transformer / PatchTST /
  MultiPatchFormer / KAN-AD.  Only their output head is switched to
  ``num_class`` (or a small classification head is added) -- no backbone
  rewrite.
- 4 traditional baselines are ported from ACS-Baseline-main.
- MyNet6View is the original-author CNN (Experiment 1).
"""
from nets.tslib_adapters import (          # noqa: E402
    InformerAdapter,
    KANADAdapter,
    MultiPatchFormerAdapter,
    NonstationaryTransformerAdapter,
    PatchTSTAdapter,
    TimesNetAdapter,
)
from nets.traditional import (             # noqa: E402
    InceptionTime,
    LSTM,
    ResNet1D_Wang,
    ViT,
)
from nets.mynet6view import MyNet6ViewModel  # noqa: E402
from nets.cnn_variants import (            # noqa: E402
    SEResNet1D_Wang,
    Res2Net1D_Wang,
    XceptionTime,
    LITEMV,
    XceptionTimeFusion,
)

MODERN = ("TimesNet", "Informer", "Nonstationary_Transformer",
          "PatchTST", "MultiPatchFormer", "KANAD")
TRADITIONAL = ("LSTM", "ResNet1D_Wang", "InceptionTime", "ViT")
CNN_VARIANTS = ("SEResNet1D_Wang", "Res2Net1D_Wang", "XceptionTime", "LITEMV")
ALL_MODELS = MODERN + TRADITIONAL + CNN_VARIANTS


def build_model(name, num_class=2, device="cuda", **overrides):
    if name in MODERN:
        from acs_utils.configs import build_configs
        cfg = build_configs(name, num_class=num_class, device=device, **overrides)
        if name == "TimesNet":
            return TimesNetAdapter(cfg)
        if name == "Informer":
            return InformerAdapter(cfg)
        if name == "Nonstationary_Transformer":
            return NonstationaryTransformerAdapter(cfg)
        if name == "PatchTST":
            return PatchTSTAdapter(cfg)
        if name == "MultiPatchFormer":
            return MultiPatchFormerAdapter(cfg)
        if name == "KANAD":
            return KANADAdapter(cfg)

    if name == "LSTM":
        return LSTM(num_classes=num_class)
    if name == "ResNet1D_Wang":
        return ResNet1D_Wang(num_classes=num_class)
    if name == "InceptionTime":
        return InceptionTime(num_classes=num_class)
    if name == "ViT":
        return ViT(num_classes=num_class)
    if name == "MyNet6View":
        return MyNet6ViewModel(num_classes=num_class)

    if name == "SEResNet1D_Wang":
        return SEResNet1D_Wang(num_classes=num_class)
    if name == "Res2Net1D_Wang":
        return Res2Net1D_Wang(num_classes=num_class)
    if name == "XceptionTime":
        return XceptionTime(num_classes=num_class)
    if name == "LITEMV":
        return LITEMV(num_classes=num_class)
    if name == "XceptionTimeFusion":
        return XceptionTimeFusion(num_classes=num_class, **overrides)

    raise ValueError(f"unknown model: {name}")
