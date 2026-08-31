"""Fast adapter regression tests that do not touch ECG data or checkpoints."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn

from fairseq_signals.modules.parameter_efficient import (
    DoRALinear,
    LoRALinear,
    enable_head,
    inject_linear_adapters,
    parameter_counts,
)


class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.encoder = nn.Module()
        self.encoder.encoder.layers = nn.ModuleList(
            [nn.Module() for _ in range(2)]
        )
        for layer in self.encoder.encoder.layers:
            layer.q = nn.Linear(16, 16)
            layer.ffn = nn.Linear(16, 32)
        self.proj = nn.Linear(16, 5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/ecg_lfm/joint_adapter_contract.json"))
    args = parser.parse_args()
    torch.manual_seed(2026)
    x = torch.randn(5, 16)
    base = nn.Linear(16, 8)
    base.eval()
    expected = base(x)
    checks = {}
    for name, cls in (("lora", LoRALinear), ("dora", DoRALinear)):
        wrapped = cls(copy.deepcopy(base), rank=4, alpha=8.0, dropout=0.0)
        wrapped.eval()
        actual = wrapped(x)
        checks[f"{name}_zero_update_max_abs_error"] = float((expected - actual).abs().max())

    target_model = DummyEncoder()
    lora_summary = inject_linear_adapters(target_model, "lora", rank=4, alpha=8.0, dropout=0.0, prefix="encoder.encoder.layers.")
    enable_head(target_model)
    checks["target_module_count"] = len(lora_summary.target_modules)
    checks["target_parameter_counts"] = {
        "trainable": parameter_counts(target_model)[0],
        "total": parameter_counts(target_model)[1],
    }
    checks["all_targets_wrapped"] = all(
        isinstance(module, LoRALinear)
        for name, module in target_model.named_modules()
        if name.startswith("encoder.encoder.layers.") and name.rsplit(".", 1)[-1] in {"q", "ffn"}
    )
    checks["passed"] = bool(
        checks["lora_zero_update_max_abs_error"] < 1e-6
        and checks["dora_zero_update_max_abs_error"] < 1e-6
        and checks["target_module_count"] == 4
        and checks["all_targets_wrapped"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False))
    if not checks["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
