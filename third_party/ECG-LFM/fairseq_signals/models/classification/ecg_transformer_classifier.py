import contextlib

import torch
import torch.nn as nn

from dataclasses import dataclass, field
from omegaconf import MISSING

from fairseq_signals.models import register_model
from fairseq_signals.models.ecg_transformer import (
    ECGTransformerFinetuningModel,
    ECGTransformerFinetuningConfig
)

from fairseq_signals.utils import utils
from fairseq_signals.modules.parameter_efficient import (
    enable_head,
    inject_linear_adapters,
)

@dataclass
class ECGTransformerClassificationConfig(ECGTransformerFinetuningConfig):
    num_labels: int = field(
        default=MISSING, metadata={"help": "number of labels to be classified"}
    )
    # Parameter-efficient adaptation is opt-in.  The default ``none`` path is
    # intentionally identical to the historical full-finetuning baseline.
    adapter_method: str = field(
        default="none",
        metadata={"help": "none, lora, or dora; adapters are injected after checkpoint load"},
    )
    adapter_rank: int = field(default=8, metadata={"help": "LoRA/DoRA rank"})
    adapter_alpha: float = field(default=16.0, metadata={"help": "LoRA/DoRA scaling alpha"})
    adapter_dropout: float = field(default=0.05, metadata={"help": "adapter input dropout"})

@register_model("ecg_transformer_classifier", dataclass=ECGTransformerClassificationConfig)
class ECGTransformerClassificationModel(ECGTransformerFinetuningModel):
    def __init__(self, cfg, encoder):
        super().__init__(cfg, encoder)

        self.proj = nn.Linear(cfg.encoder_embed_dim, cfg.num_labels)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)

        self.adapter_summary = None
        method = str(getattr(cfg, "adapter_method", "none")).lower()
        if method != "none":
            self.adapter_summary = inject_linear_adapters(
                self,
                method=method,
                rank=int(getattr(cfg, "adapter_rank", 8)),
                alpha=float(getattr(cfg, "adapter_alpha", 16.0)),
                dropout=float(getattr(cfg, "adapter_dropout", 0.05)),
            )
            # Keep the four-label head fully trainable while every pretrained
            # encoder/convolution parameter remains frozen.
            enable_head(self)

    def get_logits(self, net_output, normalize=False, **kwargs):
        logits = net_output["out"]
        
        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)
        
        return logits

    # def get_targets(self, sample, net_output, **kwargs):
    #     if isinstance(sample["age"], torch.Tensor):
    #         return sample["age"].float()
    #     else:
    #         return sample["age"]
    def get_targets(self, sample, net_output, **kwargs):
        if isinstance(sample["label"], torch.Tensor):
            return sample["label"].float()
        else:
            return sample["label"]

    def forward(self, **kwargs): #**kwargs
        res = super().forward(**kwargs)
        x = res["x"]
        padding_mask = res["padding_mask"]
        
        x = self.final_dropout(x)
        if padding_mask is not None and padding_mask.any():
            x[padding_mask] = 0

        x = torch.div(x.sum(dim=1), (x != 0).sum(dim=1))

        x = self.proj(x)

        return {
            "encoder_out": res["x"].detach(),
            "padding_mask": padding_mask,
            "out": x,
        }

    # def forward(self, a): #**kwargs
    #     kwargs = {}
    #     kwargs['source'] = a
    #     #kwargs['padding_mask'] = torch.zeros(1, 12, 2500, dtype=torch.bool)

    #     res = super().forward(**kwargs)
    #     x = res["x"]
    #     padding_mask = res["padding_mask"]
        
    #     x = self.final_dropout(x)
    #     if padding_mask is not None and padding_mask.any():
    #         x[padding_mask] = 0

    #     x = torch.div(x.sum(dim=1), (x != 0).sum(dim=1))

    #     x = self.proj(x)

    #     return x
    #     # return {
    #     #     "encoder_out": res["x"].detach(),
    #     #     "padding_mask": padding_mask,
    #     #     "out": x,
    #     # }
