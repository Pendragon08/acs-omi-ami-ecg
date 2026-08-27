"""Build a TSLib-style configs Namespace for the 3 modern models.

The 3 modern backbones (TimesNet / Informer / Nonstationary_Transformer) are
imported directly from Time-Series-Library and only need their output head
switched to ``num_class`` (multi-label -> 2).  This module provides every
attribute their ``Model(configs)`` constructors read, with sensible ECG-sized
defaults.  Only the backbone differs across models -- all other settings
(optimizer, LR, epochs, batch, seed) are unified by the trainer.
"""
import argparse

# Backbone-specific defaults.  Keep d_model % n_heads == 0.
MODEL_DEFAULTS = {
    "TimesNet": dict(
        d_model=64, d_ff=256, e_layers=2, n_heads=8,
        top_k=5, num_kernels=6, factor=3,
    ),
    "Informer": dict(
        d_model=64, d_ff=256, e_layers=2, n_heads=8, factor=3,
    ),
    "Nonstationary_Transformer": dict(
        d_model=64, d_ff=256, e_layers=2, n_heads=8, factor=3,
        p_hidden_dims=[128, 128], p_hidden_layers=2,
    ),
    "PatchTST": dict(
        d_model=64, d_ff=256, e_layers=2, n_heads=8, factor=3,
    ),
    "MultiPatchFormer": dict(
        d_model=64, d_ff=256, e_layers=2, n_heads=8,
    ),
    "KANAD": dict(
        d_model=8,  # KAN periodic-basis order (memory-light); head is Linear(enc_in, num_class)
    ),
}


def build_configs(model: str, num_class: int = 2, device: str = "cuda",
                  seq_len: int = 1000, enc_in: int = 12, **overrides) -> argparse.Namespace:
    cfg = argparse.Namespace()

    # --- task / shape -----------------------------------------------------
    cfg.task_name = "classification"
    cfg.model = model
    cfg.seq_len = seq_len
    cfg.label_len = 0
    cfg.pred_len = 0
    cfg.enc_in = enc_in
    cfg.dec_in = enc_in
    cfg.c_out = num_class
    cfg.num_class = num_class

    # --- common TSLib model params ---------------------------------------
    cfg.d_model = 64
    cfg.n_heads = 8
    cfg.e_layers = 2
    cfg.d_layers = 1
    cfg.d_ff = 256
    cfg.moving_avg = 25
    cfg.factor = 1
    cfg.distil = True
    cfg.dropout = 0.1
    cfg.embed = "timeF"
    cfg.freq = "h"
    cfg.activation = "gelu"
    cfg.top_k = 5
    cfg.num_kernels = 6
    cfg.p_hidden_dims = [128, 128]
    cfg.p_hidden_layers = 2

    # --- device -----------------------------------------------------------
    cfg.device = device
    cfg.use_gpu = device != "cpu"
    cfg.use_multi_gpu = False
    cfg.device_ids = [0]

    # --- training (overridden by trainer, kept for completeness) ----------
    cfg.train_epochs = 100
    cfg.batch_size = 64
    cfg.learning_rate = 1e-3
    cfg.lradj = "type1"
    cfg.patience = 15
    cfg.num_workers = 4
    cfg.checkpoints = "./checkpoints"

    cfg.__dict__.update(MODEL_DEFAULTS.get(model, {}))
    cfg.__dict__.update(overrides)
    return cfg
