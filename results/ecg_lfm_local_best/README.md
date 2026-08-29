# ECG-LFM LoRA/DoRA local-best results

`lora/summary/` and `dora/summary/` contain aggregate five-fold OOF metrics,
thresholds, NSTEMI-vs-UA diagnostics and ensemble summaries from the same
historical patient-level split. LoRA also contains the external-platform
aggregate metric response. The accompanying report documents the effective
trainable-parameter counts and the limitation that parameter reduction is not
identical to total VRAM or wall-clock reduction.

Checkpoints, per-record predictions, split indices and raw training logs stay
on the controlled execution server.
