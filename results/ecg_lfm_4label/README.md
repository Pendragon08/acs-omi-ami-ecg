# ECG-LFM four-label Baseline results

This directory contains aggregate artifacts for the official ECG-LFM
Baseline (`finetune_paper`). The model emits independent STEMI, NSTEMI, UA and
OMI logits. The `summary/` files contain five-fold OOF metrics, frozen OOF
thresholds, external-platform aggregate metrics, and comparisons with the
historical E4 model. Fold-level `fold_metrics.csv` files are included for
auditability.

Per-record predictions, fairseq JSONL streams, checkpoints, split indices and
waveforms are intentionally excluded; see the report and repository manifest.
