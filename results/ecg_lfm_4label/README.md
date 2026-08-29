# LEGACY INVALID — ECG-LFM independent-sigmoid four-label Baseline

This directory contains historical artifacts for the official ECG-LFM
Baseline (`finetune_paper`). The old model emitted independent STEMI, NSTEMI,
UA and OMI logits, which is invalid for mutually exclusive ACS subtypes.  Do
not report these values as current results.  The corrected categorical-ACS +
OMI rerun is under `results/corrected_ecg_lfm_joint` and documented in
`docs/reports/corrected_ecg_lfm_joint_20260829.md`.

Per-record predictions, fairseq JSONL streams, checkpoints, split indices and
waveforms are intentionally excluded; see the report and repository manifest.
