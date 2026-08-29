# Corrected joint architecture comparison artifacts

These are aggregate, version-safe artifacts for the corrected task:

`ACS categorical (OTHER/STEMI/NSTEMI/UA) + independent OMI binary`.

Each model directory contains the fold histories, aggregate `summary.json`,
`fold_metrics.json`, `training_curves.csv`, training/validation-metric/LR
figure, ROC/PR figure, confusion-matrix figure, external-platform metric JSON
and prediction provenance/hash JSON. The pooled OOF summaries use the
model-specific OMI Youden threshold; ACS uses argmax.

Raw ECG, patient-level tables, checkpoints, record-level OOF/external
predictions and raw server logs are intentionally excluded from GitHub. Their
controlled-server paths are recorded in `docs/logs/EXPERIMENT_LOG_INDEX.md`.
