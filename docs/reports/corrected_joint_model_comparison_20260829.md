# Corrected traditional architecture comparison

**Status:** queued behind the protected GPU jobs; no final metric is claimed.

`experiments/acs_joint_model_comparison.py` will rerun InceptionTime,
ResNet1D-Wang, LSTM, ViT and Transformer1D with the same categorical ACS plus
OMI binary targets, patient-level folds, preprocessing, optimizer budget and
validation threshold protocol.  Results, per-epoch loss/LR curves and model
summaries will be written below `results/corrected_joint_model_comparison/`.
