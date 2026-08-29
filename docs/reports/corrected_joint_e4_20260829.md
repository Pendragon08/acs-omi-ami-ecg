# Corrected E4 joint ACS/OMI experiment

**Status:** running on physical GPU1; no final metric is claimed yet.

The run uses `experiments/acs_joint_e4.py` with the historical patient-level
five-fold split, fold-fitted robust scaling, training-only augmentation, E4
HMoE/WavelNet/KAN backbone, categorical ACS `OTHER/STEMI/NSTEMI/UA` loss plus
independent OMI BCE, and validation-only OMI threshold selection.  The raw
screen log is `logs/corrected/acs_joint_e4_gpu1.log`; outputs will be under
`results/corrected_joint_acs_omi_e4/` and include JSON/CSV provenance, curves,
ROC/PR, confusion matrices and a corrected external submission.

After completion this report will be populated from `summary.json` without
manual transcription.
