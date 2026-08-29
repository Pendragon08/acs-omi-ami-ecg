# Corrected ECG-LFM joint ACS/OMI experiment

**Status:** queued; no final metric is claimed.

Official ECG-LFM fine-tuning is rerun with five logits and the registered
`acs_joint` criterion: summed categorical ACS cross-entropy plus numerically
stable OMI BCE.  Baseline, LoRA and DoRA retain their historical optimizer,
learning-rate and adapter settings.  Fold-specific training counts and all
configuration/checkpoint hashes are written to
`results/corrected_ecg_lfm_joint/`; the record-level evaluator will report
four-way ACS, conditional three-way ACS and OMI binary metrics separately.
