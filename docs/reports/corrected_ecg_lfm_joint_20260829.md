# Corrected ECG-LFM joint ACS/OMI experiment

**Status:** running on physical GPU1; no final metric is claimed yet.

Official ECG-LFM fine-tuning is rerun with five logits and the registered
`acs_joint` criterion: summed categorical ACS cross-entropy plus numerically
stable OMI BCE. Baseline, LoRA and DoRA retain their historical optimizer,
learning-rate and adapter settings. An A6000 memory smoke test showed that
`batch_size=64` is safe; `update_freq=4` preserves the historical effective
batch size of 256. The full queue is running with this setting under
`results/corrected_ecg_lfm_joint_bs64/` and
`logs/corrected_ecg_lfm_joint_bs64/`.

Fold-specific training counts, curves, configuration/checkpoint hashes and
the record-level evaluator will be archived after all 15 runs. The evaluator
will report four-way ACS, conditional three-way ACS and independent OMI
binary metrics separately; no legacy independent-sigmoid result is reused.
