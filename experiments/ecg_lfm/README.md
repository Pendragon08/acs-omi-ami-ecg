# ECG-LFM four-label experiment (LEGACY—invalid task formulation)

> **2026-08-29 correction:** the original files in this directory use four
> independent sigmoid outputs, including mutually-exclusive ACS subtypes.
> Use `run_joint_fold.py`, `evaluate_joint.py` and `configs/ecg_lfm/corrected_joint_*.yaml`
> for the canonical categorical ACS + independent OMI task.

This directory is an independent experiment branch. It does not modify the
historical E4 HMoE-KAN code or results.

Historical task (invalid): four independent binary outputs in the fixed order
`STEMI,NSTEMI,UA,OMI`. That formulation incorrectly treated mutually exclusive
ACS subtypes as independent labels and is retained only for provenance.

Canonical task: one categorical ACS head with
`OTHER/STEMI/NSTEMI/UA` and one independent OMI/NOMI binary head. The model
therefore emits five logits: four ACS logits trained with cross-entropy and one
OMI logit trained with BCE-with-logits. Explicit subtype `[0,0,0]` records remain
valid `OTHER` states, including the valid `OTHER + OMI` combination. The
external 1,995-record cohort is isolated from fitting,
checkpoint selection, threshold selection, and all internal model decisions.

Data gates:

1. Download the official Figshare raw WFDB archive and verify MD5.
2. Pair all `.hea/.dat` files and verify 19,955 records, 500 Hz, 12 leads,
   expected 5,000 samples, and canonical lead membership. The two official
   records whose data contain only 3,500 samples are audited and zero-padded
   with the source validation code's rule; they are not silently discarded.
3. Join raw records to `meta_train.npz`, `meta_test.npz`, official CSVs, and
   the historical E4 `folds_indices`. Any cohort discrepancy is fatal.
4. Apply the independent native pipeline: 1 Hz high-pass, 50/60 Hz notch,
   30 Hz low-pass, per-lead z-score, and two non-overlapping 5 s segments.
5. Average the two segment logits at record level; apply softmax to the ACS
   head and sigmoid only to the OMI head.

Generated audit artifacts include `artifacts/ecg_lfm/raw_500hz_manifest.csv`,
`raw_500hz_download_report.json`, `ecg_lfm_master_index.csv`, and the
machine-readable cohort/preprocessing audits under `manifests/ecg_lfm/`.
`artifacts/ecg_lfm/official_source_integrity.json` separates synchronized
CRLF/LF-only differences from semantic source changes and proves the only
semantic upstream worktree change is the documented strict-best checkpoint
patch.
`contract_tests.py` proves the official criterion equals unweighted PyTorch
BCE-with-logits, fixes label order, checks two-segment aggregation, and rejects
patient/external leakage before any model run.
The 64-record `run_overfit_gate.py` uses repeated train/valid diagnostic data
with label coverage and a deliberately fast head-only learning rate. It is a
software gate only; none of its metrics are mixed into the scientific folds.

Corrected training uses the official `ecg_transformer_classifier`, a five-logit
joint head, ACS cross-entropy plus stable OMI `BCEWithLogitsLoss`, Adam
`(0.9,0.98)`, LR `1e-6`,
effective batch 256, validation-loss checkpointing, and patience 10. The main
configuration approximates epoch-wise exponential decay through the official
registered step scheduler (`112` updates per epoch, factor `0.99`). A fixed-LR
official-repository ablation remains separate. The memory-tested physical
batches are 64 with accumulation 4 for L0 and 32 with accumulation 8 for
L1a/L1b, so optimizer update count and effective batch remain unchanged.

Long runs must be launched in `screen` with one explicitly bound GPU. See the
top-level report for exact commands, source commits, checksums, gates, and
results.

## Local-best LoRA/DoRA optimization

`configs/ecg_lfm/local_best_baseline.yaml`, `lora.yaml`, and `dora.yaml` are
an isolated adaptation ablation.  They reuse the exact historical
patient-level folds, raw 500-Hz pipeline, the joint CE+BCE criterion, optimizer,
effective batch, checkpoint rule, and OOF threshold code above.  LoRA and DoRA
are injected only into the 216 Conformer context linear projections (24
layers × 9 attention/FFN projections); the feature extractor, convolutional
blocks, layer norms, and all other backbone weights are frozen, while the
five-logit joint head remains trainable.  LoRA uses `r=8, alpha=16, dropout=0.05,
lr=1e-4`; DoRA uses the same adapter settings with `lr=5e-5`.

The wrappers are implemented in
`third_party/ECG-LFM/fairseq_signals/modules/parameter_efficient.py` and are
opt-in through `model.adapter_method`.  The default `none` path leaves the
historical full-parameter baseline untouched.  Launch the GPU-1 five-fold
queue with `scripts/run_ecg_lfm_local_best_adapters.sh`; it is resumable by
fold and writes OOF, thresholds, curves, external ensemble files, and
`reports/ecg_lfm_local_best_lora_dora_optimization_20260828.md`.
