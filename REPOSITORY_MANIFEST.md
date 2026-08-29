# Repository manifest

This manifest defines what is synchronized to the GitHub research archive and
what remains on the controlled execution environment.

The canonical task is the corrected joint formulation: categorical
`OTHER/STEMI/NSTEMI/UA` ACS subtype plus independent binary OMI/NOMI. The
former four-independent-sigmoid artifacts remain only as legacy-invalid
provenance.

## Included

| Category | GitHub location | Contents |
|---|---|---|
| Historical and current source | `experiments/`, `models/`, `scripts/`, `scripts/ecg_lfm/`, `preprocessing/`, `data_provider/`, `losses/`, `acs_utils/`, `nets/`, `baselines/` | Training, inference, evaluation, split, loss, and reporting code |
| ECG-LFM upstream source | `third_party/ECG-LFM/` | Code-only snapshot used by the ECG-LFM experiments; no checkpoints or run output |
| Configurations | `configs/`, `configs/ecg_lfm/` | YAML/JSON experiment settings and environment specifications |
| Research plans | `docs/plans/` | ACS, HMoE/KAN, ECG-LFM, LoRA/DoRA, and ablation specifications |
| Reports | `docs/reports/` | Final and historical experiment reports, tables, and machine-readable aggregate provenance |
| Aggregate results | `results/` | Summary CSV/JSON/Markdown files and selected metric/curve figures only |
| Traceability | `docs/logs/EXPERIMENT_LOG_INDEX.md` | Curated index of raw server logs, run families, seeds, GPUs, and report/result links |

## Excluded by design

- Raw or preprocessed ECG waveforms (`.med`, `.npy`, `.npz`) and controlled
  `data/`/`med_data/` directories.
- Train/test CSV files, patient-level tables, split-index files, and any row-
  level OOF, validation, external-test, or submission predictions.
- Checkpoints and downloaded pretrained weights (`.pt`, `.pth`, `.bin`).
- Raw `screen`/training logs and fairseq JSONL event streams. The server keeps
  them; the repository keeps the curated index and aggregate curves instead.
- Temporary browser/SSH/archive files, caches, virtual environments, and
  unrelated projects.
- Credentials, passwords, access tokens, and private keys. Do not add them to
  configuration files or commit messages.

The enforced patterns are in [`.gitignore`](.gitignore). Before every push,
inspect the staged file list and verify that no controlled data, checkpoint,
credential, or row-level prediction has entered the index.

## Provenance policy

Every report should identify the task definition, data/split contract, seed,
configuration, model/loss variant, threshold-selection rule, metrics, and
source result family. Absolute server paths may be retained when needed for
internal reproducibility, but credentials and record-level identifiers must
never be included. The corrected rerun registry is documented in
`docs/reports/corrected_task_formulation_20260829.md`; the historical
ECG-LFM LoRA/DoRA parameter-efficiency analysis is explicitly legacy-invalid.
