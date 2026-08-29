# Loss/feature ablation integrity audit

Overall: **WARN** (provisional deterministic audit; no fresh independent semantic reviewer was authorized).

- Parsed and recomputed 63 run JSON files and their validation prediction CSVs.
- Baseline checkpoint exactly reproduces the historical checkpoint: True.
- Ablation runner references the external test: False.
- Final loss seed sets: {'baseline': [42, 2024, 2025, 2026, 3407], 'weighted_bce': [42, 2024, 2025, 2026, 3407], 'focal': [42, 2024, 2025, 2026, 3407], 'cb_focal': [42, 2024, 2025, 2026, 3407]}.
- Deterministic errors: none.

WARN is retained even when deterministic checks pass because the historical scalers include internal-validation data and the required fresh-agent semantic audit could not be performed under the active no-delegation constraint.
