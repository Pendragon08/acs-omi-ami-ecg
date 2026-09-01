# RDR upstream provenance

- Repository: <https://github.com/GoodMorningPeter/RDR>
- Pinned commit: `bb55ea9f906f00a2d8938a97543fc232b5d26645`
- License: MIT (`LICENSE`)
- Paper: *Revive Re-weighting in Imbalanced Learning by Density Ratio Estimation*, NeurIPS 2024
- Canonical implementation: `main.py`, functions `rdr_weights` and `update_Phi_nu`

Official defaults observed at the pinned commit:

- training epochs: 200;
- warm-up: 160 epochs;
- prototype momentum `m`: 0.9;
- temperature `gamma`: 1 in the paper (the released code omits an explicit exponent because the default is one);
- density ratio: per-class minibatch Gram inverse, falling back to `pinverse` on `RuntimeError`;
- lower stabilization clamp: `1e-9`;
- normalization: global minibatch mean weight equals one;
- absent minibatch class: no sample weight and no prototype update; with experiment logging enabled, the upstream implementation has an undefined-local `r` edge case for an absent class.

The ECG-LFM port lives in `fairseq_signals/modules/rdr_official.py`. It preserves the equation and weighting semantics while computing in float32 and explicitly rejecting non-finite output.
