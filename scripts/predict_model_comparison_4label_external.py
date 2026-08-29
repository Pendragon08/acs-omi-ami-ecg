"""Compatibility entry point for external predictions.

The former script applied four independent sigmoid thresholds to
STEMI/NSTEMI/UA/OMI and is invalid because the three ACS subtypes are
mutually exclusive.  Keep this filename for reproducibility of old commands,
but route all new invocations to the corrected categorical-ACS + OMI
predictor.  The corrected predictor uses ACS softmax/argmax and a validation-
selected OMI threshold without reading external labels.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.predict_corrected_models_external import main  # noqa: E402


if __name__ == "__main__":
    main()
