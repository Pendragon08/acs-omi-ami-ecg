"""Compatibility entry point for the corrected architecture comparison.

The historical implementation compared four independent sigmoid outputs and
is invalid for mutually-exclusive ACS subtypes.  New runs are delegated to
``acs_joint_model_comparison`` (categorical ACS + independent OMI).
"""
from experiments.acs_joint_model_comparison import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
