"""Compatibility entry point for the corrected ACS/OMI joint experiment.

The former implementation trained four independent sigmoid heads for
STEMI/NSTEMI/UA/OMI.  That formulation allowed impossible simultaneous ACS
subtypes and is retained only in Git history as a legacy diagnostic.  New
invocations are routed to :mod:`experiments.acs_joint_e4`, which uses a
categorical ACS head (including explicit ``OTHER`` for ``[0,0,0]``) and a
separate OMI binary head.
"""
from experiments.acs_joint_e4 import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
