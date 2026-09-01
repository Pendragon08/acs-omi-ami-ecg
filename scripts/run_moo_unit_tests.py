"""Dependency-light runner for the MOO tests on the execution server."""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

_path = Path(__file__).resolve().parents[1] / "tests" / "test_multi_objective.py"
_spec = importlib.util.spec_from_file_location("test_multi_objective", _path)
assert _spec and _spec.loader
test_multi_objective = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(test_multi_objective)


def main() -> None:
    names = sorted(name for name in dir(test_multi_objective) if name.startswith("test_"))
    for name in names:
        function = getattr(test_multi_objective, name)
        if inspect.isfunction(function):
            function()
            print(f"PASS {name}")
    print("MOO_UNIT_TESTS_PASS")


if __name__ == "__main__":
    main()
