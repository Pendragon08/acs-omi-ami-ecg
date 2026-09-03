from experiments.ecg_lfm.audit_joint10_labels import (
    JOINT10_CLASSES,
    canonical_assignment_hash,
    candidate_joint10,
    resolve_culprit_group,
)


def row(acs="STEMI", omi=1, **segments):
    values = {name: "0" for name in ("STEMI", "NSTEMI", "UA")}
    if acs != "OTHER":
        values[acs] = "1"
    values["OMI"] = str(omi)
    for name in ("LM", "PLAD", "MLAD", "DLAD", "DB", "PLCX", "MLCX", "DLCX", "OM", "PRCA", "MRCA", "DRCA"):
        values[name] = str(segments.get(name, 0))
    return values


def test_joint10_class_order():
    assert JOINT10_CLASSES == (
        "OTHER", "UA", "STEMI+nOMI", "NSTEMI+nOMI", "STEMI+OMI+LM_LAD",
        "STEMI+OMI+LCX", "STEMI+OMI+RCA", "NSTEMI+OMI+LM_LAD",
        "NSTEMI+OMI+LCX", "NSTEMI+OMI+RCA",
    )


def test_joint10_class_mapping():
    assert candidate_joint10(row("STEMI", 0))["joint10_id"] == 2
    assert candidate_joint10(row("NSTEMI", 0))["joint10_id"] == 3
    assert candidate_joint10(row("STEMI", 1, PLAD=1))["joint10_id"] == 4
    assert candidate_joint10(row("NSTEMI", 1, PLCX=1))["joint10_id"] == 8
    assert candidate_joint10(row("NSTEMI", 1, PRCA=1))["joint10_id"] == 9


def test_ua_omi_zero_support_guard():
    result = candidate_joint10(row("UA", 1))
    assert result["status"] == "UA_OMI_INVALID"
    assert result["joint10_id"] is None


def test_other_omi_audit():
    result = candidate_joint10(row("OTHER", 1))
    assert result["joint10_id"] == 0
    assert result["exception"] == "OTHER_OMI"
    assert result["omi"] == 1


def test_cto_not_used_as_acute_omi():
    result = resolve_culprit_group(row("STEMI", 1, PLAD=2))
    assert result["status"] == "NO_GROUP"
    assert result["culprit_group"] is None
    assert result["cto_fields"] == ["PLAD"]


def test_same_group_multisegment_allowed():
    result = resolve_culprit_group(row("STEMI", 1, PLAD=1, MLAD=1))
    assert result["status"] == "UNIQUE_GROUP"
    assert result["culprit_group"] == "LM_LAD"


def test_multigroup_unresolved_not_guessed():
    result = resolve_culprit_group(row("STEMI", 1, PLAD=1, PRCA=1))
    assert result["status"] == "MULTI_GROUP"
    assert result["culprit_group"] is None
    assert result["acute_groups"] == ["LM_LAD", "RCA"]


def test_canonical_assignment_hash_is_order_independent():
    rows = [
        {"fold": "0", "subset": "outer_test", "record_id": "00002", "patient_id": "P2"},
        {"fold": "0", "subset": "inner_train", "record_id": "00001", "patient_id": "P1"},
    ]
    assert canonical_assignment_hash(rows) == canonical_assignment_hash(list(reversed(rows)))


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"{len(tests)} tests passed (direct assertion runner)")
