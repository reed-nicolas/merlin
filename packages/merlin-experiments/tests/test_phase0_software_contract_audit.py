"""The SW audit reports missing evidence without certifying a frozen corpus."""

from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase0.software_contract_audit import audit_corpus


def _capsule(root: Path, name: str, screen: dict) -> None:
    path = root / name / "capsule.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump({"label": "public", "software_screen": screen}))


def test_audit_binds_unknown_axes_to_frozen_manifest_without_promoting_review(tmp_path):
    root = tmp_path / "capsules"
    root.mkdir()
    generated = ["isa/matmul", "model/whole", "model_slices/host", "_diagnostic/refused"]
    (root / "MANIFEST.yaml").write_text(
        yaml.safe_dump(
            {
                "generated": generated,
                "phase0_evidence": {
                    "software_spec": {"target": "test_device", "status": "unreviewed", "sha256": "frozen"}
                },
                "performance_generation": {
                    "test_device": {
                        "families": [
                            {"family": "PN_selected", "requirement_basis": {"axis": "scope.performance.required"}}
                        ],
                        "counts": {
                            "by_family": {"PN_selected": {"admitted_members": 2, "written_members": 0}}
                        },
                        "errors": [
                            {"family": "PN_selected", "member": "PN_selected00", "error_type": "ImportError"},
                            {"family": "PN_derived", "member": "PN_derived00", "error_type": "ImportError"},
                        ],
                    }
                },
            }
        )
    )
    _capsule(
        root,
        "isa/matmul",
        {
            "status": "unknown",
            "decisions": [
                {
                    "role": "carrier",
                    "status": "unknown",
                    "unresolved_constraints": ["placement", "tails", "shape_bounds.K"],
                    "review_status": "unreviewed",
                }
            ],
        },
    )
    _capsule(root, "model/whole", {"status": "unknown", "decisions": []})
    _capsule(
        root,
        "model_slices/host",
        {
            "status": "unknown",
            "decisions": [
                {
                    "role": "host",
                    "status": "unknown",
                    "profiles": [
                        {
                            "status": "unknown",
                            "decisions": [
                                {
                                    "status": "unknown",
                                    "review_status": "unreviewed",
                                    "unresolved_constraints": ["ordered_operand_dtypes"],
                                }
                            ],
                        }
                    ],
                }
            ],
        },
    )
    _capsule(root, "_diagnostic/refused", {"status": "unsupported", "decisions": []})

    audit = audit_corpus(root)
    assert audit == audit_corpus(root)
    assert audit["verification_status"] == "not_established"
    assert audit["selected_software_spec"]["status"] == "unreviewed"
    assert audit["screen_status_counts"] == {"unknown": 3, "unsupported": 1}
    assert audit["explicit_refusals"] == ["_diagnostic/refused"]
    performance = audit["performance_materialization"]["test_device"]
    assert performance["status"] == "incomplete"
    assert performance["required_scope_shortfalls"] == [
        {"family": "PN_selected", "admitted_members": 2, "written_members": 0}
    ]
    assert performance["unattributed_error_families"] == ["PN_derived"]
    axes = {row["axis"]: row for row in audit["unresolved_axes"]}
    assert axes["placement"]["capsules"] == ["isa/matmul"]
    assert axes["per_operation_inventory"]["capsules"] == ["model/whole"]
    assert axes["host_capabilities"]["capsules"] == ["model_slices/host"]
    assert axes["ordered_operand_dtypes"]["capsules"] == ["model_slices/host"]
    assert "boundary" in axes["shape_bounds.K"]["independent_evidence_needed"]
    assert axes["contract_review"]["count"] == 4
    assert [row["axis"] for row in audit["global_contract_obligations"]] == [
        "numerical_semantics",
        "transfer_contracts",
    ]


def test_audit_rejects_manifest_path_escape(tmp_path):
    root = tmp_path / "capsules"
    root.mkdir()
    escaped = tmp_path / "escaped" / "capsule.yaml"
    escaped.parent.mkdir()
    escaped.write_text("name: escaped\n")
    (root / "MANIFEST.yaml").write_text(yaml.safe_dump({"generated": ["../escaped"]}))
    with pytest.raises(ValueError, match="escapes frozen corpus"):
        audit_corpus(root)
