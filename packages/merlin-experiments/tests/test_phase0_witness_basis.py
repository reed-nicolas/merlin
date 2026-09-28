"""A witness basis minimizes observed signatures, not execution obligations."""

from merlin_experiments.phase0.witness_basis import build_witness_basis


def _capsules(*names):
    return [{"name": name, "sha256": name, "program_sha256": name, "status": "inventoried"} for name in names]


def test_singleton_witnesses_prove_exact_minimum_with_typed_edges():
    applications = {
        "model": {
            "operations": [
                {"id": "op:a", "role": "compute_placement", "status": "missing", "witnesses": ["a"]},
                {"id": "op:b", "role": "support_lowering", "status": "missing", "witnesses": ["b"]},
                {"id": "op:shared", "status": "missing", "witnesses": ["a", "b", "c"]},
            ],
            "transfers": [
                {"id": "edge:0", "status": "pending_placement", "witnesses": ["a", "b"]},
            ],
        }
    }
    basis = build_witness_basis(applications, _capsules("a", "b", "c"))
    selection = basis["selection"]
    assert basis["universe"]["n_total"] == 4
    assert basis["uncovered_obligations"] == []
    assert selection["claim"] == "exact_minimum"
    assert selection["lower_bound"] == selection["n_selected"] == 2
    assert [row["name"] for row in selection["selected_capsules"]] == ["a", "b"]
    assert set(selection["forced_by_singleton_obligation"]) == {"a", "b"}
    assert basis["universe"]["obligations"][-1]["kind"] == "typed_edge_candidate"
    assert basis["formal_proof"]["eligibility"] == "not_assessed"
    assert basis["formal_proof"]["verdict"] == "not_proven"


def test_nonforced_cover_is_only_an_upper_bound_and_unwitnessed_work_stays_visible():
    applications = {
        "model": {
            "operations": [
                {"id": "one", "witnesses": ["a", "b"]},
                {"id": "two", "witnesses": ["b", "c"]},
                {"id": "three", "witnesses": ["a", "c"]},
                {"id": "unwitnessed", "witnesses": ["outside_cohort"]},
                {"id": "not_inventoried", "witnesses": ["pending"]},
            ],
            "transfers": [],
        }
    }
    basis = build_witness_basis(
        applications,
        [*_capsules("a", "b", "c"), {"name": "pending", "status": "not_available"}],
    )
    assert basis["universe"]["n_witnessable"] == 3
    assert [row["source_id"] for row in basis["uncovered_obligations"]] == ["unwitnessed", "not_inventoried"]
    assert basis["selection"]["claim"] == "upper_bound"
    assert basis["selection"]["lower_bound"] == 1
    assert basis["selection"]["n_selected"] == 2
    assert "No capsule is removed" in basis["qualification"]
