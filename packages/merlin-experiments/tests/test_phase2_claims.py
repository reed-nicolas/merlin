"""Canonical decision owners preserve frozen analyzer identities and refusals."""

import re
import sys
from copy import deepcopy
from types import ModuleType
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase0 import sweeps
from merlin_experiments.phase2.claims import affine, dispatch, paired, pk, pr
from merlin_experiments.phase2.claims.oracle_acceptance import selected_acceptance

from merlin.common.paths import repo_root
from merlin.perf.claim_reach import analyzer_identity


@pytest.mark.parametrize(
    ("declared", "owner"),
    [
        (pk._ACCEPTANCE_BASE["analyzer"], pk),
        (pk._CURRENT_ACCEPTANCE_BASE["analyzer"], pk),
        (pr.supported_acceptance()["analyzer"], pr),
        (pr._CURRENT_ACCEPTANCE_BASE["analyzer"], pr),
        (affine.ANALYZER, affine),
        (paired.ANALYZER, paired),
    ],
)
def test_frozen_identity_resolves_canonical_owner_not_legacy_shadow(declared, owner, monkeypatch):
    identity = analyzer_identity({"acceptance": {"analyzer": declared}})
    monkeypatch.setitem(sys.modules, identity.module, ModuleType(identity.module))
    resolved = dispatch.resolve([{"performance": {"acceptance": {"analyzer": declared}}}])
    assert resolved.identity.declared == declared
    assert resolved.identity.module == identity.module
    assert resolved.module is owner
    assert resolved.analyze is getattr(owner, identity.function)
    assert dispatch._registry()[declared] is resolved.analyze


def test_unknown_declared_owner_stays_unavailable():
    with pytest.raises(dispatch.DispatchError, match="unavailable"):
        dispatch.resolve([{"performance": {"acceptance": {"analyzer": "absent_claim_owner.decide/v1"}}}])


@pytest.mark.parametrize("module", [pk, pr])
def test_dispatch_validates_full_target_selected_oracle_contract(module):
    template = module._CURRENT_ACCEPTANCE_BASE
    declared = deepcopy(template)
    declared["evidence"]["correctness_simulator"] = "reference_sim"
    declared["evidence"]["timing_simulator"] = "selected_rtl"
    declared["evidence"]["timing_oracle_kind"] = "rtl_selected_rtl"
    declared["fit"]["dependent_metric"] = "selected_rtl_L3_cycles"
    frozen = selected_acceptance(template, declared)
    dispatch.verify_supported_acceptance(module, frozen, "fixture")
    frozen["evidence"]["resolved_from"].pop("timing_oracle_kind")
    with pytest.raises(dispatch.StageGateError, match="differs from the supported claim contract"):
        dispatch.verify_supported_acceptance(module, frozen, "fixture")


@pytest.mark.parametrize(("family", "module"), [("PK", pk), ("PR", pr)])
def test_phase0_selected_oracle_contract_matches_phase2_analyzer(family, module, monkeypatch):
    from merlin.targetgen import target_experiment

    monkeypatch.setattr(target_experiment, "load_capability_manifest", lambda _target: SimpleNamespace(
        contract={"runner": {"tier_sim": {"L2": "reference_sim", "L3": "elaborated_rtl"}}},
    ))
    document = yaml.safe_load((repo_root() / "experiments/templates/phase0/performance.yaml").read_text())
    declaration = deepcopy(next(row for row in document["sweeps"] if row["id"] == family)["base"]["performance"])
    resolved = sweeps._resolve_target_oracle_evidence(
        declaration, "fixture", oracle_selection={"L2": "reference_sim", "L3": "selected_rtl"}
    )["acceptance"]
    assert resolved["evidence"]["resolved_from"] == {
        "correctness_simulator": "$target_oracle:L2",
        "timing_simulator": "$target_oracle:L3",
        "timing_oracle_kind": "$target_oracle_kind:L3",
    }
    assert resolved == module.supported_acceptance(resolved)
    dispatch.verify_supported_acceptance(module, resolved, family)


def test_explicit_external_module_is_not_rewritten(monkeypatch):
    module = ModuleType("fixture_external_claim")
    module.preflight_external = lambda descriptors: {"ready": True}
    module.decide = lambda descriptors, results: {"verdict": "REFUSED"}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    declared = "fixture_external_claim.decide/v1"
    resolved = dispatch.resolve([{"performance": {"acceptance": {"analyzer": declared}}}])
    assert resolved.module is module
    assert resolved.identity.declared == declared


def test_structural_name_validation_matches_original_ascii_contract():
    samples = [None, 1, "", "valid-name.0_A", "line\n", "é", "中", "a/b"]
    samples.extend("prefix" + chr(codepoint) + "suffix" for codepoint in range(256))
    for name in samples:
        expected = isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9._-]+", name) is not None
        with pytest.raises(pk._Refusal) as raised:
            pk._descriptor_point({"name": name})
        accepted_name = "simple non-empty name" not in str(raised.value)
        assert accepted_name == expected, repr(name)
