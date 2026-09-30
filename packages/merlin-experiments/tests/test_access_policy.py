"""Packaged historical denials must survive source eviction and installed execution."""

import json
from importlib.resources import files

import pytest
from merlin_experiments import access_policy as policy

from merlin.common import access as shared


def test_extension_ships_and_loads_legacy_denials_without_source_files():
    resource = files("merlin_experiments").joinpath(policy.RESOURCE)
    assert resource.is_file()
    assert policy.load_legacy_access()
    assert set(shared.declared_modules("oracle")) < set(policy.declared_modules("oracle"))
    assert set(shared.declared_modules("grader")) < set(policy.declared_modules("grader"))
    assert len(policy.MODULE_ACCESS) == len(shared.MODULE_ACCESS) + len(policy.load_legacy_access())


def test_each_historical_target_identity_is_extension_owned_and_denied():
    expected = {
        ("merlin.targetgen.muon_oracles", "oracle", False),
        ("merlin.targetgen.eval.gemmini_conformance", "grader", False),
        ("merlin.targetgen.eval.gemmini_suite", "grader", False),
        ("merlin.targetgen.eval.gemmini_dispatcher", "grader", False),
        ("merlin.targetgen.agent.gemmini_kernel_slot", "grader", False),
        ("gemmini_conformance", "grader", True),
        ("merlin.targetgen.oracle_helpers.npu_emit", "oracle", False),
        ("atlas_program_emit", "oracle", False),
    }
    assert expected <= {(item.identity, item.origin, item.directory) for item in policy.MODULE_ACCESS}
    assert not {identity for identity, _, _ in expected} & {item.identity for item in shared.MODULE_ACCESS}


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "{}",
        '{"schema":"merlin.access.legacy.v1","modules":[]}',
        '{"schema":"future","modules":[{}]}',
        '{"schema":"merlin.access.legacy.v1","modules":[{"identity":"../escape","origin":"grader","directory":false,"aliases":[]}]}',
    ],
)
def test_missing_or_invalid_legacy_registry_refuses(tmp_path, payload):
    source = tmp_path / "access.json"
    if payload is not None:
        source.write_text(payload)
    with pytest.raises(policy.AccessPolicyUnavailable):
        policy.load_legacy_access(source)


def test_duplicate_legacy_identity_refuses(tmp_path):
    document = json.loads(policy.resource_path().read_text())
    document["modules"].append(document["modules"][0])
    source = tmp_path / "access.json"
    source.write_text(json.dumps(document))
    with pytest.raises(policy.AccessPolicyUnavailable, match="duplicate"):
        policy.load_legacy_access(source)


def test_loaded_deny_set_cannot_outlive_a_changed_resource(tmp_path, monkeypatch):
    document = json.loads(policy.resource_path().read_text())
    document["modules"].pop()
    source = tmp_path / "changed.json"
    source.write_text(json.dumps(document))
    monkeypatch.setattr(policy, "resource_path", lambda: source)
    with pytest.raises(policy.AccessPolicyUnavailable, match="changed after"):
        policy.require_current_policy()


def test_packaged_policy_byte_drift_refuses_even_when_denials_are_unchanged(tmp_path, monkeypatch):
    source = tmp_path / "same-denials.json"
    source.write_bytes(policy.resource_path().read_bytes() + b"\n")
    monkeypatch.setattr(policy, "resource_path", lambda: source)
    with pytest.raises(policy.AccessPolicyUnavailable, match="changed after"):
        policy.require_current_policy()
