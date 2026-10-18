"""A second provider edge uses the same graph and measured-selection APIs."""
from dataclasses import replace
import importlib.util
import sys

import pytest
pytest.importorskip('egglog')
from merlin.common.paths import repo_root


@pytest.fixture(scope='module')
def provider():
    path = repo_root() / 'examples/k1_cpu/egglog_relu.py'
    spec = importlib.util.spec_from_file_location('k1_integer_egglog_example', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def contract():
    # Minimal semantic fields at this explicit target edge. The retained CLI
    # campaign additionally binds the actual canonical YAML and snapshot hash.
    return {'name': 'k1_cpu', 'capabilities': {'ops': ['relu']}, 'compute_units': [
        {'kind': 'scalar', 'semantic_capabilities': [{'family': 'elementwise_map', 'dtypes': ['int8']}]}]}


def test_every_i8_encoding_and_repetition_induction_witness(provider):
    proof = provider.exhaustive_proof()
    assert proof['encodings'] == 256
    assert provider.signed_i8(128) == -128 and provider.signed_i8(255) == -1
    assert provider.signed_i8(127) == 127 and provider.signed_i8(0) == 0
    assert provider.relu(-128) == 0 and provider.relu(127) == 127
    for raw in range(256):
        value = provider.signed_i8(raw)
        assert provider.evaluate(provider.repeated(128), value) == (raw if raw < 128 else 0)


@pytest.mark.parametrize('depth', [1, 2, 5, 16, 128])
def test_real_shared_egglog_extraction_collapses_chain(provider, depth):
    node, report = provider.egglog_candidate(provider.repeated(depth))
    assert node == provider.repeated(1)
    assert report['stop_reason'] == 'fixed_point'
    assert report['dag_cost'] == 1


def test_native_hand_egglog_measured_outputs_and_explicit_cost_tier(provider, contract):
    result = provider.experiment(contract, 'a' * 64)
    assert result['status'] == 'PASS'
    assert result['tied_best'] == ['egglog', 'hand']
    assert result['device_timing'] == 'NOT_RUN_NO_BOARD'
    assert result['hardware_execution'] is False
    assert result['evidence_tier'] == 'host_functional_cost'
    assert result['executions']['native']['primitive_evaluations'] == [2925] * 3
    assert result['executions']['hand']['primitive_evaluations'] == [585] * 3
    assert result['executions']['egglog']['primitive_evaluations'] == [585] * 3
    assert result['executions']['hand']['program_sha256'] == result['executions']['egglog']['program_sha256']
    for execution in result['executions'].values():
        assert len(execution['cases']) == 18
        assert all(case['output_sha256'] == case['oracle_sha256'] and case['guards'] for case in execution['cases'])


def test_contract_unknown_semantics_and_floating_point_fail_closed(provider, contract):
    for invalid in ({}, {**contract, 'name': 'different'}, {**contract, 'compute_units': []}):
        with pytest.raises(ValueError):
            provider.experiment(invalid, 'a' * 64)
    for invalid in (replace(provider.repeated(1), type='f32'),
                    provider.Node('identity', (provider.INPUT,)),
                    provider.Node('anchor', attrs=(('id', 'forged'),)),
                    replace(provider.repeated(1), attrs=(('assume_finite', 'true'),))):
        with pytest.raises(ValueError):
            provider.serialize(invalid)


@pytest.mark.parametrize('value', [-1, 256, True, 1.0])
def test_raw_input_validation(provider, value):
    with pytest.raises(ValueError):
        provider.signed_i8(value)


def test_executed_output_difference_aborts_instead_of_receiving_cost(provider):
    with pytest.raises(provider.selection.CorrectnessFailure):
        provider.measure(b'def kernel(value, relu):\n    return 0\n')


def test_serialization_roundtrip_and_compiled_program_binding(provider):
    node = provider.repeated(3)
    assert provider.deserialize(provider.serialize(node)) == node
    candidate = provider.selection.RewriteCandidate('example', provider.serialize(node), 'a' * 64, (), provider.estimate(node))
    program = provider.compile_candidate(candidate)
    receipt, evidence = provider.measure(program)
    assert receipt.program_sha256 == provider.selection.digest(program)
    assert receipt.values == (1755,) * 3
    assert evidence['primitive_evaluations'] == [1755] * 3
