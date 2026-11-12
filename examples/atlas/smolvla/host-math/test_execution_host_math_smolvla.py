"""Checks for the SmolVLA validation-only host math selection and packaging.

The real tiny-application test uses the generated evidence under HOST_MATH_OUT and skips without it.
"""
import copy
import ctypes as ct
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest

import execution_host_math_smolvla as consumer
from merlin.compile.c_sources import c_source_pair

F32_ANGLES = (0, 0x3f800000)
F64_ANGLES = (0, 0x3ff0000000000000)


def digest(data):
    return sha256(data).hexdigest()


def domain_bytes():
    rows32 = b''.join(struct.pack('<III', angle, 0, 0) for angle in F32_ANGLES)
    rows64 = b''.join(struct.pack('<QQQ', angle, 0, 0) for angle in F64_ANGLES)
    return rows32 + rows64


def gate_for(contract, rv64_output):
    angle32 = b''.join(struct.pack('<I', a) for a in F32_ANGLES)
    angle64 = b''.join(struct.pack('<Q', a) for a in F64_ANGLES)
    hashes = {'f32_angle_sha256': digest(angle32), 'f64_angle_sha256': digest(angle64)}
    return {'schema': consumer.GATE_SCHEMA, 'gate': 'PASS', 'rounding': 'RNE', 'linkage': contract,
            'execution_host_archive_sha256': contract['archive_sha256'],
            'production_rv64_archive_sha256': consumer.RV64_ARCHIVE,
            'domain': {'f32': {'rows': 2, 'compared_f32_trig_words': 4}, 'f64': {'rows': 2, 'compared_f64_trig_words': 4}},
            **{key: 0 for key in consumer.ZERO_COUNTS}, 'results': copy.deepcopy(consumer.ZERO_RESULTS),
            'rv64_output_sha256': digest(rv64_output.read_bytes()),
            'execution_host_output_sha256': digest(rv64_output.read_bytes()),
            'output_hashes': {engine: dict(hashes) for engine in ('rv64', 'execution_host')}}


@pytest.fixture
def case(tmp_path, monkeypatch):
    production = tmp_path / 'application/c-package'
    execution = tmp_path / 'runtime/c-package'
    production.mkdir(parents=True)
    archive = tmp_path / 'native.a'
    archive.write_bytes(b'synthetic native supplier')
    monkeypatch.setattr(consumer, 'RV64_ARCHIVE', 'b' * 64)
    contract = {'schema': 'merlin.host_linkage_contract.v1', 'supplier': 'selected_math_archive',
                'archive_sha256': digest(archive.read_bytes()), 'symbols': consumer.SYMBOLS}
    rv64_output = tmp_path / 'domain.bin'
    rv64_output.write_bytes(domain_bytes())
    gate_path = tmp_path / 'equivalence-receipt.json'
    gate_path.write_text(json.dumps(gate_for(contract, rv64_output)))
    prod_contract = {**contract, 'archive_sha256': consumer.RV64_ARCHIVE}
    nodes = [{'id': 'in32', 'type': {'dtype': 'f32', 'shape': [2], 'numerical_policy': 'raw-bit-pattern'}},
             {'id': 'in64', 'type': {'dtype': 'f64', 'shape': [2], 'numerical_policy': 'raw-bit-pattern'}}]
    regions, headers, sources = [], {}, {}
    for index, (op, (dtype, symbol)) in enumerate(sorted(consumer.OPERATIONS.items())):
        name = f'region_{index:03d}'
        member, header = f'regions/{name}/{name}.c', f'regions/{name}/{name}.h'
        headers[header] = f'int {name}(void *,const void *const *,void *const *);\n'
        sources[member] = (f'#include "{name}.h"\n#if !defined(__riscv)\n#error RV64 required\n#endif\n'
                           f'int {name}(void *c,const void *const *i,void *const *o) {{ return 7; }}\n')
        for path, text in ((member, sources[member]), (header, headers[header])):
            target = production / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        source_input = 'in32' if dtype == 'f32' else 'in64'
        nodes.append({'id': name, 'inputs': [source_input], 'op': op, 'type': nodes[0 if dtype == 'f32' else 1]['type']})
        desc = {'arithmetic': 'selected-library-source-math', 'source_intrinsic': op, 'host_isa': 'rv64gc',
                'host_abi': 'lp64d', 'numeric_policy': 'selected-libm-rne-finite-v1',
                'effects': 'enabled-fpu-rne-unobserved-flags-and-errno-v1', 'source_request_digest': 'c' * 64,
                'implementation': name, 'host_linkage': {**prod_contract, 'symbols': [symbol]}}
        regions.append({'description': desc, 'placement': 'host', 'native_programs': [], 'outputs': [name],
                        'inputs': [source_input], 'source_request_digest': 'c' * 64})
    sources['retained.c'] = 'int retained(void) { return 91; }\n'
    package = c_source_pair('combined', public_headers=headers, private_headers={}, sources=sources)
    for path, text in package.items():
        (production / path).write_text(text)
    request = production.parent / 'source-request.json'
    request.write_text(json.dumps({'nodes': nodes}))
    inventory = {'c-package/' + str(path.relative_to(production)): digest(path.read_bytes())
                 for path in production.rglob('*') if path.is_file()}
    inventory['source-request.json'] = digest(request.read_bytes())
    manifest = {'application_interface': {'source': 'combined.c', 'host_linkage': [prod_contract]},
                'regions': regions, 'stage_artifacts': inventory}
    return manifest, production, execution, gate_path, archive, rv64_output


def test_replaces_all_four_components_and_preserves_production(case):
    manifest, production, execution, gate, archive, rv64_output = case
    before = {str(p): p.read_bytes() for p in production.rglob('*') if p.is_file()}
    manifest_before = copy.deepcopy(manifest)
    record = consumer.prepare_execution_host_math(*case)
    result = (execution / 'combined.c').read_text()
    assert '#error RV64 required' not in result
    assert 'int retained(void) { return 91; }' in result
    assert {c['symbol'] for c in record['components']} == set(consumer.SYMBOLS)
    assert result.count('static double (*volatile const') == 2 and result.count('static float (*volatile const') == 2
    assert record['qualified_distinct_encodings'] == {'f32': 2, 'f64': 2}
    assert '-Wl,--trace-symbol=sin' in record['link_flags'] and '-Wl,--trace-symbol=cosf' in record['link_flags']
    assert manifest == manifest_before
    assert all(Path(path).read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize('change', ['archive', 'pin', 'gate', 'results', 'domain', 'angle_hash', 'component',
                                    'f64_only_symbols', 'missing_f64_region', 'nonfinite'])
def test_refuses_changed_evidence(case, change):
    manifest, production, execution, gate_path, archive, rv64_output = case
    gate = json.loads(gate_path.read_text())
    if change == 'archive':
        archive.write_bytes(b'wrong archive')
    elif change == 'pin':
        manifest['regions'][0]['description']['host_linkage']['archive_sha256'] = 'd' * 64
    elif change == 'gate':
        gate['f64_angle_input_mismatch_count'] = 1
    elif change == 'results':
        gate['results']['sin']['f32_mismatch_count_after_cast'] = 1
    elif change == 'domain':
        rv64_output.write_bytes(rv64_output.read_bytes()[:-24])
    elif change == 'angle_hash':
        gate['output_hashes']['execution_host']['f64_angle_sha256'] = 'e' * 64
    elif change == 'component':
        (production / 'regions/region_000/region_000.c').write_text('changed')
    elif change == 'f64_only_symbols':
        gate['linkage']['symbols'] = ['cos', 'sin']
    elif change == 'missing_f64_region':
        manifest['regions'] = [r for r in manifest['regions'] if r['description']['source_intrinsic'].startswith('source_f32')]
    else:
        data = bytearray(rv64_output.read_bytes())
        data[24:32] = struct.pack('<Q', 0x7ff0000000000000)
        rv64_output.write_bytes(bytes(data))
        gate['rv64_output_sha256'] = gate['execution_host_output_sha256'] = digest(bytes(data))
        angle64 = struct.pack('<QQ', 0x7ff0000000000000, F64_ANGLES[1])
        for engine in gate['output_hashes'].values():
            engine['f64_angle_sha256'] = digest(angle64)
    gate_path.write_text(json.dumps(gate))
    with pytest.raises(ValueError):
        consumer.prepare_execution_host_math(*case)
    assert not (execution / 'combined.c').exists()


@pytest.mark.skipif(platform.machine() != 'x86_64' or not shutil.which('cc'), reason='needs x86_64 cc')
def test_binary64_component_checks_before_writes(tmp_path):
    source = tmp_path / 'unit.c'
    source.write_text(consumer._native_source64('unary', 'sin', 2, list(F64_ANGLES)) + '''
int set_round(int mode) { return fesetround(mode ? FE_DOWNWARD : FE_TONEAREST); }
void set_mxcsr(unsigned value) { _mm_setcsr(value); }
unsigned get_mxcsr(void) { return _mm_getcsr(); }
''')
    library = tmp_path / 'unit.so'
    subprocess.run(['nice', '-n', '10', 'cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-frounding-math',
                    '-ffp-contract=off', '-shared', '-fPIC', str(source), '-lm', '-o', str(library)], check=True)
    lib = ct.CDLL(str(library))
    lib.unary.argtypes = [ct.c_void_p, ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_void_p)]
    lib.get_mxcsr.restype = ct.c_uint
    inp = (ct.c_uint64 * 2)(*F64_ANGLES)
    sentinel = [0xa5a5a5a5a5a5a5a5] * 2
    out = (ct.c_uint64 * 2)(*sentinel)
    inputs = (ct.c_void_p * 1)(ct.addressof(inp)); outputs = (ct.c_void_p * 1)(ct.addressof(out))
    assert lib.unary(None, inputs, outputs) == 0
    assert list(out) == [0, struct.unpack('<Q', struct.pack('<d', __import__('math').sin(1.0)))[0]]
    for bad, code in ((0x4000000000000000, -4), (0x7ff8000000000000, -2), (0x7ff0000000000000, -2)):
        inp[1] = bad; out[:] = sentinel
        assert lib.unary(None, inputs, outputs) == code
        assert list(out) == sentinel
    inp[1] = F64_ANGLES[1]
    assert lib.unary(None, inputs, inputs) == -1
    out[:] = sentinel
    assert lib.set_round(1) == 0
    try:
        assert lib.unary(None, inputs, outputs) == -3
        assert list(out) == sentinel
    finally:
        lib.set_round(0)
    original = lib.get_mxcsr()
    try:
        for bit in (1 << 6, 1 << 15, 1 << 13):
            lib.set_mxcsr(original | bit)
            assert lib.unary(None, inputs, outputs) == -3
            assert list(out) == sentinel
    finally:
        lib.set_mxcsr(original)


@pytest.fixture
def tiny_application(tmp_path):
    out = Path(os.environ.get('HOST_MATH_OUT', Path(__file__).resolve().parent / 'out'))
    evidence = (out / 'execution-host-math/libcore-math-smolvla-execution-host.a',
                out / 'execution-host-math/equivalence-receipt.json', out / 'rv64-host-math.bin',
                out / 'core-math/libcore-math-smolvla.a')
    if any(not path.is_file() for path in evidence):
        pytest.skip('real SmolVLA packaging needs the generated evidence under HOST_MATH_OUT')
    if platform.machine() != 'x86_64' or not shutil.which('cc'):
        pytest.skip('real packaging requires x86_64 and cc')
    archive, equivalence, rv64_output, production_archive = evidence
    assert digest(production_archive.read_bytes()) == consumer.RV64_ARCHIVE
    from merlin.compile.c_unary_math import emit_unary_math_c
    from merlin.compile.semantic_application import CompiledCRegion, _render
    from merlin.compile.semantic_regions import extract_semantic_region
    from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
    from merlin.xdsl_dialects.lowering.global_plan import (
        BufferRepresentation, CycleInterval, GlobalPlan, RegionAlternative, ValueRepresentation)
    types = {dtype: TensorType((2,), dtype, 'raw-bit-pattern') for dtype in ('f32', 'f64')}
    inputs = {dtype: SemanticNode('angles_' + dtype, 'input', (), types[dtype], effect='input') for dtype in types}
    operations = tuple(SemanticNode(
        f'{operation}_{dtype}', f'source_{dtype}_{operation}_unqualified', (inputs[dtype].id,), types[dtype],
        (('numeric_semantics', f'source-{dtype}-{operation}-unqualified-v1'),), index_maps=pointwise_identity_maps(1, 1))
        for dtype in ('f32', 'f64') for operation in ('sin', 'cos'))
    source = KernelRequest((*inputs.values(), *operations), tuple(node.id for node in operations),
                           ('external',) * 4, tuple((node.id, 'external') for node in inputs.values()), 'synthetic')
    plan = GlobalPlan(tuple(RegionAlternative(
        node.id, (index,), 'selected_math', 'host', CycleInterval.unknown('test'),
        (BufferRepresentation(node.inputs[0], ValueRepresentation('memory', 'dense', node.type.dtype)),),
        (BufferRepresentation(node.id, ValueRepresentation('memory', 'dense', node.type.dtype)),),
    ) for index, node in enumerate(operations)), (), CycleInterval.unknown('test'))
    linkage = dict(schema='merlin.host_linkage_contract.v1', supplier='selected_math_archive',
                   archive_sha256=consumer.RV64_ARCHIVE, symbols=consumer.SYMBOLS)
    policy = dict(schema='merlin.compiled_unary_math.v1', policy='selected-libm-rne-finite-v1', isa='rv64gc', abi='lp64d',
                  effects='enabled-fpu-rne-unobserved-flags-and-errno-v1',
                  operations=sorted(node.op for node in operations), linkage=linkage)
    compiled = []
    for index, node in enumerate(operations):
        region = extract_semantic_region(source, (node.id,))
        name = f'region_{index:03d}'
        description, generated = emit_unary_math_c(region, name, policy)
        compiled.append(CompiledCRegion(name, name + '.h', node.inputs, (node.id,),
                        {member: data.encode() for member, data in generated.items()}, (), (), description))
    _, regions, interface, links, files = _render(source, SimpleNamespace(plan=lambda _: plan), 'capsule', 4096, tuple(compiled))
    application = tmp_path / 'application'
    for member, data in files.items():
        path = application / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = dict(schema='merlin.semantic_application.v1', application_name='capsule', application_interface=interface,
                    application_memory_bytes=4096, request_digest=source.digest(), regions=regions, segments=[],
                    link_sources=links, stage_artifacts={member: digest(data) for member, data in files.items()})
    assert interface['host_linkage'] == [linkage]
    before = {member: (application / member).read_bytes() for member in files}
    return SimpleNamespace(application=application, manifest=manifest, before=before, archive=archive,
                           equivalence=equivalence, rv64_output=rv64_output, runtime=tmp_path / 'runtime',
                           inputs=[node.id for node in inputs.values()], outputs=[node.id for node in operations])


def test_real_tiny_application_runs_selected_archive(tiny_application, monkeypatch):
    case = tiny_application
    monkeypatch.setitem(sys.modules, 'execution_host_math', consumer)
    import compile_capsule
    shutil.copytree(case.application / 'c-package', case.runtime)
    real_run = subprocess.run
    monkeypatch.setattr(compile_capsule.subprocess, 'run',
                        lambda command, **kwargs: real_run(['nice', '-n', '10', *command], **kwargs))
    compile_capsule.build_host_library(case.application, case.manifest, package=case.runtime,
        execution_host_math={'functional_validation': True, 'archive': case.archive,
                             'equivalence': case.equivalence, 'rv64_output': case.rv64_output})
    selection = json.loads((case.runtime / 'execution-host-math.json').read_text())
    assert selection['validation_only'] is True and selection['deployment_qualified'] is False
    assert {c['symbol'] for c in selection['components']} == set(consumer.SYMBOLS)
    assert set(selection['link_supplier_observation']['symbols']) == set(consumer.SYMBOLS)
    for supplier in selection['link_supplier_observation']['symbols'].values():
        assert supplier['definition'] == str(case.archive.resolve()) + '(wrapper.o)'
    gate = json.loads(case.equivalence.read_text())
    data = case.rv64_output.read_bytes()
    rows32 = list(struct.iter_unpack('<III', data[:gate['domain']['f32']['rows'] * 12]))
    rows64 = list(struct.iter_unpack('<QQQ', data[gate['domain']['f32']['rows'] * 12:]))
    pick32, pick64 = (rows32[33], rows32[-1]), (rows64[360 * 250 + 7], rows64[-1])  # time 0.25 lane 7; last row
    loaded = ct.CDLL(str(case.runtime / 'capsule.so'))
    loaded.run.argtypes = [ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_void_p)]
    loaded.run.restype = ct.c_int
    a32 = (ct.c_uint32 * 2)(*(row[0] for row in pick32))
    a64 = (ct.c_uint64 * 2)(*(row[0] for row in pick64))
    buffers = {'angles_f32': a32, 'angles_f64': a64}
    outputs = [(ct.c_uint32 * 2)() if name.endswith('f32') else (ct.c_uint64 * 2)() for name in case.outputs]
    argv_in = (ct.c_void_p * 2)(*(ct.addressof(buffers[name]) for name in case.inputs))
    argv_out = (ct.c_void_p * 4)(*(ct.addressof(buffer) for buffer in outputs))
    assert loaded.run(argv_in, argv_out) == 0
    expected = {'sin_f32': [r[1] for r in pick32], 'cos_f32': [r[2] for r in pick32],
                'sin_f64': [r[1] for r in pick64], 'cos_f64': [r[2] for r in pick64]}
    assert {name: list(buffer) for name, buffer in zip(case.outputs, outputs)} == expected
    a64[1] = a64[1] ^ 1  # one ULP away from a qualified angle
    assert loaded.run(argv_in, argv_out) != 0
    assert {member: (case.application / member).read_bytes() for member in case.before} == case.before
