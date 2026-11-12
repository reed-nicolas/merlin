"""Fail closed unless x86-64 selected-archive trig is bitwise equal to RV64 over the declared domains."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.dont_write_bytecode = True
from common import ENV, OUT, SYMBOLS, digest


def bf16(bits):
    bits = bits.astype(np.uint64)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def sections(data, rows32, rows64):
    if len(data) != rows32 * 12 + rows64 * 24:
        raise ValueError('probe output size differs from the declared domain')
    f32 = np.frombuffer(data, dtype='<u4', count=rows32 * 3).reshape(rows32, 3)
    f64 = np.frombuffer(data, dtype='<u8', offset=rows32 * 12).reshape(rows64, 3)
    return f32, f64


def sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def declared_angles(domain, record):
    positions = record['f32_rotary']['positions'][1] + 1
    timescale = np.fromfile(domain / 'timescales_f32.bin', dtype='<f4')
    times = np.fromfile(domain / 'times_f32.bin', dtype='<f4').astype(np.float64)
    scaling = np.fromfile(domain / 'scaling_f64.bin', dtype='<f8').reshape(-1, 360)
    with np.errstate(all='raise'):
        angles32 = np.arange(positions, dtype=np.float32)[:, None] / timescale[None, :]
        angles64 = scaling[:, None, :] * times[None, :, None]
    return angles32.astype('<f4').view('<u4').ravel(), angles64.astype('<f8').view('<u8').ravel()


def main():
    build = json.loads((OUT / 'execution-host-math/build-receipt.json').read_text())
    production_build = json.loads((OUT / 'core-math/build-receipt.json').read_text())
    run_receipt = json.loads((OUT / 'rv64/run-receipt.json').read_text())
    archive, production = Path(build['archive_path']), Path(production_build['archive_path'])
    executable = OUT / 'execution-host-math/host_math_probe'
    if (not digest(archive) == build['archive_sha256'] == build['linkage']['archive_sha256']
            or not digest(production) == production_build['archive_sha256'] == build['production_archive_sha256']
            or run_receipt['archive_sha256'] != production_build['archive_sha256']
            or digest(executable) != build['executable_sha256'] or build['linkage']['symbols'] != SYMBOLS):
        raise ValueError('execution-host build evidence differs')
    domain = OUT / 'domain'
    record = json.loads((domain / 'domain.json').read_text())
    if digest(domain / 'domain.json') != run_receipt['domain_sha256']:
        raise ValueError('RV64 run used a different domain')
    for name, expected in record['files'].items():
        if digest(domain / name) != expected:
            raise ValueError(f'domain input changed: {name}')
    rv_path = OUT / 'rv64-host-math.bin'
    if digest(rv_path) != run_receipt['output_sha256']:
        raise ValueError('RV64 output differs from its run receipt')
    native_path = OUT / 'execution-host-math/execution-host-math.bin'
    command = ['nice', '-n', '10', str(executable), str(run_receipt['positions']),
               *(str(domain / name) for name in ('timescales_f32.bin', 'times_f32.bin', 'scaling_f64.bin'))]
    with native_path.open('wb') as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.PIPE, env=ENV)
    with (OUT / 'execution-host-math/commands.jsonl').open('a') as log:
        log.write(json.dumps({'label': 'execute domain probe', 'command': command, 'returncode': result.returncode,
                              'stderr': result.stderr.decode()}) + '\n')
    if result.returncode:
        raise RuntimeError(f'execution-host probe failed: {result.returncode}')
    rows32, rows64 = record['f32_rotary']['rows'], record['f64_timestep']['rows']
    rv32, rv64 = sections(rv_path.read_bytes(), rows32, rows64)
    nat32, nat64 = sections(native_path.read_bytes(), rows32, rows64)
    expected32, expected64 = declared_angles(domain, record)
    counts = {
        'f32_angle_mismatch_count': int(np.count_nonzero(rv32[:, 0] != nat32[:, 0])),
        'f32_angle_input_mismatch_count': int(np.count_nonzero((rv32[:, 0] != expected32) | (nat32[:, 0] != expected32))),
        'f64_angle_mismatch_count': int(np.count_nonzero(rv64[:, 0] != nat64[:, 0])),
        'f64_angle_input_mismatch_count': int(np.count_nonzero((rv64[:, 0] != expected64) | (nat64[:, 0] != expected64))),
    }
    results = {}
    examples = []
    for name, lane in (('sin', 1), ('cos', 2)):
        f32_diff = rv32[:, lane] != nat32[:, lane]
        f64_diff = rv64[:, lane] != nat64[:, lane]
        # The graph casts the binary64 embedding to binary32 (RNE) after cat; rotary f32 feeds f32 multiplies directly.
        rv_cast = rv64[:, lane].view('<f8').astype('<f4').view('<u4')
        nat_cast = nat64[:, lane].view('<f8').astype('<f4').view('<u4')
        results[name + 'f'] = {'f32_mismatch_count': int(np.count_nonzero(f32_diff)),
                               'bf16_mismatch_count': int(np.count_nonzero(bf16(rv32[:, lane]) != bf16(nat32[:, lane])))}
        results[name] = {'f64_mismatch_count': int(np.count_nonzero(f64_diff)),
                         'f32_mismatch_count_after_cast': int(np.count_nonzero(rv_cast != nat_cast))}
        for index in np.nonzero(f32_diff)[0][:8]:
            examples.append({'function': name + 'f', 'row': int(index), 'angle': hex(int(rv32[index, 0])),
                             'rv64': hex(int(rv32[index, lane])), 'execution_host': hex(int(nat32[index, lane]))})
        for index in np.nonzero(f64_diff)[0][:8]:
            examples.append({'function': name, 'row': int(index), 'angle': hex(int(rv64[index, 0])),
                             'rv64': hex(int(rv64[index, lane])), 'execution_host': hex(int(nat64[index, lane]))})
    output_hashes = {engine: {'f32_angle_sha256': sha(f32[:, 0]), 'f64_angle_sha256': sha(f64[:, 0]),
                              'f32_rows_sha256': sha(f32), 'f64_rows_sha256': sha(f64)}
                     for engine, (f32, f64) in (('rv64', (rv32, rv64)), ('execution_host', (nat32, nat64)))}
    total = sum(counts.values()) + sum(sum(v.values()) for v in results.values())
    failed = bool(total) or digest(native_path) != digest(rv_path)
    receipt = {'schema': 'smolvla.execution-host-math-domain-check.v1', 'gate': 'FAIL' if failed else 'PASS',
               'qualification': ('x86-64 selected-archive sinf/cosf/sin/cos bitwise equal to RV64 over the declared domains.'
                                 if not failed else 'execution-host trig equivalence failed.'),
               'scope': 'functional validation only; no deployment qualification',
               'core_math_revision': build['revision'], 'rounding': 'RNE',
               'execution_host_archive_path': str(archive), 'execution_host_archive_sha256': digest(archive),
               'production_rv64_archive_path': str(production), 'production_rv64_archive_sha256': digest(production),
               'linkage': build['linkage'], 'build_receipt_sha256': digest(OUT / 'execution-host-math/build-receipt.json'),
               'rv64_run_receipt_sha256': digest(OUT / 'rv64/run-receipt.json'),
               'domain_sha256': digest(domain / 'domain.json'),
               'execution_host_output_sha256': digest(native_path), 'rv64_output_sha256': digest(rv_path),
               'domain': {'f32': {'positions': record['f32_rotary']['positions'], 'timescales': 32, 'rows': rows32,
                                  'distinct_angles': int(np.unique(rv32[:, 0]).size), 'compared_f32_trig_words': 2 * rows32},
                          'f64': {'times': record['f64_timestep']['times']['count'],
                                  'scaling_vectors': record['f64_timestep']['scaling_vectors'], 'rows': rows64,
                                  'distinct_angles': int(np.unique(rv64[:, 0]).size), 'compared_f64_trig_words': 2 * rows64}},
               **counts, 'results': results, 'first_mismatches': examples, 'output_hashes': output_hashes,
               'mismatch_total': total, 'probe_sha256': digest(executable)}
    (OUT / 'execution-host-math/equivalence-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({key: receipt[key] for key in ('gate', 'domain', 'results', 'mismatch_total', *counts)}, indent=2))
    if failed:
        raise SystemExit('execution-host selected-archive equivalence gate failed')


if __name__ == '__main__':
    main()
