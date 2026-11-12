"""Write the declared SmolVLA host trig input domains from the captured source arithmetic.

binary32 rotary: radians = f32(position) / timescale[j] (division, as apply_rope and the
captured graph compute it), timescale = pow(10000, f32(j) * 0.03125) for j in 0..31 and
every integer position 0..POSITIONS-1.

binary64 timestep embedding: angle = scaling[k] * f64(time) for k in 0..359 and every
declared binary32 time, with scaling = (1 / (pow(1000, linspace(0, 1, 360)) * 0.004)) * 1.0 * 2 * pi.
Two scaling vectors are declared: torch eager source arithmetic, and the captured MLIR chain
(decomposed linspace, constant pow folded correctly rounded by Merlin's MPFR helper).
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

POSITIONS = 512
TIME_GRID = 1000
FIXTURE_TIMES = (0.25, 0.75)
HERE = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('HOST_MATH_OUT', HERE / 'out')).resolve()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def f32_bits(values):
    return np.asarray(values, dtype='<f4').view('<u4')


def f64_bits(values):
    return np.asarray(values, dtype='<f8').view('<u8')


def timescales():
    exponent = torch.arange(32, dtype=torch.float32) * 0.03125
    source = torch.pow(10000, exponent)
    assert source.dtype == torch.float32
    return exponent.numpy(), source.numpy()


def time_values():
    grid = torch.tensor([k / TIME_GRID for k in range(TIME_GRID + 1)], dtype=torch.float32).numpy()
    for value in FIXTURE_TIMES:
        assert np.float32(value) in grid, 'fixture time must be in the declared grid'
    return grid


def scaling_tail(power):
    # mul 0.004, reciprocal, mul 1.0, mul 2, mul pi; each a separate IEEE binary64 operation.
    with np.errstate(all='raise'):
        value = power * np.float64(0.004)
        value = np.float64(1.0) / value
        value = value * np.float64(1.0)
        value = value * np.float64(2.0)
        return value * np.float64(3.141592653589793)


def scalings(helper):
    eager_linspace = torch.linspace(0.0, 1.0, 360, dtype=torch.float64)
    eager = torch.pow(1000.0, eager_linspace)
    eager = torch.reciprocal(eager * 0.004) * 1.0 * 2 * 3.141592653589793
    # Captured MLIR: rg < 180 ? 0.0 + step*f64(rg) : 1.0 - step*f64(359 - rg).
    step = np.float64(0.0027855153203342618)
    assert step == np.float64(1.0) / np.float64(359)
    rg = np.arange(360, dtype=np.int64)
    lower = step * rg.astype(np.float64) + np.float64(0.0)
    upper = np.float64(1.0) - step * (np.int64(359) - rg).astype(np.float64)
    decomposed = np.where(rg < 180, lower, upper)
    import torch._refs as refs
    assert np.array_equal(f64_bits(decomposed), f64_bits(refs.linspace(0.0, 1.0, 360, dtype=torch.float64).numpy()))
    base = int(f64_bits([1000.0])[0])
    power = np.array([helper.power(64, base, int(bits), 1024)[0] for bits in f64_bits(decomposed)], dtype='<u8')
    compiled = scaling_tail(power.view('<f8'))
    eager_tail = scaling_tail(torch.pow(1000.0, eager_linspace).numpy())
    assert np.array_equal(f64_bits(eager_tail), f64_bits(eager.numpy())), 'numpy tail differs from torch'
    return {'torch_eager': eager.numpy(), 'captured_mlir_correctly_rounded_pow': compiled}, \
        {'eager_linspace': eager_linspace.numpy(), 'decomposed_linspace': decomposed}


def main():
    from merlin.semantic_compiler.constant_math_backend import MathHelper
    helper = MathHelper(os.environ['MERLIN_CONSTANT_MATH_HELPER'])
    out = OUT / 'domain'
    out.mkdir(parents=True, exist_ok=True)
    exponent, timescale = timescales()
    base32 = int(f32_bits([10000.0])[0])
    correctly_rounded = [helper.power(32, base32, int(bits), 1024)[0] for bits in f32_bits(exponent)]
    timescale_mismatch = [j for j in range(32) if correctly_rounded[j] != int(f32_bits(timescale)[j])]
    if timescale_mismatch:
        raise SystemExit(f'f32 timescale torch and correctly rounded pow differ at {timescale_mismatch}')
    times = time_values()
    vectors, linspaces = scalings(helper)
    names = sorted(vectors)
    stacked = np.stack([vectors[name] for name in names]).astype('<f8')
    files = {'timescales_f32.bin': timescale.astype('<f4').tobytes(), 'times_f32.bin': times.astype('<f4').tobytes(),
             'scaling_f64.bin': stacked.tobytes()}
    for name, data in files.items():
        (out / name).write_bytes(data)
    positions = np.arange(POSITIONS, dtype=np.float32)
    with np.errstate(all='raise'):
        angles32 = positions[:, None] / timescale[None, :]
        angles64 = stacked[:, None, :] * times.astype(np.float64)[None, :, None]
    distinct32 = np.unique(f32_bits(angles32.ravel()))
    distinct64 = np.unique(f64_bits(angles64.ravel()))
    differing = [int(k) for k in np.nonzero(f64_bits(vectors[names[0]]) != f64_bits(vectors[names[1]]))[0]]
    record = {
        'schema': 'smolvla.host-math-domain.v1',
        'f32_rotary': {
            'arithmetic': 'f32(position) / pow(10000, f32(j) * 0.03125) in binary32 RNE; division as in lerobot apply_rope and captured aten.div',
            'positions': [0, POSITIONS - 1], 'timescales': 32,
            'timescale_hex': [hex(int(bits)) for bits in f32_bits(timescale)],
            'timescale_torch_equals_correctly_rounded_pow': True,
            'rows': int(angles32.size), 'distinct_angles': int(distinct32.size),
            'distinct_angle_sha256': digest(distinct32.astype('<u4').tobytes()),
            'bundle_positions_covered': [0, 162]},
        'f64_timestep': {
            'arithmetic': 'scaling[k] * f64(time) in binary64 RNE; scaling = (1 / (pow(1000, linspace(0, 1, 360)) * 0.004)) * 1.0 * 2 * pi',
            'times': {'definition': f'binary32 nearest of binary64 k/{TIME_GRID} for k in 0..{TIME_GRID}', 'count': int(times.size),
                      'fixture_times_included': list(FIXTURE_TIMES)},
            'scaling_vectors': names,
            'scaling_vector_note': ('torch_eager is the source framework value; captured_mlir_correctly_rounded_pow '
                                    'follows the captured MLIR decomposed linspace with pow folded correctly rounded'),
            'scaling_lanes_differing_between_vectors': differing,
            'scaling_hex': {name: [hex(int(bits)) for bits in f64_bits(vectors[name])] for name in names},
            'linspace_lanes_differing': [int(k) for k in np.nonzero(f64_bits(linspaces['eager_linspace']) != f64_bits(linspaces['decomposed_linspace']))[0]],
            'rows': int(angles64.size), 'distinct_angles': int(distinct64.size),
            'distinct_angle_sha256': digest(distinct64.astype('<u8').tobytes())},
        'files': {name: digest(data) for name, data in files.items()},
        'constant_math_helper': helper.identity, 'torch_version': torch.__version__, 'numpy_version': np.__version__}
    (out / 'domain.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({key: record[key] for key in ('files',)}, indent=2))
    print(json.dumps({'f32_distinct': record['f32_rotary']['distinct_angles'], 'f64_distinct': record['f64_timestep']['distinct_angles'],
                      'scaling_lanes_differing': differing}))


if __name__ == '__main__':
    main()
