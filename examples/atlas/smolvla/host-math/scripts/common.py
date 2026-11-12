"""Shared pins and helpers for the SmolVLA selected host math build and checks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('HOST_MATH_OUT', HERE / 'out')).resolve()
VENDOR = HERE.parents[1] / 'rotary-host/third_party/core-math'
REVISION = 'a055b9b9a86b857e49573e339cf68306ad92c831'
SOURCES = ('sinf.c', 'cosf.c', 'sin.c', 'cos.c')
SOURCE_SHA256 = {
    'sinf.c': 'f27c4baa849b1d9982153c69719c912a3058bc9d86d7d01e7385e2aee56c4278',
    'cosf.c': '3ea4c6d03d32e225a8fd31ee37eb785cefda0f8940c4e826352fc9103a275dcf',
    'sin.c': 'c8a94aff114e7c2fa314b4f74575a0c2dfcdb8fb116c48a6244ab914304fb905',
    'cos.c': 'a64b819dc4e1b7da24dbe0f3076b389bd6f67d671f9c0708a13d89d6f888f259',
}
SYMBOLS = ['cos', 'cosf', 'sin', 'sinf']
DEFINED = sorted(SYMBOLS + ['cr_' + symbol for symbol in SYMBOLS])
WRAPPER = ('extern float cr_sinf(float);\nextern float cr_cosf(float);\n'
           'extern double cr_sin(double);\nextern double cr_cos(double);\n'
           'float sinf(float x) { return cr_sinf(x); }\nfloat cosf(float x) { return cr_cosf(x); }\n'
           'double sin(double x) { return cr_sin(x); }\ndouble cos(double x) { return cr_cos(x); }\n')
STRICT = ['-O2', '-std=c11', '-fno-fast-math', '-frounding-math', '-ffp-contract=off',
          '-fno-strict-aliasing', '-fno-math-errno']
RV64_FLAGS = ['-march=rv64gc', '-mabi=lp64d', *STRICT]
NATIVE_FLAGS = ['-march=x86-64-v3', '-m64', '-fPIC', *STRICT]
ENV = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vendored_sources():
    """Return the four unmodified upstream sources after checking revision and bytes."""
    if (VENDOR / 'REVISION').read_text().strip() != REVISION:
        raise ValueError('CORE-MATH revision marker differs')
    paths = [VENDOR / name for name in SOURCES]
    for path in paths:
        if digest(path) != SOURCE_SHA256[path.name]:
            raise ValueError(f'vendored CORE-MATH source changed: {path.name}')
    return paths


def run(command, label, log, **kwargs):
    actual = ['nice', '-n', '10', *map(str, command)]
    result = subprocess.run(actual, capture_output=True, text=True, env=ENV, **kwargs)
    with Path(log).open('a') as handle:
        handle.write(json.dumps({'label': label, 'command': actual, 'returncode': result.returncode,
                                 'stdout': result.stdout[-4000:], 'stderr': result.stderr[-4000:]}) + '\n')
    if result.returncode:
        raise RuntimeError(f'{label} failed: {result.stderr}')
    return result


def symbols(nm_text, kind):
    return sorted({line.split()[-1] for line in nm_text.splitlines() if f' {kind} ' in line})
