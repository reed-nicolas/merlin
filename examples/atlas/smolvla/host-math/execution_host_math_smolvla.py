"""SmolVLA selection of validated native host unary math for execution packaging.

Same contract as whole-model/unchanged_fx/execution_host_math.py, generalized to the SmolVLA
archive (cos, cosf, sin, sinf) and to two declared domains: binary32 rotary angles and binary64
timestep-embedding angles. Component validation and text replacement are the shared helper.
"""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

import execution_host_math as base
from merlin.llvmlower.link_supplier_trace import trace_symbol_flags
from merlin.targetgen.host_linkage_contract import validate_linkage_contract

RV64_ARCHIVE = 'b97472d87fbe9dc1e755b7b86cb6e79cc73084e77d75118acae6dbc6f1ff54b9'
SYMBOLS = ['cos', 'cosf', 'sin', 'sinf']
OPERATIONS = {'source_f32_sin_unqualified': ('f32', 'sinf'), 'source_f32_cos_unqualified': ('f32', 'cosf'),
              'source_f64_sin_unqualified': ('f64', 'sin'), 'source_f64_cos_unqualified': ('f64', 'cos')}
GATE_SCHEMA = 'smolvla.execution-host-math-domain-check.v1'
ZERO_COUNTS = ('f32_angle_mismatch_count', 'f32_angle_input_mismatch_count', 'f64_angle_mismatch_count',
               'f64_angle_input_mismatch_count', 'mismatch_total')
ZERO_RESULTS = {'sinf': {'f32_mismatch_count': 0, 'bf16_mismatch_count': 0},
                'cosf': {'f32_mismatch_count': 0, 'bf16_mismatch_count': 0},
                'sin': {'f64_mismatch_count': 0, 'f32_mismatch_count_after_cast': 0},
                'cos': {'f64_mismatch_count': 0, 'f32_mismatch_count_after_cast': 0}}


def _native_source64(name, symbol, count, angles):
    """Binary64 counterpart of execution_host_math._native_source."""
    table = ','.join(f'0x{value:016x}ull' for value in angles)
    size = count * 8
    return f'''#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include <fenv.h>
#include <xmmintrin.h>
#if !defined(__x86_64__) || defined(__FAST_MATH__)
#error validation unary math requires x86_64 without fast-math
#endif
_Static_assert(CHAR_BIT==8 && FLT_RADIX==2 && DBL_MANT_DIG==53 && sizeof(double)==8 && SIZE_MAX>={size}ULL,"validation unary representation");
static const uint64_t {name}_angles[]={{{table}}};
static double (*volatile const {name}_call)(double)={symbol};
int {name}(void *context,const void *const *inputs,void *const *outputs) {{
  (void)context;
  if (!inputs || !outputs || !inputs[0] || !outputs[0]) return -1;
  uintptr_t a=(uintptr_t)inputs[0],b=(uintptr_t)outputs[0];
  if (a>UINTPTR_MAX-{size}u || b>UINTPTR_MAX-{size}u || (a<b+{size}u && b<a+{size}u)) return -1;
  if (fegetround()!=FE_TONEAREST || (_mm_getcsr() & ((3u<<13)|(1u<<15)|(1u<<6)))) return -3;
  for(size_t i=0;i<{count}u;++i) {{
    uint64_t bits; memcpy(&bits,(const unsigned char *)inputs[0]+8u*i,8);
    if ((bits&0x7fffffffffffffffull)>=0x7ff0000000000000ull) return -2;
    size_t lo=0,hi={len(angles)}u;
    while(lo<hi) {{ size_t mid=lo+(hi-lo)/2; if({name}_angles[mid]<bits) lo=mid+1; else hi=mid; }}
    if(lo=={len(angles)}u || {name}_angles[lo]!=bits) return -4;
  }}
  for(size_t i=0;i<{count}u;++i) {{
    double x,y; memcpy(&x,(const unsigned char *)inputs[0]+8u*i,8);
    y={name}_call(x); memcpy((unsigned char *)outputs[0]+8u*i,&y,8);
  }}
  return 0;
}}
'''


def _count(value):
    if type(value) is not int or value <= 0:
        raise ValueError('declared domain size is invalid')
    return value


def _selection(manifest, equivalence_receipt, archive, rv64_output):
    gate_bytes = equivalence_receipt.read_bytes()
    gate = json.loads(gate_bytes)
    native = validate_linkage_contract(gate['linkage'])
    native_archive_sha256 = native['archive_sha256']
    domain = gate.get('domain', {})
    rows32, rows64 = _count(domain.get('f32', {}).get('rows')), _count(domain.get('f64', {}).get('rows'))
    if (gate.get('schema') != GATE_SCHEMA or gate.get('gate') != 'PASS' or gate.get('rounding') != 'RNE'
            or native['symbols'] != SYMBOLS
            or gate.get('execution_host_archive_sha256') != native_archive_sha256
            or base._hash(archive.read_bytes()) != native_archive_sha256
            or gate.get('production_rv64_archive_sha256') != RV64_ARCHIVE
            or domain['f32'].get('compared_f32_trig_words') != 2 * rows32
            or domain['f64'].get('compared_f64_trig_words') != 2 * rows64
            or any(gate.get(key) != 0 for key in ZERO_COUNTS)
            or gate.get('results') != ZERO_RESULTS
            or gate.get('execution_host_output_sha256') != gate.get('rv64_output_sha256')
            or not gate.get('rv64_output_sha256')):
        raise ValueError('native host math equivalence gate or archive differs')
    data = rv64_output.read_bytes()
    if len(data) != rows32 * 12 + rows64 * 24 or base._hash(data) != gate['rv64_output_sha256']:
        raise ValueError('qualified angle domain size differs')
    f32 = np.frombuffer(data, dtype='<u4', count=rows32 * 3).reshape(rows32, 3)[:, 0].copy()
    f64 = np.frombuffer(data, dtype='<u8', offset=rows32 * 12).reshape(rows64, 3)[:, 0].copy()
    hashes = {'f32': base._hash(f32.tobytes()), 'f64': base._hash(f64.tobytes())}
    if any(gate.get('output_hashes', {}).get(engine, {}).get(width + '_angle_sha256') != hashes[width]
           for engine in ('rv64', 'execution_host') for width in hashes):
        raise ValueError('qualified angle domain hash differs')
    angles = {'f32': [int(v) for v in np.unique(f32)], 'f64': [int(v) for v in np.unique(f64)]}
    if (any((bits & 0x7fffffff) >= 0x7f800000 for bits in angles['f32'])
            or any((bits & 0x7fffffffffffffff) >= 0x7ff0000000000000 for bits in angles['f64'])):
        raise ValueError('qualified domain contains nonfinite angles')
    production_linkage = [validate_linkage_contract(value) for value in manifest['application_interface']['host_linkage']]
    if production_linkage != [{**native, 'archive_sha256': RV64_ARCHIVE}]:
        raise ValueError('production unary archive pin differs')
    return ({'production_linkage': production_linkage, 'execution_linkage': [native],
             'equivalence_receipt_sha256': sha256(gate_bytes).hexdigest(), 'qualified_angle_sha256': hashes,
             'qualified_distinct_encodings': {width: len(values) for width, values in angles.items()},
             'archive': str(archive.resolve()),
             'link_flags': ['-Wl,-Bsymbolic-functions', '-Wl,--exclude-libs,ALL', *trace_symbol_flags(SYMBOLS)]}, angles)


def validate_execution_host_math_selection(manifest, equivalence_receipt: Path, archive: Path, rv64_output: Path):
    """Authenticate the explicit selection before any runtime package is copied."""
    return _selection(manifest, equivalence_receipt, archive, rv64_output)[0]


def prepare_execution_host_math(manifest, production_package: Path, execution_package: Path,
                                equivalence_receipt: Path, archive: Path, rv64_output: Path):
    """Replace validated component text; return link/provenance data for this run."""
    if (execution_package.resolve().is_relative_to(production_package.resolve())
            or production_package.resolve().is_relative_to(execution_package.resolve())):
        raise ValueError('validation package must be separate from production')
    selection, angles = _selection(manifest, equivalence_receipt, archive, rv64_output)

    def native_source(name, symbol, count, dtype):
        if dtype == 'f32':
            return base._native_source(name, symbol, count, angles['f32'])
        return _native_source64(name, symbol, count, angles['f64'])

    source_name, original, selected, components, seen = base.replace_unary_components(
        manifest, production_package, selection['production_linkage'][0], OPERATIONS, native_source)
    if seen != set(SYMBOLS):
        raise ValueError('execution selection must cover every qualified unary symbol')
    destination = base._member(execution_package, source_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(selected.encode())
    return {**selection, 'scope': 'validation-only native host unary component text replacement', 'source': source_name,
            'production_source_sha256': base._hash(original), 'execution_source_sha256': base._hash(selected.encode()),
            'components': components}
