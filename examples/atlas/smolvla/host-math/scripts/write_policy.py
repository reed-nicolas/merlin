"""Write the host_unary_math precision-policy fragment for the gated RV64 archive."""
import json
import sys

sys.dont_write_bytecode = True
from common import HERE, OUT, SYMBOLS, digest


def main():
    from merlin.compile.c_unary_math import validate_policy
    sys.path.insert(0, str(HERE))
    from execution_host_math_smolvla import RV64_ARCHIVE
    build = json.loads((OUT / 'core-math/build-receipt.json').read_text())
    gates = [json.loads((OUT / name).read_text()) for name in ('execution-host-math/equivalence-receipt.json', 'mpfr-reference-receipt.json')]
    if (any(gate['gate'] != 'PASS' for gate in gates)
            or gates[0]['production_rv64_archive_sha256'] != RV64_ARCHIVE
            or not digest(build['archive_path']) == build['archive_sha256'] == RV64_ARCHIVE):
        raise ValueError('policy needs the passing gates for the pinned RV64 archive')
    policy = validate_policy({
        'schema': 'merlin.compiled_unary_math.v1', 'policy': 'selected-libm-rne-finite-v1', 'isa': 'rv64gc', 'abi': 'lp64d',
        'effects': 'enabled-fpu-rne-unobserved-flags-and-errno-v1',
        'operations': ['source_f32_cos_unqualified', 'source_f32_sin_unqualified',
                       'source_f64_cos_unqualified', 'source_f64_sin_unqualified'],
        'linkage': {'schema': 'merlin.host_linkage_contract.v1', 'supplier': 'selected_math_archive',
                    'archive_sha256': RV64_ARCHIVE, 'symbols': SYMBOLS}})
    text = json.dumps(policy, indent=2) + '\n'
    (OUT / 'host-unary-math-policy.json').write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
