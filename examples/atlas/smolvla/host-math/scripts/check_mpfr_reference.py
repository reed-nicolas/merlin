"""Independently check every distinct declared-domain result against correctly rounded MPFR sin/cos."""
import json
import os
import sys

import numpy as np
import torch

sys.dont_write_bytecode = True
from check_execution_host_math import sections
from common import OUT, digest


def main():
    from merlin.semantic_compiler.constant_math_backend import MathHelper
    torch.set_num_threads(1)
    helper = MathHelper(os.environ['MERLIN_CONSTANT_MATH_HELPER'])
    gate = json.loads((OUT / 'execution-host-math/equivalence-receipt.json').read_text())
    rv_path = OUT / 'rv64-host-math.bin'
    if gate['gate'] != 'PASS' or digest(rv_path) != gate['rv64_output_sha256']:
        raise ValueError('run the passing equivalence gate first')
    rv32, rv64 = sections(rv_path.read_bytes(), gate['domain']['f32']['rows'], gate['domain']['f64']['rows'])
    report = {'schema': 'smolvla.host-math-mpfr-reference.v1', 'reference': 'MPFR correctly rounded RNE via Merlin MathHelper.unary_reference',
              'helper': helper.identity, 'rv64_output_sha256': gate['rv64_output_sha256'],
              'equivalence_receipt_sha256': digest(OUT / 'execution-host-math/equivalence-receipt.json'),
              'torch_version': torch.__version__, 'results': {}}
    failed = False
    for width, rows, dtype, utype, suffix in ((32, rv32, '<f4', '<u4', 'f'), (64, rv64, '<f8', '<u8', '')):
        angles, first = np.unique(rows[:, 0], return_index=True)
        for name, lane in (('sin', 1), ('cos', 2)):
            values = rows[first, lane]
            # Every row of an angle must carry the same result before one representative is checked.
            lookup = dict(zip(angles.tolist(), values.tolist()))
            consistent = all(lookup[a] == v for a, v in zip(rows[:, 0].tolist(), rows[:, lane].tolist()))
            precision, mismatches, count = {}, [], 0
            for angle, value in zip(angles.tolist(), values.tolist()):
                reference, bits = helper.unary_reference(width, int(angle), name)
                precision[bits] = precision.get(bits, 0) + 1
                if reference != value:
                    count += 1
                    if len(mismatches) < 16:
                        mismatches.append({'angle': hex(angle), 'archive': hex(value), 'mpfr': hex(reference)})
            source = getattr(torch, name)(torch.from_numpy(angles.astype(utype).view(dtype).copy())).numpy().view(utype)
            entry = {'distinct_angles': int(angles.size), 'mpfr_mismatch_count': count, 'rows_consistent': consistent,
                     'mpfr_precision_bits_histogram': {str(k): v for k, v in sorted(precision.items())},
                     'first_mismatches': mismatches,
                     'torch_cpu_mismatch_count_informational': int(np.count_nonzero(source != values))}
            if width == 64:
                cast = values.view('<f8').astype('<f4').view('<u4')
                torch_cast = source.view('<f8').astype('<f4').view('<u4')
                entry['torch_cpu_mismatch_count_after_f32_cast_informational'] = int(np.count_nonzero(torch_cast != cast))
            report['results'][name + suffix] = entry
            failed |= bool(count) or not consistent
    report['gate'] = 'FAIL' if failed else 'PASS'
    (OUT / 'mpfr-reference-receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if failed:
        raise SystemExit('archive results differ from correctly rounded MPFR')


if __name__ == '__main__':
    main()
