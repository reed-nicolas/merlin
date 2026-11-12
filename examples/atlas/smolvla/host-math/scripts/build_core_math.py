"""Build the RV64GC/lp64d selected archive from the four reviewed CORE-MATH sources and our wrapper."""
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from common import DEFINED, OUT, REVISION, RV64_FLAGS, WRAPPER, digest, run, symbols, vendored_sources


def main():
    riscv = Path(os.environ['RISCV'])
    tool = lambda name: str(riscv / 'bin' / ('riscv64-unknown-elf-' + name))
    out = OUT / 'core-math'
    out.mkdir(parents=True, exist_ok=True)
    log = out / 'commands.jsonl'
    log.unlink(missing_ok=True)
    wrapper = out / 'wrapper.c'
    wrapper.write_text(WRAPPER)
    inputs = [*vendored_sources(), wrapper]
    objects = []
    for source in inputs:
        obj = out / (source.stem + '.o')
        run([tool('gcc'), *RV64_FLAGS, '-c', source, '-o', obj], 'compile ' + source.name, log)
        objects.append(obj)
    archive = out / 'libcore-math-smolvla.a'
    archive.unlink(missing_ok=True)  # fresh deterministic archive, no stale members
    run([tool('ar'), 'rcsD', archive, *objects], 'archive', log)
    nm = run([tool('nm'), '-g', archive], 'archive symbols', log).stdout
    (out / 'symbols.txt').write_text(nm)
    assembly = run([tool('objdump'), '-d', archive], 'archive disassembly', log).stdout
    (out / 'disassembly.txt').write_text(assembly)
    defined = symbols(nm, 'T')
    if defined != DEFINED:
        raise ValueError(f'archive defines {defined}, expected {DEFINED}')
    fused = [line.strip() for line in assembly.splitlines() if any(op in line for op in ['fmadd.', 'fmsub.', 'fnmadd.', 'fnmsub.'])]
    receipt = {'schema': 'smolvla.core-math-build.v1', 'revision': REVISION, 'license': 'MIT',
               'isa': 'rv64gc', 'abi': 'lp64d', 'flags': RV64_FLAGS,
               'compiler': run([tool('gcc'), '--version'], 'compiler version', log).stdout.splitlines()[0],
               'compiler_path': tool('gcc'), 'compiler_sha256': digest(tool('gcc')),
               'source_sha256': {p.name: digest(p) for p in inputs}, 'source_paths': [str(p) for p in inputs],
               'object_sha256': {p.name: digest(p) for p in objects},
               'archive_path': str(archive), 'archive_sha256': digest(archive),
               'defined_symbols': defined, 'undefined_symbols': symbols(nm, 'U'),
               'linkage': {'schema': 'merlin.host_linkage_contract.v1', 'supplier': 'selected_math_archive',
                           'archive_sha256': digest(archive), 'symbols': ['cos', 'cosf', 'sin', 'sinf']},
               'fma': 'Implicit contraction off; explicit upstream __builtin_fma/__builtin_fmaf preserved as fused RV64 instructions.',
               'fused_instruction_count': len(fused), 'fused_instructions': fused,
               'disassembly_sha256': digest(out / 'disassembly.txt'), 'nice': 10}
    (out / 'build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({key: receipt[key] for key in ('archive_sha256', 'defined_symbols', 'undefined_symbols', 'fused_instruction_count')}, indent=2))


if __name__ == '__main__':
    main()
