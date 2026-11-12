"""Build the x86-64 execution-host archive from the identical sources; never modify the RV64 supplier."""
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys

sys.dont_write_bytecode = True
from common import DEFINED, HERE, NATIVE_FLAGS, OUT, REVISION, SYMBOLS, WRAPPER, digest, run, symbols, vendored_sources

ARCHIVE = 'libcore-math-smolvla-execution-host.a'


def main():
    assert platform.machine() == 'x86_64', 'this execution archive requires x86_64'
    out = OUT / 'execution-host-math'
    out.mkdir(parents=True, exist_ok=True)
    log = out / 'commands.jsonl'
    log.unlink(missing_ok=True)
    production_receipt = OUT / 'core-math/build-receipt.json'
    pinned = json.loads(production_receipt.read_text())
    production = Path(pinned['archive_path'])
    if pinned['revision'] != REVISION or digest(production) != pinned['archive_sha256']:
        raise ValueError('RV64 supplier differs from its build receipt')
    wrapper = out / 'wrapper.c'
    wrapper.write_text(WRAPPER)
    sources = [*vendored_sources(), wrapper]
    for source in sources:
        if digest(source) != pinned['source_sha256'][source.name]:
            raise ValueError(f'execution source differs from the RV64 build: {source.name}')
    cc = shutil.which(os.environ.get('CC', 'cc'))
    assert cc, 'native compiler unavailable'
    cc = str(Path(cc).resolve())
    objects = []
    for source in sources:
        obj = out / (source.stem + '.o')
        run([cc, *NATIVE_FLAGS, '-c', source, '-o', obj], 'compile ' + source.name, log)
        objects.append(obj)
    archive = out / ARCHIVE
    archive.unlink(missing_ok=True)
    run(['ar', 'rcsD', archive, *objects], 'archive', log)
    nm = run(['nm', '-g', archive], 'archive symbols', log).stdout
    (out / 'symbols.txt').write_text(nm)
    if symbols(nm, 'T') != DEFINED:
        raise ValueError(f'execution archive defines {symbols(nm, "T")}')
    assembly = run(['objdump', '-d', archive], 'archive disassembly', log).stdout
    (out / 'disassembly.txt').write_text(assembly)
    fused = [line.strip() for line in assembly.splitlines() if re.search(r'\bv(?:f|fn)m(?:add|sub)\w*', line)]
    assert fused, 'explicit source FMA must remain fused'
    source = HERE / 'scripts/host_math_probe.c'
    executable = out / 'host_math_probe'
    link_flags = ['-fno-builtin', '-Wl,-Map=' + str(out / 'host_math_probe.map')]
    traced = SYMBOLS + ['cr_' + symbol for symbol in SYMBOLS]
    link_flags += ['-Wl,--trace-symbol=' + symbol for symbol in traced]
    linked = run([cc, *NATIVE_FLAGS, *link_flags, source, archive, '-lm', '-o', executable], 'link probe', log)
    trace = linked.stdout + linked.stderr
    (out / 'link-symbol-trace.txt').write_text(trace)
    for symbol in traced:
        if not re.search(re.escape(str(archive)) + r'\([^\n]+\): definition of ' + symbol + r'\b', trace):
            raise ValueError(f'probe does not take {symbol} from the execution archive')
    linked_nm = run(['nm', '-g', executable], 'linked symbols', log).stdout
    (out / 'linked-symbols.txt').write_text(linked_nm)
    for symbol in traced:
        assert re.search(r' T ' + symbol + r'$', linked_nm, re.MULTILINE), symbol
    dynamic = run(['nm', '-D', '--undefined-only', executable], 'dynamic imports', log).stdout
    (out / 'dynamic-imports.txt').write_text(dynamic)
    imported = {line.split()[-1].split('@')[0] for line in dynamic.splitlines() if line.strip()}
    if imported & set(traced):
        raise ValueError('probe must not import selected trig functions from a shared library')
    receipt = {'schema': 'smolvla.execution-host-math-build.v1', 'revision': REVISION, 'license': 'MIT',
               'isa': 'x86-64-v3', 'abi': 'sysv-x86_64',
               'compiler': run([cc, '--version'], 'compiler version', log).stdout.splitlines()[0],
               'compiler_path': cc, 'compiler_sha256': digest(cc), 'flags': NATIVE_FLAGS, 'link_flags': link_flags,
               'archive_path': str(archive), 'archive_sha256': digest(archive),
               'source_sha256': {p.name: digest(p) for p in sources}, 'source_paths': [str(p) for p in sources],
               'defined_symbols': symbols(nm, 'T'), 'undefined_symbols': symbols(nm, 'U'),
               'probe_dynamic_imports': sorted(imported),
               'fma': 'Implicit contraction disabled; explicit __builtin_fma/__builtin_fmaf lowered to native fused instructions.',
               'fused_instruction_count': len(fused), 'fused_instructions': fused,
               'link_symbol_trace_sha256': digest(out / 'link-symbol-trace.txt'),
               'link_map_sha256': digest(out / 'host_math_probe.map'),
               'executable_sha256': digest(executable), 'probe_source_sha256': digest(source),
               'production_archive_path': str(production), 'production_archive_sha256': digest(production),
               'production_build_receipt_sha256': digest(production_receipt),
               'linkage': {'schema': 'merlin.host_linkage_contract.v1', 'supplier': 'selected_math_archive',
                           'archive_sha256': digest(archive), 'symbols': SYMBOLS},
               'thread_environment': {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
               'nice': 10}
    (out / 'build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (out / 'host-linkage-contract.json').write_text(json.dumps(receipt['linkage'], indent=2) + '\n')
    print(json.dumps({key: receipt[key] for key in ('archive_sha256', 'defined_symbols', 'undefined_symbols',
                                                     'probe_dynamic_imports', 'fused_instruction_count')}, indent=2))


if __name__ == '__main__':
    main()
