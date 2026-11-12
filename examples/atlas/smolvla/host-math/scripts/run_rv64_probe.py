"""Link the declared-domain probe to the RV64 archive and run it under the existing Spike/pk."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
from common import ENV, HERE, OUT, SYMBOLS, digest, run


def main():
    riscv = Path(os.environ['RISCV'])
    build = json.loads((OUT / 'core-math/build-receipt.json').read_text())
    archive = Path(build['archive_path'])
    if digest(archive) != build['archive_sha256']:
        raise ValueError('RV64 archive differs from its build receipt')
    domain = OUT / 'domain'
    record = json.loads((domain / 'domain.json').read_text())
    for name, expected in record['files'].items():
        if digest(domain / name) != expected:
            raise ValueError(f'domain input changed: {name}')
    positions = record['f32_rotary']['positions'][1] + 1
    out = OUT / 'rv64'
    out.mkdir(parents=True, exist_ok=True)
    log = out / 'commands.jsonl'
    log.unlink(missing_ok=True)
    source = HERE / 'scripts/host_math_probe.c'
    probe = out / 'host_math_probe'
    trace = [f'-Wl,--trace-symbol={symbol}' for symbol in SYMBOLS]
    command = [riscv / 'bin/riscv64-unknown-elf-gcc', '-march=rv64gc', '-mabi=lp64d', '-O2', '-fno-fast-math',
               '-ffp-contract=off', '-fno-builtin', source, archive, '-lm', '-Wl,-Map,' + str(out / 'host_math_probe.map'),
               *trace, '-o', probe]
    linked = run(command, 'link RV64 probe', log)
    (out / 'link-symbol-trace.txt').write_text(linked.stdout + linked.stderr)
    for symbol in SYMBOLS:
        if not re.search(re.escape(str(archive)) + r'\(wrapper\.o\): definition of ' + symbol + r'\b', linked.stdout + linked.stderr):
            raise ValueError(f'RV64 probe does not take {symbol} from the selected archive')
    spike, pk = riscv / 'bin/spike', riscv / 'riscv64-unknown-elf/bin/pk'
    environment = dict(ENV, PATH=f'{riscv}/../bin:' + ENV['PATH'],
                       LD_LIBRARY_PATH=f'{riscv}/../lib' + (':' + ENV['LD_LIBRARY_PATH'] if ENV.get('LD_LIBRARY_PATH') else ''))
    output = OUT / 'rv64-host-math.bin'
    arguments = ['nice', '-n', '10', str(spike), '--isa=rv64gc', str(pk), str(probe), str(positions),
                 *(str(domain / name) for name in ('timescales_f32.bin', 'times_f32.bin', 'scaling_f64.bin'))]
    with output.open('wb') as handle, (out / 'spike.log').open('wb') as errors:
        result = subprocess.run(arguments, stdout=handle, stderr=errors, env=environment)
    if result.returncode:
        raise RuntimeError(f'RV64 probe failed with {result.returncode}; see {out / "spike.log"}')
    rows32 = record['f32_rotary']['rows']
    rows64 = record['f64_timestep']['rows']
    if output.stat().st_size != rows32 * 12 + rows64 * 24:
        raise ValueError('RV64 probe output size differs from the declared domain')
    receipt = {'schema': 'smolvla.rv64-host-math-run.v1', 'archive_path': str(archive), 'archive_sha256': digest(archive),
               'build_receipt_sha256': digest(OUT / 'core-math/build-receipt.json'), 'link_command': [str(c) for c in command],
               'probe_source_sha256': digest(source), 'probe_sha256': digest(probe),
               'link_map_sha256': digest(out / 'host_math_probe.map'),
               'link_symbol_trace_sha256': digest(out / 'link-symbol-trace.txt'),
               'emulator': str(spike), 'emulator_sha256': digest(spike), 'proxy_kernel': str(pk), 'proxy_kernel_sha256': digest(pk),
               'run_command': arguments, 'domain_sha256': digest(domain / 'domain.json'), 'positions': positions,
               'rows_f32': rows32, 'rows_f64': rows64, 'output_path': str(output), 'output_sha256': digest(output)}
    (out / 'run-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({key: receipt[key] for key in ('archive_sha256', 'output_sha256', 'rows_f32', 'rows_f64')}, indent=2))


if __name__ == '__main__':
    main()
