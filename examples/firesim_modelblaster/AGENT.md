# FireSim ModelBlaster runner example

This independently installable package is an out-of-tree FireSim execution
adapter. It is not part of the Merlin core wheel and must be selected explicitly
with `MERLIN_FIRESIM_RUNNER=modelblaster` or `runner_name="modelblaster"`.

Keep ModelBlaster checkout discovery, workload/project identity, queue policy,
and simulator imports here. Do not put executable support in
`examples/saturn/target/`, which is reference metadata only. Never claim an
FPGA result from the adapter's import or preflight alone.
