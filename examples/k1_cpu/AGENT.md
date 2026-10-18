# Integer egglog diagnostic example

This handwritten example demonstrates a provider edge using shared Merlin egglog and measured-selection APIs. It is a scalar host functional diagnostic for the signed-i8 ReLU semantics admitted by the selected CPU contract, not a K1 compiler backend, board runner, RVV implementation or device performance qualification.

Keep target names, exact integer premises, code emission and execution checks in this example. Shared graph/extraction/selection mechanisms stay in `src/merlin/semantic_compiler/` without target names or arithmetic rules. Do not add runtime fallbacks or reinterpret host measurements as device timing.

Canonical metadata remains in `merlin/targets/k1_cpu/contracts/`. Generated receipts belong in the explicitly chosen artifact destination, not this directory. Tests belong in `merlin/tests/ir/` and resolve paths through `merlin.common.paths`.
