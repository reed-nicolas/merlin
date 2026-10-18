# Signed-i8 egglog diagnostic

`egglog_relu.py` demonstrates a second provider using the shared optional Python egglog core and measured selector. Its scalar semantics match the canonical CPU contract's int8 elementwise/ReLU capability. It does not emit K1 instructions, run the board, execute RVV or estimate device timing.

Supply a JSON snapshot with `contract` containing the parsed canonical `merlin/targets/k1_cpu/contracts/target_contract.yaml` and `source_sha256` containing that YAML's raw SHA-256. With the optional `egglog` dependency available, launch from the repository root:

```sh
PYTHONPATH=src python examples/k1_cpu/egglog_relu.py --contract-json /absolute/path/contract-snapshot.json
```

The command writes a JSON receipt to stdout. Redirect it to your selected artifact destination. It records exactness evidence, source/tool/contract hashes, actual shared-egglog saturation, native/hand/egglog execution checks, guards, repeated primitive counts and their evidence tier. The count metric is `relu_evaluations`, tier `host_functional_cost`; a reduction in primitive calls is not a measured K1 speedup. Hand and egglog can have identical emitted code and are reported as tied controls.

The integer proof is exhaustive over all 256 i8 encodings and extends to arbitrary positive repetition by idempotence. No floating-point rule or caller contract is inferred. The provider refuses unknown operations/types, checks its selected semantic capability, and aborts on output/oracle/guard failure. The shared mechanism and target boundary are described in [the design note](../../docs/design/guarded_egraph_selection.md).
