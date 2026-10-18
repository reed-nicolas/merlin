"""Exact signed-i8 ReLU provider edge; host functional cost, never K1 timing.

Input contract JSON is an explicitly materialized snapshot of the canonical
target YAML plus its raw SHA-256. No board, vector length or latency is guessed.
The shared Merlin modules own actual saturation/extraction and receipt ranking.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

from egglog import rewrite, ruleset, vars_
from egglog.deconstruct import get_callable_args
from merlin.semantic_compiler import egraph_terms
from merlin.semantic_compiler import rewrite_selection as selection


@dataclass(frozen=True)
class Node:
    op: str
    operands: tuple[Node, ...] = ()
    attrs: tuple[tuple[str, str], ...] = ()
    type: str = 'i8'


INPUT = Node('anchor', attrs=(('id', 'input'),))
METRIC = 'relu_evaluations'
TIER = 'host_functional_cost'
HOST_TARGET = 'scalar_i8_host_diagnostic'
PROOF = ('For signed i8 x, R(x)=max(x,0) lies in [0,127]. '
         'R(y)=y on [0,127], so R(R(x))=R(x). Induction gives '
         'R^n(x)=R(x) for every integer n>=1. No overflow, floating-point '
         'behavior, caller premise, hardware latency or vector width is used.')
COST_FACTS = {'input': 0, 'relu': 1, 'meaning': 'abstract scalar primitive evaluations; not device cycles'}


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def signed_i8(raw: int) -> int:
    if type(raw) is not int or not 0 <= raw < 256:
        raise ValueError('raw i8 requires an integer in [0,255]')
    return raw if raw < 128 else raw - 256


def relu(value: int) -> int:
    if type(value) is not int or not -128 <= value <= 127:
        raise ValueError('signed i8 value out of range')
    return value if value >= 0 else 0


def repeated(count: int) -> Node:
    if type(count) is not int or not 1 <= count <= 128:
        raise ValueError('this diagnostic bounds source ReLU depth to 1..128')
    node = INPUT
    for _ in range(count):
        node = Node('relu', (node,))
    return node


def validate(node: Node, depth: int = 0) -> None:
    if depth > 128 or not isinstance(node, Node) or node.type != 'i8':
        raise ValueError('unsupported node/type/depth')
    if node == INPUT:
        return
    if node.op != 'relu' or node.attrs or len(node.operands) != 1:
        raise ValueError('expected unary signed-i8 ReLU or the declared input anchor')
    validate(node.operands[0], depth + 1)


def evaluate(node: Node, value: int) -> int:
    return value if node == INPUT else relu(evaluate(node.operands[0], value))


def serialize(node: Node) -> bytes:
    validate(node)
    return canonical(asdict(node))


def deserialize(source: bytes) -> Node:
    def visit(data, depth=0):
        if depth > 128 or not isinstance(data, dict) or set(data) != {'op', 'operands', 'attrs', 'type'}:
            raise ValueError('invalid serialized node')
        if not isinstance(data['operands'], list) or not isinstance(data['attrs'], list):
            raise ValueError('invalid operands/attributes')
        node = Node(data['op'], tuple(visit(child, depth + 1) for child in data['operands']),
                    tuple(tuple(pair) for pair in data['attrs']), data['type'])
        validate(node, depth)
        return node
    return visit(json.loads(source))


def validate_contract(contract: dict) -> None:
    if contract.get('name') != 'k1_cpu' or 'relu' not in contract.get('capabilities', {}).get('ops', []):
        raise ValueError('select the canonical K1 contract with ReLU capability')
    units = contract.get('compute_units', [])
    if not any(unit.get('kind') == 'scalar' and any(
        capability.get('family') == 'elementwise_map' and 'int8' in capability.get('dtypes', [])
        for capability in unit.get('semantic_capabilities', [])) for unit in units):
        raise ValueError('contract lacks scalar int8 elementwise semantics')


def exhaustive_proof() -> dict:
    single = bytes(relu(signed_i8(raw)) for raw in range(256))
    witnesses = []
    for raw in range(256):
        value = signed_i8(raw)
        if relu(relu(value)) != relu(value):
            raise ArithmeticError(f'i8 idempotence witness failed: {raw}')
        for count in (1, 2, 3, 5, 16, 127, 128):
            if evaluate(repeated(count), value) != relu(value):
                raise ArithmeticError(f'i8 repetition witness failed: {raw}/{count}')
        witnesses.append([raw, value, single[raw]])
    return {'identity': PROOF, 'encodings': 256, 'repetition_depths': [1, 2, 3, 5, 16, 127, 128],
            'witness_sha256': selection.digest(canonical(witnesses)),
            'output_sha256': selection.digest(single)}


def rules():
    x, = vars_('x', egraph_terms.Value)
    def r(value):
        return egraph_terms.Value.apply('relu', '[]', 'i8', egraph_terms.forest((value,)))
    return ruleset(rewrite(r(r(x))).to(r(x)))


def egglog_candidate(native: Node):
    validate(native)
    saturated = egraph_terms.saturate({'output': egraph_terms.encode_node(native)}, rules=rules())
    roots, cost = egraph_terms.extract(saturated, marginal_cost=lambda graph, node:
        1 if get_callable_args(node, egraph_terms.Value.apply) is not None else 0)
    extracted = egraph_terms.decode_node(roots['output'], Node)
    validate(extracted)
    return extracted, {'rounds': saturated.rounds, 'function_rows': saturated.function_rows,
                       'stop_reason': saturated.stop_reason, 'dag_cost': cost}


def estimate(node: Node):
    nodes = []
    seen = {}
    def visit(current):
        if current in seen:
            return seen[current]
        operands = tuple(visit(child) for child in current.operands)
        identity = str(len(nodes))
        nodes.append(selection.CostNode(identity, 'input' if current == INPUT else 'relu', operands, (('scalar', 1),)))
        seen[current] = identity
        return identity
    root = visit(node)
    return selection.estimate_dag(nodes, (root,), operation_costs={'input': 0, 'relu': 1},
        pressure_weights={'scalar': 0}, facts_sha256=selection.digest(canonical(COST_FACTS)))


def compile_candidate(candidate: selection.RewriteCandidate) -> bytes:
    node = deserialize(candidate.source)
    def emit(current):
        return 'value' if current == INPUT else 'relu(' + emit(current.operands[0]) + ')'
    return ('def kernel(value, relu):\n    return ' + emit(node) + '\n').encode()


def cases() -> tuple[tuple[str, bytes], ...]:
    # Independent raw-byte oracle below accepts every signed encoding. Additional
    # lengths cross typical vector boundaries without claiming RVV execution.
    return (('all_i8', bytes(range(256))),) + tuple(
        (f'tail_{length}', bytes((index * 73 + 129) % 256 for index in range(length)))
        for length in (1, 7, 31, 33, 257))


def measure(program: bytes) -> tuple[selection.Measurement, dict]:
    namespace = {}
    exec(compile(program, '<scalar-i8-generated-kernel>', 'exec'), namespace)
    kernel = namespace['kernel']
    counts = []
    evidence = []
    for invocation in range(3):
        calls = 0
        def counted(value):
            nonlocal calls
            calls += 1
            return relu(value)
        for name, raw in cases():
            original = bytes(raw)
            # These sentinels lie outside signed-i8 values and remain untouched.
            output = [911, -733] + [-999] * len(raw) + [617, -419]
            for index, bits in enumerate(raw, 2):
                output[index] = kernel(signed_i8(bits), counted)
            actual = bytes(value & 255 for value in output[2:-2])
            # Independent oracle uses the sign bit in raw encodings, not relu().
            expected = bytes(bits if bits < 128 else 0 for bits in raw)
            good_guards = output[:2] == [911, -733] and output[-2:] == [617, -419] and raw == original
            if actual != expected or not good_guards:
                raise selection.CorrectnessFailure(f'host scalar output/guard mismatch: {name}/{invocation}')
            evidence.append({'case': name, 'invocation': invocation, 'elements': len(raw),
                'input_sha256': selection.digest(raw), 'output_sha256': selection.digest(actual),
                'oracle_sha256': selection.digest(expected), 'guards': True})
        counts.append(calls)
    return selection.Measurement(selection.digest(program), HOST_TARGET, METRIC, TIER,
        tuple(counts), True, True, True), {'cases': evidence, 'primitive_evaluations': counts}


def experiment(contract: dict, contract_sha256: str, *, depth: int = 5) -> dict:
    validate_contract(contract)
    if len(contract_sha256) != 64 or any(char not in '0123456789abcdef' for char in contract_sha256):
        raise ValueError('canonical contract needs its source SHA-256')
    proof = exhaustive_proof()
    certificate = selection.digest(canonical({'contract_sha256': contract_sha256, 'proof': proof}))
    native = repeated(depth)
    extracted, saturation = egglog_candidate(native)
    nodes = {'native': native, 'hand': repeated(1), 'egglog': extracted}
    candidates = [selection.RewriteCandidate(name, serialize(node), certificate,
        ('integer_relu_idempotence',), estimate(node)) for name, node in nodes.items()]
    # Hand and egglog can have identical bytes. Retain both comparison controls
    # explicitly; their equality is part of the finding, not extra diversity.
    chosen = selection.shortlist(candidates, baseline='native', required=('hand', 'egglog'), limit=3)
    executions = {}
    def admit(candidate):
        if candidate.certificate_sha256 != certificate or candidate.families != ('integer_relu_idempotence',):
            return False
        try:
            node = deserialize(candidate.source)
        except (ValueError, TypeError, KeyError, RecursionError):
            return False
        return all(evaluate(node, signed_i8(raw)) == (raw if raw < 128 else 0) for raw in range(256))
    def qualify(candidate, program, target):
        row, evidence = measure(program)
        executions[candidate.id] = {'program_sha256': selection.digest(program),
                                     'source_sha256': candidate.source_sha256, **evidence}
        return row
    result = selection.select_measured(chosen, baseline='native', targets={HOST_TARGET: (METRIC, TIER)},
        compile_candidate=compile_candidate, measure=qualify, admit=admit)
    minimum = min(row[HOST_TARGET].values[0] for row in result.measurements.values())
    return {'status': 'PASS', 'contract_name': contract['name'], 'contract_sha256': contract_sha256,
        'requested_mode': 'host_scalar_diagnostic', 'actual_mode': 'host_scalar_diagnostic',
        'evidence_tier': TIER, 'metric': METRIC, 'device_timing': 'NOT_RUN_NO_BOARD',
        'hardware_execution': False, 'host_architecture': platform.machine(),
        'source_depth': depth, 'proof': proof, 'certificate_sha256': certificate,
        'cost_facts': COST_FACTS, 'saturation': saturation,
        'estimates': {candidate.id: asdict(candidate.cost) for candidate in chosen},
        'selection': asdict(result), 'executions': executions,
        'tied_best': sorted(name for name, rows in result.measurements.items() if rows[HOST_TARGET].values[0] == minimum),
        'tools': {'python': sys.version, 'python_executable_sha256': selection.digest(Path(sys.executable).read_bytes()),
                  'egglog': version('egglog')},
        'sources': {name: selection.digest(Path(module.__file__).read_bytes()) for name, module in
                    (('egraph_terms', egraph_terms), ('rewrite_selection', selection))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract-json', type=Path, required=True,
                        help='snapshot with contract and source_sha256 keys')
    parser.add_argument('--depth', type=int, default=5)
    args = parser.parse_args()
    snapshot = json.loads(args.contract_json.read_bytes())
    report = experiment(snapshot['contract'], snapshot['source_sha256'], depth=args.depth)
    report['contract_snapshot_sha256'] = selection.digest(args.contract_json.read_bytes())
    report['provider_sha256'] = selection.digest(Path(__file__).read_bytes())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
