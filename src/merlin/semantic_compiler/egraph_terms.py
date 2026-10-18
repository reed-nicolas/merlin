"""Optional, target-neutral egglog value forest and bounded saturation.

Arithmetic rules and their guards are supplied by a target provider. Anchors
name immutable values across an external effect boundary; state and resource
handles are never encoded here. Providers must reverify decoded programs.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable, Mapping

from egglog import DagCostModel, EGraph, Expr, String, method
from egglog.deconstruct import get_callable_args, get_literal_value


class Value(Expr):
    @method(cost=0)
    @classmethod
    def anchor(cls, id: String, type: String) -> Value: ...

    @method(cost=1)
    @classmethod
    def apply(cls, op: String, attrs: String, type: String, children: Inputs) -> Value: ...


class Inputs(Expr):
    @method(cost=0)
    @classmethod
    def empty(cls) -> Inputs: ...

    @method(cost=0)
    @classmethod
    def cons(cls, head: Value, tail: Inputs) -> Inputs: ...


def attrs_json(attrs) -> str:
    return json.dumps(sorted(attrs), separators=(',', ':'), allow_nan=False)


def forest(values) -> Inputs:
    result = Inputs.empty()
    for value in reversed(tuple(values)):
        result = Inputs.cons(value, result)
    return result


def unforest(expression: Inputs) -> tuple[Value, ...]:
    values = []
    while get_callable_args(expression, Inputs.empty) is None:
        pair = get_callable_args(expression, Inputs.cons)
        if pair is None:
            raise ValueError('unknown value forest constructor')
        values.append(pair[0])
        expression = pair[1]
    return tuple(values)


def encode_node(node) -> Value:
    """Encode a provider node with op, operands, attrs and type fields."""
    if node.op == 'anchor':
        return Value.anchor(dict(node.attrs)['id'], node.type)
    return Value.apply(node.op, attrs_json(node.attrs), node.type, forest(encode_node(child) for child in node.operands))


def decode_node(value: Value, node_factory: Callable):
    """Reconstruct a provider's immutable nodes; semantic verification is next."""
    anchor = get_callable_args(value, Value.anchor)
    if anchor is not None:
        identity, type_ = (get_literal_value(arg) for arg in anchor)
        return node_factory('anchor', (), (('id', identity),), type_)
    args = get_callable_args(value, Value.apply)
    if args is None:
        raise ValueError('unknown value constructor')
    op, attributes, type_ = (get_literal_value(arg) for arg in args[:3])
    try:
        data = json.loads(attributes)
        if not isinstance(data, list) or any(not isinstance(pair, list) or len(pair) != 2 or any(not isinstance(x, str) for x in pair) for pair in data):
            raise ValueError('attributes must be string pairs')
        attrs = tuple(tuple(pair) for pair in data)
        if len(dict(attrs)) != len(attrs):
            raise ValueError('duplicate attributes')
    except (TypeError, ValueError) as error:
        raise ValueError('invalid value attributes') from error
    return node_factory(op, tuple(decode_node(child, node_factory) for child in unforest(args[3])), attrs, type_)


@dataclass(frozen=True)
class Saturation:
    graph: EGraph
    names: tuple[str, ...]
    root: Inputs
    rounds: int
    function_rows: int
    stop_reason: str


def saturate(roots: Mapping[str, Value], *, rules, register_facts: Callable[[EGraph], None] | None = None,
             max_rounds: int = 20, max_function_rows: int = 100000) -> Saturation:
    """Run provider rules to a fixed point or refuse a bounded-out result.

    The row bound is checked after each round; it is an observed growth bound,
    not a hard allocator limit inside egglog. Callers bound source size too.
    """
    if not roots or type(max_rounds) is not int or max_rounds <= 0 or type(max_function_rows) is not int or max_function_rows <= 0:
        raise ValueError('nonempty roots and positive work bounds are required')
    graph = EGraph()
    root = forest(roots.values())
    graph.register(root)
    if register_facts is not None:
        register_facts(graph)
    for round_ in range(1, max_rounds + 1):
        report = graph.run(1, ruleset=rules)
        rows = sum(size for _, size in graph.all_function_sizes())
        if rows > max_function_rows:
            raise ValueError('e-graph function-row budget exceeded')
        if not report.updated:
            return Saturation(graph, tuple(roots), root, round_, rows, 'fixed_point')
    raise ValueError('e-graph did not reach a fixed point within the round budget')


def extract(result: Saturation, *, marginal_cost=None) -> tuple[dict[str, Value], int | None]:
    """Extract the ordered forest together so shared producers count once."""
    if marginal_cost is None:
        expression = result.graph.extract(result.root, extractor='greedy-dag')
        score = None
    else:
        expression, score = result.graph.extract(result.root, include_cost=True,
            cost_model=DagCostModel(marginal_cost, identity=0), extractor='greedy-dag')
    values = unforest(expression)
    if len(values) != len(result.names):
        raise ValueError('extraction changed output arity')
    return dict(zip(result.names, values)), score
