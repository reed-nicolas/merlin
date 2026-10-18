from dataclasses import dataclass
import pytest
pytest.importorskip('egglog')
from egglog import rewrite, ruleset, vars_
from egglog.deconstruct import get_callable_args
from merlin.semantic_compiler.egraph_terms import Value, Inputs, encode_node, decode_node, forest, saturate, extract

@dataclass(frozen=True)
class Node:
    op: str
    operands: tuple = ()
    attrs: tuple = ()
    type: str = 'bytes'

def wrap(value): return Value.apply('copy','[]','bytes',forest((value,)))

def test_provider_neutral_roundtrip_and_real_saturation():
    node=Node('copy',(Node('anchor',attrs=(('id','source'),)),))
    assert decode_node(encode_node(node),Node)==node
    x,=vars_('x',Value)
    result=saturate({'out':encode_node(node)},rules=ruleset(rewrite(wrap(x)).to(x)))
    roots,score=extract(result)
    assert decode_node(roots['out'],Node)==node.operands[0]
    assert result.stop_reason=='fixed_point'

def test_joint_forest_dag_cost_charges_shared_producer_once():
    x=Value.anchor('x','bytes'); shared=wrap(x)
    result=saturate({'a':shared,'b':shared},rules=ruleset())
    roots,score=extract(result,marginal_cost=lambda graph,node:7 if get_callable_args(node,Value.apply) is not None else 0)
    assert len(roots)==2 and score==7

def test_bounds_and_malformed_attributes_are_refused():
    x=Value.anchor('x','bytes')
    with pytest.raises(ValueError):saturate({'x':wrap(x)},rules=ruleset(),max_function_rows=1)
    bad=Value.apply('copy','[["a","b"],["a","c"]]','bytes',Inputs.empty())
    with pytest.raises(ValueError):decode_node(bad,Node)
