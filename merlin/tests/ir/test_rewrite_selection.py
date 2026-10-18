from dataclasses import replace
import pytest

from merlin.semantic_compiler.rewrite_selection import (
    CandidateUnavailable, CorrectnessFailure, CostNode, Measurement,
    RewriteCandidate, digest, estimate_dag, select_measured, shortlist,
)

PIN = digest(b'cost facts')


def cost(nodes=None, roots=('result',), **kwargs):
    return estimate_dag(nodes or [CostNode('result', 'compute', resources=(('tile', 1),))], roots,
                        operation_costs={'input': 0, 'compute': 3}, pressure_weights={'tile': 2},
                        facts_sha256=PIN, **kwargs)


def candidate(name, source=None):
    return RewriteCandidate(name, source or name.encode(), digest(b'exact evidence'), ('idempotence',), cost())


def receipt(candidate, program, target):
    values = {'native': (30, 20), 'a': (18, 17), 'b': (20, 19)}
    n = values[candidate.id][target == 't2']
    return Measurement(digest(program), target, 'cycles', 'model', (n, n-1), True, True, True)


def select(candidates=None, measure=receipt, compile_candidate=lambda c: c.source, **kw):
    return select_measured(candidates or [candidate(n) for n in ('native','a','b')], baseline='native',
                           targets={'t1': ('cycles','model'), 't2': ('cycles','model')},
                           compile_candidate=compile_candidate, measure=measure, admit=lambda c: True, **kw)


def test_shared_producer_charged_once_and_roots_remain_live():
    nodes = [CostNode('x','input',resources=(('tile',1),)),
             CostNode('a','compute',('x',),(('tile',1),)),
             CostNode('b','compute',('x',),(('tile',1),)),
             CostNode('result','compute',('a','b'),(('tile',1),))]
    c = cost(nodes, roots=('x','result'), capacities={'tile':3}, anchored_pressure={'tile':1})
    assert c.operation_cost == 9
    assert dict(c.peak_pressure) == {'tile':5}
    assert c.exceeds_capacity == ('tile',)
    assert c.score == 19


def test_last_use_release_reduces_peak():
    nodes = [CostNode(str(i), 'compute', (() if i == 0 else (str(i-1),)), (('tile',1),)) for i in range(100)]
    assert dict(cost(nodes,roots=('99',)).peak_pressure) == {'tile':2}


@pytest.mark.parametrize('nodes,roots', [
    ([CostNode('result','missing')],('result',)),
    ([CostNode('result','compute',('unknown',))],('result',)),
    ([CostNode('result','compute'),CostNode('result','compute')],('result',)),
    ([CostNode('result','compute',resources=(('tile',-1),))],('result',)),
    ([CostNode('result','compute')],('unknown',)),
])
def test_unknown_or_invalid_costs_fail_closed(nodes, roots):
    with pytest.raises(ValueError): cost(nodes,roots)


def test_shortlist_retains_native_and_hand_control_and_deduplicates():
    native, hand, alias, other = candidate('native'),candidate('hand'),candidate('alias',b'hand'),candidate('other')
    chosen=shortlist([alias,other,hand,native], baseline='native', required=('hand',), limit=4)
    assert [c.id for c in chosen] == ['native','hand','other']
    with pytest.raises(ValueError): shortlist([native,hand],baseline='native',required=('hand',),limit=1)


def test_measurement_wins_over_estimate_and_tracks_regret():
    result=select()
    assert result.winner_by_target == {'t1':'a','t2':'a'}
    assert result.minimax_candidate == 'a'
    assert result.worst_regret['native'] > 0


@pytest.mark.parametrize('field,value', [('outputs',False),('oracle',False),('guards',False),('outputs',1)])
def test_correctness_difference_aborts_before_later_candidates(field,value):
    seen=[]
    def measure(c,p,t):
        seen.append(c.id)
        return replace(receipt(c,p,t),**{field:value})
    with pytest.raises(CorrectnessFailure): select(measure=measure)
    assert seen == ['native']


@pytest.mark.parametrize('field,value', [('program_sha256','0'*64),('target','wrong'),('metric','nanoseconds'),('tier','silicon'),('values',()),('values',(True,)),('values',(0,))])
def test_unbound_or_unqualified_receipts_refused(field,value):
    with pytest.raises(ValueError): select(measure=lambda c,p,t:replace(receipt(c,p,t),**{field:value}))


def test_compile_refusal_is_separate_from_correctness_failure():
    def compile(c):
        if c.id == 'a': raise CandidateUnavailable('register allocation failed')
        return c.source
    result=select(compile_candidate=compile)
    assert result.rejected == {'a':'register allocation failed'}
    assert result.winner_by_target == {'t1':'b','t2':'b'}
    with pytest.raises(CandidateUnavailable):select(compile_candidate=lambda c: (_ for _ in ()).throw(CandidateUnavailable('missing tool')))


def test_partial_grid_cannot_win():
    def measure(c,p,t):
        if c.id=='a' and t=='t2': raise CandidateUnavailable('harness unavailable')
        return receipt(c,p,t)
    result=select(measure=measure)
    assert 'a' not in result.complete_candidates
    assert result.winner_by_target == {'t1':'b','t2':'b'}


def test_certificate_is_checked_before_compilation():
    called=[]
    with pytest.raises(CorrectnessFailure):
        select_measured([candidate('native')],baseline='native', targets={'t1':('cycles','model')},
                        admit=lambda c: False, compile_candidate=lambda c:called.append(c),measure=receipt)
    assert not called
