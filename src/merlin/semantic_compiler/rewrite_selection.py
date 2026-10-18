"""Target-supplied exact candidates with shared DAG costs and measured selection.

Providers own arithmetic proofs, e-graph encoding/rules, materialization and
execution. This module owns no target names, numeric identities or latencies.
Estimated cost is a shortlist heuristic; only qualified receipts choose a winner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Callable, Mapping, Sequence


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def _natural(value: int, name: str, *, positive: bool = False) -> None:
    if type(value) is not int or value < int(positive):
        raise ValueError(f"{name} must be a {'positive' if positive else 'nonnegative'} integer")


@dataclass(frozen=True)
class CostNode:
    id: str
    operation: str
    inputs: tuple[str, ...] = ()
    resources: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class CostEstimate:
    operation_cost: int
    peak_pressure: tuple[tuple[str, int], ...]
    pressure_penalty: int
    exceeds_capacity: tuple[str, ...]
    facts_sha256: str

    @property
    def score(self) -> int:
        return self.operation_cost + self.pressure_penalty


def estimate_dag(
    nodes: Sequence[CostNode], roots: Sequence[str], *,
    operation_costs: Mapping[str, int], pressure_weights: Mapping[str, int],
    facts_sha256: str, capacities: Mapping[str, int] | None = None,
    anchored_pressure: Mapping[str, int] | None = None,
) -> CostEstimate:
    """Charge each named producer once and estimate simultaneous live buffers.

    The input is a topological materialization order, not an e-class traversal.
    Results overlap their operands at issue. Roots live through the boundary.
    Anchored state resources come from the provider and are conservatively added.
    This does not prove allocation, bank legality, schedule length or timing.
    """
    if len(facts_sha256) != 64 or any(c not in '0123456789abcdef' for c in facts_sha256):
        raise ValueError('cost facts need a SHA-256 identity')
    seen: set[str] = set()
    last: dict[str, int] = {}
    total = 0
    demands: dict[str, dict[str, int]] = {}
    for index, node in enumerate(nodes):
        if not node.id or node.id in seen or not node.operation:
            raise ValueError('cost nodes need unique identities and an operation')
        if any(child not in seen for child in node.inputs):
            raise ValueError('cost graph must be topological with known inputs')
        if node.operation not in operation_costs:
            raise ValueError(f'unknown operation cost: {node.operation}')
        cost = operation_costs[node.operation]
        _natural(cost, 'operation cost')
        total += cost
        for child in node.inputs:
            last[child] = index
        resource_map = dict(node.resources)
        if len(resource_map) != len(node.resources):
            raise ValueError('duplicate resource demand')
        for name, amount in resource_map.items():
            if not name or name not in pressure_weights:
                raise ValueError('resource demand needs a declared pressure weight')
            _natural(amount, 'resource demand')
        demands[node.id] = resource_map
        last[node.id] = index
        seen.add(node.id)
    if not roots or any(root not in seen for root in roots):
        raise ValueError('cost roots must name existing outputs')
    for root in roots:
        last[root] = len(nodes)
    for value in pressure_weights.values():
        _natural(value, 'pressure weight')
    live: dict[str, int] = {}
    peak: dict[str, int] = {}
    expires: dict[int, list[str]] = {}
    for name, end in last.items():
        expires.setdefault(end, []).append(name)
    for index, node in enumerate(nodes):
        for name, amount in demands[node.id].items():
            live[name] = live.get(name, 0) + amount
            peak[name] = max(peak.get(name, 0), live[name])
        for expired in expires.get(index, ()):
            for name, amount in demands[expired].items():
                live[name] -= amount
    for name, amount in (anchored_pressure or {}).items():
        _natural(amount, 'anchored pressure')
        if name not in pressure_weights:
            raise ValueError('anchored pressure needs a declared weight')
        peak[name] = peak.get(name, 0) + amount
    for name, amount in (capacities or {}).items():
        _natural(amount, 'capacity')
    exceeds = tuple(sorted(name for name, capacity in (capacities or {}).items() if peak.get(name, 0) > capacity))
    return CostEstimate(total, tuple(sorted(peak.items())), sum(pressure_weights[k] * v for k, v in peak.items()), exceeds, facts_sha256)


@dataclass(frozen=True)
class RewriteCandidate:
    id: str
    source: bytes
    certificate_sha256: str
    families: tuple[str, ...]
    cost: CostEstimate

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.source, bytes) or not self.source:
            raise ValueError('candidate needs an identity and nonempty source bytes')
        if len(self.certificate_sha256) != 64 or any(c not in '0123456789abcdef' for c in self.certificate_sha256):
            raise ValueError('candidate needs an exactness certificate digest')

    @property
    def source_sha256(self) -> str:
        return digest(self.source)


def shortlist(candidates: Sequence[RewriteCandidate], *, baseline: str, limit: int,
              required: Sequence[str] = ()) -> tuple[RewriteCandidate, ...]:
    """Keep the native control and required hand controls, then rank by cost.

    Pressure is a soft penalty: the target allocator, not this estimate, decides
    feasibility. This retains possible rematerializations and fallback choices.
    """
    _natural(limit, 'shortlist limit', positive=True)
    by_id = {candidate.id: candidate for candidate in candidates}
    if len(by_id) != len(candidates):
        raise ValueError('duplicate candidate id')
    mandatory = tuple(dict.fromkeys((baseline, *required)))
    if len(mandatory) > limit or any(name not in by_id for name in mandatory):
        raise ValueError('shortlist cannot retain all required controls')
    chosen = [by_id[name] for name in mandatory]
    identities = {candidate.source_sha256 for candidate in chosen}
    for candidate in sorted(candidates, key=lambda c: (c.cost.score, c.source_sha256, c.id)):
        if len(chosen) >= limit:
            break
        if candidate.id in mandatory or candidate.source_sha256 in identities:
            continue
        chosen.append(candidate)
        identities.add(candidate.source_sha256)
    return tuple(chosen)


@dataclass(frozen=True)
class Measurement:
    program_sha256: str
    target: str
    metric: str
    tier: str
    values: tuple[int, ...]
    outputs: bool
    oracle: bool
    guards: bool


class CorrectnessFailure(RuntimeError):
    """Abort this experiment; never turn a numerical failure into a slow score."""


class CandidateUnavailable(RuntimeError):
    """Explicit compile/target infrastructure refusal, separately reported."""


@dataclass(frozen=True)
class SelectionResult:
    winner_by_target: dict[str, str]
    measurements: dict[str, dict[str, Measurement]]
    rejected: dict[str, str]
    complete_candidates: tuple[str, ...]
    minimax_candidate: str
    worst_regret: dict[str, float]


def select_measured(
    candidates: Sequence[RewriteCandidate], *, baseline: str,
    targets: Mapping[str, tuple[str, str]],
    compile_candidate: Callable[[RewriteCandidate], bytes],
    measure: Callable[[RewriteCandidate, bytes, str], Measurement],
    admit: Callable[[RewriteCandidate], bool],
) -> SelectionResult:
    """Compile each admitted candidate and require exact, binary-bound receipts.

    Target values declare (metric, evidence tier), preventing accidental ranking
    of a host functional measurement as device timing. The first invocation is
    the score, matching the explicit cold-invocation comparison policy.
    """
    ids = [c.id for c in candidates]
    if len(set(ids)) != len(ids) or baseline not in ids or not targets:
        raise ValueError('selection needs unique candidates, baseline and targets')
    if any(not target or not metric or not tier for target, (metric, tier) in targets.items()):
        raise ValueError('measurement target, metric and tier must be explicit')
    measurements: dict[str, dict[str, Measurement]] = {}
    rejected: dict[str, str] = {}
    ordered = sorted(candidates, key=lambda c: (c.id != baseline, c.id))
    for candidate in ordered:
        # Admission must recheck the provider's actual certificate, not its name.
        if admit(candidate) is not True:
            raise CorrectnessFailure(f'candidate certificate rejected: {candidate.id}')
        try:
            program = compile_candidate(candidate)
            if not isinstance(program, bytes) or not program:
                raise ValueError('compiler must return the measured program bytes')
            rows = measurements.setdefault(candidate.id, {})
            for target, (metric, tier) in targets.items():
                row = measure(candidate, program, target)
                if row.outputs is not True or row.oracle is not True or row.guards is not True:
                    raise CorrectnessFailure(f'output/oracle/guard failure: {candidate.id}/{target}')
                if (row.program_sha256 != digest(program) or row.target != target
                        or row.metric != metric or row.tier != tier):
                    raise ValueError('receipt binary/target/metric/tier mismatch')
                if not row.values:
                    raise ValueError('empty measurement')
                for value in row.values:
                    _natural(value, 'measurement value', positive=True)
                rows[target] = row
        except CandidateUnavailable as error:
            rejected[candidate.id] = str(error)
            if candidate.id == baseline:
                raise CandidateUnavailable('baseline unavailable: ' + str(error)) from error
    complete = tuple(sorted(name for name, rows in measurements.items() if set(rows) == set(targets) and name not in rejected))
    if baseline not in complete:
        raise CandidateUnavailable('baseline lacks a complete measurement grid')
    best = {target: min(measurements[name][target].values[0] for name in complete) for target in targets}
    winners = {target: min(complete, key=lambda name: (measurements[name][target].values[0], name)) for target in targets}
    regret = {name: max(measurements[name][target].values[0] / best[target] - 1 for target in targets) for name in complete}
    return SelectionResult(winners, measurements, rejected, complete, min(complete, key=lambda name: (regret[name], name)), regret)
