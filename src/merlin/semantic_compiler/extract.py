"""Merlin-owned buffer-aware enumerative candidate extraction.

The bounded traversal is informed by ACT section 5.3, but is an independently
implemented multi-root extractor. It requests each e-class in a particular
physical storage and keeps alternatives after an initial allocation failure.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256
from time import monotonic

from .egg_bridge import Exploration
from .model import KernelRequest
from .rules import RuleProgram


class ExtractionTimeout(RuntimeError):
    pass


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and monotonic() >= deadline:
        raise ExtractionTimeout("native candidate extraction exceeded its deadline")


@dataclass(frozen=True)
class Choice:
    eclass: int
    symbol: str
    storage: str
    children: tuple[Choice, ...] = ()
    bundle_instance: str = ""
    bundle_result_index: int = 0
    bundle_result_count: int = 1

    def __post_init__(self) -> None:
        if type(self.bundle_instance) is not str:
            raise ValueError("selected instruction needs a string bundle identity")
        if self.bundle_instance:
            if (
                type(self.bundle_result_index) is not int
                or type(self.bundle_result_count) is not int
                or self.bundle_result_count <= 1
                or not 0 <= self.bundle_result_index < self.bundle_result_count
            ):
                raise ValueError("selected multi-result instruction has malformed projection identity")
        elif self.bundle_result_index != 0 or self.bundle_result_count != 1:
            raise ValueError("single-result choice has a multi-result projection index")

    def key(self) -> tuple:
        base = (self.eclass, self.symbol, self.storage, tuple(child.key() for child in self.children))
        return base if not self.bundle_instance else (
            *base, self.bundle_instance, self.bundle_result_index, self.bundle_result_count,
        )

    def physical_instance(self) -> tuple | None:
        if not self.bundle_instance:
            return None
        return (self.bundle_instance, tuple(child.key() for child in self.children))

    def instruction_count(self) -> int:
        return Candidate((self,)).instruction_count()


@dataclass(frozen=True)
class Candidate:
    outputs: tuple[Choice, ...]

    def digest(self) -> str:
        data = json.dumps([choice.key() for choice in self.outputs], separators=(",", ":"), sort_keys=True)
        return sha256(data.encode()).hexdigest()

    def instruction_count(self) -> int:
        """Charge one shared instruction once across all ordered roots."""
        seen: set[tuple] = set()
        instances: set[tuple] = set()

        def visit(choice: Choice) -> int:
            key = choice.key()
            if key in seen:
                return 0
            seen.add(key)
            instance = choice.physical_instance()
            count = int(choice.symbol.startswith("i_") and (instance is None or instance not in instances))
            if instance is not None:
                instances.add(instance)
            return count + sum(visit(child) for child in choice.children)

        return sum(visit(output) for output in self.outputs)

    def bundles_complete(self) -> bool:
        """A selected bundle retains every result of one physical issue."""
        groups: dict[tuple, tuple[int, dict[int, Choice]]] = {}
        visited: set[tuple] = set()

        def visit(choice: Choice) -> bool:
            if choice.key() in visited:
                return True
            visited.add(choice.key())
            instance = choice.physical_instance()
            if instance is not None:
                count, projections = groups.setdefault(instance, (choice.bundle_result_count, {}))
                if count != choice.bundle_result_count or not 0 <= choice.bundle_result_index < count:
                    return False
                previous = projections.setdefault(choice.bundle_result_index, choice)
                if previous.key() != choice.key():
                    return False
            return all(visit(child) for child in choice.children)

        return all(visit(output) for output in self.outputs) and all(
            set(projections) == set(range(count)) for count, projections in groups.values()
        )


def _choices(
    exploration: Exploration,
    request: KernelRequest,
    program: RuleProgram,
    eclass: int,
    storage: str,
    required_layout: str,
    budget: int,
    ancestors: frozenset[tuple[int, str, str]],
    deadline: float | None,
) -> Iterator[Choice]:
    _check_deadline(deadline)
    if budget < 0:
        return
    # A no-op semantic rewrite can put a copy instruction and its source in
    # one e-class. Revisited classes may still terminate at an input/constant
    # leaf; only further instruction expansion would make a recursive cycle.
    recursive = (eclass, storage, required_layout) in ancestors
    next_ancestors = ancestors | {(eclass, storage, required_layout)}
    inputs = dict(request.input_storages)
    for node in exploration.classes.get(eclass, ()):
        _check_deadline(deadline)
        metadata = program.symbols.get(node.symbol)
        if metadata is None:
            continue
        kind = metadata["kind"]
        if kind == "input":
            if inputs[metadata["source_node"]] == storage and (
                not required_layout or request.boundary_layout(metadata["source_node"]) == required_layout
            ):
                yield Choice(eclass, node.symbol, storage)
            continue
        if kind == "constant":
            # Constant payloads are compiler inputs and must be initialized by
            # the materializer; they are never an uninitialized scratchpad leaf.
            if storage == "external" and (
                not required_layout or request.boundary_layout(metadata["source_node"]) == required_layout
            ):
                yield Choice(eclass, node.symbol, storage)
            continue
        if kind == "view":
            descriptor = metadata["descriptor"]
            required_extents = (
                tuple(descriptor["source_extents"])
                if descriptor.get("kind") == "concat"
                else (descriptor.get("source_extent", descriptor["extent"]),)
            )
            if (
                recursive or len(node.children) != len(required_extents) or descriptor["storage"] != storage
                or required_layout and descriptor["physical_layout"] != required_layout
            ):
                continue

            def view_children(index: int, chosen: tuple[Choice, ...]) -> Iterator[tuple[Choice, ...]]:
                _check_deadline(deadline)
                if index == len(node.children):
                    yield chosen
                    return
                for child in _choices(
                    exploration, request, program, node.children[index], storage,
                    descriptor["physical_layout"], budget,
                    next_ancestors, deadline,
                ):
                    child_metadata = program.symbols[child.symbol]
                    child_extent = (
                        child_metadata["descriptor"]["extent"]
                        if child_metadata["kind"] in {"instruction", "view"}
                        else child_metadata.get("extent")
                    )
                    if child_extent != required_extents[index]:
                        continue
                    if Candidate((*chosen, child)).instruction_count() > budget:
                        continue
                    yield from view_children(index + 1, (*chosen, child))

            for children in view_children(0, ()):
                yield Choice(eclass, node.symbol, storage, children)
            continue
        if kind != "instruction":
            continue
        if recursive:
            continue
        if budget == 0:
            continue
        descriptor = metadata["descriptor"]
        if descriptor["output_storage"] != storage or (
            required_layout and descriptor.get("output_layout", "") != required_layout
        ):
            continue
        required = descriptor["input_storages"]
        # Earlier request schemas do not state a boundary representation.
        # Descriptor layouts become obligations only for layout-qualified v4
        # requests (or when a view explicitly requests a physical layout).
        input_layouts = descriptor.get("input_layouts", ()) if (
            request.boundary_layouts or required_layout
        ) else ()
        if request.boundary_layouts and len(input_layouts) != len(required):
            continue
        if len(required) != len(node.children):
            continue
        if not node.children:
            yield Choice(
                eclass, node.symbol, storage,
                bundle_instance=metadata.get("bundle_instance", ""),
                bundle_result_index=metadata.get("bundle_result_index", 0),
                bundle_result_count=metadata.get("bundle_result_count", 1),
            )
            continue

        def combinations(index: int, chosen: tuple[Choice, ...]) -> Iterator[tuple[Choice, ...]]:
            _check_deadline(deadline)
            if index == len(node.children):
                yield chosen
                return
            for child in _choices(
                exploration,
                request,
                program,
                node.children[index],
                required[index],
                input_layouts[index] if input_layouts else "",
                budget - 1,
                next_ancestors,
                deadline,
            ):
                # The same selected producer can satisfy two input ports.
                # Charge the prospective DAG, not the unfolded child trees.
                if 1 + Candidate((*chosen, child)).instruction_count() > budget:
                    continue
                yield from combinations(index + 1, (*chosen, child))

        for children in combinations(0, ()):
            yield Choice(
                eclass, node.symbol, storage, children,
                metadata.get("bundle_instance", ""),
                metadata.get("bundle_result_index", 0),
                metadata.get("bundle_result_count", 1),
            )


def enumerate_candidates(
    exploration: Exploration,
    request: KernelRequest,
    program: RuleProgram,
    *,
    node_budget: int,
    max_candidates: int,
    deadline: float | None = None,
) -> Iterator[Candidate]:
    if node_budget <= 0 or max_candidates <= 0:
        raise ValueError("candidate work bounds must be positive")
    if len(exploration.roots) != len(request.outputs):
        raise ValueError("e-graph roots differ from source outputs")
    seen: set[str] = set()
    emitted = 0

    def combine(index: int, chosen: tuple[Choice, ...]) -> Iterator[Candidate]:
        nonlocal emitted
        _check_deadline(deadline)
        if emitted >= max_candidates:
            return
        if index == len(exploration.roots):
            candidate = Candidate(chosen)
            digest = candidate.digest()
            if digest not in seen and candidate.bundles_complete():
                seen.add(digest)
                emitted += 1
                yield candidate
            return
        root = exploration.roots[index]
        storage = request.output_storages[index]
        layout = request.output_layouts[index] if request.output_layouts else ""
        for choice in _choices(
            exploration, request, program, root, storage, layout, node_budget, frozenset(), deadline,
        ):
            if emitted >= max_candidates:
                return
            if Candidate((*chosen, choice)).instruction_count() <= node_budget:
                yield from combine(index + 1, (*chosen, choice))

    yield from combine(0, ())
