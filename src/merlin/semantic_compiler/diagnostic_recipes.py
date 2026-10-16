"""Explicit diagnostic recipe exploration through the existing native bridge.

This is an adapter boundary, not an admitted production rewrite pass. A trusted
target validator must check numerical equivalence, domains and physical recipe
legality before returning a certificate. Matching tensor types alone proves
none of those obligations. Rust remains an unmodified general e-graph executor.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from hashlib import sha256

from .model import KernelRequest, SemanticNode
from .rules import InstructionDescriptor, Rule, RuleProgram, generate_rules


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class DiagnosticRecipe:
    source_node: str
    nodes: tuple[SemanticNode, ...]
    result: str
    contract: str

    def record(self) -> dict:
        return {"source_node": self.source_node, "nodes": [node.record() for node in self.nodes],
                "result": self.result, "contract": self.contract}

    def digest(self) -> str:
        return _digest(self.record())


@dataclass(frozen=True)
class RecipeCertificate:
    """Binding returned by the target's trusted validator, never a proof itself."""

    request_digest: str
    recipe_digest: str
    target_identity: str
    proof_identity: str


def build_diagnostic_recipe_program(
    request: KernelRequest,
    descriptors: tuple[InstructionDescriptor, ...],
    recipes: tuple[DiagnosticRecipe, ...],
    *,
    admit: Callable[[KernelRequest, DiagnosticRecipe], RecipeCertificate],
) -> tuple[KernelRequest, RuleProgram, tuple[RecipeCertificate, ...]]:
    """Add separately validated pure recipe branches to a diagnostic request.

    Original nodes, ordered roots, boundaries and descriptors retain their
    identities. Every primitive recipe step is visible to the real multi-root
    extractor; there is no opaque one-instruction cost for a compound recipe.
    Allocation/checking can use the returned expanded request. Production
    search and target materialization do not invoke this adapter automatically.
    """
    if request.lowering_policy != "diagnostic":
        raise ValueError("recipe exploration requires an explicit diagnostic request")
    nodes = list(request.nodes)
    existing = {node.id: node for node in nodes}
    original_sources = set(existing)
    certificates = []
    for recipe in recipes:
        if type(recipe.contract) is not str or not recipe.contract or not recipe.nodes or recipe.source_node not in original_sources:
            raise ValueError("diagnostic recipe needs a contract and existing source")
        source = request.node(recipe.source_node)
        if source.effect != "pure":
            raise ValueError("diagnostic recipes cannot replace a boundary or state")
        certificate = admit(request, recipe)
        if not isinstance(certificate, RecipeCertificate) or (
            certificate.request_digest != request.digest()
            or certificate.recipe_digest != recipe.digest()
            or certificate.target_identity != request.target_identity
            or type(certificate.proof_identity) is not str or not certificate.proof_identity
        ):
            raise ValueError("recipe admission certificate has a stale or incomplete binding")
        for node in recipe.nodes:
            if node.effect != "pure" or node.id in existing or any(child not in existing for child in node.inputs):
                raise ValueError("recipe nodes must be fresh, pure and topologically ordered")
            nodes.append(node)
            existing[node.id] = node
        if recipe.result not in {node.id for node in recipe.nodes} or existing[recipe.result].type != source.type:
            raise ValueError("recipe result must retain the complete source tensor type")
        certificates.append(certificate)
    expanded = replace(request, nodes=tuple(nodes), source_identity=_digest({
        "request": request.digest(), "recipes": [recipe.record() for recipe in recipes],
        "certificates": [vars(certificate) for certificate in certificates],
    }))
    program = generate_rules(expanded, descriptors)
    # Indexed initial nodes, rather than guessed hashes or graph-specific
    # names, identify the full typed source expressions.
    symbols_by_source = {node.id: program.nodes[index]["symbol"]
                         for index, node in enumerate(expanded.nodes)}
    rewrites = list(program.rewrites)
    for recipe in recipes:
        source, result = request.node(recipe.source_node), expanded.node(recipe.result)
        # Ground expressions pin exact operands. A rule for one source site must
        # never match an unrelated expression with the same typed signature.
        def expression(node_id: str) -> str:
            node = expanded.node(node_id)
            symbol = symbols_by_source[node_id]
            return symbol if not node.inputs else f"({symbol} {' '.join(expression(child) for child in node.inputs)})"
        identity = recipe.digest()
        rewrites.append(Rule("diagnostic_recipe_" + identity, expression(source.id), expression(result.id),
                             source.id, "<diagnostic-recipe>", identity))
    return expanded, replace(program, rewrites=tuple(rewrites)), tuple(certificates)
