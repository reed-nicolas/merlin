"""Portable diagnostic recipe example for one explicitly selected target source.

This is not installed support or automatic production admission. The generic
recipe executor belongs to Merlin; promotion of these target facts, primitive
semantics and virtual materialization belongs to the selected OOT provider.
Callers supply ordinary KernelRequests and retain every numerical policy gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

from merlin.semantic_compiler.diagnostic_recipes import DiagnosticRecipe, RecipeCertificate
from merlin.semantic_compiler.model import IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
from merlin.semantic_compiler.rules import AxisBound, InstructionDescriptor

TARGET_IDENTITY = "atlas-rtl-0079c0541111197741a231c002e3843fa6f545b2:movement-mxu01-vpu-pack-tile-v21"
CONTRACT = "finite-bf16-pack-vs-clamp-push-pop-e8m0-v1"
TILE = (32, 32)
BF16_LAYOUT = "atlas.tensor.bf16-pair-halves-32x32.v1"
FP8_LAYOUT = "atlas.tensor.fp8-e4m3-row-major-32x32.v1"
RESET_MAPS = (IndexMap(3, ((1,0,0),(0,0,1)), (0,0)),
              IndexMap(3, ((0,1,0),(0,0,1)), (0,0)),
              IndexMap(3, ((1,0,0),(0,1,0)), (0,0)))
PACK_DOMAINS = {
    "fp8-e4m3-selected-pack-all-scales-v1": ("bf16-finite-all-codes-v1", 0, 255),
    "fp8-e4m3-scaled-e8m0-rtl-rne-flush-clamp-122-128-v1": ("bf16-all-bits-rtl-pack-v1", 122, 128),
    "fp8-e4m3-scaled-e8m0-exact-normal-122-128-v1": ("bf16-positive-zero-or-exact-normal-under-scale-122-128-v1", 122, 128),
    "fp8-e4m3-unit-e8m0-seven-codes-v1": ("bf16-seven-powers-and-zero-v1", 127, 127),
}
FINITE_RESET_POLICIES = frozenset({"mxu0-e4m3-ordered-bf16-finite-v1", "mxu1-e4m3-anchor-bf16-finite-reset-v1"})


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class FiniteProof:
    request_digest: str
    value: str
    dtype: str
    shape: tuple[int, ...]
    layout: str
    reason: str
    origin: str = "derived"
    finite: bool = True
    may_nan: bool = False
    may_inf: bool = False
    nonnegative: bool = False


def prove_finite(request: KernelRequest, value: str) -> FiniteProof | None:
    """Small source-semantic proof, refusing overflow-capable arithmetic.

    A 32-term reset from raw FP8 bytes has magnitude bounded by
    32*480*480*(129/128)^32 < 2**25 under per-step BF16 rounding.
    No property of an f32 source model establishes this target-program fact.
    """
    facts = {}
    def visit(node_id):
        if node_id in facts:
            return facts[node_id]
        node = request.node(node_id)
        reason = None
        if node.type.dtype == "bf16" and node.type.shape == TILE:
            attrs = dict(node.attrs)
            if node.op == "bf16_splat" and set(attrs) == {"bits"} and type(attrs["bits"]) is int:
                if 0 <= attrs["bits"] <= 65535 and attrs["bits"] & 0x7f80 != 0x7f80:
                    reason = "finite BF16 splat bits"
            elif node.op == "constant":
                binding = next((item for item in request.constants if item.node_id == node.id), None)
                if binding is not None and binding.encoding == "bf16-le-raw":
                    payload = bytes.fromhex(binding.data_hex)
                    if all(int.from_bytes(payload[index:index+2], "little") & 0x7f80 != 0x7f80
                           for index in range(0, len(payload), 2)):
                        reason = "finite exact BF16 constant bytes"
            elif (node.op in {"identity", "mov", "relu"} and len(node.inputs) == 1
                  and not node.attrs and node.type.numerical_policy in {"raw-bit-pattern", *FINITE_RESET_POLICIES}
                  and node.index_maps in ((), pointwise_identity_maps(2, 1)) and visit(node.inputs[0])):
                reason = "finite bit-preserving copy or sign clamp"
            elif (node.op == "matmul" and node.type.numerical_policy in FINITE_RESET_POLICIES
                  and not node.attrs and node.index_maps == RESET_MAPS
                  and len(node.inputs) == 2 and all(request.node(child).type.dtype == "fp8_e4m3"
                      and request.node(child).type.shape == TILE for child in node.inputs)):
                reason = "raw FP8 fixed32 reset bound below2^25"
        facts[node_id] = reason
        return reason
    reason = visit(value)
    if reason is None:
        return None
    return FiniteProof(request.digest(), value, "bf16", TILE, BF16_LAYOUT, reason,
                       nonnegative=request.node(value).op == "relu")


def conversion_site(request: KernelRequest, node_id: str):
    if request.target_identity != TARGET_IDENTITY:
        raise ValueError("conversion target identity differs from selected source")
    node = request.node(node_id)
    if node.effect != "pure" or node.op != "pack_fp8" or len(node.inputs) != 1:
        raise ValueError("recipe source is not one pure PACK conversion")
    source = request.node(node.inputs[0])
    attrs = dict(node.attrs)
    scale = attrs.get("scale_code")
    if (source.type.dtype != "bf16" or node.type.dtype != "fp8_e4m3"
            or source.type.shape != TILE or node.type.shape != TILE
            or type(scale) is not int or not 0 <= scale <= 255
            or node.index_maps != pointwise_identity_maps(2, 1)):
        raise ValueError("conversion dtype, shape, maps or scale is outside the explicit domain")
    domain = PACK_DOMAINS.get(node.type.numerical_policy)
    if (domain is None or set(attrs) != {"input_domain", "scale_code"}
            or attrs["input_domain"] != domain[0] or not domain[1] <= scale <= domain[2]):
        raise ValueError("conversion numerical policy/domain/scale differs")
    return node, source, scale


def make_recipe(request: KernelRequest, node_id: str, engine: int) -> DiagnosticRecipe:
    node, source, scale = conversion_site(request, node_id)
    if type(engine) is not int or engine not in (0, 1):
        raise ValueError("conversion engine must be0 or1")
    prefix = "conv_" + digest((node.semantic_key(), node.inputs, engine))[:16]
    nodes = []
    def append(suffix, op, inputs, typed, attrs=()):
        made = SemanticNode(prefix + suffix, op, tuple(inputs), typed, tuple(attrs),
                            index_maps=pointwise_identity_maps(2, len(inputs)) if inputs else ())
        nodes.append(made)
        return made.id
    value = source.id
    typed = source.type
    if scale <= 246:
        typed = TensorType(TILE, "bf16", "raw-bit-pattern")
        cap = ((scale + 8) << 7) | 96
        nonnegative = (proof := prove_finite(request, source.id)) is not None and proof.nonnegative
        low = None if nonnegative else append("_low", "bf16_splat", (), typed, (("bits", cap | 0x8000),))
        high = append("_high", "bf16_splat", (), typed, (("bits", cap),))
        if low is not None:
            value = append("_max", "max_bf16", (value, low), typed)
        value = append("_min", "min_bf16", (value, high), typed)
    pushed = append("_push", "push_acc_bf16", (value,), typed, (("engine", engine),))
    result = append("_pop", "pop_acc_fp8", (pushed,), node.type,
                    (("engine", engine), ("inverse_scale", 254 - min(scale, 254))))
    return DiagnosticRecipe(node_id, tuple(nodes), result, CONTRACT)


def validator(proofs: dict[str, FiniteProof]):
    def admit(request, recipe):
        node, source, scale = conversion_site(request, recipe.source_node)
        proof = proofs.get(node.id)
        source_attrs = dict(source.attrs)
        if (source.op == "bf16_splat" and type(source_attrs.get("bits")) is int
                and source_attrs["bits"] & 0x7f80 == 0x7f80):
            raise ValueError("a caller contract cannot override a known NaN/Inf source")
        if (proof is None or proof.request_digest != request.digest() or proof.value != source.id
                or proof.dtype != "bf16" or proof.shape != TILE or proof.layout != BF16_LAYOUT
                or proof.finite is not True or proof.may_nan is not False or proof.may_inf is not False
                or proof.origin not in {"derived", "caller_contract"} or not proof.reason):
            raise ValueError("PACK alternative lacks a matching finite/NaN/Inf/layout proof")
        if source.op == "constant":
            binding = next((item for item in request.constants if item.node_id == source.id), None)
            if binding is not None and binding.encoding == "bf16-le-raw":
                payload = bytes.fromhex(binding.data_hex)
                if any(int.from_bytes(payload[index:index+2], "little") & 0x7f80 == 0x7f80
                       for index in range(0, len(payload), 2)):
                    raise ValueError("a caller contract cannot override known NaN/Inf constant bytes")
        if proof.origin == "derived" and proof != prove_finite(request, source.id):
            raise ValueError("derived finite proof does not replay from source semantics")
        attrs = dict(recipe.nodes[-1].attrs) if recipe.nodes else {}
        engine = attrs.get("engine")
        if recipe != make_recipe(request, node.id, engine):
            raise ValueError("recipe structure or clamp/scale/engine certificate differs")
        return RecipeCertificate(request.digest(), recipe.digest(), request.target_identity,
                                 digest({"contract": CONTRACT, "finite_proof": vars(proof),
                                         "scale": scale, "cap": ((scale + 8) << 7) | 96 if scale <= 246 else None,
                                         "inverse": 254 - min(scale, 254), "engine": engine}))
    return admit


def recipe_descriptors(request, recipes):
    """Primitive-step descriptors for extraction; timings deliberately unqualified."""
    descriptors = {}
    nodes = {node.id: node for node in request.nodes}
    for recipe in recipes:
        for node in recipe.nodes:
            sources = tuple(nodes[child] for child in node.inputs)
            attrs = dict(node.attrs)
            engine = attrs.get("engine")
            output_storage = "acc" + str(engine) if node.op == "push_acc_bf16" else (
                "tensor_fp8" if node.op == "pop_acc_fp8" else "tensor_bf16")
            input_storages = ("acc" + str(engine),) if node.op == "pop_acc_fp8" else ("tensor_bf16",) * len(sources)
            input_layouts = tuple(("atlas.mxu" + str(engine) + ".accumulator-slot-32x32-bf16.v1")
                                 if node.op == "pop_acc_fp8" else BF16_LAYOUT for _ in sources)
            output_layout = ("atlas.mxu" + str(engine) + ".accumulator-slot-32x32-bf16.v1") if (
                node.op == "push_acc_bf16") else FP8_LAYOUT if node.op == "pop_acc_fp8" else BF16_LAYOUT
            key = digest((node.op, node.type.record(), [source.type.record() for source in sources], node.attrs))[:20]
            descriptor = InstructionDescriptor("recipe_" + node.op + "_" + key, node.op, input_storages,
                output_storage, node.type.dtype, node.type.numerical_policy, (2,), required_attrs=node.attrs,
                input_dtypes=tuple(source.type.dtype for source in sources),
                input_numerical_policies=tuple(source.type.numerical_policy for source in sources),
                input_ranks=(2,) * len(sources), index_maps=node.index_maps,
                output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                input_axis_bounds=((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),) * len(sources),
                input_layouts=input_layouts, output_layout=output_layout)
            descriptors[descriptor.name] = descriptor
            nodes[node.id] = node
    return tuple(descriptors.values())


def native_pack_descriptor(request, node_id):
    node, source, _ = conversion_site(request, node_id)
    return InstructionDescriptor("native_pack_" + digest((source.type.record(), node.attrs))[:20],
        node.op, ("tensor_bf16",), "tensor_fp8", node.type.dtype, node.type.numerical_policy, (2,),
        required_attrs=node.attrs, input_dtypes=(source.type.dtype,),
        input_numerical_policies=(source.type.numerical_policy,), input_ranks=(2,),
        index_maps=node.index_maps,
        output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
        input_axis_bounds=((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
        input_layouts=(BF16_LAYOUT,), output_layout=FP8_LAYOUT)
