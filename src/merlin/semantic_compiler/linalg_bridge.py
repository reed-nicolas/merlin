"""Conservative parsed-Linalg entry to Merlin's typed semantic kernel graph.

The admitted region uses static integer tensors for contractions, signed i64
indices for checked embedding lookup, opaque f32/BF16/FP8 tensors for
bit-preserving movement including static same-rank slices, exact signed-i64
constant casts to f32, strict f32
projected pointwise arithmetic and explicitly ordered reduction, bounded BF16 ReLU/add forms, and
annotated ordered FP8/BF16 reset and continuation contractions. Every source
operation is accounted for; unsupported bodies and initializers raise. Ordinary
f32 matmul and reductions without a declared order remain source-only nodes
until explicit numerical decisions replace them. Their reduction order is not
inferred from the Linalg operation name.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from math import prod
from struct import pack, unpack

from .model import ConstantBinding, IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
from .reference import TensorValue


class LinalgBridgeError(ValueError):
    pass


@dataclass(frozen=True)
class LinalgTranslation:
    request: KernelRequest
    constants: dict[str, TensorValue]
    source_operations: tuple[str, ...]


def _maps() -> tuple[IndexMap, ...]:
    return (
        IndexMap(3, ((1, 0, 0), (0, 0, 1)), (0, 0)),
        IndexMap(3, ((0, 0, 1), (0, 1, 0)), (0, 0)),
        IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
        IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
    )


def _transpose_maps(permutation: tuple[int, ...]) -> tuple[IndexMap, ...]:
    rank = len(permutation)
    inverse = tuple(permutation.index(axis) for axis in range(rank))
    source = IndexMap(
        rank,
        tuple(tuple(int(loop == inverse[axis]) for loop in range(rank)) for axis in range(rank)),
        (0,) * rank,
    )
    output = IndexMap(
        rank,
        tuple(tuple(int(loop == axis) for loop in range(rank)) for axis in range(rank)),
        (0,) * rank,
    )
    return source, output


def _oriented_matmul_maps() -> tuple[IndexMap, ...]:
    return (
        IndexMap(3, ((1, 0, 0), (0, 0, 1)), (0, 0)),
        IndexMap(3, ((0, 1, 0), (0, 0, 1)), (0, 0)),
        IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
    )


def _oriented_matmul_accumulate_maps() -> tuple[IndexMap, ...]:
    maps = _oriented_matmul_maps()
    return (maps[0], maps[1], maps[2], maps[2])


def _source_f32_batched_matmul(op) -> tuple[object, object, object, tuple[IndexMap, ...], str]:
    """Recognize a batched contraction while retaining its unresolved f32 order."""
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import DenseArrayBase
    from xdsl.dialects.linalg.attrs import IteratorType, IteratorTypeAttr
    from xdsl.dialects.linalg.ops import GenericOp, YieldOp
    from xdsl.ir.affine import AffineDimExpr

    if (
        type(op) is not GenericOp or len(op.inputs) != 2 or len(op.outputs) != 1
        or len(op.results) != 1 or op.successors
        or set(op.properties) != {"indexing_maps", "iterator_types", "operandSegmentSizes"}
        or any(not key.startswith("prov.") for key in op.attributes)
        or len(op.regions) != 1 or len(op.regions[0].blocks) != 1
    ):
        raise LinalgBridgeError("source f32 batched matmul needs one closed Linalg region")
    segments = op.properties["operandSegmentSizes"]
    if (not isinstance(segments, DenseArrayBase) or str(segments.elt_type) != "i32"
            or tuple(segments.iter_values()) != (2, 1)):
        raise LinalgBridgeError("source f32 batched matmul has invalid operand segments")
    lhs, rhs, init = (*op.inputs, op.outputs[0])
    lhs_type, rhs_type, init_type, result_type = (
        _type(value.type) for value in (lhs, rhs, init, op.results[0])
    )
    result_rank = len(result_type.shape)
    if (
        result_rank < 3 or any(value.dtype != "f32" or value.numerical_policy != "raw-bit-pattern"
                               for value in (lhs_type, rhs_type, init_type, result_type))
        or any(len(value.shape) != result_rank for value in (lhs_type, rhs_type, init_type))
        or init_type != result_type
        or lhs_type.shape[:-2] != result_type.shape[:-2]
        or rhs_type.shape[:-2] != result_type.shape[:-2]
        or lhs_type.shape[-2] != result_type.shape[-2]
        or rhs_type.shape[-1] != result_type.shape[-1]
        or lhs_type.shape[-1] != rhs_type.shape[-2]
    ):
        raise LinalgBridgeError("source f32 batched matmul has incompatible shapes or types")
    iterators = op.iterator_types.data
    if (len(iterators) != result_rank + 1
            or any(not isinstance(iterator, IteratorTypeAttr) for iterator in iterators)
            or tuple(iterator.data for iterator in iterators)
            != (IteratorType.PARALLEL,) * result_rank + (IteratorType.REDUCTION,)):
        raise LinalgBridgeError("source f32 batched matmul has invalid reduction iterators")
    batch_axes = tuple(range(result_rank - 2))
    m_axis, n_axis, k_axis = result_rank - 2, result_rank - 1, result_rank
    expected_axes = (
        (*batch_axes, m_axis, k_axis),
        (*batch_axes, k_axis, n_axis),
        (*batch_axes, m_axis, n_axis),
    )
    maps = tuple(attribute.data for attribute in op.get_indexing_maps())
    if len(maps) != 3 or any(
        mapping.num_dims != result_rank + 1 or mapping.num_symbols != 0
        or len(mapping.results) != len(axes)
        or any(type(expression) is not AffineDimExpr or expression.position != axis
               for expression, axis in zip(mapping.results, axes, strict=True))
        for mapping, axes in zip(maps, expected_axes, strict=True)
    ):
        raise LinalgBridgeError("source f32 batched matmul has nonstandard index maps")
    block = op.regions[0].block
    body = tuple(block.ops)
    if (tuple(str(argument.type) for argument in block.args) != ("f32",) * 3
            or len(body) != 3 or tuple(nested.name for nested in body)
            != ("arith.mulf", "arith.addf", "linalg.yield")
            or type(body[2]) is not YieldOp
            or any(nested.regions or nested.successors
                   or any(not key.startswith("prov.") for key in nested.attributes)
                   for nested in body)
            or any(set(arithmetic.properties) - {"fastmath"} for arithmetic in body[:2])
            or any((flags := arithmetic.properties.get("fastmath")) is not None
                   and (not isinstance(flags, arith.FastMathFlagsAttr) or flags.data)
                   for arithmetic in body[:2])
            or body[2].properties
            or any(len(arithmetic.results) != 1 or str(arithmetic.results[0].type) != "f32"
                   for arithmetic in body[:2])
            or tuple(body[0].operands) != tuple(block.args[:2])
            or tuple(body[2].operands) != (body[1].results[0],)):
        raise LinalgBridgeError("source f32 batched matmul has an unqualified scalar body")
    add_operands = tuple(body[1].operands)
    if add_operands == (body[0].results[0], block.args[2]):
        add_order = "product-accumulator"
    elif add_operands == (block.args[2], body[0].results[0]):
        add_order = "accumulator-product"
    else:
        raise LinalgBridgeError("source f32 batched matmul adds the wrong scalar operands")
    try:
        op.verify()
    except Exception as exc:  # noqa: BLE001 - xDSL owns source verification
        raise LinalgBridgeError(f"source f32 batched matmul failed verification: {exc}") from exc
    index_maps = tuple(
        IndexMap(result_rank + 1,
                 tuple(tuple(int(loop == axis) for loop in range(result_rank + 1)) for axis in axes),
                 (0,) * len(axes))
        for axes in (*expected_axes, expected_axes[-1])
    )
    return lhs, rhs, init, index_maps, add_order


def _f32_reduce_maps(rank: int, axes: tuple[int, ...]) -> tuple[IndexMap, ...]:
    source = IndexMap(
        rank,
        tuple(tuple(int(loop == dim) for loop in range(rank)) for dim in range(rank)),
        (0,) * rank,
    )
    projection = IndexMap(
        rank,
        tuple(tuple(int(loop == dim) for loop in range(rank))
              for dim in range(rank) if dim not in axes),
        (0,) * (rank - len(axes)),
    )
    return source, projection, projection


def _recognize_f32_reduce(op) -> tuple[object, object, tuple[int, ...], str, str, str]:
    """Check one f32 add or maximum reduction without inventing its order.

    The attribute is a source promise. Parsing cannot prove that a prior
    producer or an external model intended this execution order.
    """
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import DenseArrayBase, StringAttr
    from xdsl.dialects.linalg.ops import ReduceOp, YieldOp

    if (
        type(op) is not ReduceOp or set(op.properties) != {"dimensions"}
        or any(key != "merlin.reduction_order" and not key.startswith("prov.")
               for key in op.attributes)
        or op.successors or len(op.operands) != 2
        or tuple(op.operands) != (op.input, op.init)
        or len(op.results) != 1 or len(op.regions) != 1
        or len(op.regions[0].blocks) != 1
    ):
        raise LinalgBridgeError("ordered f32 reduction needs a closed registered linalg.reduce")
    order = op.attributes.get("merlin.reduction_order")
    if order is not None and (not isinstance(order, StringAttr) or order.data != "axis-ascending"):
        raise LinalgBridgeError("f32 linalg.reduce has an unqualified numerical-order declaration")
    source_type, init_type, result_type = (_type(value.type) for value in (op.input, op.init, op.results[0]))
    dimensions = op.properties["dimensions"]
    if not isinstance(dimensions, DenseArrayBase) or str(dimensions.elt_type) != "i64":
        raise LinalgBridgeError("ordered f32 reduction needs a typed i64 axis")
    axes = tuple(dimensions.iter_values())
    rank = len(source_type.shape)
    if (
        not axes or len(axes) >= rank or len(set(axes)) != len(axes)
        or any(type(axis) is not int or not 0 <= axis < rank for axis in axes)
        or order is not None and len(axes) != 1
        or rank < 2 or source_type.dtype != "f32"
        or source_type.numerical_policy != "raw-bit-pattern"
        or init_type != result_type
        or result_type != TensorType(
            tuple(extent for dim, extent in enumerate(source_type.shape) if dim not in axes),
            "f32", "raw-bit-pattern",
        )
    ):
        raise LinalgBridgeError("ordered f32 reduction has an unsupported axis, shape, or type")
    block = op.regions[0].block
    body = tuple(block.ops)
    if len(block.args) != 2 or len(body) != 2:
        raise LinalgBridgeError("ordered f32 reduction needs one add and one yield")
    add, yld = body
    flags = add.properties.get("fastmath")
    if (
        add.name not in {"arith.addf", "arith.maximumf"} or type(yld) is not YieldOp
        or any(nested.regions or nested.successors
               or any(not key.startswith("prov.") for key in nested.attributes)
               for nested in body)
        or set(add.properties) - {"fastmath"} or yld.properties
        or flags is not None and (not isinstance(flags, arith.FastMathFlagsAttr) or flags.data)
        or tuple(str(arg.type) for arg in block.args) != ("f32", "f32")
        or tuple(add.operands) not in (tuple(block.args), tuple(reversed(block.args)))
        or len(add.results) != 1 or str(add.results[0].type) != "f32"
        or tuple(yld.operands) != (add.results[0],)
    ):
        raise LinalgBridgeError("f32 reduction scalar body or fast-math flags are unqualified")
    add_order = ("input-accumulator" if tuple(add.operands) == tuple(block.args)
                 else "accumulator-input")
    if order is not None and (add.name != "arith.addf" or add_order != "input-accumulator"):
        raise LinalgBridgeError("ordered f32 reduction changes its declared scalar add order or operation")
    try:
        op.verify()
    except Exception as exc:  # noqa: BLE001 - xDSL owns source verification
        raise LinalgBridgeError(f"ordered f32 reduction failed verification: {exc}") from exc
    return op.input, op.init, axes, "unresolved" if order is None else order.data, add_order, add.name


def _linear_affine_expression(expression, rank: int) -> tuple[tuple[int, ...], int]:
    """Accept only integer-affine dimension/add/constant-multiply expressions."""
    from xdsl.ir.affine import AffineBinaryOpExpr, AffineBinaryOpKind, AffineConstantExpr, AffineDimExpr

    if isinstance(expression, AffineDimExpr) and 0 <= expression.position < rank:
        return tuple(int(axis == expression.position) for axis in range(rank)), 0
    if isinstance(expression, AffineConstantExpr):
        return (0,) * rank, expression.value
    if isinstance(expression, AffineBinaryOpExpr):
        left, left_offset = _linear_affine_expression(expression.lhs, rank)
        right, right_offset = _linear_affine_expression(expression.rhs, rank)
        if expression.kind is AffineBinaryOpKind.Add:
            return tuple(a + b for a, b in zip(left, right)), left_offset + right_offset
        if expression.kind is AffineBinaryOpKind.Mul:
            if not any(left):
                return tuple(left_offset * value for value in right), left_offset * right_offset
            if not any(right):
                return tuple(right_offset * value for value in left), right_offset * left_offset
    raise LinalgBridgeError("indexed copy has a nonlinear or unsupported affine expression")


def _static_index_copy_map(op, source_type: TensorType, result_type: TensorType) -> tuple[IndexMap, ...]:
    """Check a one-input all-parallel Linalg region yielding its input scalar."""
    from xdsl.dialects.linalg.attrs import IteratorType

    if (
        len(op.inputs) != 1 or len(op.outputs) != 1 or len(op.results) != 1
        or set(op.properties) != {"indexing_maps", "iterator_types", "operandSegmentSizes"}
        or tuple(op.properties["operandSegmentSizes"].get_values()) != (1, 1)
        or any(not key.startswith("prov.") for key in op.attributes)
        or len(op.body.blocks) != 1
    ):
        raise LinalgBridgeError("indexed copy has an unsupported Linalg signature")
    body = op.body.block
    operations = tuple(body.ops)
    if (
        len(body.args) != 2 or len(operations) != 1 or operations[0].name != "linalg.yield"
        or len(operations[0].operands) != 1 or operations[0].operands[0] is not body.args[0]
        or operations[0].properties
        or any(not key.startswith("prov.") for key in operations[0].attributes)
        or str(body.args[0].type) != source_type.dtype
        or str(body.args[1].type) != result_type.dtype
    ):
        raise LinalgBridgeError("indexed copy body must yield the unchanged input scalar")
    rank = len(result_type.shape)
    maps = tuple(attribute.data for attribute in op.get_indexing_maps())
    if (
        len(maps) != 2
        or len(op.get_iterator_types()) != rank
        or any(attribute.data is not IteratorType.PARALLEL for attribute in op.get_iterator_types())
        or any(mapping.num_dims != rank or mapping.num_symbols != 0 for mapping in maps)
        or len(maps[0].results) != len(source_type.shape)
        or len(maps[1].results) != rank
    ):
        raise LinalgBridgeError("indexed copy needs all-parallel loops and static affine maps")
    converted = []
    for mapping in maps:
        rows = tuple(_linear_affine_expression(expression, rank) for expression in mapping.results)
        converted.append(IndexMap(rank, tuple(row for row, _ in rows), tuple(offset for _, offset in rows)))
    identity = IndexMap(
        rank, tuple(tuple(int(axis == dim) for axis in range(rank)) for dim in range(rank)),
        (0,) * rank,
    )
    if converted[1] != identity:
        raise LinalgBridgeError("indexed copy destination map must cover every output element once")
    for row, offset, source_extent in zip(
        converted[0].coefficients, converted[0].offsets, source_type.shape,
    ):
        minimum = offset + sum(min(0, coefficient * (extent - 1))
                               for coefficient, extent in zip(row, result_type.shape))
        maximum = offset + sum(max(0, coefficient * (extent - 1))
                               for coefficient, extent in zip(row, result_type.shape))
        if minimum < 0 or maximum >= source_extent:
            raise LinalgBridgeError("indexed copy input map exceeds source bounds")
    return tuple(converted)


def _static_f32_square_maps(op, source_type: TensorType, result_type: TensorType) -> tuple[IndexMap, ...]:
    """Recognize one strict f32 multiplication of each source element by itself."""
    from xdsl.dialects import arith
    from xdsl.dialects.linalg.attrs import IteratorType

    if (
        len(op.inputs) != 1 or len(op.outputs) != 1 or len(op.results) != 1
        or set(op.properties) != {"indexing_maps", "iterator_types", "operandSegmentSizes"}
        or tuple(op.properties["operandSegmentSizes"].get_values()) != (1, 1)
        or any(not key.startswith("prov.") for key in op.attributes)
        or len(op.body.blocks) != 1
        or source_type != result_type
        or source_type.dtype != "f32" or source_type.numerical_policy != "raw-bit-pattern"
    ):
        raise LinalgBridgeError("f32 square has an unsupported type or Linalg signature")
    body = op.body.block
    operations = tuple(body.ops)
    if len(body.args) != 2 or len(operations) != 2:
        raise LinalgBridgeError("f32 square needs one multiplication and one yield")
    multiply, yield_op = operations
    flags = multiply.properties.get("fastmath")
    if (
        (multiply.name, yield_op.name) != ("arith.mulf", "linalg.yield")
        or any(nested.regions or nested.successors
               or any(not key.startswith("prov.") for key in nested.attributes)
               for nested in operations)
        or set(multiply.properties) - {"fastmath"} or yield_op.properties
        or flags is not None and (not isinstance(flags, arith.FastMathFlagsAttr) or flags.data)
        or tuple(multiply.operands) != (body.args[0], body.args[0])
        or len(multiply.results) != 1 or str(multiply.results[0].type) != "f32"
        or tuple(yield_op.operands) != (multiply.results[0],)
        or any(str(arg.type) != "f32" for arg in body.args)
    ):
        raise LinalgBridgeError("f32 square must multiply its input by itself without fast math")
    rank = len(result_type.shape)
    maps = tuple(attribute.data for attribute in op.get_indexing_maps())
    if (
        len(maps) != 2 or len(op.get_iterator_types()) != rank
        or any(attribute.data is not IteratorType.PARALLEL for attribute in op.get_iterator_types())
        or any(mapping.num_dims != rank or mapping.num_symbols != 0 for mapping in maps)
    ):
        raise LinalgBridgeError("f32 square needs all-parallel static indexing")
    converted = tuple(
        IndexMap(rank, tuple(row for row, _ in rows), tuple(offset for _, offset in rows))
        for rows in (
            tuple(_linear_affine_expression(expression, rank) for expression in mapping.results)
            for mapping in maps
        )
    )
    if converted != pointwise_identity_maps(rank, 1):
        raise LinalgBridgeError("f32 square requires identical source and output index maps")
    return pointwise_identity_maps(rank, 2)


def _embedding_lookup_operands(op) -> tuple[object, object, object]:
    """Recognize a checked tensor-table read indexed by one signed i64 tensor."""
    from xdsl.dialects.linalg.attrs import IteratorType
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    if (
        len(op.inputs) != 1 or len(op.outputs) != 1 or len(op.results) != 1
        or set(op.properties) != {"indexing_maps", "iterator_types", "operandSegmentSizes"}
        or tuple(op.properties["operandSegmentSizes"].get_values()) != (1, 1)
        or any(not key.startswith("prov.") for key in op.attributes)
        or len(op.body.blocks) != 1
    ):
        raise LinalgBridgeError("embedding lookup has an unsupported Linalg signature")
    indices, destination = op.inputs[0], op.outputs[0]
    body = op.body.block
    body_operations = tuple(body.ops)
    if len(body_operations) != 4:
        raise LinalgBridgeError("embedding lookup body must read one table row and one output channel")
    cast, channel, extract, yield_op = body_operations
    if (
        len(body.args) != 2 or tuple(arg.name for arg in (cast, channel, extract, yield_op))
        != ("arith.index_cast", "linalg.index", "tensor.extract", "linalg.yield")
        or any(nested.regions or nested.successors
               or any(not key.startswith("prov.") for key in nested.attributes)
               for nested in (cast, channel, extract, yield_op))
        or cast.properties or extract.properties or yield_op.properties
        or set(channel.properties) != {"dim"}
        or str(body.args[0].type) != "i64" or str(body.args[1].type) != "f32"
        or tuple(cast.operands) != (body.args[0],) or len(cast.results) != 1
        or str(cast.results[0].type) != "index"
        or channel.operands or len(channel.results) != 1 or str(channel.results[0].type) != "index"
        or len(extract.operands) != 3 or len(extract.results) != 1
        or tuple(extract.operands[1:]) != (cast.results[0], channel.results[0])
        or str(extract.results[0].type) != "f32"
        or tuple(yield_op.operands) != (extract.results[0],)
    ):
        raise LinalgBridgeError("embedding lookup body must read one table row and one output channel")
    table = extract.operands[0]
    indices_type, table_type, destination_type, result_type = (
        _type(value.type) for value in (indices, table, destination, op.results[0])
    )
    output_rank = len(result_type.shape)
    if (
        indices_type.dtype != "i64" or indices_type.numerical_policy != "signed-i64"
        or table_type.dtype != "f32" or table_type.numerical_policy != "raw-bit-pattern"
        or len(table_type.shape) != 2 or result_type != destination_type
        or result_type != TensorType((*indices_type.shape, table_type.shape[1]), "f32", "raw-bit-pattern")
        or getattr(getattr(channel.properties["dim"], "value", None), "data", None) != output_rank - 1
        or len(op.get_iterator_types()) != output_rank
        or any(item.data is not IteratorType.PARALLEL for item in op.get_iterator_types())
    ):
        raise LinalgBridgeError("embedding lookup has an unsupported shape, type, or channel index")
    dimensions = tuple(AffineDimExpr(dim) for dim in range(output_rank))
    expected = (
        AffineMap(output_rank, 0, dimensions[:-1]),
        AffineMap(output_rank, 0, dimensions),
    )
    if tuple(item.data for item in op.get_indexing_maps()) != expected:
        raise LinalgBridgeError("embedding lookup maps differ from the output prefix and channel")
    return indices, table, destination


def _type(attribute) -> TensorType:
    from xdsl.dialects.builtin import TensorType as XdslTensorType

    if not isinstance(attribute, XdslTensorType):
        raise LinalgBridgeError("only ranked tensor values are admitted")
    shape = tuple(int(dimension) for dimension in attribute.get_shape())
    dtype = str(attribute.element_type)
    if dtype == "i8":
        policy = "signed-i8"
    elif dtype == "i32":
        policy = "i32-wrap-k-ascending"
    elif dtype == "i64":
        policy = "signed-i64"
    elif dtype == "bf16":
        policy = "raw-bit-pattern"
    elif dtype == "f32":
        # Source f32 arithmetic needs a separate declared precision policy.
        # Its boundary values and bit-preserving moves still have exact types.
        policy = "raw-bit-pattern"
    elif dtype == "f8E4M3FN":
        dtype = "fp8_e4m3"
        policy = "raw-bit-pattern"
    else:
        raise LinalgBridgeError(f"no admitted numerical policy for {dtype}")
    try:
        return TensorType(shape, dtype, policy)
    except ValueError as exc:
        raise LinalgBridgeError(f"unadmitted tensor type {attribute}: {exc}") from exc


def _scalar_integer(op) -> int:
    attr = op.properties.get("value", op.attributes.get("value"))
    value = getattr(getattr(attr, "value", None), "data", None)
    if type(value) is not int or str(op.results[0].type) != "i32":
        raise LinalgBridgeError("fill scalar must be an explicit i32 integer constant")
    if not -(1 << 31) <= value < (1 << 31):
        raise LinalgBridgeError("fill constant is outside signed i32")
    return value


def _scalar_i64(op) -> int:
    attr = op.properties.get("value", op.attributes.get("value"))
    value = getattr(getattr(attr, "value", None), "data", None)
    if type(value) is not int or str(op.results[0].type) != "i64":
        raise LinalgBridgeError("fill scalar must be an explicit i64 integer constant")
    if not -(1 << 63) <= value < (1 << 63):
        raise LinalgBridgeError("fill constant is outside signed i64")
    return value


def _i64_to_f32_bits(value: int) -> int:
    """Round a signed i64 exactly to binary32, ties to even."""
    if not -(1 << 63) <= value < (1 << 63):
        raise LinalgBridgeError("signed integer cast is outside i64")
    magnitude = abs(value)
    if magnitude == 0:
        return 0
    exponent = magnitude.bit_length() - 1
    shift = max(0, exponent - 23)
    significand = magnitude >> shift
    if shift:
        remainder = magnitude - (significand << shift)
        halfway = 1 << (shift - 1)
        if remainder > halfway or remainder == halfway and significand & 1:
            significand += 1
    else:
        significand <<= 23 - exponent
    if significand == 1 << 24:
        significand >>= 1
        exponent += 1
    return ((int(value < 0) << 31) | ((exponent + 127) << 23)
            | (significand & ((1 << 23) - 1)))


def _scalar_f32_bits(op) -> int:
    from math import isnan

    from xdsl.dialects.builtin import FloatAttr

    attr = op.properties.get("value", op.attributes.get("value"))
    if str(op.results[0].type) != "f32" or not isinstance(attr, FloatAttr):
        raise LinalgBridgeError("fill scalar must be an explicit f32 constant")
    value = float(attr.value.data)
    if isnan(value):
        raise LinalgBridgeError("f32 NaN scalar payload cannot be preserved by this bridge")
    try:
        return unpack("<I", pack("<f", value))[0]
    except OverflowError as exc:
        raise LinalgBridgeError("f32 fill scalar is outside its storage range") from exc


def _named_matmul_maps_are_default(op) -> bool:
    from xdsl.dialects.linalg.attrs import IteratorType
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    d0, d1, d2 = (AffineDimExpr(i) for i in range(3))
    expected = (
        AffineMap(3, 0, (d0, d2)),
        AffineMap(3, 0, (d2, d1)),
        AffineMap(3, 0, (d0, d1)),
    )
    return tuple(attr.data for attr in op.get_indexing_maps()) == expected and tuple(
        attr.data for attr in op.get_iterator_types()
    ) == (IteratorType.PARALLEL, IteratorType.PARALLEL, IteratorType.REDUCTION)


def translate_linalg_text(
    source: str,
    *,
    entry: str,
    target_identity: str,
    lowering_policy: str = "strict-native",
    boundary_storage: str = "external",
    argument_constants: Mapping[int, bytes] | None = None,
) -> LinalgTranslation:
    """Translate one admitted function, preserving its init and ordered returns.

    The source is parsed by Merlin's established xDSL frontend. The shared
    generic-matmul recognizer checks maps, iterators, scalar body,
    signed extension, widths, and operand connections before translation.
    """
    from xdsl.dialects.builtin import DictionaryAttr, IntegerAttr, StringAttr
    from xdsl.dialects.linalg.attrs import IteratorType

    from merlin.frontends.linalg_bf16_add import (
        recognize_bf16_f32_chop_add,
        recognize_bf16_f32_chop_sub,
    )
    from merlin.frontends.linalg_bf16_mul import recognize_bf16_product_rne
    from merlin.frontends.linalg_bf16_relu import recognize_bf16_sign_clamp_relu
    from merlin.frontends.linalg_composite_math import recognize_static_composite_math_body
    from merlin.frontends.linalg_fp8_cast import recognize_bf16_fp8_truncation
    from merlin.frontends.linalg_math_patterns import recognize_static_f32_exp_rsqrt_body
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.frontends.linalg_ordered_fp8 import (
        is_positive_bf16_zero,
        recognize_ordered_fp8_bf16_reset_matmul,
    )
    from merlin.frontends.linalg_patterns import (
        InvalidLinalgPattern,
        recognize_f32_relu_ordered_select,
        recognize_signed_i8_i32_matmul,
        recognize_static_f32_division,
        recognize_static_projected_pointwise,
    )

    module = parse_mlir_text(source)
    functions = [op for op in module.walk() if op.name == "func.func" and op.sym_name.data == entry]
    if len(functions) != 1:
        raise LinalgBridgeError(f"expected exactly one entry function @{entry}")
    function = functions[0]
    if len(function.body.blocks) != 1:
        raise LinalgBridgeError("entry function must have one block")
    block = function.body.block
    argument_constants = {} if argument_constants is None else dict(argument_constants)
    if any(
        type(index) is not int or not 0 <= index < len(block.args) or not isinstance(data, bytes)
        for index, data in argument_constants.items()
    ):
        raise LinalgBridgeError("lifted argument constants have invalid indices or bytes")
    operations = tuple(block.ops)
    if not operations or operations[-1].name != "func.return":
        raise LinalgBridgeError("entry function must end in func.return")

    nodes: list[SemanticNode] = []
    env: dict[object, str | None] = {}
    scalars: dict[object, int] = {}
    i64_scalars: dict[object, int] = {}
    f32_scalars: dict[object, int] = {}
    bf16_zeros: set[object] = set()
    zero_fills: set[object] = set()
    constants: dict[str, TensorValue] = {}
    constant_bindings: list[ConstantBinding] = []
    boundaries: list[tuple[str, str]] = []
    source_operations: list[str] = []
    argument_attrs = function.arg_attrs.data if function.arg_attrs is not None else (None,) * len(block.args)
    if len(argument_attrs) != len(block.args):
        raise LinalgBridgeError("entry argument attributes do not match the function signature")
    for index, (arg, attributes) in enumerate(zip(block.args, argument_attrs)):
        type_ = _type(arg.type)
        node_id = f"arg{index}"
        domain_attrs: tuple[tuple[str, str], ...] = ()
        if attributes is not None:
            if not isinstance(attributes, DictionaryAttr) or any(
                key != "merlin.input_domain" and not key.startswith("prov.") for key in attributes.data
            ):
                raise LinalgBridgeError("entry argument has unknown attributes")
            domain = attributes.data.get("merlin.input_domain")
            if domain is not None:
                if not isinstance(domain, StringAttr) or not domain.data:
                    raise LinalgBridgeError("entry input domain must be a nonempty string")
                domain_attrs = (("input_domain", domain.data),)
        data = argument_constants.get(index)
        if data is None:
            nodes.append(SemanticNode(node_id, "input", (), type_, attrs=domain_attrs, effect="input"))
            boundaries.append((node_id, boundary_storage))
        else:
            encoding, width, signed = {
                "i8": ("i8", 1, True),
                "i32": ("i32-le", 4, True),
                "i64": ("i64-le", 8, True),
                "bf16": ("bf16-le-raw", 2, False),
                "f32": ("f32-le-raw", 4, False),
                "fp8_e4m3": ("fp8_e4m3-raw", 1, False),
            }[type_.dtype]
            if len(data) != prod(type_.shape) * width:
                raise LinalgBridgeError(f"lifted argument {index} differs from its parsed tensor type")
            nodes.append(SemanticNode(node_id, "constant", (), type_, attrs=domain_attrs, effect="constant"))
            constants[node_id] = TensorValue(type_, tuple(
                int.from_bytes(data[offset:offset + width], "little", signed=signed)
                for offset in range(0, len(data), width)
            ))
            constant_bindings.append(ConstantBinding(node_id, encoding, data.hex()))
        env[arg] = node_id

    for ordinal, op in enumerate(operations):
        source_operations.append(op.name)
        if op.name == "arith.constant":
            if len(op.results) != 1:
                raise LinalgBridgeError("scalar constant has an unsupported result count")
            if is_positive_bf16_zero(op):
                bf16_zeros.add(op.results[0])
            elif str(op.results[0].type) == "f32":
                f32_scalars[op.results[0]] = _scalar_f32_bits(op)
            elif str(op.results[0].type) == "i64":
                i64_scalars[op.results[0]] = _scalar_i64(op)
            else:
                scalars[op.results[0]] = _scalar_integer(op)
        elif op.name == "arith.sitofp":
            if (len(op.operands) != 1 or len(op.results) != 1 or op.properties
                    or any(not key.startswith("prov.") for key in op.attributes)
                    or str(op.operands[0].type) != "i64" or str(op.results[0].type) != "f32"
                    or op.operands[0] not in i64_scalars):
                raise LinalgBridgeError("scalar signed-i64-to-f32 cast needs a declared constant")
            try:
                op.verify()
            except Exception as exc:  # noqa: BLE001 - xDSL owns source verification
                raise LinalgBridgeError(f"scalar signed-i64-to-f32 cast failed verification: {exc}") from exc
            f32_scalars[op.results[0]] = _i64_to_f32_bits(i64_scalars[op.operands[0]])
        elif op.name == "tensor.empty":
            if len(op.results) != 1:
                raise LinalgBridgeError("tensor.empty has an unsupported result count")
            _type(op.results[0].type)
            env[op.results[0]] = None
        elif op.name == "tensor.splat":
            if len(op.operands) != 1 or len(op.results) != 1 or op.properties or any(
                not key.startswith("prov.") for key in op.attributes
            ):
                raise LinalgBridgeError("tensor.splat has an unsupported signature or attributes")
            scalar = op.operands[0]
            type_ = _type(op.results[0].type)
            node_id = f"op{ordinal}r0"
            count = prod(type_.shape)
            if type_.dtype == "f32" and scalar in f32_scalars:
                bits = f32_scalars[scalar]
                value, encoding, data = bits, "f32-le-raw", bits.to_bytes(4, "little")
            elif type_.dtype == "i32" and scalar in scalars:
                value = scalars[scalar]
                encoding, data = "i32-le", value.to_bytes(4, "little", signed=True)
            elif type_.dtype == "i64" and scalar in i64_scalars:
                value = i64_scalars[scalar]
                encoding, data = "i64-le", value.to_bytes(8, "little", signed=True)
            elif type_.dtype == "bf16" and scalar in bf16_zeros:
                value, encoding, data = 0, "bf16-le-raw", b"\x00\x00"
            else:
                raise LinalgBridgeError("tensor.splat requires an admitted constant with matching element type")
            attrs = (("value", value),) if type_.dtype in {"i32", "i64"} else ()
            nodes.append(SemanticNode(node_id, "constant", (), type_, attrs=attrs, effect="constant"))
            constants[node_id] = TensorValue(type_, (value,) * count)
            constant_bindings.append(ConstantBinding(node_id, encoding, (data * count).hex()))
            env[op.results[0]] = node_id
        elif op.name in {"tensor.expand_shape", "tensor.collapse_shape"}:
            expected_properties = (
                {"reassociation", "static_output_shape"}
                if op.name == "tensor.expand_shape" else {"reassociation"}
            )
            if (len(op.operands) != 1 or len(op.results) != 1 or set(op.properties) != expected_properties
                or any(not key.startswith("prov.") for key in op.attributes)):
                raise LinalgBridgeError("tensor reshape has an unsupported signature or attributes")
            source_value = op.operands[0]
            if source_value not in env or env[source_value] is None:
                raise LinalgBridgeError("tensor reshape reads an undefined or uninitialized source")
            source_type, result_type = _type(source_value.type), _type(op.results[0].type)
            if (source_type.dtype, source_type.numerical_policy) != (
                result_type.dtype, result_type.numerical_policy
            ):
                raise LinalgBridgeError("tensor reshape changes element type or numerical policy")
            groups = tuple(tuple(int(axis.value.data) for axis in group.data) for group in op.reassociation.data)
            expanded = op.name == "tensor.expand_shape"
            partitioned_shape = result_type.shape if expanded else source_type.shape
            grouped_shape = source_type.shape if expanded else result_type.shape
            if (len(groups) != len(grouped_shape) or any(not group for group in groups)
                or tuple(axis for group in groups for axis in group) != tuple(range(len(partitioned_shape)))
                or any(prod(partitioned_shape[axis] for axis in group) != extent
                       for group, extent in zip(groups, grouped_shape))
                or prod(source_type.shape) != prod(result_type.shape)
                or (expanded and tuple(op.static_output_shape.get_values()) != result_type.shape)):
                raise LinalgBridgeError("tensor reshape has invalid static reassociation or shape")
            node_id = f"op{ordinal}r0"
            nodes.append(SemanticNode(node_id, "reshape", (env[source_value],), result_type))
            env[op.results[0]] = node_id
        elif op.name == "tensor.concat":
            if (len(op.operands) < 2 or len(op.results) != 1 or set(op.properties) != {"dim"}
                    or op.regions or op.successors
                    or any(not key.startswith("prov.") for key in op.attributes)):
                raise LinalgBridgeError("tensor.concat needs two or more tensors and one static axis")
            dim = op.properties["dim"]
            if not isinstance(dim, IntegerAttr) or str(dim.type) != "i64":
                raise LinalgBridgeError("tensor.concat axis must be a typed i64 integer")
            axis = dim.value.data
            result_type = _type(op.results[0].type)
            source_types = tuple(_type(operand.type) for operand in op.operands)
            if (not 0 <= axis < len(result_type.shape)
                    or any(operand not in env or env[operand] is None for operand in op.operands)
                    or any(len(source.shape) != len(result_type.shape)
                           or (source.dtype, source.numerical_policy)
                           != (result_type.dtype, result_type.numerical_policy)
                           or any(source.shape[dim] != result_type.shape[dim]
                                  for dim in range(len(result_type.shape)) if dim != axis)
                           for source in source_types)
                    or sum(source.shape[axis] for source in source_types) != result_type.shape[axis]):
                raise LinalgBridgeError("tensor.concat has incompatible source shape, type, or axis")
            node_id = f"op{ordinal}r0"
            nodes.append(SemanticNode(
                node_id, "concat", tuple(env[operand] for operand in op.operands), result_type,
                attrs=(("axis", axis),),
            ))
            env[op.results[0]] = node_id
        elif op.name == "tensor.extract_slice":
            if (
                len(op.operands) != 1 or len(op.results) != 1
                or set(op.properties) != {
                    "static_offsets", "static_sizes", "static_strides", "operandSegmentSizes",
                }
                or tuple(op.properties["operandSegmentSizes"].get_values()) != (1, 0, 0, 0)
                or op.offsets or op.sizes or op.strides
                or any(not key.startswith("prov.") for key in op.attributes)
            ):
                raise LinalgBridgeError("tensor.extract_slice needs one source and fully static indices")
            source_value = op.operands[0]
            if source_value not in env or env[source_value] is None:
                raise LinalgBridgeError("tensor.extract_slice reads an undefined or uninitialized source")
            source_type, result_type = _type(source_value.type), _type(op.results[0].type)
            rank = len(source_type.shape)
            offsets = tuple(op.properties["static_offsets"].get_values())
            sizes = tuple(op.properties["static_sizes"].get_values())
            strides = tuple(op.properties["static_strides"].get_values())
            if (
                len(result_type.shape) != rank or sizes != result_type.shape
                or (source_type.dtype, source_type.numerical_policy)
                != (result_type.dtype, result_type.numerical_policy)
                or any(len(values) != rank for values in (offsets, sizes, strides))
            ):
                raise LinalgBridgeError("tensor.extract_slice has a rank-reduced or mismatched result")
            if any(offset < 0 or stride <= 0 for offset, stride in zip(offsets, strides)):
                raise LinalgBridgeError("tensor.extract_slice has an invalid offset or stride")
            if any(
                offset + (size - 1) * stride >= extent
                for offset, size, stride, extent in zip(offsets, sizes, strides, source_type.shape)
            ):
                raise LinalgBridgeError("tensor.extract_slice exceeds source bounds")
            source_map = IndexMap(
                rank,
                tuple(tuple(strides[dim] if axis == dim else 0 for axis in range(rank))
                      for dim in range(rank)),
                offsets,
            )
            identity = IndexMap(
                rank, tuple(tuple(int(axis == dim) for axis in range(rank)) for dim in range(rank)),
                (0,) * rank,
            )
            node_id = f"op{ordinal}r0"
            nodes.append(SemanticNode(
                node_id, "gather", (env[source_value],), result_type,
                index_maps=(source_map, identity),
            ))
            env[op.results[0]] = node_id
        elif op.name == "tensor.insert_slice":
            if (
                len(op.operands) != 2 or len(op.results) != 1
                or set(op.properties) != {
                    "static_offsets", "static_sizes", "static_strides", "operandSegmentSizes",
                }
                or tuple(op.properties["operandSegmentSizes"].get_values()) != (1, 1, 0, 0, 0)
                or op.offsets or op.sizes or op.strides
                or any(not key.startswith("prov.") for key in op.attributes)
            ):
                raise LinalgBridgeError("tensor.insert_slice needs a fully static source and destination")
            source_value, destination = op.source, op.dest
            if source_value not in env or env[source_value] is None:
                raise LinalgBridgeError("tensor.insert_slice reads an undefined source")
            if destination not in env or env[destination] is None:
                raise LinalgBridgeError("tensor.insert_slice reads an uninitialized destination")
            source_type, destination_type, result_type = (
                _type(value.type) for value in (source_value, destination, op.results[0])
            )
            rank = len(source_type.shape)
            offsets = tuple(op.properties["static_offsets"].get_values())
            sizes = tuple(op.properties["static_sizes"].get_values())
            strides = tuple(op.properties["static_strides"].get_values())
            if (
                len(destination_type.shape) != rank or destination_type != result_type
                or (source_type.dtype, source_type.numerical_policy)
                != (destination_type.dtype, destination_type.numerical_policy)
                or sizes != source_type.shape
                or any(len(values) != rank for values in (offsets, sizes, strides))
            ):
                raise LinalgBridgeError("tensor.insert_slice sizes differ from source or result type")
            if any(offset < 0 or stride <= 0 for offset, stride in zip(offsets, strides)):
                raise LinalgBridgeError("tensor.insert_slice has an invalid offset or stride")
            if any(
                offset + (size - 1) * stride >= extent
                for offset, size, stride, extent in zip(
                    offsets, sizes, strides, destination_type.shape,
                )
            ):
                raise LinalgBridgeError("tensor.insert_slice exceeds destination bounds")
            identity = IndexMap(
                rank, tuple(tuple(int(axis == dim) for axis in range(rank)) for dim in range(rank)),
                (0,) * rank,
            )
            placement = IndexMap(
                rank,
                tuple(tuple(strides[dim] if axis == dim else 0 for axis in range(rank))
                      for dim in range(rank)),
                offsets,
            )
            node_id = f"op{ordinal}r0"
            nodes.append(SemanticNode(
                node_id, "insert_slice", (env[source_value], env[destination]), result_type,
                index_maps=(identity, placement, placement),
            ))
            env[op.results[0]] = node_id
        elif op.name == "linalg.fill":
            if len(op.operands) != 2 or len(op.results) != 1:
                raise LinalgBridgeError("linalg.fill requires one scalar and one destination")
            scalar, destination = op.operands
            if destination not in env:
                raise LinalgBridgeError("linalg.fill requires a declared destination")
            type_ = _type(op.results[0].type)
            if type_ != _type(destination.type):
                raise LinalgBridgeError("linalg.fill result/destination has unsupported type")
            if type_.dtype == "bf16" and scalar in bf16_zeros and env[destination] is None:
                if any(not key.startswith("prov.") for key in op.attributes):
                    raise LinalgBridgeError("BF16 zero fill has unknown attributes")
                zero_fills.add(op.results[0])
                env[op.results[0]] = None
                continue
            if type_.dtype == "f32" and scalar in f32_scalars:
                node_id = f"op{ordinal}r0"
                bits = f32_scalars[scalar]
                nodes.append(SemanticNode(node_id, "constant", (), type_, effect="constant"))
                constants[node_id] = TensorValue(type_, (bits,) * prod(type_.shape))
                constant_bindings.append(ConstantBinding(
                    node_id, "f32-le-raw", (bits.to_bytes(4, "little") * prod(type_.shape)).hex(),
                ))
                env[op.results[0]] = node_id
                continue
            if type_.dtype == "i64" and scalar in i64_scalars:
                node_id = f"op{ordinal}r0"
                value = i64_scalars[scalar]
                nodes.append(SemanticNode(
                    node_id, "constant", (), type_, attrs=(("value", value),), effect="constant",
                ))
                constants[node_id] = TensorValue(type_, (value,) * prod(type_.shape))
                constant_bindings.append(ConstantBinding(
                    node_id, "i64-le", (value.to_bytes(8, "little", signed=True) * prod(type_.shape)).hex(),
                ))
                env[op.results[0]] = node_id
                continue
            if scalar not in scalars or type_.dtype != "i32":
                raise LinalgBridgeError("linalg.fill requires an admitted scalar and destination")
            node_id = f"op{ordinal}r0"
            value = scalars[scalar]
            nodes.append(SemanticNode(node_id, "constant", (), type_, attrs=(("value", value),), effect="constant"))
            constants[node_id] = TensorValue(type_, (value,) * prod(type_.shape))
            constant_bindings.append(ConstantBinding(
                node_id, "i32-le", (value.to_bytes(4, "little", signed=True) * prod(type_.shape)).hex(),
            ))
            env[op.results[0]] = node_id
        elif op.name == "linalg.copy":
            if len(op.inputs) != 1 or len(op.outputs) != 1 or len(op.results) != 1 or op.attributes:
                raise LinalgBridgeError("copy has an unsupported signature or attributes")
            source_value, destination = op.inputs[0], op.outputs[0]
            if source_value not in env or env[source_value] is None or destination not in env:
                raise LinalgBridgeError("copy reads an undefined or uninitialized input or destination")
            source_type = _type(source_value.type)
            if source_type != _type(destination.type) or source_type != _type(op.results[0].type):
                raise LinalgBridgeError("copy requires a type-preserving input and result")
            node_id = f"op{ordinal}r0"
            nodes.append(SemanticNode(node_id, "identity", (env[source_value],), source_type))
            env[op.results[0]] = node_id
        elif op.name == "linalg.transpose":
            if len(op.inputs) != 1 or len(op.outputs) != 1 or len(op.results) != 1:
                raise LinalgBridgeError("transpose has an unsupported signature")
            source_value, destination = op.inputs[0], op.outputs[0]
            if source_value not in env or env[source_value] is None or destination not in env:
                raise LinalgBridgeError("transpose reads an undefined or uninitialized input or destination")
            # The named transpose writes every result element, so an empty
            # destination contributes storage but no semantic input value.
            source_type = _type(source_value.type)
            destination_type = _type(destination.type)
            result_type = _type(op.results[0].type)
            permutation = tuple(op.permutation.get_values())
            rank = len(source_type.shape)
            if (
                rank < 1
                or len(permutation) != rank
                or sorted(permutation) != list(range(rank))
                or set(op.properties) != {"permutation"}
                or any(not name.startswith("prov.") for name in op.attributes)
                or result_type != destination_type
                or result_type.dtype != source_type.dtype
                or result_type.numerical_policy != source_type.numerical_policy
                or result_type.shape != tuple(source_type.shape[axis] for axis in permutation)
            ):
                raise LinalgBridgeError("transpose permutation, shape, or numerical policy is unsupported")
            try:
                op.verify()
            except Exception as exc:  # noqa: BLE001 - xDSL verifier is a fail-closed boundary
                raise LinalgBridgeError(f"transpose failed xDSL verification: {exc}") from exc
            node_id = f"op{ordinal}r0"
            nodes.append(
                SemanticNode(node_id, "transpose", (env[source_value],), result_type,
                             index_maps=_transpose_maps(permutation))
            )
            env[op.results[0]] = node_id
        elif op.name == "linalg.generic" and len(op.inputs) == 1 and (
            _type(op.inputs[0].type).dtype == "i64"
        ):
            indices, table, destination = _embedding_lookup_operands(op)
            if (
                indices not in env or env[indices] is None
                or table not in env or env[table] is None
                or destination not in env or env[destination] is not None
            ):
                raise LinalgBridgeError("embedding lookup needs defined indices/table and an empty destination")
            table_type = _type(table.type)
            node_id = f"op{ordinal}r0"
            nodes.append(SemanticNode(
                node_id, "embedding_lookup", (env[indices], env[table]), _type(op.results[0].type),
                attrs=(("index_domain", "nonnegative-below-table-rows"),
                       ("minimum_index_bits", max(2, (table_type.shape[0] - 1).bit_length() + 1))),
            ))
            env[op.results[0]] = node_id
        elif op.name == "linalg.generic" and len(op.inputs) == 2 and (
            "merlin.numeric_semantics" in op.attributes
            and "merlin.reduction_order" not in op.attributes
        ):
            try:
                policy = op.attributes["merlin.numeric_semantics"]
                if not isinstance(policy, StringAttr):
                    raise LinalgBridgeError("BF16 binary numerical policy must be a string")
                if policy.data == "selected-f32-rne-bf16-chop-v1":
                    body = (tuple(op.regions[0].block.ops)
                            if len(op.regions) == 1 and len(op.regions[0].blocks) == 1 else ())
                    if len(body) > 2 and body[2].name == "arith.subf":
                        semantic_op = "sub_bf16"
                        lhs, rhs, destination, semantics, domain = recognize_bf16_f32_chop_sub(op)
                    else:
                        semantic_op = "add_bf16"
                        lhs, rhs, destination, semantics, domain = recognize_bf16_f32_chop_add(op)
                elif policy.data == "bf16-product-rne-v1":
                    semantic_op = "mul_bf16"
                    lhs, rhs, destination, semantics, domain = recognize_bf16_product_rne(op)
                else:
                    raise LinalgBridgeError(f"unadmitted BF16 binary numerical policy {policy.data!r}")
            except InvalidLinalgPattern as exc:
                raise LinalgBridgeError(str(exc)) from exc
            if any(value not in env or env[value] is None for value in (lhs, rhs)) or (
                destination not in env or env[destination] is not None
            ):
                raise LinalgBridgeError("BF16 binary operation needs two defined sources and an empty destination")
            source_ids = (env[lhs], env[rhs])
            source_nodes = {node.id: node for node in nodes}
            if any(
                source_nodes[source_id].op == "input"
                and dict(source_nodes[source_id].attrs).get("input_domain") != domain
                for source_id in source_ids
            ):
                raise LinalgBridgeError("BF16 binary premise differs from its source boundary")
            result_type = _type(op.results[0].type)
            node_id = f"op{ordinal}r0"
            nodes.append(SemanticNode(
                node_id, semantic_op, source_ids, result_type,
                attrs=(("input_domain", domain), ("numeric_semantics", semantics)),
                index_maps=pointwise_identity_maps(2, 2),
            ))
            env[op.results[0]] = node_id
        elif op.name == "linalg.generic" and len(op.inputs) == 2 and (
            "merlin.reduction_order" in op.attributes
        ):
            try:
                lhs, rhs, init, semantics, domain = recognize_ordered_fp8_bf16_reset_matmul(op)
            except InvalidLinalgPattern as exc:
                raise LinalgBridgeError(str(exc)) from exc
            if lhs not in env or rhs not in env:
                raise LinalgBridgeError("ordered FP8 matmul needs declared inputs")
            source_ids = (env[lhs], env[rhs])
            if any(source_id is None for source_id in source_ids):
                raise LinalgBridgeError("ordered FP8 matmul reads uninitialized operands")
            source_nodes = {node.id: node for node in nodes}
            if any(
                (
                    source_nodes[source_id].op == "input"
                    and dict(source_nodes[source_id].attrs).get("input_domain") != domain
                ) or (
                    source_nodes[source_id].op != "input"
                    and not (
                        source_nodes[source_id].op == "pack_fp8"
                        and source_nodes[source_id].type.dtype == "fp8_e4m3"
                        and source_nodes[source_id].type.numerical_policy != "raw-bit-pattern"
                    )
                )
                for source_id in source_ids
            ):
                raise LinalgBridgeError(
                    "ordered FP8 matmul needs declared boundary inputs or a computed FP8 pack"
                )
            # Preserve a computed pack's policy and def-use edge. The selected
            # target must establish that its result satisfies this matmul's
            # declared input domain; parsing alone cannot prove that closure.
            node_id = f"op{ordinal}r0"
            result_type = _type(op.results[0].type)
            accumulator_id = None if init in zero_fills else env.get(init)
            if init not in zero_fills:
                producer = source_nodes.get(accumulator_id)
                if (
                    producer is None or producer.op not in {"matmul", "matmul_accumulate"}
                    or producer.type != TensorType(result_type.shape, result_type.dtype, semantics)
                ):
                    raise LinalgBridgeError(
                        "ordered FP8 continuation needs a matching prior ordered accumulator"
                    )
            nodes.append(SemanticNode(
                node_id, "matmul" if accumulator_id is None else "matmul_accumulate",
                source_ids if accumulator_id is None else (*source_ids, accumulator_id),
                TensorType(result_type.shape, result_type.dtype, semantics),
                index_maps=(_oriented_matmul_maps() if accumulator_id is None
                            else _oriented_matmul_accumulate_maps()),
            ))
            env[op.results[0]] = node_id
        elif op.name == "linalg.generic" and len(op.inputs) == 1 and (
            _type(op.results[0].type).dtype == "fp8_e4m3"
        ):
            try:
                source_value, destination, semantics, domain, scale_code = recognize_bf16_fp8_truncation(op)
            except InvalidLinalgPattern as exc:
                raise LinalgBridgeError(str(exc)) from exc
            if (source_value not in env or env[source_value] is None
                or destination not in env or env[destination] is not None):
                raise LinalgBridgeError("BF16-to-FP8 truncation needs a defined source and empty destination")
            source_id = env[source_value]
            source_node = next(node for node in nodes if node.id == source_id)
            if (source_node.op == "input"
                and dict(source_node.attrs).get("input_domain") != domain):
                raise LinalgBridgeError("BF16-to-FP8 input domain differs from its source boundary")
            # A computed producer may discharge the declared input-domain
            # premise through a target rule. Keep its def-use edge intact;
            # the frontend does not infer numerical closure from an op name.
            result_type = _type(op.results[0].type)
            node_id = f"op{ordinal}r0"
            nodes.append(SemanticNode(
                node_id, "pack_fp8", (source_id,),
                TensorType(result_type.shape, result_type.dtype, semantics),
                attrs=(("input_domain", domain), ("scale_code", scale_code)),
                index_maps=pointwise_identity_maps(2, 1),
            ))
            env[op.results[0]] = node_id
        elif op.name == "linalg.generic" and len(op.inputs) == 1 and (
            _type(op.results[0].type).dtype == "bf16"
        ):
            try:
                source_value, destination = recognize_bf16_sign_clamp_relu(op)
            except InvalidLinalgPattern as exc:
                raise LinalgBridgeError(str(exc)) from exc
            if source_value not in env or env[source_value] is None or destination not in env:
                raise LinalgBridgeError("BF16 ReLU reads an undefined or uninitialized tensor")
            source_type = _type(source_value.type)
            destination_type = _type(destination.type)
            result_type = _type(op.results[0].type)
            if (
                source_type.dtype != "bf16"
                or destination_type != result_type
                or result_type.dtype != "bf16"
                or source_type.shape != result_type.shape
            ):
                raise LinalgBridgeError("BF16 ReLU result or destination differs from its source shape and dtype")
            node_id = f"op{ordinal}r0"
            nodes.append(SemanticNode(
                node_id, "relu", (env[source_value],), result_type,
                attrs=(("input_domain", "finite-normal-or-zero-bf16"),
                       ("numeric_semantics", "bf16-sign-clamp-v1")),
                index_maps=pointwise_identity_maps(2, 1),
            ))
            env[op.results[0]] = node_id
        elif (op.name == "linalg.generic" and len(op.inputs) == 2 and len(op.results) == 1
              and _type(op.results[0].type).dtype == "f32"
              and any(iterator.data == IteratorType.REDUCTION
                      for iterator in op.iterator_types.data)):
            lhs, rhs, init, maps, add_order = _source_f32_batched_matmul(op)
            if any(value not in env or env[value] is None for value in (lhs, rhs, init)):
                raise LinalgBridgeError("source f32 batched matmul reads an undefined initializer or operand")
            node_id = f"op{ordinal}r0"
            nodes.append(SemanticNode(
                node_id, "source_f32_matmul_unqualified",
                tuple(env[value] for value in (lhs, rhs, init)), _type(op.results[0].type),
                attrs=(("reduction_order", "unresolved"),
                       ("numeric_semantics", "source-f32-mul-add-unqualified-v1"),
                       ("scalar_add_order", add_order)),
                index_maps=maps,
            ))
            env[op.results[0]] = node_id
        elif op.name == "linalg.generic" and _type(op.results[0].type).dtype == "f32":
            if len(op.inputs) == 2 and len(op.outputs) == 1:
                try:
                    pointwise = recognize_static_projected_pointwise(op)
                except InvalidLinalgPattern as exc:
                    try:
                        pointwise = recognize_static_f32_division(op)
                    except InvalidLinalgPattern as division_exc:
                        raise LinalgBridgeError(
                            "f32 linalg.generic needs an admitted semantic decomposition: "
                            f"{exc}; {division_exc}"
                        ) from division_exc
                operation = {
                    "arith.addf": "f32_add",
                    "arith.subf": "f32_sub",
                    "arith.mulf": "f32_mul",
                    "arith.divf": "f32_div",
                }.get(pointwise.operation)
                if operation is None:
                    raise LinalgBridgeError("f32 linalg.generic needs an admitted semantic decomposition")
                lhs, rhs = op.inputs
                destination = op.outputs[0]
                result_type = _type(op.results[0].type)
                if (
                    lhs not in env or env[lhs] is None or rhs not in env or env[rhs] is None
                    or destination not in env or env[destination] is not None
                    or any(_type(value.type).dtype != "f32"
                           or _type(value.type).numerical_policy != "raw-bit-pattern"
                           for value in (lhs, rhs))
                    or _type(destination.type) != result_type
                ):
                    raise LinalgBridgeError("f32 arithmetic needs defined inputs and an empty output")
                rank = len(result_type.shape)
                maps = tuple(
                    IndexMap(
                        rank,
                        tuple(row for row, _ in rows),
                        tuple(offset for _, offset in rows),
                    )
                    for rows in (
                        tuple(_linear_affine_expression(expression, rank) for expression in mapping.data.results)
                        for mapping in op.get_indexing_maps()
                    )
                )
                node_id = f"op{ordinal}r0"
                nodes.append(SemanticNode(
                    node_id, operation, (env[lhs], env[rhs]), result_type,
                    attrs=(("numeric_semantics", f"ieee754-f32-{operation[4:]}-rne-v1"),),
                    index_maps=maps,
                ))
                env[op.results[0]] = node_id
            elif len(op.inputs) != 1 or len(op.outputs) != 1:
                raise LinalgBridgeError("f32 linalg.generic needs an admitted semantic decomposition")
            else:
                source_value, destination = op.inputs[0], op.outputs[0]
                if (
                    source_value not in env or env[source_value] is None
                    or destination not in env or env[destination] is not None
                ):
                    raise LinalgBridgeError("indexed copy needs a defined source and an empty destination")
                source_type, result_type = _type(source_value.type), _type(op.results[0].type)
                if (
                    source_type.dtype != "f32" or result_type != _type(destination.type)
                    or source_type.numerical_policy != result_type.numerical_policy
                ):
                    raise LinalgBridgeError("indexed copy changes type or numerical policy")
                try:
                    maps = _static_index_copy_map(op, source_type, result_type)
                    semantic_op = "gather"
                    inputs = (env[source_value],)
                    attrs = ()
                except LinalgBridgeError as copy_error:
                    try:
                        maps = _static_f32_square_maps(op, source_type, result_type)
                    except LinalgBridgeError as square_error:
                        try:
                            checked_source, checked_destination = recognize_f32_relu_ordered_select(op)
                        except InvalidLinalgPattern as relu_error:
                            try:
                                math_source = recognize_static_f32_exp_rsqrt_body(op)
                            except InvalidLinalgPattern as math_error:
                                try:
                                    composite = recognize_static_composite_math_body(op)
                                except InvalidLinalgPattern as composite_error:
                                    raise LinalgBridgeError(
                                        "f32 linalg.generic needs an admitted semantic decomposition: "
                                        f"{copy_error}; {square_error}; {relu_error}; "
                                        f"{math_error}; {composite_error}"
                                    ) from composite_error
                                semantic_op = "source_f32_composite_unqualified"
                                maps = pointwise_identity_maps(len(result_type.shape), 1)
                                inputs = (env[source_value],)
                                attrs = (("numeric_semantics", "source-f32-composite-unqualified-v1"),
                                         ("operation", composite.operation),
                                         ("literal_bits", ",".join(composite.literal_bits)))
                            else:
                                kind = math_source.operation.removeprefix("math.")
                                semantic_op = f"source_f32_{kind}_unqualified"
                                maps = pointwise_identity_maps(len(result_type.shape), 1)
                                inputs = (env[source_value],)
                                attrs = (("numeric_semantics", f"source-f32-{kind}-unqualified-v1"),)
                        else:
                            if checked_source is not source_value or checked_destination is not destination:
                                raise LinalgBridgeError("f32 zero select operands differ from the parsed source")
                            semantic_op = "f32_relu_ordered_select"
                            maps = pointwise_identity_maps(len(result_type.shape), 1)
                            inputs = (env[source_value],)
                            attrs = (("numeric_semantics", "f32-ordered-lt-positive-zero-select-v1"),)
                    else:
                        semantic_op = "f32_mul"
                        inputs = (env[source_value], env[source_value])
                        attrs = (("numeric_semantics", "ieee754-f32-mul-rne-v1"),)
                node_id = f"op{ordinal}r0"
                nodes.append(SemanticNode(
                    node_id, semantic_op, inputs, result_type, attrs=attrs, index_maps=maps,
                ))
                env[op.results[0]] = node_id
        elif op.name in {"linalg.generic", "linalg.matmul"}:
            if op.name == "linalg.generic":
                try:
                    lhs, rhs, init = recognize_signed_i8_i32_matmul(op)
                except InvalidLinalgPattern as exc:
                    raise LinalgBridgeError(str(exc)) from exc
            else:
                if len(op.inputs) != 2 or len(op.outputs) != 1 or len(op.results) != 1:
                    raise LinalgBridgeError("named matmul has an unsupported signature")
                if not _named_matmul_maps_are_default(op):
                    raise LinalgBridgeError("named matmul has nonstandard indexing or iterators")
                lhs, rhs, init = (*op.inputs, op.outputs[0])
            if any(value not in env or env[value] is None for value in (lhs, rhs, init)):
                raise LinalgBridgeError("matmul reads an undefined or uninitialized operand")
            lhs_type, rhs_type, init_type, result_type = (
                _type(value.type) for value in (lhs, rhs, init, op.results[0])
            )
            if lhs_type.dtype != rhs_type.dtype or lhs_type.numerical_policy != rhs_type.numerical_policy:
                raise LinalgBridgeError("matmul input dtypes or numerical policies differ")
            if op.name == "linalg.matmul" and lhs_type.dtype not in {"i32", "f32"}:
                raise LinalgBridgeError("named matmul needs i32 operands or a source-only f32 graph node")
            if len(lhs_type.shape) != 2 or len(rhs_type.shape) != 2 or (
                lhs_type.shape[1] != rhs_type.shape[0]
                or result_type.shape != (lhs_type.shape[0], rhs_type.shape[1])
                or init_type != result_type
                or result_type.dtype != ("f32" if lhs_type.dtype == "f32" else "i32")
            ):
                raise LinalgBridgeError("matmul shape or initialized result type differs")
            node_id = f"op{ordinal}r0"
            if op.name == "linalg.matmul" and lhs_type.dtype == "f32":
                from .precision_intake import _check_matmul_body

                if (set(op.properties) != {"indexing_maps", "operandSegmentSizes"}
                    or any(not key.startswith("prov.") for key in op.attributes)):
                    raise LinalgBridgeError("source f32 matmul has an unknown semantic attribute")
                _check_matmul_body(op)
                try:
                    op.verify()
                except Exception as exc:  # noqa: BLE001 - xDSL owns source verification
                    raise LinalgBridgeError(f"source f32 matmul failed verification: {exc}") from exc
                nodes.append(SemanticNode(
                    node_id, "source_f32_matmul_unqualified",
                    tuple(env[value] for value in (lhs, rhs, init)), result_type,
                    attrs=(("reduction_order", "unresolved"),
                           ("numeric_semantics", "source-f32-mul-add-unqualified-v1")),
                    index_maps=_maps(),
                ))
                env[op.results[0]] = node_id
                continue
            nodes.append(SemanticNode(
                node_id, "matmul_accumulate", tuple(env[value] for value in (lhs, rhs, init)),
                result_type, index_maps=_maps(),
            ))
            env[op.results[0]] = node_id
        elif op.name == "func.return":
            if ordinal != len(operations) - 1 or not op.operands:
                raise LinalgBridgeError("func.return must be final and return tensors")
            if tuple(value.type for value in op.operands) != tuple(function.function_type.outputs.data):
                raise LinalgBridgeError("returned tensor types differ from the entry signature")
            outputs = tuple(env.get(value) for value in op.operands)
            if any(output is None for output in outputs):
                raise LinalgBridgeError("function returns an undefined or uninitialized tensor")
            request = KernelRequest(
                nodes=tuple(nodes), outputs=outputs, output_storages=(boundary_storage,) * len(outputs),
                input_storages=tuple(boundaries), target_identity=target_identity,
                lowering_policy=lowering_policy, source_identity=sha256(source.encode()).hexdigest(),
                constants=tuple(constant_bindings),
            )
            return LinalgTranslation(request, constants, tuple(source_operations))
        elif op.name == "linalg.reduce" and op.results and _type(op.results[0].type).dtype == "f32":
            source_value, init, axes, order, add_order, scalar_op = _recognize_f32_reduce(op)
            if any(value not in env or env[value] is None for value in (source_value, init)):
                raise LinalgBridgeError("ordered f32 reduction needs a defined input and initializer")
            source_type = _type(source_value.type)
            node_id = f"op{ordinal}r0"
            maximum = scalar_op == "arith.maximumf"
            nodes.append(SemanticNode(
                node_id, ("source_f32_reduce_max_unqualified" if maximum else
                          "f32_reduce_add" if order == "axis-ascending" else
                          "source_f32_reduce_add_unqualified"),
                (env[source_value], env[init]),
                _type(op.results[0].type),
                attrs=(("axis", axes[0]) if len(axes) == 1 else ("axes", ",".join(map(str, axes))),
                       ("reduction_order", order),
                       ("numeric_semantics", "source-f32-maximum-unqualified-v1" if maximum else
                        "ieee754-f32-add-rne-v1" if order == "axis-ascending" else
                        "source-f32-add-unqualified-v1"),
                       *((("scalar_operand_order" if maximum else "scalar_add_order", add_order),)
                         if order == "unresolved" else ())),
                index_maps=_f32_reduce_maps(len(source_type.shape), axes),
            ))
            env[op.results[0]] = node_id
        else:
            raise LinalgBridgeError(f"no admitted translation for source operation {op.name}")
    raise LinalgBridgeError("entry function has no return")
