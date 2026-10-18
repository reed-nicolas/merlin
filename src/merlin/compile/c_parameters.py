"""Specialize a C call with a staged model's immutable parameters.

Runtime arguments keep their captured order. Mutable model buffers require an
explicit state interface and cannot become static constants through this API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from merlin.frontends.compile_inputs import load_compile_inputs, load_lifted_argument_bytes
from merlin.llvmlower.c_runtime import embed_array


@dataclass(frozen=True)
class FrozenCArguments:
    declarations: str
    arguments: tuple[str, ...]
    runtime_inputs: tuple[dict, ...]
    parameters: tuple[dict, ...]
    weights_sha256: str | None


def freeze_c_arguments(
    directory: str | Path,
    *,
    name: str,
    source_sha256: str,
    max_embedded_bytes: int = 64 * 1024 * 1024,
) -> FrozenCArguments:
    """Embed declared f32 parameters, returning expressions in function order.

    The source identity must name the exact staged IR used by the caller's
    compiler. The checked staging API reopens IR, signature and weights; runtime
    samples and reference outputs are not compiler inputs. The byte budget
    limits source generation and is not a device-memory capacity declaration.
    """
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if not name or name[0] not in letters or any(char not in letters + "0123456789_" for char in name):
        raise ValueError("parameter namespace must be a C identifier")
    if type(max_embedded_bytes) is not int or max_embedded_bytes < 0:
        raise ValueError("parameter source budget must be a nonnegative byte count")
    staged = load_compile_inputs(directory)
    if staged.program_sha256 != source_sha256:
        raise ValueError("parameter source differs from compiled IR")
    if not staged.argument_bindings:
        raise ValueError("parameter export requires a staged argument signature")
    if any(row["kind"] == "buffer" for row in staged.argument_bindings):
        raise ValueError("mutable model buffers require an explicit state interface")
    if any(row["dtype"] != "f32" for row in staged.argument_bindings):
        raise ValueError("C parameter export currently requires f32 source arguments")
    payloads = load_lifted_argument_bytes(directory, staged)
    declarations, arguments, parameters = [], [], []
    symbols: dict[str, str] = {}
    runtime: dict[int, dict] = {}
    byte_count = 0
    for index, row in enumerate(staged.argument_bindings):
        if row["kind"] == "input":
            ordinal = row["input_index"]
            runtime[ordinal] = dict(row)
            arguments.append(f"inputs[{ordinal}]")
            continue
        weight = row["weight"]
        if weight not in symbols:
            payload = payloads[index]
            byte_count += len(payload)
            if byte_count > max_embedded_bytes:
                raise ValueError("embedded parameters exceed the C source byte budget")
            values = np.frombuffer(payload, dtype="<f4")
            if not values.size:
                raise ValueError("C parameter export requires nonempty parameter tensors")
            symbol = f"{name}_parameter_{len(symbols)}"
            symbols[weight] = symbol
            declarations.append(f"static const float {symbol}[{values.size}] = {{{embed_array(values, 'f32')}}};")
            parameters.append({**row, "symbol": symbol, "bytes": len(payload)})
        arguments.append(symbols[weight])
    return FrozenCArguments(
        "\n".join(declarations),
        tuple(arguments),
        tuple(runtime[index] for index in range(len(runtime))),
        tuple(parameters),
        staged.weights_sha256,
    )
