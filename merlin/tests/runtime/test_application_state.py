"""Explicit state ports, exact captured initialization and refusal of partial updates."""

import ctypes
import shutil
import subprocess
from dataclasses import replace

import pytest

from merlin.compile.application_state import StateDeclaration
from merlin.frontends.compile_inputs import StagedCompileInputs
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType


def state_source():
    type_ = TensorType((2,), "f32", "raw-bit-pattern")
    nodes = tuple(SemanticNode(f"arg{i}", "input", (), type_, effect="input") for i in range(3))
    source = KernelRequest(
        nodes,
        ("arg2", "arg0", "arg1"),
        ("external",) * 3,
        tuple((node.id, "external") for node in nodes),
        "synthetic",
        "diagnostic",
    )
    staged = StagedCompileInputs(
        "source",
        "forward",
        None,
        None,
        "0" * 64,
        "1" * 64,
        None,
        (
            {"kind": "buffer", "weight": "history", "dtype": "f32", "shape": [2]},
            {"kind": "buffer", "weight": "scale", "dtype": "f32", "shape": [2]},
            {"kind": "input", "name": "value", "input_index": 0, "dtype": "f32", "shape": [2]},
        ),
    )
    declaration = {
        "schema": "merlin.application_state.v1",
        "source_program_sha256": staged.program_sha256,
        "ports": [
            {
                "argument_index": 0,
                "name": "history",
                "mode": "updated",
                "output_index": 1,
                "initialization": "captured",
            },
            {
                "argument_index": 1,
                "name": "scale",
                "mode": "read_only",
                "output_index": None,
                "initialization": "captured",
            },
        ],
    }
    initial = {0: bytes.fromhex("0100c07f00000080"), 1: bytes.fromhex("0000803f00000040")}
    return source, staged, declaration, initial


def test_state_inputs_updates_and_initialization_have_exact_source_bindings():
    source, staged, record, initial = state_source()
    declaration = StateDeclaration.from_record(record, staged)
    state = declaration.bind(source, initial)
    assert state.source_request_digest == source.digest()
    assert [port.input_index for port in state.ports] == [0, 1]
    assert state.record()["ports"][0]["output_index"] == 1
    assert not state.record()["in_place_updates"]
    state.check(source)
    wrong_port = replace(state.ports[0], input_index=2)
    with pytest.raises(ValueError, match="ports differ"):
        replace(state, ports=(wrong_port, state.ports[1])).check(source)
    caller = {**record, "ports": [{**port, "initialization": "caller"} for port in record["ports"]]}
    caller_declaration = StateDeclaration.from_record(caller, staged)
    assert caller_declaration.digest() != declaration.digest()
    assert declaration.compilation_identity(initial) != declaration.compilation_identity({**initial, 0: bytes(8)})
    assert caller_declaration.compilation_identity(initial) == caller_declaration.compilation_identity({})
    caller_state = caller_declaration.bind(source, {})
    metadata, files = caller_state.emit_initializers("model")
    assert metadata == {"captured_initializers": []}
    assert "model_initialize_captured_state" not in files["model_state.h"]
    assert all(port.initial is None for port in caller_state.ports)
    with pytest.raises(ValueError, match="initialization bytes"):
        declaration.bind(source, {0: bytes(4), 1: initial[1]})
    wrong_output = replace(source.nodes[2], type=TensorType((1, 2), "f32", "raw-bit-pattern"))
    changed = replace(source, nodes=(*source.nodes[:2], wrong_output), outputs=("arg0", "arg2", "arg1"))
    with pytest.raises(ValueError, match="shape or type"):
        declaration.bind(changed, initial)


@pytest.mark.parametrize("fault", ["source", "omitted", "parameter", "name", "duplicate", "mode", "output", "unknown"])
def test_invalid_state_declarations_refuse(fault):
    source, staged, record, initial = state_source()
    ports = [dict(row) for row in record["ports"]]
    if fault == "source":
        record = {**record, "source_program_sha256": "2" * 64}
    elif fault == "omitted":
        ports.pop()
    elif fault == "parameter":
        staged = replace(
            staged, argument_bindings=({**staged.argument_bindings[0], "kind": "param"}, *staged.argument_bindings[1:])
        )
    elif fault == "name":
        ports[0]["name"] = "other"
    elif fault == "duplicate":
        ports.append(dict(ports[0]))
    elif fault == "mode":
        ports[0]["mode"] = []
    elif fault == "output":
        ports[0]["output_index"] = 99
    else:
        ports[0]["extra"] = True
    with pytest.raises(ValueError):
        StateDeclaration.from_record({**record, "ports": ports}, staged).bind(source, initial)


def test_compiled_initializers_preserve_bits_and_validate_all_destinations(tmp_path):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("C compiler required")
    source, staged, record, initial = state_source()
    state = StateDeclaration.from_record(record, staged).bind(source, initial)
    metadata, files = state.emit_initializers("model")
    assert [row["input_index"] for row in metadata["captured_initializers"]] == [0, 1]
    for filename, text in files.items():
        (tmp_path / filename).write_text(text)
    subprocess.run(
        [
            cc,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            "-fPIC",
            str(tmp_path / "model_state.c"),
            "-o",
            str(tmp_path / "state.so"),
        ],
        check=True,
        capture_output=True,
    )
    lib = ctypes.CDLL(str(tmp_path / "state.so"))
    call = lib.model_initialize_captured_state
    call.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
    a, b = (ctypes.create_string_buffer(b"x" + bytes([0x71]) * 8 + b"z") for _ in range(2))
    destinations = (ctypes.c_void_p * 2)(ctypes.addressof(a) + 1, ctypes.addressof(b) + 1)
    assert call(destinations, (ctypes.c_size_t * 2)(8, 7)) == -1
    assert a.raw == b.raw == b"x" + bytes([0x71]) * 8 + b"z\0"
    overlap = (ctypes.c_void_p * 2)(ctypes.addressof(a) + 1, ctypes.addressof(a) + 2)
    assert call(overlap, (ctypes.c_size_t * 2)(8, 8)) == -1
    assert a.raw == b.raw
    assert call(destinations, (ctypes.c_size_t * 2)(8, 8)) == 0
    assert a.raw[1:9] == initial[0] and b.raw[1:9] == initial[1]
    assert a.raw[:1] == b.raw[:1] == b"x" and a.raw[9:] == b.raw[9:] == b"z\0"
    # Initializing one destination may overwrite the caller's pointer table.
    # Later destinations must still use the pointers validated before copying.
    table = (ctypes.c_void_p * 2)()
    table[0], table[1] = ctypes.addressof(table) + ctypes.sizeof(ctypes.c_void_p), ctypes.addressof(b) + 1
    assert call(table, (ctypes.c_size_t * 2)(8, 8)) == 0
    assert ctypes.string_at(ctypes.addressof(table) + ctypes.sizeof(ctypes.c_void_p), 8) == initial[0]
    assert b.raw[1:9] == initial[1]
