"""Publish the bounded, one-invocation deployment selected by the compiler.

Provider reservations remain opaque. All caller-owned arrays are placed after
them without aliasing; execution and image changes stay in generated runtime C.
"""

from __future__ import annotations

import json
from hashlib import sha256
from math import prod
from pathlib import Path

from merlin.llvmlower.c_runtime import DT_BYTES
from merlin.semantic_compiler.codegen import current_codegen_options
from merlin.semantic_compiler.target_binding import NativeCompilationError


WINDOW_ADDRESS = 0x200C0000
WINDOW_BYTES = 786432


def selected():
    options = current_codegen_options()
    return (options.io_memory_base == WINDOW_ADDRESS and options.io_memory_bytes == WINDOW_BYTES
            and options.io_memory_capacity_base == WINDOW_ADDRESS
            and options.io_memory_capacity_bytes == WINDOW_BYTES)


def _refuse(reason, message):
    status = "resource_limit" if reason.endswith("_overflow") else "unsupported_semantics"
    raise NativeCompilationError(status, reason + ": " + message)


def _integer(value, name, *, positive=False):
    if type(value) is not int or value < int(positive):
        _refuse("deployment_provider_contract", "invalid provider " + name)
    return value


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def place(source, adapter, regions, interface, native_root):
    """Choose absolute public/context bindings only for the exact on-chip map."""
    if not selected():
        return None
    if current_codegen_options().application_residency:
        _refuse("deployment_unsupported_residency", "on-chip deployment requires disjoint nonresident storage")
    factory = getattr(adapter, "deployment_requirements", None)
    if factory is None or native_root is None:
        _refuse("deployment_provider_contract", "on-chip deployment requires provider requirements")
    provider = json.loads(json.dumps(factory(regions, native_root), allow_nan=False))
    if provider.get("schema") != "atlas.provider_deployment.v1" or provider.get("window") != {
        "address": WINDOW_ADDRESS, "bytes": WINDOW_BYTES,
    }:
        _refuse("deployment_provider_contract", "provider deployment window differs from selected geometry")
    reservations = provider.get("reservations")
    if not isinstance(reservations, list):
        _refuse("deployment_provider_contract", "provider reservations must be explicit")
    occupied, names = [], set()
    cursor = WINDOW_ADDRESS
    for row in reservations:
        if not isinstance(row, dict) or not all(key in row for key in (
            "name", "address", "bytes", "role", "lifetime", "promotion_receipts",
        )) or not isinstance(row["name"], str) or not row["name"] or row["name"] in names:
            _refuse("deployment_provider_contract", "invalid provider reservation identity")
        names.add(row["name"])
        address = _integer(row["address"], "reservation address")
        size = _integer(row["bytes"], "reservation bytes", positive=True)
        if address < WINDOW_ADDRESS or address + size > WINDOW_ADDRESS + WINDOW_BYTES:
            _refuse("deployment_provider_contract", "provider reservation outside deployment window")
        for start, end, lifetime in occupied:
            if address < end and start < address + size:
                current = row["lifetime"]
                if (not isinstance(lifetime, dict) or not isinstance(current, dict)
                        or lifetime.get("scope") != "region_call" or current.get("scope") != "region_call"
                        or type(lifetime.get("context_index")) is not int
                        or type(current.get("context_index")) is not int
                        or lifetime["context_index"] == current["context_index"]):
                    _refuse("deployment_provider_contract", "provider reservations overlap within a call or lack distinct call lifetimes")
        occupied.append((address, address + size, row["lifetime"]))
        cursor = max(cursor, address + size)
    allocations = []

    def allocate(name, size, role, alignment=64):
        nonlocal cursor
        _integer(size, name + " bytes")
        if name in names or type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
            _refuse("deployment_provider_contract", "invalid deployment allocation identity or alignment")
        names.add(name)
        cursor = (cursor + alignment - 1) // alignment * alignment
        row = dict(name=name, address=cursor, bytes=size, role=role, alignment=alignment)
        allocations.append(row)
        cursor += size
        return row

    allocate("device_scratch", max(1, _integer(provider.get("raw_scratch_bytes"), "raw scratch")), "private_scratch")
    allocate("conversion_scratch", max(1, _integer(provider.get("conversion_scratch_bytes"), "conversion scratch")),
             "private_scratch")
    allocate("workspace", max(1, interface["workspace_bytes"]), "application_workspace",
             max(64, interface["workspace_alignment"]))
    state = interface.get("model_state", {}).get("ports", [])
    state_inputs = {row["node_id"]: row for row in state}
    state_outputs = {row["output_index"]: row for row in state if row["mode"] == "updated"}

    def boundary(key, index, kind):
        tensor = source.node(key).type
        size = prod(tensor.shape) * DT_BYTES[tensor.dtype]
        state_port = state_inputs.get(key) if kind == "input" else state_outputs.get(index)
        role = ("persistent_state_input" if kind == "input" else "persistent_state_update") if state_port else (
            "public_input" if kind == "input" else "public_output")
        row = allocate(kind + ":" + key, size, role)
        return {**row, "id": key, "index": index, "type": tensor.record(), "layout": "dense",
                **({"path": f"inputs/input_{index:03d}.bin"} if kind == "input" else {}),
                "endianness": "little", "ownership": "caller", **({"state": state_port["name"]} if state_port else {})}

    inputs = [boundary(key, i, "input") for i, key in enumerate(interface["inputs"])]
    outputs = [boundary(key, i, "output") for i, key in enumerate(interface["outputs"])]
    parameters = interface.get("model_parameters")
    parameter = None
    if parameters:
        parameter = {**allocate("parameters", parameters["payload_bytes"], "immutable_parameters"),
                     "path": "handoff/parameters.bin", "sha256": parameters["payload_sha256"]}
    required = cursor - WINDOW_ADDRESS
    if required > WINDOW_BYTES:
        _refuse("deployment_window_overflow",
                f"disjoint on-chip deployment requires {required} bytes; window {WINDOW_BYTES} bytes; "
                f"overflow {required - WINDOW_BYTES} bytes")
    deployment = {
        "schema": "merlin.onchip_deployment.v1", "request_digest": source.digest(),
        "target_identity": source.target_identity, "window": provider["window"], "provider": provider,
        "allocations": allocations, "inputs": inputs, "outputs": outputs, "parameters": parameter,
        "calls": [{"region": row["id"], "context_index": row["context_index"], "placement": row["placement"],
                   "native_programs": row["native_programs"],
                   "launches_per_invocation": row["description"].get("launches_per_invocation",
                       row["description"].get("streaming", row["description"].get("native_export", {}).get("streaming", {}))
                       .get("launch_count", 1)),
                   **{key: row["description"][key] for key in
                      ("device_launch_schedule", "launches_per_invocation", "launches_per_call")
                      if key in row["description"]}} for row in regions],
        "window_required_bytes": required, "window_remaining_bytes": WINDOW_BYTES - required,
        "storage_policy": "disjoint; no lifetime sharing", "completion": "synchronous generated application run",
        "host_map": {"imem_address": 0x20000, "imem_bytes": 131072, "csr_address": 0x40000},
        "max_polls": 10000000,
        "polling_bound": "host polls per generated device wait; independent of functional-model instruction limit",
        "host_allocation": {
            "address": 0x08000000, "bytes": 65536, "measurement": "deferred_to_linker_PT_LOAD_and_zero_fill",
            "reservations": [
                {"name": "code", "bytes": 36864, "includes": "embedded instruction images and constants"},
                {"name": "data", "bytes": 8192}, {"name": "context", "bytes": 4096},
                {"name": "stack", "bytes": 12288}, {"name": "heap", "bytes": 4096},
            ], "reserved_bytes": 65536,
        },
    }
    interface["deployment"] = deployment
    return deployment


def adapter_c(name, deployment):
    """A fixed consumer ABI; all model-specific binding stays generated."""
    declarations = """
typedef struct { const char *name; const char *role; uintptr_t address; size_t bytes; } sp5_hash_range;
extern const sp5_hash_range sp5_hash_ranges[];
extern const size_t sp5_hash_count;
int sp5_bind(atlas_host *,atlas_tile_memory_io,size_t max_polls);
int sp5_run(void);
"""
    def pointer(row, const=False):
        return f"({'const ' if const else ''}void *)(uintptr_t)UINT64_C({row['address']})"

    inputs = ",".join(pointer(row, True) for row in deployment["inputs"]) or "0"
    outputs = ",".join(pointer(row) for row in deployment["outputs"]) or "0"
    hashes = ",\n".join(
        f"  {{{json.dumps(row['id'])},{json.dumps(row['role'])},UINT64_C({row['address']}),{row['bytes']}u}}"
        for row in deployment["outputs"]
    )
    parameter = deployment["parameters"]
    bind_parameters = (f"  status={name}_bind_parameters(&sp5_context,{pointer(parameter, True)},{parameter['bytes']}u);\n"
                       "  if (status) return status;\n") if parameter else ""
    implementation = f"""
static {name}_context sp5_context;
_Static_assert(sizeof(sp5_context)<=4096u,"deployment context reservation exceeded");
const sp5_hash_range sp5_hash_ranges[]={{{hashes}}};
const size_t sp5_hash_count={len(deployment['outputs'])}u;
int sp5_bind(atlas_host *host,atlas_tile_memory_io io,size_t max_polls) {{
  int status={name}_bind(&sp5_context,host,io,max_polls);
  if (status) return status;
{bind_parameters}  return 0;
}}
int sp5_run(void) {{
  const void *inputs[]={{{inputs}}};
  void *outputs[]={{{outputs}}};
  return {name}_run(&sp5_context,inputs,outputs);
}}
"""
    return declarations, implementation


def publish(deployment, interface, files, native_root):
    """Authenticate payload identities and serialize the single invocation."""
    if interface.get("platform_binding", {}).get("completion") != "synchronous":
        _refuse("deployment_completion_contract", "on-chip deployment needs synchronous generated completion")
    steps = []
    for row in deployment["inputs"]:
        steps.append({"operation": "window_preload", "name": row["id"], "path": row["path"],
                      "source_offset": 0, "address": row["address"], "bytes": row["bytes"],
                      "alignment": row["alignment"], "role": row["role"], "lifetime": "through_run_completion",
                      "identity": "required_runtime_input", "completion": "before_bind"})
    if deployment["parameters"]:
        row = deployment["parameters"]
        steps.append({"operation": "window_preload", **row, "source_offset": 0,
                      "lifetime": "immutable_through_completion_and_recovery", "completion": "before_bind"})
    images = deployment["provider"].get("images")
    if not isinstance(images, list) or not images:
        _refuse("deployment_provider_contract", "on-chip deployment requires instruction images")
    image_library = {}
    for row in images:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            _refuse("deployment_provider_contract", "invalid deployment image")
        path = Path(row["path"])
        if path.is_absolute() or path.as_posix() != row["path"] or ".." in path.parts:
            _refuse("deployment_provider_contract", "unsafe deployment image path")
        root = Path(native_root)
        if any((root / Path(*path.parts[:i])).is_symlink() for i in range(1, len(path.parts) + 1)):
            _refuse("deployment_provider_contract", "linked deployment image")
        payload = (root / path).read_bytes()
        if not _digest(row.get("sha256")) or sha256(payload).hexdigest() != row["sha256"] or len(payload) != row.get("bytes"):
            _refuse("deployment_provider_contract", "deployment image identity differs")
        if (row.get("imem_offset_bytes") != 0 or row.get("capacity_bytes") != 131072
                or row.get("upload_convention") != "counted_ensure_loaded"):
            _refuse("deployment_provider_contract", "unsupported deployment image upload convention")
        # The counted loader uploads exactly these words; it appends no epilogue.
        if len(payload) > row["capacity_bytes"]:
            _refuse("deployment_imem_overflow", "instruction image exceeds 131072 bytes")
        image_library[row["path"]] = {"operation": "install_image", **row, "source_offset": 0,
                      "address": deployment["host_map"]["imem_address"] + row["imem_offset_bytes"], "owner": "generated_runtime",
                      "completion": "ensure_loaded_before_start", "final_upload_bytes": len(payload)}
    runtime_events = []
    for call in deployment["calls"]:
        if call["placement"] == "host":
            runtime_events.append({"operation": "host_call", **call, "owner": "generated_runtime"})
            continue
        paths = [(Path(program) / "program.bin").as_posix() for program in call["native_programs"]]
        if any(path not in image_library for path in paths):
            _refuse("deployment_runtime_contract", "generated call image missing from provider library")
        installs = [image_library[path] for path in paths]
        event = {"operation": "region_call", **call, "owner": "generated_runtime"}
        if "device_launch_schedule" in call:
            from .launch_schedule import validate_launch_schedule

            event["device_launch_schedule"] = validate_launch_schedule(
                call["device_launch_schedule"], image_sha256s=[row["sha256"] for row in installs],
                placement=call["placement"])
            event["images"] = installs
        else:
            count = call.get("launches_per_invocation", call.get("launches_per_call"))
            if len(installs) != 1 or type(count) is not int or count <= 0:
                _refuse("deployment_runtime_contract", "generated call needs an explicit launch count or launch schedule")
            event["events"] = [installs[0], {
                "operation": "launch_group", "repeat": count, "events": [
                    {"operation": "write_packed_inputs", "destinations": "provider reservations for this call"},
                    {"operation": "start", "entry": "atlas_host_start"},
                    {"operation": "wait", "entry": "atlas_host_wait", "max_polls": deployment["max_polls"]},
                    {"operation": "read_outputs", "completion": "successful wait and generated status checks"},
                ]}, {"operation": "restore_dense_outputs"}]
        runtime_events.append(event)
    steps.extend([
        {"operation": "bind", "entry": "sp5_bind", "completion": "synchronous"},
        {"operation": "invoke", "entry": "sp5_run", "generated_entry": interface["platform_binding"]["invoke"],
         "completion": "synchronous", "runtime_events": runtime_events,
         "failure": "preserve buffers until device completion or reset"},
        {"operation": "hash", "ranges": deployment["outputs"], "completion": "after_successful_run"},
    ])
    record = {**deployment, "steps": steps, "hash_ranges": deployment["outputs"],
              "image_library": list(image_library.values()), "reload_policy": "generated counted ensure_loaded",
              "adapter_header": "handoff/deployment_adapter.h", "timeout": "max_polls per synchronous device wait"}
    files["deployment.json"] = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    header = ("#pragma once\n#include \"" + Path(interface["header"]).name + "\"\n"
              f"#define SP5_WINDOW_BASE UINT64_C({WINDOW_ADDRESS})\n"
              f"#define SP5_IMEM_BASE UINT64_C({deployment['host_map']['imem_address']})\n"
              f"#define SP5_CSR_BASE UINT64_C({deployment['host_map']['csr_address']})\n"
              f"#define SP5_IMEM_BYTES {deployment['host_map']['imem_bytes']}u\n"
              f"#define SP5_MAX_POLLS {deployment['max_polls']}u\n").encode()
    files["c-package/combined/deployment_adapter.h"] = header
    files["handoff/deployment_adapter.h"] = header
