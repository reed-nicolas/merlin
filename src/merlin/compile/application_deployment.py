"""Publish the bounded, one-invocation deployment selected by the compiler.

Provider reservations remain opaque. All caller-owned arrays are placed after
them without aliasing; execution and image changes stay in generated runtime C.
"""

from __future__ import annotations

import json
import re
from hashlib import sha256
from math import prod
from pathlib import Path

from merlin.llvmlower.c_runtime import DT_BYTES
from merlin.semantic_compiler.codegen import current_codegen_options
from merlin.semantic_compiler.target_binding import NativeCompilationError

from .launch_schedule import validate_launch_schedule


def selected():
    return getattr(current_codegen_options(), "application_deployment", None) == "onchip"


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
    """Choose bindings within the provider-published on-chip map."""
    if not selected():
        return None
    if current_codegen_options().application_residency:
        _refuse("deployment_unsupported_residency", "on-chip deployment requires disjoint nonresident storage")
    factory = getattr(adapter, "deployment_requirements", None)
    if factory is None or native_root is None:
        _refuse("deployment_provider_contract", "on-chip deployment requires provider requirements")
    supplied = factory(regions, native_root)
    if supplied is None:
        _refuse("deployment_geometry_mismatch", "provider does not support the selected deployment geometry")
    provider = json.loads(json.dumps(supplied, allow_nan=False))
    if not isinstance(provider, dict) or provider.get("schema") != "merlin.provider_deployment.v1":
        _refuse("deployment_provider_contract", "unsupported provider deployment schema")
    window = provider.get("window")
    if not isinstance(window, dict):
        _refuse("deployment_provider_contract", "provider deployment window must be explicit")
    window_address = _integer(window.get("address"), "window address")
    window_bytes = _integer(window.get("bytes"), "window bytes", positive=True)
    options = current_codegen_options()
    if (options.io_memory_base != window_address or options.io_memory_bytes != window_bytes
            or options.io_memory_capacity_base != window_address or options.io_memory_capacity_bytes != window_bytes):
        _refuse("deployment_geometry_mismatch", "provider deployment window differs from selected geometry")
    host_map = provider.get("host_map")
    if not isinstance(host_map, dict):
        _refuse("deployment_provider_contract", "provider host map must be explicit")
    for key in ("imem_address", "imem_bytes", "csr_address"):
        _integer(host_map.get(key), key, positive=key == "imem_bytes")
    runtime = provider.get("runtime")
    if not isinstance(runtime, dict) or any(
            not isinstance(runtime.get(key), str) or re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", runtime[key]) is None
            for key in ("host_type", "io_type", "start", "wait")):
        _refuse("deployment_provider_contract", "provider runtime identifiers must be explicit")
    for key in ("host_context_reservation_bytes", "max_polls", "allocation_alignment"):
        _integer(provider.get(key), key, positive=True)
    host_allocation = provider.get("host_allocation")
    if not isinstance(host_allocation, dict) or host_allocation.get("measurement") != "declared_reservations":
        _refuse("deployment_provider_contract", "provider host allocation must declare reservations")
    _integer(host_allocation.get("address"), "host allocation address")
    host_bytes = _integer(host_allocation.get("bytes"), "host allocation bytes", positive=True)
    reserved_bytes = _integer(host_allocation.get("reserved_bytes"), "reserved host bytes")
    declared = host_allocation.get("declared_reservations")
    if not isinstance(declared, list):
        _refuse("deployment_provider_contract", "provider host reservations must be explicit")
    host_reservations = {}
    for row in declared:
        if (not isinstance(row, dict) or not isinstance(row.get("name"), str) or not row["name"]
                or row["name"] in host_reservations):
            _refuse("deployment_provider_contract", "invalid provider host reservation identity")
        host_reservations[row["name"]] = _integer(row.get("bytes"), "host reservation bytes")
    if sum(host_reservations.values()) != reserved_bytes or reserved_bytes > host_bytes:
        _refuse("deployment_provider_contract", "provider host reservations exceed or differ from declared capacity")
    if host_reservations.get("context", 0) < provider["host_context_reservation_bytes"]:
        _refuse("deployment_provider_contract", "provider context budget exceeds its host reservation")
    if provider.get("imem_address_convention") != "host_map.imem_address + image.imem_offset_bytes":
        _refuse("deployment_provider_contract", "unsupported instruction address convention")
    published_regions = provider.get("regions")
    if not isinstance(published_regions, list) or len(published_regions) != len(regions):
        _refuse("deployment_provider_contract", "provider must publish every region launch count")
    calls = []
    for region, published in zip(regions, published_regions):
        if (not isinstance(published, dict) or published.get("context_index") != region["context_index"]
                or published.get("placement") != region["placement"]):
            _refuse("deployment_provider_contract", "provider region launch identity differs")
        count = _integer(published.get("launches_per_invocation"), "region launch count")
        if (region["placement"] == "host" and count != 0) or (region["placement"] != "host" and count == 0):
            _refuse("deployment_provider_contract", "provider region launch count differs from placement")
        calls.append({"region": region["id"], "context_index": region["context_index"],
                      "placement": region["placement"], "native_programs": region["native_programs"],
                      "launches_per_invocation": count,
                      **({"device_launch_schedule": published["device_launch_schedule"]}
                         if "device_launch_schedule" in published else {})})
    reservations = provider.get("reservations")
    if not isinstance(reservations, list):
        _refuse("deployment_provider_contract", "provider reservations must be explicit")
    occupied, names = [], set()
    cursor = window_address
    for row in reservations:
        if not isinstance(row, dict) or not all(key in row for key in (
            "name", "address", "bytes", "role", "lifetime", "promotion_receipts",
        )) or not isinstance(row["name"], str) or not row["name"] or row["name"] in names:
            _refuse("deployment_provider_contract", "invalid provider reservation identity")
        names.add(row["name"])
        address = _integer(row["address"], "reservation address")
        size = _integer(row["bytes"], "reservation bytes", positive=True)
        if address < window_address or address + size > window_address + window_bytes:
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

    def allocate(name, size, role, alignment=None):
        nonlocal cursor
        if alignment is None:
            alignment = provider["allocation_alignment"]
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
             max(provider["allocation_alignment"], interface["workspace_alignment"]))
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
    required = cursor - window_address
    if required > window_bytes:
        _refuse("deployment_window_overflow",
                f"disjoint on-chip deployment requires {required} bytes; window {window_bytes} bytes; "
                f"overflow {required - window_bytes} bytes")
    deployment = {
        "schema": "merlin.onchip_deployment.v1", "request_digest": source.digest(),
        "target_identity": source.target_identity, "window": provider["window"], "provider": provider,
        "allocations": allocations, "inputs": inputs, "outputs": outputs, "parameters": parameter,
        "calls": calls,
        "window_required_bytes": required, "window_remaining_bytes": window_bytes - required,
        "storage_policy": "disjoint; no lifetime sharing", "completion": "synchronous generated application run",
        "host_map": host_map, "max_polls": provider["max_polls"],
        "polling_bound": "host polls per generated device wait; independent of functional-model instruction limit",
        "host_allocation": provider["host_allocation"],
    }
    interface["deployment"] = deployment
    return deployment


def adapter_c(name, deployment):
    """A fixed consumer ABI; all model-specific binding stays generated."""
    runtime = deployment["provider"]["runtime"]
    declarations = f"""
typedef struct {{ const char *name; const char *role; uintptr_t address; size_t bytes; }} merlin_deployment_hash_range;
extern const merlin_deployment_hash_range merlin_deployment_hash_ranges[];
extern const size_t merlin_deployment_hash_count;
int merlin_deployment_bind({runtime["host_type"]} *,{runtime["io_type"]},size_t max_polls);
int merlin_deployment_run(void);
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
    bind_parameters = (f"  status={name}_bind_parameters(&merlin_deployment_context,{pointer(parameter, True)},{parameter['bytes']}u);\n"
                       "  if (status) return status;\n") if parameter else ""
    implementation = f"""
static {name}_context merlin_deployment_context;
_Static_assert(sizeof(merlin_deployment_context)<={deployment["provider"]["host_context_reservation_bytes"]}u,"deployment context reservation exceeded");
const merlin_deployment_hash_range merlin_deployment_hash_ranges[]={{{hashes}}};
const size_t merlin_deployment_hash_count={len(deployment['outputs'])}u;
int merlin_deployment_bind({runtime["host_type"]} *host,{runtime["io_type"]} io,size_t max_polls) {{
  int status={name}_bind(&merlin_deployment_context,host,io,max_polls);
  if (status) return status;
{bind_parameters}  return 0;
}}
int merlin_deployment_run(void) {{
  const void *inputs[]={{{inputs}}};
  void *outputs[]={{{outputs}}};
  return {name}_run(&merlin_deployment_context,inputs,outputs);
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
        if (row.get("imem_offset_bytes") != 0 or row.get("capacity_bytes") != deployment["host_map"]["imem_bytes"]
                or row.get("upload_convention") != "counted_ensure_loaded"):
            _refuse("deployment_provider_contract", "unsupported deployment image upload convention")
        # The counted loader uploads exactly these words; it appends no epilogue.
        if len(payload) > row["capacity_bytes"]:
            _refuse("deployment_imem_overflow", f"instruction image exceeds {row['capacity_bytes']} bytes")
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
            event["device_launch_schedule"] = validate_launch_schedule(
                call["device_launch_schedule"], image_sha256s=[row["sha256"] for row in installs],
                placement=call["placement"])
            if event["device_launch_schedule"]["total_launches"] != call["launches_per_invocation"]:
                _refuse("deployment_runtime_contract", "provider launch count differs from schedule")
            event["images"] = installs
        else:
            count = call["launches_per_invocation"]
            if len(installs) != 1 or type(count) is not int or count <= 0:
                _refuse("deployment_runtime_contract", "generated call needs an explicit launch count or launch schedule")
            event["events"] = [installs[0], {
                "operation": "launch_group", "repeat": count, "events": [
                    {"operation": "write_packed_inputs", "destinations": "provider reservations for this call"},
                    {"operation": "start", "entry": deployment["provider"]["runtime"]["start"]},
                    {"operation": "wait", "entry": deployment["provider"]["runtime"]["wait"], "max_polls": deployment["max_polls"]},
                    {"operation": "read_outputs", "completion": "successful wait and generated status checks"},
                ]}, {"operation": "restore_dense_outputs"}]
        runtime_events.append(event)
    steps.extend([
        {"operation": "bind", "entry": "merlin_deployment_bind", "completion": "synchronous"},
        {"operation": "invoke", "entry": "merlin_deployment_run", "generated_entry": interface["platform_binding"]["invoke"],
         "completion": "synchronous", "runtime_events": runtime_events,
         "failure": "preserve buffers until device completion or reset"},
        {"operation": "hash", "ranges": deployment["outputs"], "completion": "after_successful_run"},
    ])
    record = {**deployment, "steps": steps, "hash_ranges": deployment["outputs"],
              "image_library": list(image_library.values()), "reload_policy": "generated counted ensure_loaded",
              "adapter_header": "handoff/deployment_adapter.h", "timeout": "max_polls per synchronous device wait"}
    files["deployment.json"] = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    header = ("#pragma once\n#include \"" + Path(interface["header"]).name + "\"\n"
              f"#define MERLIN_DEPLOYMENT_WINDOW_BASE UINT64_C({deployment['window']['address']})\n"
              f"#define MERLIN_DEPLOYMENT_WINDOW_BYTES {deployment['window']['bytes']}u\n"
              f"#define MERLIN_DEPLOYMENT_IMEM_BASE UINT64_C({deployment['host_map']['imem_address']})\n"
              f"#define MERLIN_DEPLOYMENT_CSR_BASE UINT64_C({deployment['host_map']['csr_address']})\n"
              f"#define MERLIN_DEPLOYMENT_IMEM_BYTES {deployment['host_map']['imem_bytes']}u\n"
              f"#define MERLIN_DEPLOYMENT_MAX_POLLS {deployment['max_polls']}u\n").encode()
    files["c-package/combined/deployment_adapter.h"] = header
    files["handoff/deployment_adapter.h"] = header
