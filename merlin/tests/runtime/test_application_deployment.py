"""Exercise publication bounds, image integrity, and the generated fixed ABI."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from merlin.compile.application_deployment import WINDOW_ADDRESS, WINDOW_BYTES, adapter_c, place, publish
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType


class DeploymentTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        tensor = TensorType((8,), "bf16", "raw-bit-pattern")
        self.source = KernelRequest((SemanticNode("x", "input", (), tensor, effect="input"),
            SemanticNode("y", "identity", ("x",), tensor)), ("y",), ("external",),
            (("x", "external"),), "test-target")
        self.interface = dict(inputs=["x"], outputs=["y"], workspace_bytes=128, workspace_alignment=64,
                              header="combined/app.h", platform_binding=dict(completion="synchronous", invoke="app_run"))
        payload = b"\x01\0\0\0"
        (self.root / "program.bin").write_bytes(payload)
        self.provider = dict(schema="atlas.provider_deployment.v1",
            window=dict(address=WINDOW_ADDRESS, bytes=WINDOW_BYTES),
            reservations=[dict(name="device_frame", address=WINDOW_ADDRESS, bytes=4096,
                               role="device_scratch", lifetime="through_run_completion", promotion_receipts=[])],
            raw_scratch_bytes=4096, conversion_scratch_bytes=128,
            images=[dict(path="program.bin", sha256=sha256(payload).hexdigest(), bytes=len(payload),
                         imem_offset_bytes=0, capacity_bytes=131072, upload_convention="counted_ensure_loaded")])
        self.adapter = SimpleNamespace(deployment_requirements=lambda *_: deepcopy(self.provider))
        self.regions = [dict(id="r", context_index=0, placement="accelerator", native_programs=["."],
                             description=dict(launches_per_invocation=1))]
        self.options = CodegenOptions(io_memory_base=WINDOW_ADDRESS, io_memory_bytes=WINDOW_BYTES,
            io_memory_capacity_base=WINDOW_ADDRESS, io_memory_capacity_bytes=WINDOW_BYTES)

    def placement(self):
        with codegen_options(self.options):
            return place(self.source, self.adapter, self.regions, self.interface, self.root)

    def test_default_preserves_interface_and_does_not_call_provider(self):
        self.adapter.deployment_requirements = lambda *_: self.fail("default provider call")
        before = deepcopy(self.interface)
        self.assertIsNone(place(self.source, self.adapter, [], self.interface, self.root))
        self.assertEqual(self.interface, before)
        with codegen_options(CodegenOptions(io_memory_base=WINDOW_ADDRESS, io_memory_bytes=WINDOW_BYTES)):
            self.assertIsNone(place(self.source, self.adapter, [], self.interface, self.root))

    def test_disjoint_arrays_public_ranges_and_hash_roles(self):
        self.interface["model_state"] = {"ports": [dict(node_id="x", output_index=0, mode="updated", name="cache")]}
        deployment = self.placement()
        allocations = deployment["allocations"]
        cursor = WINDOW_ADDRESS + 4096
        for row in allocations:
            self.assertGreaterEqual(row["address"], cursor)
            self.assertEqual(row["address"] % row["alignment"], 0)
            cursor = row["address"] + row["bytes"]
        self.assertEqual([row["name"] for row in allocations[:3]], ["device_scratch", "conversion_scratch", "workspace"])
        self.assertEqual(deployment["outputs"][0]["role"], "persistent_state_update")
        self.assertEqual(sum(row["bytes"] for row in deployment["host_allocation"]["reservations"]), 65536)
        files = {}
        publish(deployment, self.interface, files, self.root)
        record = json.loads(files["deployment.json"])
        self.assertEqual([row["operation"] for row in record["steps"]],
                         ["window_preload", "bind", "invoke", "hash"])
        self.assertEqual(record["steps"][2]["runtime_events"][0]["events"][0]["final_upload_bytes"], 4)
        self.assertEqual(record["steps"][2]["runtime_events"][0]["events"][0]["address"], 0x20000)
        self.assertEqual(record["hash_ranges"], deployment["outputs"])
        declarations, implementation = adapter_c("app", deployment)
        self.assertIn("int sp5_bind(atlas_host *,atlas_tile_memory_io,size_t max_polls)", declarations)
        self.assertIn("sizeof(sp5_context)<=4096u", implementation)
        self.assertIn("persistent_state_update", implementation)

    def test_overflow_names_required_window_and_excess(self):
        self.provider["raw_scratch_bytes"] = WINDOW_BYTES
        with self.assertRaisesRegex(RuntimeError, r"requires [0-9]+ bytes; window 786432 bytes; overflow [0-9]+ bytes"):
            self.placement()

    @unittest.skipUnless(shutil.which("cc"), "requires C compiler")
    def test_fixed_adapter_c_binds_and_runs_published_addresses(self):
        deployment = self.placement()
        declarations, implementation = adapter_c("app", deployment)
        expected_input = deployment["inputs"][0]["address"]
        expected_output = deployment["outputs"][0]["address"]
        code = f"""
#include <stdint.h>
#include <stddef.h>
typedef struct {{ int initialized; }} atlas_host;
typedef struct {{ int unused; }} atlas_tile_memory_io;
typedef struct {{ int initialized; }} app_context;
static int app_bind(app_context *ctx,atlas_host *host,atlas_tile_memory_io io,size_t polls) {{
  (void)io; ctx->initialized=host->initialized && polls; return !ctx->initialized;
}}
static int app_run(app_context *ctx,const void *const *inputs,void *const *outputs) {{
  return !ctx->initialized || (uintptr_t)inputs[0]!={expected_input}u ||
         (uintptr_t)outputs[0]!={expected_output}u;
}}
{declarations}
{implementation}
int main(void) {{
  atlas_host host={{1}};
  atlas_tile_memory_io io={{0}};
  return sp5_bind(&host,io,100u) || sp5_run() || sp5_hash_count!=1u ||
         sp5_hash_ranges[0].address!={expected_output}u || sp5_hash_ranges[0].bytes!=16u;
}}
"""
        source = self.root / "adapter.c"
        executable = self.root / "adapter"
        source.write_text(code)
        subprocess.run(["cc", "-std=c11", "-Wall", "-Werror", str(source), "-o", str(executable)], check=True)
        subprocess.run([str(executable)], check=True)

    def test_bad_provider_reservations_and_mutated_image_refuse(self):
        self.provider["reservations"].append({**self.provider["reservations"][0], "name": "overlap"})
        with self.assertRaisesRegex(RuntimeError, "reservations overlap"):
            self.placement()
        self.provider["reservations"].pop()
        deployment = self.placement()
        (self.root / "program.bin").write_bytes(b"\0\0\0\0")
        with self.assertRaisesRegex(RuntimeError, "image identity differs"):
            publish(deployment, self.interface, {}, self.root)

    def test_existing_per_call_reuse_and_two_image_runtime_order(self):
        first = self.provider["reservations"][0]
        first["lifetime"] = dict(scope="region_call", context_index=0)
        self.provider["reservations"].append({**first, "name": "second_frame",
            "lifetime": dict(scope="region_call", context_index=1)})
        folder = self.root / "second"
        folder.mkdir()
        payload = b"\x02\0\0\0"
        (folder / "program.bin").write_bytes(payload)
        self.provider["images"].append({**self.provider["images"][0], "path": "second/program.bin",
                                        "sha256": sha256(payload).hexdigest()})
        self.regions.append(dict(id="r2", context_index=1, placement="accelerator", native_programs=["second"],
                                 description=dict(streaming=dict(launch_count=3))))
        deployment = self.placement()
        self.assertEqual(deployment["allocations"][0]["address"], WINDOW_ADDRESS + 4096)
        files = {}
        publish(deployment, self.interface, files, self.root)
        record = json.loads(files["deployment.json"])
        events = record["steps"][2]["runtime_events"]
        self.assertEqual([row["region"] for row in events], ["r", "r2"])
        self.assertEqual([row["events"][0]["path"] for row in events], ["program.bin", "second/program.bin"])
        self.assertEqual([row["events"][1]["repeat"] for row in events], [1, 3])
        self.provider["reservations"][1]["lifetime"]["context_index"] = 0
        with self.assertRaisesRegex(RuntimeError, "reservations overlap within a call"):
            self.placement()

    def test_counted_upload_accepts_exact_capacity_and_refuses_one_word_over(self):
        for size in (131072, 131076):
            payload = bytes(size)
            (self.root / "program.bin").write_bytes(payload)
            self.provider["images"][0].update(bytes=size, sha256=sha256(payload).hexdigest())
            deployment = self.placement()
            if size == 131072:
                publish(deployment, self.interface, {}, self.root)
            else:
                with self.assertRaisesRegex(RuntimeError, "image exceeds 131072"):
                    publish(deployment, self.interface, {}, self.root)


if __name__ == "__main__":
    unittest.main()
