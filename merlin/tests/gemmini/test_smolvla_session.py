"""SmolVLA artifacts, session routing and native-backend diagnostics."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from merlin.common.paths import repo_root


SOURCE = repo_root() / "examples/gemmini/comparisons/tvm/smolvla_session.py"
sys.path.insert(0, str(SOURCE.parent))
SPEC = importlib.util.spec_from_file_location("smolvla_session_tested", SOURCE)
FACTORY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FACTORY)
PREP_SPEC = importlib.util.spec_from_file_location("smolvla_observation_tested", SOURCE.with_name("prepare_smolvla_observation.py"))
PREPARATION = importlib.util.module_from_spec(PREP_SPEC)
PREP_SPEC.loader.exec_module(PREPARATION)
NUMERIC_SPEC = importlib.util.spec_from_file_location("smolvla_bf16_diagnostic_tested", SOURCE.with_name("bf16_export.py"))
NUMERICS = importlib.util.module_from_spec(NUMERIC_SPEC)
NUMERIC_SPEC.loader.exec_module(NUMERICS)
DIAGNOSTIC_SPEC = importlib.util.spec_from_file_location("smolvla_backends_tested", SOURCE.with_name("diagnose_smolvla_backends.py"))
DIAGNOSTIC = importlib.util.module_from_spec(DIAGNOSTIC_SPEC)
DIAGNOSTIC_SPEC.loader.exec_module(DIAGNOSTIC)


class SmolVLAArtifactTests(unittest.TestCase):
    def config(self):
        return {"type": "smolvla", **FACTORY.GEOMETRY,
                "input_features": {"observation.state": {"type": "STATE", "shape": [6]},
                    **{f"observation.images.camera{index}": {"type": "VISUAL", "shape": [3, 256, 256]}
                       for index in range(1, 4)}},
                "output_features": {"action": {"type": "ACTION", "shape": [6]}}}

    def test_absent_checkpoint_fails_before_framework_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                FACTORY.checkpoint_artifacts(directory)

    def test_nonofficial_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(self.config()))
            (root / "model.safetensors").write_text("fixture only; never a model")
            with self.assertRaisesRegex(ValueError, "pinned official full"):
                FACTORY.checkpoint_artifacts(root)

    def test_geometry_truncation_and_precision_changes_are_rejected(self):
        for change in ({"num_steps": 1}, {"num_vlm_layers": 2}, {"load_vlm_weights": False}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "config.json").write_text(json.dumps({**self.config(), **change}))
                (root / "model.safetensors").write_text("fixture only; never a model")
                identity = {"path": str(root / "model.safetensors"), "sha256": FACTORY.WEIGHT_SHA256}
                with mock.patch.object(FACTORY, "file_identity", return_value=identity):
                    with self.assertRaisesRegex(ValueError, "complete pinned"):
                        FACTORY.checkpoint_artifacts(root)

    def test_unsupported_precision_and_unattributed_dataset_fail_early(self):
        base = {"checkpoint_dir": "/missing", "backbone_dir": "/missing", "model2mlir_root": "/missing"}
        for change in ({"precision": "float32"}, {"cpu_threads": 32}, {"fixture_kind": "dataset"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                FACTORY.get_model_and_inputs(**base, **change)

    def test_source_mean_requires_explicit_integer_session_and_one_native_thread(self):
        base = {"checkpoint_dir": "/missing", "backbone_dir": "/missing", "model2mlir_root": "/missing"}
        for change in ({"source_mean_policy": "unknown"}, {"source_mean_policy": "torch_cpu_mean720_v1"}, {"source_mean_policy": "torch_cpu_mean720_v1", "quantized": True, "integer_diagnostic_dir": "/missing", "cpu_threads": 2}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "Source mean"):
                FACTORY.get_model_and_inputs(**base, **change)

    def test_restricted_component_is_rejected_before_read(self):
        with self.assertRaisesRegex(ValueError, "Restricted path component"):
            FACTORY.directory_artifacts("/tmp/" + "vl" + "si" + "-fixture")


class SmolVLATensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)
        cls.config = SimpleNamespace(resize_imgs_with_padding=(2, 2), tokenizer_max_length=3,
                                    max_state_dim=4, max_action_dim=4, chunk_size=5,
                                    robot_state_feature=SimpleNamespace(shape=[2]))

    def test_synthetic_fixture_is_repeatable_and_pads_only_physical_state(self):
        first = FACTORY.fixture_inputs(self.config, 3, seed=42)
        second = FACTORY.fixture_inputs(self.config, 3, seed=42)
        self.assertEqual(first[0].shape, (1, 3, 3, 2, 2))
        self.assertTrue(all(self.torch.equal(a, b) for a, b in zip(first, second)))
        self.assertEqual(self.torch.count_nonzero(first[4][:, 2:]).item(), 0)

    def test_npz_precision_is_never_coerced(self):
        import numpy as np
        values = FACTORY.fixture_inputs(self.config, 3, seed=42)
        names = ("images", "image_masks", "language_tokens", "language_mask", "state", "noise")
        arrays = {name: value.numpy() for name, value in zip(names, values)}
        arrays["images"] = arrays["images"].astype(np.float16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.npz"
            np.savez(path, **arrays)
            with self.assertRaisesRegex(ValueError, "dtype"):
                FACTORY.fixture_inputs(self.config, 3, input_npz=path)

    def test_complete_schedule_exact_time_stream_and_physical_decode(self):
        root = os.environ.get("MODEL2MLIR_ROOT")
        if not root:
            self.skipTest("MODEL2MLIR_ROOT selects the actual loader/protocol for this contract test")
        from verify_session import load_module, load_protocol, verify_session
        import numpy as np
        torch = self.torch
        load_protocol(Path(root))
        loader = load_module("smolvla_contract_loader", Path(root) / "workloads/smolvla/loader.py")

        class Backbone(torch.nn.Module):
            def forward(self, **kwargs):
                embeds = kwargs["inputs_embeds"][0]
                cache = {index: {"key_states": embeds, "value_states": embeds}
                         for index in range(16)}
                return (embeds, None), cache

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vlm_with_expert = Backbone()
                self.config = SimpleNamespace(num_vlm_layers=16, num_steps=10, chunk_size=5,
                                              action_feature=SimpleNamespace(shape=[2]))

            def embed_prefix(self, images, masks, tokens, language_mask, state):
                embeds = state[:, None, :] + sum(image.mean() for image in images)
                return embeds, torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1)

            def denoise_step(self, prefix_masks, cache, flow, timestep):
                return flow * 0.2 + timestep[:, None, None] + cache[0]["key_states"].mean()

            def sample_actions(self, images, masks, tokens, language_mask, state, noise):
                embeds, _, _ = self.embed_prefix(images, masks, tokens, language_mask, state)
                value = noise
                for index in range(self.config.num_steps):
                    time = torch.tensor(1.0 + index * (-1.0 / self.config.num_steps)).expand(1)
                    value = value - (1.0 / self.config.num_steps) * (
                        value * 0.2 + time[:, None, None] + embeds.mean())
                return value

        model = Model().eval()
        inputs = FACTORY.fixture_inputs(self.config, 3, seed=42)
        provenance = {}
        session = FACTORY.make_policy_session(model, inputs, loader, provenance)
        self.assertTrue(provenance["upstream_sample_actions_equivalence"]["exact"])
        self.assertEqual([program.steps for program in session.programs], [1, 10, 1])
        self.assertEqual(session.streams[0].values.shape, (10, 1))
        self.assertEqual(session.programs[-1].module.action_dim, 2)
        self.assertFalse(any(route.name == "timestep" for route in session.routes))

        def compile_eager(program, _inputs, _outputs):
            def run(values):
                with torch.no_grad():
                    return program.module(*(torch.from_numpy(value) for value in values))
            return run

        result = verify_session(session, compile_eager, np, torch)
        counts = {name: sum(row["program"] == name for row in result["checks"])
                  for name in ("prefix_encode", "flow_denoise", "action_decode")}
        self.assertEqual(counts, {"prefix_encode": 2, "flow_denoise": 20, "action_decode": 2})
        self.assertTrue(result["reset_verified"])


class SmolVLAObservationTests(unittest.TestCase):
    def schema(self):
        return {"codebase_version": "v3.0", "robot_type": "so100", "fps": 30,
                "total_episodes": 50, "total_frames": 19631, "total_tasks": 1,
                "features": {**{name: {"dtype": "video", "shape": [480, 640, 3],
                                       "names": ["height", "width", "channels"]}
                                for name in PREPARATION.CAMERA_MAP},
                             **{name: {"dtype": "float32", "shape": [6],
                                       "names": PREPARATION.STATE_NAMES}
                                for name in ("observation.state", "action")}}}

    def test_official_camera_coordinate_schema_and_frozen_episode_split(self):
        PREPARATION.validate_schema(self.schema())
        counts = {split: sum(PREPARATION.episode_split(index) == split for index in range(50))
                  for split in ("train", "development", "heldout")}
        self.assertEqual(counts, {"train": 30, "development": 10, "heldout": 10})
        for index in (-1, 50, True):
            with self.subTest(index=index), self.assertRaises(ValueError):
                PREPARATION.episode_split(index)
        changed = self.schema()
        changed["features"]["observation.state"]["names"] = list(reversed(PREPARATION.STATE_NAMES))
        with self.assertRaisesRegex(ValueError, "coordinate schema"):
            PREPARATION.validate_schema(changed)
        changed = self.schema()
        changed["features"]["observation.images.fake"] = changed["features"]["observation.images.top"]
        with self.assertRaisesRegex(ValueError, "exactly its recorded"):
            PREPARATION.validate_schema(changed)

    def test_recorded_preparation_uses_original_processor_without_inventing_camera(self):
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        config = SimpleNamespace(image_features={name: {} for name in (
            "observation.images.camera1", "observation.images.camera2", "observation.images.camera3")},
            empty_cameras=0, resize_imgs_with_padding=(2, 2), max_state_dim=32,
            max_action_dim=32, chunk_size=50)
        policy = SimpleNamespace(config=config)
        policy.prepare_images = lambda batch: SmolVLAPolicy.prepare_images(policy, batch)
        policy.prepare_state = lambda batch: SmolVLAPolicy.prepare_state(policy, batch)
        observation = {"task": "Pick up the cube", "observation.state": torch.arange(6, dtype=torch.float32),
                       "action": torch.full((6,), 99.0),
                       "observation.images.top": torch.ones(3, 480, 640) * 0.25,
                       "observation.images.wrist": torch.ones(3, 480, 640) * 0.75}
        passed = []

        def processor(batch):
            passed.append(batch)
            return {**{key: value.unsqueeze(0) for key, value in batch.items() if torch.is_tensor(value)},
                    "observation.state": (batch["observation.state"] - 2.0).unsqueeze(0),
                    "observation.language.tokens": torch.zeros(1, 48, dtype=torch.int64),
                    "observation.language.attention_mask": torch.ones(1, 48, dtype=torch.bool)}

        rng = torch.random.get_rng_state().clone()
        first = PREPARATION.prepare_tensors(policy, processor, observation, noise_seed=7, torch=torch)
        second = PREPARATION.prepare_tensors(policy, processor, observation, noise_seed=7, torch=torch)
        self.assertEqual(first[0].shape, (1, 2, 3, 2, 2))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))
        self.assertTrue(torch.equal(first[4][0, :6], observation["observation.state"] - 2))
        self.assertEqual(first[4][0, 6:].count_nonzero().item(), 0)
        self.assertTrue(first[1].all())
        self.assertNotIn("action", passed[0])
        self.assertNotIn("observation.images.camera3", passed[0])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(observation["observation.state"], torch.arange(6, dtype=torch.float32)))
        observation["observation.images.top"] = observation["observation.images.top"].to(torch.float16)
        with self.assertRaisesRegex(ValueError, "FP32 CHW"):
            PREPARATION.prepare_tensors(policy, processor, observation, noise_seed=7, torch=torch)

    def test_observation_manifest_checks_npz_and_camera_route_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "observation.npz"
            fixture.write_bytes(b"identity control only")
            manifest = {
                "version": 1, "fixture_kind": "dataset", "dataset_id": PREPARATION.DATASET_ID,
                "dataset_revision": PREPARATION.DATASET_REVISION, "checkpoint_revision": FACTORY.MODEL_REVISION,
                "precision": "original_lerobot", "schema_verified": True,
                "selected_cameras": list(PREPARATION.CAMERA_MAP.values()), "camera_mapping": PREPARATION.CAMERA_MAP,
                "state_names": PREPARATION.STATE_NAMES, "active_gate_changed": False,
                "model_quality_qualified": False, "episode_index": 0, "split": "train",
                "input_fixture": FACTORY.file_identity(fixture, FACTORY.safe_path),
                "bindings": [{"path": str(root / relative), "sha256": digest}
                             for relative, digest in PREPARATION.PINNED_FILES.items()]}
            path = root / "observation.json"
            path.write_text(json.dumps(manifest))
            identities = {item["path"]: item for item in manifest["bindings"] + [manifest["input_fixture"]]}
            with mock.patch.object(FACTORY, "file_identity", side_effect=lambda name, _: identities[str(name)]):
                self.assertEqual(FACTORY.observation_manifest(fixture, path)["selected_cameras"],
                                 ["observation.images.camera1", "observation.images.camera2"])
                manifest["selected_cameras"].reverse()
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "pinned preparation"):
                    FACTORY.observation_manifest(fixture, path)
            manifest["selected_cameras"].reverse()
            path.write_text(json.dumps(manifest))
            fixture.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "NPZ identity"):
                FACTORY.observation_manifest(fixture, path)


class SmolVLASourceMeanTests(unittest.TestCase):
    def test_session_hook_keeps_default_table_and_selects_only_declared_policy(self):
        import torch
        import bf16_export
        import tinyllama_integer
        original = tinyllama_integer.export_options(torch)["custom_translation_table"]
        with mock.patch.object(tinyllama_integer, "export_options", side_effect=lambda _: {"custom_translation_table": dict(original)}), mock.patch.object(bf16_export, "source_mean_translations", return_value={torch.ops.aten.mean.dim: "selected"}) as select:
            self.assertEqual(FACTORY.get_onnx_export_options(torch)["custom_translation_table"], original)
            model = SimpleNamespace(session=SimpleNamespace(metadata={"provenance": {}}))
            self.assertEqual(FACTORY.get_onnx_export_options(torch, model)["custom_translation_table"], original)
            select.assert_not_called()
            model.session.metadata["provenance"]["source_mean_policy"] = {"name": "torch_cpu_mean720_v1"}
            table = FACTORY.get_onnx_export_options(torch, model)["custom_translation_table"]
            self.assertEqual(table, {**original, torch.ops.aten.mean.dim: "selected"})
            select.assert_called_once_with(torch, "torch_cpu_mean720_v1")

    def test_native_policy_needs_no_source_bindings_and_unknown_policy_is_rejected(self):
        self.assertIsNone(NUMERICS.source_mean_contract(None))
        self.assertEqual(NUMERICS.source_mean_translations(None), {})
        with self.assertRaisesRegex(ValueError, "Unknown source mean"):
            NUMERICS.source_mean_contract(None, "unknown")

    def test_source_mean_refuses_unproved_runtime_and_modified_binary(self):
        runtime = SimpleNamespace(version=SimpleNamespace(git_version="449b1768410104d3ed79d3bcfe4ba1d65c7f22c0"), backends=SimpleNamespace(cpu=SimpleNamespace(get_cpu_capability=lambda: "AVX512")), get_num_threads=lambda: 1, __file__="/tmp/pinned-torch/__init__.py")
        with mock.patch("verify_resnet.file_identity", return_value={"sha256": "changed"}), self.assertRaisesRegex(ValueError, "independently proved"):
            NUMERICS.source_mean_contract(runtime, "torch_cpu_mean720_v1")
        for field, value in (("get_num_threads", lambda: 2), ("version", SimpleNamespace(git_version="different")), ("backends", SimpleNamespace(cpu=SimpleNamespace(get_cpu_capability=lambda: "AVX2")))):
            with self.subTest(field=field), mock.patch.object(runtime, field, value), self.assertRaisesRegex(ValueError, "pinned PyTorch"):
                NUMERICS.source_mean_contract(runtime, "torch_cpu_mean720_v1")

    def test_export_selects_only_bound_fp32_last_axis_keepdim_geometry(self):
        import torch
        class Means(torch.nn.Module):
            def forward(self, value, other):
                return value.mean(-1, keepdim=True), value.mean(-1), value.mean(1, keepdim=True), other.mean(-1, keepdim=True)
        for dtype in (torch.float32, torch.bfloat16):
            inputs = (torch.arange(36000, dtype=torch.float32).reshape(1, 50, 720).to(dtype), torch.ones(1, 2, 720, dtype=dtype))
            before = tuple(value.clone() for value in inputs)
            with mock.patch.object(NUMERICS, "source_mean_contract", return_value={"name": "torch_cpu_mean720_v1"}):
                table = NUMERICS.source_mean_translations(torch, "torch_cpu_mean720_v1")
            graph = torch.onnx.export(Means().eval(), inputs, dynamo=True, opset_version=18, optimize=False, custom_translation_table=table).model_proto
            selected = [node for node in graph.graph.node if node.domain == "merlin.source_preserving"]
            self.assertEqual([(node.op_type, len(node.input), len(node.output), len(node.attribute)) for node in selected], [("TorchMean720Fp32", 1, 1, 0)] if dtype == torch.float32 else [])
            self.assertEqual(sum(node.op_type == "ReduceMean" for node in graph.graph.node), 3 if dtype == torch.float32 else 4)
            self.assertTrue(all(torch.equal(original, value) for original, value in zip(before, inputs)))


class SmolVLAArithmeticDiagnosticTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("TVM_LIBRARY_PATH"), "Requires explicit LLVM-enabled TVM host build")
    def test_fused_bf16_silu_preserves_one_output_cast_and_fp32_default(self):
        from decimal import Decimal, localcontext
        import numpy as np
        import onnx
        import torch
        import tvm
        from verify_session import StageCompiler, array

        class Silu(torch.nn.Module):
            def forward(self, value):
                return torch.nn.functional.silu(value)

        values = torch.tensor([0.0, -0.0, 0.01171875, 0.01953125, 1.015625, 2.015625, -1.0078125, -2.015625, -15.75, -14.75], dtype=torch.bfloat16)
        source = Silu()(values)
        self.assertTrue(torch.equal(source.view(torch.uint16), Silu()(values.float()).bfloat16().view(torch.uint16)))
        staged = values * values.sigmoid()
        self.assertTrue(torch.any(staged[values > 0].view(torch.uint16) != source[values > 0].view(torch.uint16)))
        self.assertTrue(torch.any(staged[values < 0].view(torch.uint16) != source[values < 0].view(torch.uint16)))
        for value, output in zip(values[2:], source[2:], strict=True):
            word = int(output.view(torch.uint16))
            with localcontext() as context:
                context.prec = 80
                exact_input = Decimal.from_float(float(value))
                ideal = exact_input / (1 + (-exact_input).exp())
                distances = [(abs(ideal - Decimal.from_float(float(torch.tensor(candidate, dtype=torch.uint16).view(torch.bfloat16)))), candidate) for candidate in range(word - 2, word + 3)]
                ordered = sorted(distances, key=lambda item: (item[0], item[1] & 1))
                self.assertEqual(ordered[0][1], word)
                self.assertGreater(ordered[1][0] - ordered[0][0], Decimal("1e-65"))
        inputs = (array(values, np, torch),)
        for mode in ("baseline", "optimized"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                compiler = StageCompiler(Path(directory), mode, np, torch, onnx, tvm, exporter="dynamo", opset=18, bf16_fused_export=True)
                run = compiler(SimpleNamespace(name="bf16_silu", module=Silu()), inputs, 1)
                actual = array(run(inputs), np, torch)
                self.assertEqual(actual.tobytes(), array(source, np, torch).tobytes())
                self.assertEqual(array(run(inputs), np, torch).tobytes(), actual.tobytes())
                graph = onnx.load(compiler.records[0]["onnx_artifacts"][0]["path"])
                self.assertEqual([node.op_type for node in graph.graph.node], ["Cast", "Sigmoid", "Mul", "Cast"])
        # FP32 dispatcher and default BF16 export remain their built-in graphs.
        for dtype, fused, expected in ((torch.float32, False, ["Sigmoid", "Mul"]), (torch.float32, True, ["Sigmoid", "Mul"]), (torch.bfloat16, False, ["Sigmoid", "Mul"])):
            graph = torch.onnx.export(Silu(), (values.to(dtype),), dynamo=True, opset_version=18, optimize=False, custom_translation_table=NUMERICS.translations(torch) if fused else None).model_proto
            self.assertEqual([node.op_type for node in graph.graph.node], expected)
        self.assertEqual(int(source[0].view(torch.uint16)), 0)
        self.assertEqual(int(source[1].view(torch.uint16)), 0x8000)

    def test_proposal_refuses_heldout_unbound_or_nonfinite_metadata_and_never_accepts(self):
        fixture = {"split": "train", "schema_verified": True, "input_fixture": {"path": "/chosen/observation.npz", "sha256": "0" * 64}, "state_names": ["first", "second"]}
        proposal = DIAGNOSTIC.numerical_contract_proposal(fixture, {}, {}, evidence=[{"status": "passed"}])
        self.assertEqual(proposal["qualification_verdict"], "NO_ACCEPTANCE")
        self.assertFalse(proposal["thresholds_adopted"])
        self.assertFalse(proposal["acceptance_contract_selected"])
        self.assertFalse(proposal["source"]["fixture_binding_verified_by_producer"])
        self.assertFalse(proposal["source"]["source_closure_verified_by_producer"])
        self.assertEqual(proposal["existing_gate"]["rtol"], 1e-4)
        self.assertIsNone(proposal["action_quality"]["policy_coordinates"]["limits"]["rms_error"])
        self.assertFalse(proposal["claim_scoped_gaps"]["functional_compiler_checks"]["training_camera_or_backbone_identity_required"])
        proposal["source"]["fixture"]["state_names"].append("third")
        self.assertEqual(fixture["state_names"], ["first", "second"])
        for change in ({"split": "heldout"}, {"split": "synthetic"}, {"schema_verified": False}, {"input_fixture": None}, {"input_fixture": {"path": "/chosen/observation.npz", "sha256": "not a hash"}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                DIAGNOSTIC.numerical_contract_proposal({**fixture, **change}, {}, {})
        with self.assertRaises(ValueError):
            DIAGNOSTIC.numerical_contract_proposal(fixture, {"epsilon": float("nan")}, {})

    def test_action_statistics_preserve_raw_signed_zero_encoding_distinction(self):
        import torch
        result = DIAGNOSTIC.statistics(torch.tensor([0.0]), torch.tensor([-0.0]), torch)
        self.assertEqual(result["exactly_unequal_count"], 0)
        self.assertEqual(result["raw_encoding_unequal_count"], 1)
        self.assertEqual(result["outside_existing_pointwise_comparison_count"], 0)

    def test_original_processor_provenance_binds_selected_config_state_and_classes(self):
        import torch
        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
        features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)), "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))}
        norm_map = {FeatureType.STATE: NormalizationMode.MEAN_STD, FeatureType.ACTION: NormalizationMode.MEAN_STD}
        before = NormalizerProcessorStep(features=features, norm_map=norm_map, stats={})
        after = UnnormalizerProcessorStep(features=features, norm_map=norm_map, stats={})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = DIAGNOSTIC.original_processor_diagnostic(root, root, torch, physical_dim=2)
            self.assertFalse(result["available"])
            (root / "stats.safetensors").write_bytes(b"selected bytes; mocked processor loading")
            for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
                (root / name).write_text(json.dumps({"steps": [{"state_file": "stats.safetensors"}]}))
            with mock.patch.object(PolicyProcessorPipeline.__origin__, "from_pretrained", side_effect=[SimpleNamespace(steps=[before]), SimpleNamespace(steps=[after])]) as load:
                result = DIAGNOSTIC.original_processor_diagnostic(root, root, torch, physical_dim=2)
            self.assertEqual(load.call_count, 2)
            self.assertTrue(result["available"])
            self.assertTrue(result["bindings_unchanged"])
            bound = {Path(item["path"]).name for item in result["bindings"]}
            self.assertTrue({"policy_preprocessor.json", "policy_postprocessor.json", "stats.safetensors", "normalize_processor.py", "pipeline.py", "processor_smolvla.py"} <= bound)
            self.assertFalse(result["normalization_qualified"])
            self.assertTrue(result["action_probe_transform_is_identity"])

    def test_original_processor_namespaced_statistics_are_not_silently_remapped(self):
        import torch
        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
        from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
        features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)), "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))}
        norm_map = {FeatureType.STATE: NormalizationMode.MEAN_STD, FeatureType.ACTION: NormalizationMode.MEAN_STD}
        stats = {"so100.buffer.action": {"mean": [3.0, 7.0], "std": [2.0, 4.0]}}
        before = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)
        after = UnnormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)
        result = DIAGNOSTIC.processor_transform_diagnostic(SimpleNamespace(steps=[before]), SimpleNamespace(steps=[after]), torch, physical_dim=2)
        self.assertTrue(result["state_probe_transform_is_identity"])
        self.assertTrue(result["action_probe_transform_is_identity"])
        self.assertFalse(result["plain_action_statistics_present"])
        self.assertFalse(result["statistics_remapped"])
        self.assertEqual(sorted(after._tensor_stats), ["so100.buffer.action"])
        inventory = result["feature_statistics_inventory"]["postprocessor"]
        self.assertIsNone(inventory["features"]["action"]["effective_statistics_key"])
        self.assertEqual(inventory["statistics"]["so100.buffer.action"]["mean"]["shape"], [2])
        self.assertEqual(inventory["statistics"]["so100.buffer.action"]["mean"]["dtype"], "torch.float32")
        after = UnnormalizerProcessorStep(features=features, norm_map=norm_map, stats={"action": stats["so100.buffer.action"]})
        result = DIAGNOSTIC.processor_transform_diagnostic(SimpleNamespace(steps=[before]), SimpleNamespace(steps=[after]), torch, physical_dim=2)
        self.assertTrue(result["plain_action_statistics_present"])
        self.assertFalse(result["action_probe_transform_is_identity"])
        self.assertFalse(result["physical_action_units_qualified"])

    def test_action_diagnostic_separates_policy_coordinates_from_explicit_physical_units(self):
        import torch
        source = torch.zeros(1, 3, 2)
        candidate = source + torch.tensor([0.25, -0.5])
        result = DIAGNOSTIC.action_quality_diagnostic(source, candidate, torch)
        self.assertIsNone(result["physical"])
        self.assertEqual(len(result["policy_coordinates_by_coordinate"]), 2)
        self.assertEqual(len(result["policy_coordinates_by_horizon"]), 3)
        self.assertFalse(result["normalization_qualified"])
        self.assertFalse(result["acceptance_contract_selected"])
        self.assertFalse(result["model_quality_qualified"])
        self.assertFalse(result["robot_success_qualified"])
        result = DIAGNOSTIC.action_quality_diagnostic(source, candidate, torch, physical_transform=lambda value: value * torch.tensor([2.0, 10.0]) + 7.0, physical_units=["first coordinate units", "second coordinate units"])
        self.assertEqual(result["physical"]["by_coordinate"][0]["max_absolute_error"], 0.5)
        self.assertEqual(result["physical"]["by_coordinate"][1]["max_absolute_error"], 5.0)
        self.assertIn("mixed BF16/FP32", result["reference_forward_precision"])

    def test_action_diagnostic_rejects_missing_units_nonfinite_and_mutating_transforms(self):
        import torch
        source = torch.zeros(1, 2, 2)
        for kwargs in ({"physical_transform": lambda value: value}, {"physical_units": ["units", "units"]}, {"physical_transform": lambda value: value, "physical_units": ["units", " "]}):
            with self.subTest(kwargs=list(kwargs)), self.assertRaises(ValueError):
                DIAGNOSTIC.action_quality_diagnostic(source, source, torch, **kwargs)
        with self.assertRaises(ValueError):
            DIAGNOSTIC.action_quality_diagnostic(source, source + float("nan"), torch)
        rng = torch.random.get_rng_state().clone()
        def mutate(value):
            value.add_(1)
            torch.rand(1)
            return value
        with self.assertRaisesRegex(AssertionError, "mutated its inputs"):
            DIAGNOSTIC.action_quality_diagnostic(source, source, torch, physical_transform=mutate, physical_units=["units", "units"])
        self.assertEqual(source.count_nonzero().item(), 0)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_declared_integer_projection_bias_cast_and_source_preservation(self):
        import copy
        import torch
        source = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=True, dtype=torch.bfloat16))
        with torch.no_grad():
            source[0].weight.copy_(torch.tensor([[0.25, -0.75, 0.5], [1.0, 0.125, -0.5]]))
            source[0].bias.copy_(torch.tensor([0.125, -0.25]))
        before = {name: value.clone() for name, value in source.state_dict().items()}
        target = copy.deepcopy(source)
        records, projections, recipe = FACTORY.install_integer_projections(target, torch)
        value = torch.tensor([[0.25, 1.5, -0.5], [0.0, 0.0, 0.0]], dtype=torch.bfloat16)
        output = target(value)
        check = FACTORY.check_integer_projection(projections["0"], value, output, torch)
        self.assertTrue(check["float_bias_and_output_cast_exact"])
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(projections["0"].integer.weight_kn.dtype, torch.int8)
        self.assertEqual(projections["0"].weight.dtype, torch.bfloat16)
        self.assertEqual(projections["0"].weight.numel(), 0)
        self.assertTrue(all(torch.equal(before[name], value) for name, value in source.state_dict().items()))
        self.assertEqual(records[0]["float_interface_dtype"], "torch.bfloat16")
        self.assertEqual(recipe["accumulator"], "int32")
        self.assertNotEqual(recipe["attention_and_state"], "float32_cpu")
        with self.assertRaisesRegex(ValueError, "original floating interface"):
            target(value.float())

    def test_linear_bound_handles_cancellation_and_one_declared_cast(self):
        import torch
        left = torch.tensor([[1.0, -1.0, 0.5], [0.25, 0.5, -0.125]], dtype=torch.bfloat16)
        weight = torch.tensor([[1.0, 1.0, 0.0], [0.5, -0.75, 0.25]], dtype=torch.bfloat16)
        bias = torch.tensor([0.0, 0.125], dtype=torch.bfloat16)
        result = NUMERICS.bf16_linear_error_diagnostic(left, weight, bias, torch)
        actual = (left.float() @ weight.float().T + bias.float()).bfloat16().double()
        self.assertTrue(torch.all((actual - result["higher_precision_value"]).abs()
                                  <= result["absolute_error_bound_to_fp64_diagnostic"]))
        self.assertEqual(result["higher_precision_value"][0, 0].item(), 0.0)
        self.assertGreater(result["fp32_arithmetic_error_bound"][0, 0].item(), 0.0)
        self.assertEqual(result["operation_bound"], 7)
        self.assertTrue(result["diagnostic_only"])
        self.assertFalse(result["acceptance_contract_selected"])

    def test_linear_diagnostic_rejects_dtype_changes_and_unbounded_overflow(self):
        import torch
        for left, weight in ((torch.ones(1, 2), torch.ones(1, 2, dtype=torch.bfloat16)),
                             (torch.full((1, 2), 1e30, dtype=torch.bfloat16),
                              torch.full((1, 2), 1e30, dtype=torch.bfloat16))):
            with self.subTest(left=left.dtype), self.assertRaises(ValueError):
                NUMERICS.bf16_linear_error_diagnostic(left, weight, None, torch)


class SmolVLABackendArgumentTests(unittest.TestCase):
    def test_invalid_seeds_attribution_and_existing_outputs_fail_early(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = ["--checkpoint-dir", str(root / "policy"), "--backbone-dir", str(root / "backbone"), "--output-dir", str(root / "new")]
            for seeds in (("0", "0"), ("-1",), (str(2**63),)):
                with self.subTest(seeds=seeds), self.assertRaises(SystemExit):
                    DIAGNOSTIC.parse_args(base + ["--seeds", *seeds])
            with self.assertRaises(SystemExit):
                DIAGNOSTIC.parse_args(base + ["--seeds", "0", "--fixture-kind", "dataset"])
            with self.assertRaises(SystemExit):
                DIAGNOSTIC.parse_args(base + ["--seeds", "0", "1", "--input-npz", str(root / "fixture.npz")])
            for attributes in ([], ["--fixture-kind", "synthetic"], ["--fixture-kind", "synthetic", "--input-source", " "]):
                with self.subTest(attributes=attributes), self.assertRaises(SystemExit):
                    DIAGNOSTIC.parse_args(base + ["--seeds", "0", "--input-npz", str(root / "fixture.npz"), *attributes])
            (root / "new").mkdir()
            with mock.patch("merlin.common.paths.artifacts_dir", return_value=root), self.assertRaises(FileExistsError):
                DIAGNOSTIC.parse_args(base + ["--seeds", "0"])
            alias = root / "alias"
            alias.symlink_to(root / "new", target_is_directory=True)
            with mock.patch("merlin.common.paths.artifacts_dir", return_value=root), self.assertRaises(FileExistsError):
                DIAGNOSTIC.parse_args(base[:4] + ["--output-dir", str(alias), "--seeds", "0"])

    def test_supplied_fixtures_require_truthful_explicit_attribution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = ["--checkpoint-dir", str(root / "policy"), "--backbone-dir", str(root / "backbone"), "--output-dir", str(root / "new"), "--seeds", "0", "--input-npz", str(root / "fixture.npz")]
            for kind in ("dataset", "synthetic"):
                with self.subTest(kind=kind), mock.patch("merlin.common.paths.artifacts_dir", return_value=root):
                    args = DIAGNOSTIC.parse_args(base + ["--fixture-kind", kind, "--input-source", "Attributed supplied development fixture"])
                self.assertEqual(args.fixture_kind, kind)
                self.assertEqual(args.input_source, "Attributed supplied development fixture")


class SmolVLABackendCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        cls.np, cls.torch = np, torch
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.original_threads)

    def fixture(self, *, mutate_cache=False, integer_linears=False):
        torch = self.torch

        class Backbone(torch.nn.Module):
            def forward(self, **kwargs):
                embeddings = kwargs["inputs_embeds"][0]
                cache = {index: {"key_states": embeddings.clone(), "value_states": embeddings.clone()} for index in range(2)}
                return (embeddings, None), cache

        class Source(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(num_steps=10, num_vlm_layers=2, chunk_size=50, action_feature=SimpleNamespace(shape=[2]))
                self.vlm_with_expert = Backbone()
                self.register_buffer("constant", torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) / 16)
                if integer_linears:
                    self.prefix_projection = torch.nn.Linear(4, 4, dtype=torch.bfloat16)
                    self.flow_projection = torch.nn.Linear(4, 4, dtype=torch.float32)

            def embed_prefix(self, images, masks, tokens, language_mask, state):
                outputs = [torch.nn.functional.scaled_dot_product_attention(self.constant + image.mean(), self.constant, self.constant) for image in images]
                embedding = sum(outputs).reshape(1, 4, 4).bfloat16()
                if integer_linears:
                    embedding = self.prefix_projection(embedding)
                return embedding, torch.ones(1, 4, dtype=torch.bool), torch.zeros(1, 4, dtype=torch.bool)

            def denoise_step(self, *, x_t, prefix_pad_masks, past_key_values, timestep):
                if mutate_cache:
                    past_key_values[0]["key_states"].add_(1)
                flow = self.flow_projection(x_t) if integer_linears else x_t
                return flow * 0.2 + timestep[:, None, None] + past_key_values[0]["value_states"].float().mean()

            def sample_actions(self, images, masks, tokens, language_mask, state, noise):
                embedding, padding, _ = self.embed_prefix(images, masks, tokens, language_mask, state)
                _, cache = self.vlm_with_expert.forward(inputs_embeds=[embedding, None], fill_kv_cache=True)
                value = noise
                for index in range(10):
                    timestep = torch.tensor(1.0 + index * (-1.0 / 10)).expand(noise.shape[0])
                    velocity = self.denoise_step(x_t=value, prefix_pad_masks=padding, past_key_values=cache, timestep=timestep)
                    value = value + (-1.0 / 10) * velocity
                return value

        config = SimpleNamespace(resize_imgs_with_padding=(2, 2), tokenizer_max_length=3, max_state_dim=4, max_action_dim=4, chunk_size=50, robot_state_feature=SimpleNamespace(shape=[2]))
        return Source().eval(), DIAGNOSTIC.factory.fixture_inputs(config, 3, seed=7)

    def test_complete_capture_dispatch_noise_cache_and_repeatability(self):
        torch, np = self.torch, self.np
        model, inputs = self.fixture()
        state_before = DIAGNOSTIC.state_identity(model, torch)
        flags_before = DIAGNOSTIC.backend_flags(torch)
        rng_before = torch.random.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            runs = {}
            for backend in ("default", "math"):
                first, metadata = DIAGNOSTIC.run_source_policy(model, inputs, backend, 7, 0, directory, np, torch, profile=True)
                second, _ = DIAGNOSTIC.run_source_policy(model, inputs, backend, 7, 1, directory, np, torch)
                self.assertTrue(all(torch.equal(first[name], second[name]) for name in first))
                self.assertEqual(first["final_actions"].shape, (1, 50, 2))
                self.assertEqual(len(metadata["prefix_cache_checks"]), 10)
                self.assertTrue(all(item["before"] == item["after"] for item in metadata["prefix_cache_checks"]))
                self.assertEqual(metadata["captures"]["step_00_input_state"], metadata["input_bindings"]["noise"])
                self.assertEqual(metadata["attention_profile"]["native_sdpa_calls_by_stage"], dict(zip(DIAGNOSTIC.SCOPES, (3, 0, 0))))
                self.assertTrue(metadata["temporary_wrappers_restored"])
                with np.load(metadata["output_file"]["path"], allow_pickle=False) as saved:
                    self.assertEqual(saved["prefix_key_cache"].dtype, np.uint16)
                    for index in range(10):
                        np.testing.assert_array_equal(saved[f"step_{index:02d}_output_state"], saved[f"step_{index:02d}_input_state"] + (-1.0 / 10) * saved[f"step_{index:02d}_velocity"])
                runs[backend] = metadata
            self.assertIn("aten::_scaled_dot_product_flash_attention_for_cpu", runs["default"]["attention_profile"]["operators"])
            self.assertIn("aten::_scaled_dot_product_attention_math", runs["math"]["attention_profile"]["operators"])
        self.assertEqual(DIAGNOSTIC.state_identity(model, torch), state_before)
        self.assertEqual(DIAGNOSTIC.backend_flags(torch), flags_before)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))

    def test_cache_failure_restores_context_hooks_and_rng(self):
        torch = self.torch
        model, inputs = self.fixture(mutate_cache=True)
        flags_before = DIAGNOSTIC.backend_flags(torch)
        rng_before = torch.random.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(AssertionError, "mutated the prefix cache"):
            DIAGNOSTIC.run_source_policy(model, inputs, "math", 7, 0, directory, self.np, torch)
        self.assertEqual(DIAGNOSTIC.backend_flags(torch), flags_before)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
        self.assertNotIn("embed_prefix", model.__dict__)
        self.assertNotIn("denoise_step", model.__dict__)
        self.assertNotIn("forward", model.vlm_with_expert.__dict__)

    def test_velocity_replay_separates_prefix_from_state_feedback(self):
        torch = self.torch
        model, inputs = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            captures, _ = DIAGNOSTIC.run_source_policy(model, inputs, "default", 7, 0,
                                                       directory, self.np, torch)
        original = {name: DIAGNOSTIC.tensor_identity(value, torch) for name, value in captures.items()}
        observed_states = []

        def candidate(**kwargs):
            observed_states.append(kwargs["x_t"].clone())
            return model.denoise_step(**kwargs) + 0.125

        _, shared = DIAGNOSTIC.compare_identical_state_velocities(
            model.denoise_step, candidate, captures, torch, expected_steps=10)
        self.assertTrue(shared["identical_prefix"])
        self.assertEqual(shared["steps"], 10)
        for index, value in enumerate(observed_states):
            self.assertTrue(torch.equal(value, captures[f"step_{index:02d}_input_state"]))
        alternate = (captures["prefix_pad_mask"], captures["prefix_key_cache"],
                     captures["prefix_value_cache"] + 0.25)
        _, conditioned = DIAGNOSTIC.compare_identical_state_velocities(
            model.denoise_step, model.denoise_step, captures, torch,
            expected_steps=10, candidate_prefix=alternate)
        self.assertFalse(conditioned["identical_prefix"])
        self.assertTrue(all(row["comparison"]["max_absolute_error"] > 0.2
                            for row in conditioned["comparisons"]))
        self.assertEqual(original, {name: DIAGNOSTIC.tensor_identity(value, torch)
                                    for name, value in captures.items()})
        self.assertFalse(shared["acceptance_contract_selected"])

    def test_velocity_replay_refuses_incomplete_or_mutating_candidates(self):
        torch = self.torch
        model, inputs = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            captures, _ = DIAGNOSTIC.run_source_policy(model, inputs, "default", 7, 0,
                                                       directory, self.np, torch)
        incomplete = dict(captures)
        del incomplete["step_09_input_state"]
        with self.assertRaisesRegex(ValueError, "complete contiguous"):
            DIAGNOSTIC.compare_identical_state_velocities(
                model.denoise_step, model.denoise_step, incomplete, torch, expected_steps=10)
        rng_before = torch.random.get_rng_state().clone()

        def mutation(**kwargs):
            kwargs["past_key_values"][0]["key_states"].add_(1)
            torch.rand(1)
            return model.denoise_step(**kwargs)

        with self.assertRaisesRegex(AssertionError, "mutated state/mask/cache"):
            DIAGNOSTIC.compare_identical_state_velocities(
                model.denoise_step, mutation, captures, torch, expected_steps=10)
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        altered_mask = (torch.zeros_like(captures["prefix_pad_mask"]),
                        captures["prefix_key_cache"], captures["prefix_value_cache"])
        with self.assertRaisesRegex(ValueError, "routing mask"):
            DIAGNOSTIC.compare_identical_state_velocities(
                model.denoise_step, model.denoise_step, captures, torch,
                expected_steps=10, candidate_prefix=altered_mask)

    def test_complete_integer_policy_keeps_source_and_tracks_stage_placement(self):
        import copy
        torch = self.torch
        source, inputs = self.fixture(integer_linears=True)
        before = DIAGNOSTIC.state_identity(source, torch)
        candidate = copy.deepcopy(source)
        records, projections, recipe = DIAGNOSTIC.factory.install_integer_projections(candidate, torch)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch("merlin.common.paths.artifacts_dir", return_value=root):
                result, receipt = DIAGNOSTIC.factory.run_integer_policy_diagnostic(
                    source, candidate, inputs, records, projections, root / "integer",
                    seed=7, fixture_kind="synthetic", recipe=recipe, torch=torch)
            self.assertEqual(result["denoising_steps"], 10)
            self.assertEqual(result["executed_projection_count"], 2)
            by_name = {record["name"]: record for record in result["projection_inventory"]}
            self.assertEqual(by_name["prefix_projection"]["calls_by_stage"], {"prefix_encode": 1})
            self.assertEqual(by_name["flow_projection"]["calls_by_stage"], {"flow_denoise": 10})
            self.assertTrue(all(record["integer_operator_reference"]["passed"]
                                for record in result["projection_inventory"]))
            self.assertTrue(result["source_mask_routing_exact"])
            for replay in result["identical_state_velocity_replay"].values():
                self.assertEqual(replay["steps"], 10)
                self.assertTrue(replay["reference_replay_exact"])
                self.assertTrue(replay["inputs_cache_rng_unchanged"])
            self.assertFalse(result["model_quality_qualified"])
            self.assertFalse(result["active_gate_changed"])
            self.assertFalse(result["device_execution"])
            self.assertTrue(Path(receipt["path"]).is_file())
            memory = result["memory"]
            self.assertEqual(memory["host_two_model_diagnostic_constants_bytes"],
                             memory["source_constants_bytes"] + memory["target_constants_bytes"])
        self.assertEqual(before, DIAGNOSTIC.state_identity(source, torch))

    def test_stride_zero_hash_and_signed_bf16_distances(self):
        torch = self.torch
        singleton = torch.tensor(0.5).expand(1)
        self.assertEqual(DIAGNOSTIC.tensor_identity(singleton, torch), DIAGNOSTIC.tensor_identity(torch.tensor([0.5]), torch))
        default = torch.tensor([0.0, 1.0], dtype=torch.bfloat16)
        alternate = torch.tensor([-0.0, 1.0078125], dtype=torch.bfloat16)
        result = DIAGNOSTIC.statistics(default, alternate, torch)
        self.assertEqual(result["bf16_representable_step_distance"]["max"], 1)
        self.assertEqual(result["exactly_unequal_count"], 1)
        expanded_default = torch.tensor(1.0, dtype=torch.bfloat16).expand(1)
        expanded_alternate = torch.tensor(1.0078125, dtype=torch.bfloat16).expand(1)
        expanded_result = DIAGNOSTIC.statistics(expanded_default, expanded_alternate, torch)
        self.assertEqual(expanded_result["bf16_representable_step_distance"]["max"], 1)
        self.assertEqual(DIAGNOSTIC.dense_flat_copy(expanded_default, torch).stride(), (1,))
        self.assertEqual((DIAGNOSTIC.RTOL, DIAGNOSTIC.ATOL), (1e-4, 1e-4))


if __name__ == "__main__":
    unittest.main()
