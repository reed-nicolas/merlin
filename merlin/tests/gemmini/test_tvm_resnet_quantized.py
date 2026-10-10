"""Direct unittest entry point for the explicit ResNet development recipe."""

import importlib.util
import sys
import unittest

import numpy as np
from merlin.common.paths import repo_root


MODULE_PATH = repo_root() / "examples/gemmini/comparisons/tvm/resnet_quantized.py"
sys.path.insert(0, str(MODULE_PATH.parent))
spec = importlib.util.spec_from_file_location("resnet_quantized", MODULE_PATH)
recipe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = recipe
spec.loader.exec_module(recipe)


class IntegerPolicies(unittest.TestCase):
    def test_signed_rounding_and_shifts(self):
        values = np.arange(-9, 10, dtype=np.int64)
        np.testing.assert_array_equal(recipe.round_divide(values, 2), [-5, -4, -4, -3, -3, -2, -2, -1, -1, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5])
        np.testing.assert_array_equal(recipe.rescale(values, -1), recipe.round_divide(values, 2))
        np.testing.assert_array_equal(recipe.rescale(values, 2), values * 4)
        with self.assertRaises(ValueError):
            recipe.rescale(values, 31)
        with self.assertRaises(ValueError):
            recipe.round_divide(np.array([np.iinfo(np.int64).min]), 2)
        with self.assertRaises(ValueError):
            recipe.rescale(np.array([np.iinfo(np.int64).min]), 1)
        with self.assertRaises(ValueError):
            recipe.round_divide(values, 1.5)
        with self.assertRaises(ValueError):
            recipe.rescale(values.astype(np.float64), 1)
        np.testing.assert_array_equal(recipe.round_divide(values, 3),
                                      [-3, -3, -2, -2, -2, -1, -1, -1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3])
        for operation in (lambda: recipe.round_divide(np.array([np.iinfo(np.int64).max]), 2),
                          lambda: recipe.rescale(np.array([np.iinfo(np.int64).max]), 1),
                          lambda: recipe.rescale(values, np.iinfo(np.int64).min)):
            with self.assertRaises(ValueError):
                operation()

    def test_symmetric_quantization(self):
        np.testing.assert_array_equal(recipe.quantize(np.array([-200., -1.5, -.5, 0., .5, 1.5, 200.]), 0), [-127, -2, -1, 0, 1, 2, 127])
        self.assertEqual(recipe.exponent_for(127.), 0)
        self.assertEqual(recipe.exponent_for(128.), 1)
        with self.assertRaises(ValueError):
            recipe.quantize(np.array([np.nan]), 0)

    def test_contraction_against_direct_int64(self):
        x = np.arange(-18, 18, dtype=np.int8).reshape(1, 2, 3, 6)
        w = np.arange(-16, 16, dtype=np.int8).reshape(2, 2, 2, 4)
        attrs = {"stride": [1, 2], "padding": [1, 1], "dilation": [1, 1]}
        actual = recipe.integer_contraction(x, w, attrs)
        padded = np.pad(x.astype(np.int64), ((0, 0), (0, 0), (1, 1), (1, 1)))
        expected = np.zeros((1, 2, 4, 3), dtype=np.int64)
        for o in range(2):
            for y in range(4):
                for z in range(3):
                    expected[0, o, y, z] = (padded[0, :, y:y + 2, 2 * z:2 * z + 4] * w[o].astype(np.int64)).sum()
        np.testing.assert_array_equal(actual, expected)
        a, b = x.reshape(1, -1), w.reshape(2, -1)
        np.testing.assert_array_equal(recipe.integer_contraction(a[:, :16], b[:, :16]), a[:, :16].astype(np.int64) @ b[:, :16].astype(np.int64).T)

    def test_bias_aware_scaling_independent_power_of_two_witness(self):
        # Tiny folded weights can coexist with a substantial folded bias.
        raw_weight = np.array([[2. ** -24, -2. ** -24], [-2. ** -24, 2. ** -24], [0., 0.], [1., -1.]], dtype=np.float32)
        raw_bias = np.array([4., -4., 64., 2.], dtype=np.float32)
        original_weight, original_bias = raw_weight.copy(), raw_bias.copy()
        weight, exponents, bias, changes = recipe.quantize_parameters(raw_weight, raw_bias, -1)
        # At e=-28 the first two biases have magnitude 2**31; e=-27 is
        # the first admissible scale, with exact qweights +/-8 and bias +/-2**30.
        np.testing.assert_array_equal(exponents, [-27, -27, -23, -6])
        np.testing.assert_array_equal(weight, [[8, -8], [-8, 8], [0, 0], [64, -64]])
        np.testing.assert_array_equal(bias, [2 ** 30, -(2 ** 30), 2 ** 30, 256])
        self.assertEqual([change["channel"] for change in changes], [0, 1, 2])
        for channel, change in enumerate(changes):
            self.assertEqual(change["raw_weight_sha256"], recipe.array_digest(raw_weight[channel]))
            self.assertGreater(change["initial_bound_with_bias"], recipe.LIMIT)
            self.assertEqual(change["bound_with_bias"], 127 * sum(abs(int(v)) for v in weight[channel]) + abs(int(bias[channel])))
            self.assertLessEqual(change["bound_with_bias"], recipe.LIMIT)
        for channel in (0, 1):
            self.assertEqual(changes[channel]["initial_exponent"], -30)
            self.assertEqual(changes[channel]["initial_bias"], (1 if channel == 0 else -1) * 2 ** 33)
            self.assertEqual(changes[channel]["initial_accumulation_bound"], 127 * 128)
        # Independent Python integer contraction+bias witness at both endpoints.
        for channel in range(4):
            for inputs in ((127, -127), (-127, 127)):
                result = sum(x * int(w) for x, w in zip(inputs, weight[channel])) + int(bias[channel])
                self.assertLessEqual(abs(result), recipe.LIMIT)
        np.testing.assert_array_equal(raw_weight, original_weight)
        np.testing.assert_array_equal(raw_bias, original_bias)

    def test_bias_aware_scaling_exact_int32_boundary(self):
        for sign in (-1, 1):
            for extra, expected_exponent in ((0, -24), (1, -23)):
                with self.subTest(sign=sign, extra=extra):
                    raw_bias = np.array([sign * (recipe.LIMIT + extra) * 2. ** -24])
                    weight, exponents, bias, changes = recipe.quantize_parameters(np.zeros((1, 1)), raw_bias, 0)
                    self.assertEqual(int(exponents[0]), expected_exponent)
                    self.assertEqual(int(bias[0]), sign * (recipe.LIMIT if extra == 0 else 2 ** 30))
                    self.assertEqual(len(changes), extra)
        # A representable bias also needs headroom for the contraction itself.
        weight, exponents, bias, changes = recipe.quantize_parameters(np.array([[64.]]), np.array([float(recipe.LIMIT - 8127)]), 0)
        self.assertEqual(int(exponents[0]), 1)
        self.assertEqual(changes[0]["initial_bound_with_bias"], recipe.LIMIT + 1)
        self.assertEqual(changes[0]["initial_accumulation_bound"], 8128)
        self.assertEqual(int(weight[0, 0]), 32)

    def test_bias_aware_scaling_retains_fail_closed_limits(self):
        cases = [
            (np.array([[127. * 2. ** 30]]), np.array([float(recipe.LIMIT) * 2. ** 60]), 30),
            (np.zeros((1, 131072)), np.zeros(1), 0),
            (np.array([[2. ** -40]]), np.zeros(1), 0),
            (np.array([[np.nan]]), np.zeros(1), 0),
            (np.zeros((1, 1)), np.array([np.inf]), 0),
            (np.zeros((1, 1)), np.zeros(1), 31),
        ]
        for weight, bias, exponent in cases:
            with self.subTest(shape=weight.shape, exponent=exponent), self.assertRaises(ValueError):
                recipe.quantize_parameters(weight, bias, exponent)


class FullResNetDiagnostic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import torchvision
        torch.set_num_threads(2)
        torch.manual_seed(811)
        cls.model = torchvision.models.resnet50(weights=None).cpu().eval()
        rng = np.random.default_rng(815)
        cls.calibration = rng.standard_normal((1, 3, 64, 64), dtype=np.float32)
        cls.evaluation = rng.standard_normal((1, 3, 64, 64), dtype=np.float32)
        cls.original_hash = recipe.state_digest(cls.model)
        cls.plan = recipe.calibrate_resnet50(cls.model, [cls.calibration], calibration_ids=["synthetic-calibration-815"], input_shape=cls.calibration.shape,
                                            checkpoint_declaration="random diagnostic; not pretrained")

    def test_complete_inventory_and_source_preserved(self):
        manifest = self.plan.manifest()
        self.assertFalse(manifest["paper_quality_approved"])
        self.assertEqual(manifest["recipe"], "resnet50-v1.5-symmetric-pow2-development-v2")
        for site in manifest["sites"]:
            if "bound_with_bias" in site:
                self.assertLessEqual(max(site["bound_with_bias"]), recipe.LIMIT)
                self.assertLessEqual(site["contraction_dimension_bound"], recipe.LIMIT)
        self.assertEqual(sum(site.kind == "conv2d" for site in self.plan.sites), 53)
        self.assertEqual(sum(site.kind == "linear" for site in self.plan.sites), 1)
        self.assertEqual(sum(site.kind == "add" for site in self.plan.sites), 16)
        self.assertEqual(self.original_hash, recipe.state_digest(self.model))
        with self.assertRaises(ValueError):
            recipe.reference(self.plan, self.calibration)

    def test_full_semantic_cpu_graph_matches_independent_reference(self):
        import tvm
        mod = recipe.export_relax(self.plan, return_integer_logits=True)
        executable = tvm.relax.build(mod, target="llvm")
        vm = tvm.relax.VirtualMachine(executable, tvm.cpu())
        image = recipe.quantize(self.evaluation, self.plan.input_exponent)
        outputs = vm["main"](tvm.nd.array(image))
        integer_actual, actual = outputs[0].numpy(), outputs[1].numpy()
        integer_expected, expected = recipe.reference(self.plan, self.evaluation, return_integer_logits=True)
        self.assertEqual(integer_actual.dtype, np.int32)
        np.testing.assert_array_equal(integer_actual, integer_expected)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual.shape, (1, 1000))

    def test_all_contractions_reach_device_graph(self):
        _, _, coverage = recipe.prepare_device_graph(self.plan, optimize=False)
        self.assertEqual(coverage["device_invocations"], 54)

    def test_averagepool_uses_rounded_spatial_mean(self):
        _, values = recipe.reference(self.plan, self.evaluation, return_intermediates=True)
        site = next(site for site in self.plan.sites if site.kind == "averagepool")
        data = values[site.inputs[0]]
        self.assertEqual(data.shape[2:], (2, 2))
        self.assertTrue((data >= 0).all())
        expected = ((data.astype(np.int64).sum(axis=(2, 3), keepdims=True) + 2) // 4).astype(np.int8)
        np.testing.assert_array_equal(values[site.name], expected)

    def test_unsafe_modified_plan_refused(self):
        import copy
        modified = copy.copy(self.plan)
        modified.sites = list(self.plan.sites)
        modified.sites[0] = copy.copy(modified.sites[0])
        modified.sites[0].bias = np.full_like(modified.sites[0].bias, recipe.LIMIT)
        with self.assertRaises(ValueError):
            recipe.reference(modified, self.evaluation)
        with self.assertRaises(ValueError):
            recipe.export_relax(modified)

    def test_invalid_calibration_refused(self):
        with self.assertRaises(ValueError):
            recipe.calibrate_resnet50(self.model, [self.calibration, self.calibration], calibration_ids=["one", "two"], input_shape=self.calibration.shape)

    def test_invalid_plan_parameters_and_geometry_refused(self):
        import copy
        def modify_site(kind, update):
            plan = copy.copy(self.plan)
            plan.sites = list(self.plan.sites)
            index = next(i for i, site in enumerate(plan.sites) if site.kind == kind)
            plan.sites[index] = copy.copy(plan.sites[index])
            update(plan.sites[index])
            return plan
        cases = [
            ("conv2d", lambda site: setattr(site, "weight_exponents", np.full_like(site.weight_exponents, np.iinfo(np.int64).min))),
            ("linear", lambda site: setattr(site, "weight_exponents", np.full_like(site.weight_exponents, np.iinfo(np.int64).min))),
            ("conv2d", lambda site: setattr(site, "inputs", site.inputs * 2)),
            ("conv2d", lambda site: setattr(site, "weight", site.weight[:, :2].copy())),
            ("conv2d", lambda site: setattr(site, "shape", (*site.shape[:-1], site.shape[-1] + 1))),
            ("conv2d", lambda site: setattr(site, "attrs", {**site.attrs, "stride": [0, 1]})),
            ("conv2d", lambda site: setattr(site, "exponent", 30)),
            ("relu", lambda site: setattr(site, "exponent", site.exponent + 1)),
            ("add", lambda site: setattr(site, "inputs", site.inputs[:1])),
            ("flatten", lambda site: setattr(site, "shape", (1, 1))),
            ("averagepool", lambda site: setattr(site, "shape", (*site.shape[:2], 2, 1))),
            ("maxpool", lambda site: setattr(site, "attrs", {**site.attrs, "padding": [100, 100]})),
            ("linear", lambda site: setattr(site, "shape", (1, 999))),
            ("linear", lambda site: setattr(site, "exponent", 0)),
        ]
        for kind, update in cases:
            with self.subTest(kind=kind, update=update):
                plan = modify_site(kind, update)
                with self.assertRaises(ValueError):
                    plan.validate()
                with self.assertRaises(ValueError):
                    recipe.reference(plan, self.evaluation)
        self.assertEqual(recipe.state_digest(self.model), self.original_hash)

    def test_modified_calibration_provenance_refused(self):
        import copy
        for hashes in ((), ("bad-hash",), self.plan.calibration_hashes * 2):
            plan = copy.copy(self.plan)
            plan.calibration_hashes = hashes
            with self.subTest(hashes=hashes), self.assertRaises(ValueError):
                plan.validate()
        plan = copy.copy(self.plan)
        plan.provenance = {**plan.provenance, "calibration_ids": []}
        with self.assertRaises(ValueError):
            plan.validate()


if __name__ == "__main__":
    unittest.main()
