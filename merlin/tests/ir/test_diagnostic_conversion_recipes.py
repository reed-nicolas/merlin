"""Portable target-example guard tests; no frozen tree, model or RTL needed."""
from dataclasses import replace
import importlib.util
import sys
import unittest

from merlin.common.paths import repo_root
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps

_path = repo_root() / "examples/atlas/diagnostic_conversion_recipes.py"
_spec = importlib.util.spec_from_file_location("diagnostic_conversion_example", _path)
_example = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _example
_spec.loader.exec_module(_example)


def fixture(scale=127):
    fp8 = TensorType(_example.TILE, "fp8_e4m3", "raw-bit-pattern")
    bf16 = TensorType(_example.TILE, "bf16", "mxu0-e4m3-ordered-bf16-finite-v1")
    packed = TensorType(_example.TILE, "fp8_e4m3", "fp8-e4m3-selected-pack-all-scales-v1")
    return KernelRequest((SemanticNode("a", "input", (), fp8, effect="input"),
        SemanticNode("w", "input", (), fp8, effect="input"),
        SemanticNode("value", "matmul", ("a", "w"), bf16, index_maps=_example.RESET_MAPS),
        SemanticNode("pack", "pack_fp8", ("value",), packed,
            attrs=(("input_domain", "bf16-finite-all-codes-v1"), ("scale_code", scale)),
            index_maps=pointwise_identity_maps(2, 1))),
        ("value", "pack"), ("tensor_bf16", "tensor_fp8"),
        (("a", "tensor_fp8"), ("w", "tensor_fp8")), _example.TARGET_IDENTITY, "diagnostic")


class ConversionRecipeExampleTest(unittest.TestCase):
    def test_all_scale_codes_and_both_conversion_engines(self):
        for scale in range(256):
            request = fixture(scale)
            proof = _example.prove_finite(request, "value")
            for engine in (0, 1):
                recipe = _example.make_recipe(request, "pack", engine)
                certificate = _example.validator({"pack": proof})(request, recipe)
                self.assertEqual(certificate.recipe_digest, recipe.digest())
                self.assertEqual(dict(recipe.nodes[-1].attrs),
                                 {"engine": engine, "inverse_scale": 254 - min(scale, 254)})
                self.assertEqual(len(recipe.nodes), 6 if scale <= 246 else 2)
                if scale <= 246:
                    self.assertEqual(dict(recipe.nodes[1].attrs)["bits"], ((scale + 8) << 7) | 96)

    def test_unknown_or_nan_capable_and_stale_facts_fail_closed(self):
        request = fixture()
        proof = _example.prove_finite(request, "value")
        recipe = _example.make_recipe(request, "pack", 0)
        for bad in (None, replace(proof, may_nan=True), replace(proof, may_inf=True),
                    replace(proof, request_digest="stale"), replace(proof, value="a"),
                    replace(proof, shape=(16, 16)), replace(proof, layout="other")):
            with self.assertRaises(ValueError):
                _example.validator({"pack": bad})(request, recipe)
        for op in ("matmul_accumulate", "mul_bf16", "exp_bf16", "recip_bf16"):
            changed = replace(request, nodes=(*request.nodes[:2], replace(request.nodes[2], op=op), request.nodes[3]))
            self.assertIsNone(_example.prove_finite(changed, "value"))

    def test_semantic_domain_target_scale_and_recipe_are_checked(self):
        request = fixture()
        for scale in (-1, 256, True):
            with self.assertRaises(ValueError):
                _example.make_recipe(fixture(scale), "pack", 0)
        for engine in (-1, 2, True):
            with self.assertRaises(ValueError):
                _example.make_recipe(request, "pack", engine)
        with self.assertRaises(ValueError):
            _example.make_recipe(replace(request, target_identity="other"), "pack", 0)
        recipe = _example.make_recipe(request, "pack", 0)
        bad = replace(recipe, nodes=(*recipe.nodes[:-1], replace(recipe.nodes[-1],
            attrs=(("engine", 0), ("inverse_scale", 126)))))
        with self.assertRaises(ValueError):
            _example.validator({"pack": _example.prove_finite(request, "value")})(request, bad)


if __name__ == "__main__":
    unittest.main()
