"""The diagnostic adapter requires explicit, source-bound target admission."""
from dataclasses import replace
import unittest

from merlin.semantic_compiler.diagnostic_recipes import (
    DiagnosticRecipe, RecipeCertificate, build_diagnostic_recipe_program,
)
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType


def fixture():
    typed = TensorType((2, 2), "i8", "exact-bits")
    request = KernelRequest((SemanticNode("input", "input", (), typed, effect="input"),
                             SemanticNode("result", "compute", ("input",), typed)),
                            ("result",), ("memory",), (("input", "memory"),),
                            "explicit-test-target", lowering_policy="diagnostic")
    recipe = DiagnosticRecipe("result", (SemanticNode("alternative", "other_compute", ("input",), typed),),
                              "alternative", "test-only-explicit-contract")
    return request, recipe


def admission(request, recipe):
    return RecipeCertificate(request.digest(), recipe.digest(), request.target_identity, "test-only-validator")


class DiagnosticRecipeTest(unittest.TestCase):
    def test_preserve_roots_and_boundaries_and_bind_added_rule(self):
        request, recipe = fixture()
        expanded, program, certificates = build_diagnostic_recipe_program(
            request, (), (recipe,), admit=admission)
        self.assertEqual(expanded.nodes[:len(request.nodes)], request.nodes)
        self.assertEqual(expanded.outputs, request.outputs)
        self.assertEqual(expanded.input_storages, request.input_storages)
        self.assertNotEqual(expanded.source_identity, request.source_identity)
        self.assertEqual(len(certificates), 1)
        self.assertEqual(len(program.rewrites), 1)
        rule = program.rewrites[0]
        self.assertEqual(rule.source_node, "result")
        self.assertNotIn("?", rule.lhs + rule.rhs)
        self.assertEqual(program.roots, (1,))

    def test_target_validator_must_accept_before_union(self):
        request, recipe = fixture()
        def refuse(request, recipe):
            raise ValueError("unproved domain")
        with self.assertRaisesRegex(ValueError, "unproved domain"):
            build_diagnostic_recipe_program(request, (), (recipe,), admit=refuse)
        for certificate in (replace(admission(request, recipe), request_digest="stale"),
                            replace(admission(request, recipe), recipe_digest="stale"),
                            replace(admission(request, recipe), target_identity="other"),
                            replace(admission(request, recipe), proof_identity=""),
                            replace(admission(request, recipe), proof_identity=True)):
            with self.assertRaisesRegex(ValueError, "certificate"):
                build_diagnostic_recipe_program(request, (), (recipe,),
                                                admit=lambda request, recipe: certificate)

    def test_production_and_effectful_or_untyped_alternatives_refused(self):
        request, recipe = fixture()
        for policy in ("strict-native", "hybrid"):
            with self.assertRaisesRegex(ValueError, "diagnostic"):
                build_diagnostic_recipe_program(replace(request, lowering_policy=policy), (), (), admit=admission)
        for changed in (replace(recipe, source_node="input"),
                        replace(recipe, nodes=(replace(recipe.nodes[0], id="input"),), result="input"),
                        replace(recipe, nodes=(replace(recipe.nodes[0], inputs=("missing",)),)),
                        replace(recipe, nodes=(replace(recipe.nodes[0], type=TensorType((4, 1), "i8", "exact-bits")),)),
                        replace(recipe, result="input")):
            with self.assertRaises(ValueError):
                build_diagnostic_recipe_program(request, (), (changed,), admit=admission)


if __name__ == "__main__":
    unittest.main()
