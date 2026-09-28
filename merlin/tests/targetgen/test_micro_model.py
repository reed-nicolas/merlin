"""The minimal whole-model inventory is DERIVED, and its shape is the point.

A whole-model capsule small enough to run at the cycle-accurate tier is the only way the end-to-end
claim ever gets checked against hardware. Which layers it must contain is not a matter of taste: a
hand-picked model covers whichever layers its author thought of, and this repo already shipped a corpus
whose only model capsule was a LLaMA decoder -- no tanh, no erf, no convolution, and nothing saying the
evidence was one architecture wide.
"""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest

from merlin.targetgen import boundary as BD
from merlin.targetgen import micro_model as MM


def _acc(fam, dtype="i8"):
    return MM.LayerRequirement(family=fam, dtype=dtype, side=MM.ACCELERATOR)


def _host(fam):
    return MM.LayerRequirement(family=fam, dtype=None, side=MM.HOST)


class TestInterleaving:
    def test_host_work_lands_between_accelerator_work(self):
        # A host PREFIX and a host SUFFIX compose as H->A->H, which the corpus already evidences. The
        # shape it evidences nowhere is A->H->A -- accelerator, host island, accelerator -- and that is
        # exactly the placement decision a whole-model compiler gets wrong.
        order = MM.interleave([_acc("contraction"), _acc("movement")], [_host("normalization")])
        assert [layer.side for layer in order] == [MM.ACCELERATOR, MM.HOST, MM.ACCELERATOR]

    def test_the_sequence_opens_and_closes_on_the_accelerator(self):
        order = MM.interleave(
            [_acc("contraction"), _acc("movement")], [_host("normalization"), _host("reduction"), _host("softmax")]
        )
        assert order[0].side == MM.ACCELERATOR and order[-1].side == MM.ACCELERATOR

    def test_more_host_layers_than_gaps_still_keeps_every_one_an_island(self):
        order = MM.interleave([_acc("contraction"), _acc("movement")], [_host(f"h{i}") for i in range(5)])
        assert order[0].side == MM.ACCELERATOR and order[-1].side == MM.ACCELERATOR
        assert sum(1 for layer in order if layer.side == MM.HOST) == 5

    def test_a_target_with_no_host_families_gets_no_manufactured_seam(self):
        # A target that admits everything the captures contain has no seam. Inventing one would test a
        # boundary that target does not have, which is worse than reporting there is nothing to test.
        order = MM.interleave([_acc("contraction"), _acc("movement")], [])
        assert all(layer.side == MM.ACCELERATOR for layer in order)

    def test_a_target_that_admits_nothing_is_all_host(self):
        order = MM.interleave([], [_host("normalization")])
        assert [layer.side for layer in order] == [MM.HOST]


class TestCompositionShape:
    def test_the_inventory_reports_its_own_composition_in_the_boundary_vocabulary(self):
        # One vocabulary for "how is this composed", shared with the coverage axis, so the model's
        # intended shape and the shape the corpus is measured on can never drift apart.
        s = MM.MicroModelSpec(layers=MM.interleave([_acc("contraction"), _acc("movement")], [_host("normalization")]))
        assert s.composition() == BD.A_H_A

    def test_two_host_segments_between_three_accelerator_segments_is_routing(self):
        s = MM.MicroModelSpec(
            layers=MM.interleave(
                [_acc("contraction"), _acc("elementwise_map"), _acc("movement")],
                [_host("normalization"), _host("reduction")],
            )
        )
        assert s.composition() == BD.ROUTING
        assert BD.A_H_A in BD.patterns_in_sequence(
            [BD.ACCEL if layer.side == MM.ACCELERATOR else BD.HOST for layer in s.layers]
        )


class TestSpellingsComeFromCaptures:
    def test_the_spelling_is_the_one_real_models_use(self, tmp_path):
        # Which concrete op stands for a family is a question about real networks, and the captures
        # answer it. Choosing by taste is how a corpus ends up one architecture wide.
        class _R:
            def __init__(self, op, fam):
                self.op, self._f = op, fam

            def resolved_family(self):
                return self._f

        import merlin.targetgen.model_coverage as mc

        seen = {"a": [_R("matmul", "contraction")] * 3 + [_R("conv", "contraction")]}
        old_load, old_regions = mc.load_module, mc.regions_from_module
        try:
            mc.load_module = lambda p: str(p)
            mc.regions_from_module = lambda m: tuple(seen["a"])
            got = MM.observed_spellings({"a": tmp_path / "m.mlir"})
        finally:
            mc.load_module, mc.regions_from_module = old_load, old_regions
        assert got["contraction"].most_common(1)[0] == ("matmul", 3)

    def test_an_unreadable_capture_is_skipped_not_fatal(self, tmp_path):
        assert MM.observed_spellings({"missing": tmp_path / "nope.mlir"}) == {}


class TestSpecKeepsItsUncertainty:
    def test_an_unresolvable_target_reports_rather_than_inventing_layers(self):
        s = MM.spec("definitely_not_a_target", {})
        assert s.layers == []
        # No tile edge means extents cannot be sized against the target's own geometry, and the spec
        # must say so instead of quietly emitting a plausible number.
        assert s.extent is None or s.tile_edge is not None

    def test_a_dict_round_trips_with_the_composition_named(self):
        s = MM.MicroModelSpec(target="t", layers=[_acc("contraction"), _host("reduction"), _acc("movement")])
        d = s.to_dict()
        assert d["composition"] == BD.A_H_A
        assert d["n_accelerator_layers"] == 2 and d["n_host_layers"] == 1


def test_gemmini_emission_does_not_call_host_operations_accelerator(monkeypatch):
    from merlin.common.paths import repo_root
    from merlin.targetgen import conformance as CF
    from merlin.targetgen.software_spec import load_software_spec

    monkeypatch.setattr(
        CF, "boundaries", lambda _target: SimpleNamespace(tile_edge=16, tile_edge_is_hardware_fact=True)
    )
    monkeypatch.setattr(
        CF,
        "observed",
        lambda _capture, _target: Counter(
            {
                "attention": 1,
                "elementwise_map": 1,
                "movement": 1,
                "normalization": 1,
                "reduction": 1,
            }
        ),
    )
    monkeypatch.setattr(
        MM,
        "observed_spellings",
        lambda _captures: {
            family: Counter({op: 1})
            for family, op in {
                "contraction": "matmul",
                "attention": "attention_full",
                "elementwise_map": "gelu",
                "movement": "transpose",
                "normalization": "rmsnorm",
                "reduction": "reduce_sum",
            }.items()
        },
    )

    sw_path = repo_root() / "examples/gemmini/target/software-spec.yaml"
    spec = MM.spec(
        "gemmini",
        {"capture": "model.mlir"},
        software_spec=load_software_spec(sw_path, "gemmini"),
        capture_dtype="int8",
    )
    source = MM.emit_pytorch(spec)

    for family in ("elementwise_map", "movement", "reduction"):
        assert f"# host: {family}" in source
        assert f"# accelerator: {family}" not in source
        assert any(row["family"] == family for row in spec.unexercised_capabilities)
    assert "Unexercised accelerator capabilities" in source
    assert source.count("# accelerator: contraction") == 2  # both sides of the host seam
    assert spec.composition() == BD.A_H_A


@pytest.mark.parametrize("capture_dtype", ["fp32", "fp16", "bf16"])
def test_standalone_float_capability_remains_accelerator_on_another_target(monkeypatch, capture_dtype):
    from merlin.targetgen import conformance as CF
    from merlin.targetgen import eligibility as EL
    from merlin.targetgen.compute_units import SemanticCapability

    monkeypatch.setattr(
        CF, "boundaries", lambda _target: SimpleNamespace(tile_edge=16, tile_edge_is_hardware_fact=True)
    )
    monkeypatch.setattr(
        CF,
        "admitted",
        lambda _target: {
            "contraction": (capture_dtype,),
            "elementwise_map": (capture_dtype,),
            "reduction": (capture_dtype,),
        },
    )
    monkeypatch.setattr(CF, "admitting_units", lambda _target: {})
    monkeypatch.setattr(
        CF, "observed", lambda _capture, _target: Counter({"elementwise_map": 1, "normalization": 1, "reduction": 1})
    )
    monkeypatch.setattr(
        EL,
        "capability_map_for_target",
        lambda _target: {
            family: SemanticCapability(family=family, dtypes=(capture_dtype,))
            for family in ("contraction", "elementwise_map", "reduction")
        },
    )
    software_spec = {
        "status": "reviewed",
        "numerical_semantics": {"accumulator_dtype": "f32", "readout_dtype": "f32"},
        "operations": [
            {"id": family, "families": [family], "placement": placement, "signature": {"dtypes": [capture_dtype]}}
            for family, placement in (
                ("contraction", "accelerator"),
                ("elementwise_map", "accelerator"),
                ("normalization", "host"),
                ("reduction", "host"),
            )
        ],
    }
    monkeypatch.setattr(
        MM,
        "observed_spellings",
        lambda _captures: {
            family: Counter({op: 1})
            for family, op in {
                "contraction": "matmul",
                "elementwise_map": "gelu",
                "normalization": "rmsnorm",
                "reduction": "reduce_sum",
            }.items()
        },
    )

    spec = MM.spec(
        "synthetic_simt",
        {"capture": "model.mlir"},
        software_spec=software_spec,
        capture_dtype=capture_dtype,
    )
    source = MM.emit_pytorch(spec)

    assert "# accelerator: elementwise_map" in source
    assert "# host: normalization" in source
    assert "# host: reduction" in source
    assert {row["family"] for row in spec.unexercised_capabilities} == {"reduction"}
