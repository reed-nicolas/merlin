"""Kernel-mining facts that used to be Python literals are DATA, so a new ISA family or expert corpus is a
data edit, not a core one: class marker tables live in core and target-specific tables are
explicitly selected from the target owner; expert-corpus locations and source aliases live in
the corpus registry ``merlin/contract/corpora.yaml`` (read only by ``merlin.targetgen.corpora``)."""

import json

import pytest

from merlin.common.paths import merlin_dir, repo_root
from merlin.kernels import build_asm as B
from merlin.kernels import framework_contracts as FC
from merlin.kernels import markers as M
from merlin.targetgen import corpora as C

pytestmark = pytest.mark.target("gemmini")


def _clear_marker_caches():
    FC._load_builtin_feature_contract.cache_clear()
    M._target_families.cache_clear()
    M._compiled_for_family.cache_clear()


@pytest.fixture
def feature_dir(tmp_path, monkeypatch):
    """A private feature_extraction dir, swapped in for the shipped one; caches rebuilt both ways."""
    monkeypatch.setattr(FC, "_FEATURE_DIR", tmp_path)
    _clear_marker_caches()
    yield tmp_path
    monkeypatch.undo()
    _clear_marker_caches()


def test_every_declared_motif_is_in_the_vocabulary():
    for fam in FC.feature_families():
        assert set(FC.load_feature_contract(fam).get("markers") or {}) <= set(M.MOTIFS), fam


def test_a_new_isa_family_is_one_data_file(feature_dir):
    for f in (FC.feature_families.__globals__["_DIR"] / "feature_extraction").glob("*.yaml"):
        (feature_dir / f.name).write_text(f.read_text())
    (feature_dir / "toyisa.yaml").write_text(
        "family: toyisa\ntargets: [toyisa, Toy_Accel]\nmarkers:\n  accumulator_lifetime: ['toy_mac\\w*']\n"
    )
    assert M.target_family("toy_accel") == "toyisa"
    assert M.target_family("rvv") == "rvv"  # the shipped families still resolve
    fired = M.fired_markers("for (i = 0; i < n; i++) toy_mac_acc(x);", "TOY_ACCEL")
    assert fired == {"accumulator_lifetime": ["toy_mac_acc"], "tiling_blocking": ["for ("]}


def test_target_feature_contract_requires_explicit_selection():
    path = repo_root() / "examples/gemmini/kernel-mining/feature-extraction.yaml"
    assert M.target_family("gemmini") == "generic"
    assert FC.load_feature_contract("gemmini") == {}
    assert "weight_stationary_dataflow" not in M.fired_markers("replace_gemmini_calls()", "exo_schedule")
    with FC.use_feature_contract(path) as selection:
        assert selection["family"] == "gemmini" and len(selection["sha256"]) == 64
        assert M.target_family("gemmini") == "gemmini"
        assert "compute_preloaded" in M.fired_markers("compute_preloaded();", "gemmini")["intrinsic_lowering"]
        assert "weight_stationary_dataflow" in M.fired_markers("replace_gemmini_calls()", "exo_schedule")
    assert M.target_family("gemmini") == "generic"
    assert FC.load_feature_contract("gemmini") == {}


def test_selected_target_mining_index_extract_audit(tmp_path):
    from merlin.kernels.audit import main as audit_main
    from merlin.kernels.cli_extract import main as extract_main
    from merlin.kernels.cli_index import main as index_main

    source = tmp_path / "source"
    kernels = source / "kernels"
    kernels.mkdir(parents=True)
    (kernels / "kernel_abcd.c").write_bytes(
        (merlin_dir() / "tests/data/kernels/autocomp_gemmini_matmul.c").read_bytes()
    )
    selected = repo_root() / "examples/gemmini/kernel-mining/feature-extraction.yaml"
    bare_index = tmp_path / "bare.json"
    selected_index = tmp_path / "selected.json"
    common = ["--source", "autocomp", "--repo", str(source), "--target", "gemmini"]
    with pytest.raises(SystemExit, match="requires --target or an explicit --framework-contract"):
        index_main(["--source", "autocomp", "--repo", str(source), "--out", str(tmp_path / "refused.json")])
    assert index_main([*common, "--out", str(bare_index)]) == 0
    assert index_main([*common, "--feature-contract", str(selected), "--out", str(selected_index)]) == 0
    bare = json.loads(bare_index.read_text())
    chosen = json.loads(selected_index.read_text())
    assert bare["count"] == chosen["count"] == 1
    assert "dispatch_metrics" not in bare["records"][0]["features"]
    assert chosen["records"][0]["features"]["dispatch_metrics"]["n_dispatches"] > 0
    assert chosen["feature_contract"]["sha256"]
    caller = repo_root() / "examples/gemmini/kernel-mining/autocomp-framework.yaml"
    default_index = tmp_path / "declared-default.json"
    assert (
        index_main(
            [
                "--source",
                "autocomp",
                "--repo",
                str(source),
                "--framework-contract",
                str(caller),
                "--feature-contract",
                str(selected),
                "--out",
                str(default_index),
            ]
        )
        == 0
    )
    declared = json.loads(default_index.read_text())
    assert declared["target"] == "gemmini" and declared["framework_contract"]["sha256"]
    outputs = [
        "--inputs",
        str(selected_index),
        "--out",
        str(tmp_path / "candidates.yaml"),
        "--policies",
        str(tmp_path / "policies.yaml"),
    ]
    with pytest.raises(ValueError, match="same explicitly selected"):
        extract_main(outputs)
    assert extract_main([*outputs, "--feature-contract", str(selected)]) == 0
    assert (
        audit_main(
            ["--inputs", str(selected_index), "--feature-contract", str(selected), "--out", str(tmp_path / "audit.md")]
        )
        == 0
    )


def test_selected_caller_contract_reaches_dossier_and_trace():
    from merlin.kernels.dossier import build_dossier
    from merlin.kernels.ingest.exo import _detect_target
    from merlin.kernels.trace import expert_steps_from_contract
    from merlin.kernels.types import NormalizedKernel

    caller = repo_root() / "examples/gemmini/kernel-mining/autocomp-framework.yaml"
    feature = repo_root() / "examples/gemmini/kernel-mining/feature-extraction.yaml"
    kernel = NormalizedKernel(
        source="autocomp", target="gemmini", path="k.c", op="matmul", dtype="i8", raw_text="compute_preloaded();"
    )
    assert FC.load_contract("autocomp") == {}
    assert _detect_target("from platforms.gemmini import *", None) == "unknown"
    with FC.use_framework_contract(caller), FC.use_feature_contract(feature):
        assert build_dossier(kernel).framework_contract["isa"] == "gemmini"
        steps = expert_steps_from_contract("autocomp")
        assert steps and all(s.entry.startswith(str(caller)) for s in steps)
        assert _detect_target("from platforms.gemmini import *", None) == "gemmini"
    assert FC.load_contract("autocomp") == {}


def test_a_target_claimed_by_two_families_is_refused(feature_dir):
    (feature_dir / "a.yaml").write_text("targets: [dup]\n")
    (feature_dir / "b.yaml").write_text("targets: [DUP]\n")
    with pytest.raises(ValueError, match="claimed by two"):
        M.target_family("dup")


def test_a_misspelled_motif_is_refused_not_silently_dropped(feature_dir):
    (feature_dir / "x.yaml").write_text("targets: [x]\nmarkers:\n  pakced_rhs: ['a']\n")
    with pytest.raises(ValueError, match="unknown motif"):
        M.markers_for("x")


def test_source_aliases_name_exactly_one_corpus():
    seen = {}
    for name, spec in C.kernel_corpora().items():
        for alias in spec.get("sources") or []:
            assert seen.setdefault(alias.lower(), name) == name, alias
            assert C.kernel_corpus_for_source(alias.upper()) == name
    assert C.kernel_corpus_for_source("") is None
    assert C.kernel_corpus_for_source("no_such_source") is None


def test_canonical_capsule_corpus_is_first():
    assert C.capsule_corpus_roots()[0] == merlin_dir() / "contract" / "capsules"


def test_a_new_corpus_is_a_registry_entry(tmp_path, monkeypatch):
    reg = {
        "capsule_corpora": [],
        "kernel_corpora": {
            "toyblas": {
                "sources": ["toyblas", "toy-blas"],
                "layout": "single_tu",
                "checkout": "ToyBLAS",
                "include_subdirs": ["", "inc"],
            }
        },
    }
    monkeypatch.setattr(C, "_registry", lambda: reg)
    monkeypatch.setenv(C.kernel_corpus_env("toyblas"), str(tmp_path))
    assert C.kernel_corpus_env("toyblas") == "MERLIN_TOYBLAS_REPO"
    assert B.framework_include_roots("Toy-BLAS") == [tmp_path, tmp_path / "inc"]
    assert B.benchmark_source() is None  # no standalone-benchmark corpus declared
    assert B.benchmarks_dir() is None
    assert B.corpus_root("nope") is None


def test_env_var_wins_over_every_other_location(tmp_path, monkeypatch):
    name = B.benchmark_source()
    assert name is not None
    monkeypatch.setenv(C.kernel_corpus_env(name), str(tmp_path))
    assert B.corpus_root(name) == tmp_path
    spec = C.kernel_corpora()[name]
    assert B.benchmarks_dir() == tmp_path / spec["benchmarks"]
