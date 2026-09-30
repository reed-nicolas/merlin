"""One discoverable command surface for phase definitions and existing engines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import runner
from .spec import SpecError, load_spec
from .spec import catalog as catalog


def _source(value: str, catalog_path: Path | None) -> Path:
    path = Path(value).expanduser()
    if path.exists():
        return path
    entries = catalog(catalog_path)
    if value not in entries:
        raise SpecError(f"unknown experiment {value!r}; use list or pass a definition path")
    return entries[value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="merlin experiment", description=__doc__)
    parser.add_argument("--catalog", type=Path, help="catalog YAML; paths inside it are relative to that file")
    commands = parser.add_subparsers(dest="verb", required=True)
    commands.add_parser("list", help="list the versioned experiment catalog")
    stored = commands.add_parser("runs", help="discover stored phase orchestrations (read-only)")
    stored.add_argument("--root", type=Path, help="run root; defaults to the configured out/runs")
    stored.add_argument("--target", help="filter by exact target identity")
    stored.add_argument("--experiment", help="filter by exact experiment identity")
    for verb in ("inspect", "preflight", "run"):
        child = commands.add_parser(verb)
        child.add_argument("spec", help="definition path or catalog id")
        child.add_argument("--phase", choices=("0", "1", "2", "all"), default="all")
        child.add_argument("--run-dir", type=Path, help="explicit output; otherwise use the configured run root")
        child.add_argument("--corpus-seal", type=Path, help="reviewed Phase 0 release seal for Phase 1")
        child.add_argument("--bundle-manifest", type=Path, help="reviewed replacement Phase 1 input bundle")
        child.add_argument(
            "--phase0-conformance-spec", type=Path, help="new reviewed Phase 0 requirement (select with synth profile)"
        )
        child.add_argument(
            "--phase0-synth-profile", type=Path, help="new synthesized Phase 0 profile (select with requirement)"
        )
        child.add_argument(
            "--phase0-hidden-profile",
            type=Path,
            help="operator-owned private Phase 0 profile; never put it in examples",
        )
        child.add_argument("--phase0-rtl-facts", type=Path, help="select exact extracted facts for a new Phase 0 run")
        child.add_argument(
            "--phase0-capability-contract",
            type=Path,
            help="select exact same-target capability contract for a new Phase 0 run",
        )
        child.add_argument(
            "--phase0-evidence-mode",
            choices=("diagnostic", "verified"),
            help="diagnostic preserves unknowns; verified refuses unresolved required evidence",
        )
        child.add_argument(
            "--phase0-m2m-root", type=Path, help="explicit Model2MLIR source root for diagnostic capture"
        )
        child.add_argument(
            "--phase0-m2m-python", type=Path, help="explicit Model2MLIR venv Python for diagnostic capture"
        )
    commands.add_parser("status").add_argument("run_dir", type=Path)
    commands.add_parser("lineage", help="read frozen phase inputs and handoffs without executing engines").add_argument(
        "run_dir", type=Path
    )
    child = commands.add_parser("resume")
    child.add_argument("run_dir", type=Path)
    child.add_argument("--checkpoint", type=Path, help="sealed native checkpoint for a new model_portfolio segment")
    corpus = commands.add_parser(
        "corpus", help="derive capsule groups, inspect run coverage, or prepare and review a corpus release"
    )
    operations = corpus.add_subparsers(dest="operation", required=True)
    derive = operations.add_parser("derive", help="deterministic requirements and complete census; no agent execution")
    derive.add_argument("definition", help="explicit experiment definition or catalog id")
    derive.add_argument("--application-capture", action="append", required=True, metavar="LABEL=PATH")
    derive.add_argument(
        "--application-capture-selection", action="append", default=[], metavar="LABEL=PATH@SHA256",
        help="pre-execution selection for each selected capture; omitted legacy captures remain diagnostic",
    )
    derive.add_argument(
        "--native-qualification",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="optional exact generated native-host qualification receipt; never grants RVV support",
    )
    derive.add_argument(
        "--rtl-facts", type=Path, required=True, help="exact extraction artifact; never re-extract implicitly"
    )
    derive.add_argument("--output", type=Path, required=True, help="new immutable artifact root")
    capture = operations.add_parser("capture", help="preselect and issue one fresh sealed CPU capture")
    capture_ops = capture.add_subparsers(dest="capture_operation", required=True)
    select_capture = capture_ops.add_parser("select", help="freeze source/runtime/tool bytes before capture")
    select_capture.add_argument("--m2m-root", type=Path, required=True)
    select_capture.add_argument("--workload-root", type=Path, required=True)
    select_capture.add_argument("--venv", type=Path, required=True)
    select_capture.add_argument("--dtype", choices=("fp32", "int8"), default="fp32")
    select_capture.add_argument("--recipe", type=Path)
    select_capture.add_argument("--run-dir", type=Path, required=True)
    select_capture.add_argument("--output", type=Path, required=True, help="fresh owner-only selection directory")
    select_capture.add_argument("--bwrap", type=Path)
    issue_capture = capture_ops.add_parser("issue", help="capture only from an exact preselected identity")
    issue_capture.add_argument("--selection", type=Path, required=True)
    issue_capture.add_argument("--expected-sha256", required=True)
    from merlin.targetgen import group_capsules

    groups = operations.add_parser(
        "groups",
        help="derive capsules from captured compute groups (does not seal or approve)",
        description=group_capsules.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group_capsules.configure_parser(groups)
    compare = operations.add_parser("compare", help="compare public recipes from explicit definitions; prints JSON")
    compare.add_argument("definitions", nargs="+", help="definition paths or catalog ids")
    coverage = operations.add_parser("coverage", help="inspect public coverage of one completed Phase 0 run")
    coverage.add_argument("run_dir", type=Path)
    coverage.add_argument("--spec", type=Path, required=True, help="explicit conformance requirement YAML")
    prepare = operations.add_parser("prepare")
    prepare.add_argument("run_dir", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument(
        "--generated-only",
        action="store_true",
        help="use only the selected Phase-0 output for public capsules; never read the historical corpus",
    )
    prepare.add_argument(
        "--private-baseline",
        type=Path,
        help="operator-owned hidden capsule category when it is absent from the public source checkout",
    )
    prepare.add_argument(
        "--retirements",
        type=Path,
        help="reviewed public baseline-generated members intentionally retired by this derivation",
    )
    operations.add_parser("inspect").add_argument("release", type=Path)
    seal = operations.add_parser("seal")
    seal.add_argument("release", type=Path)
    seal.add_argument("--expected-digest", required=True)
    seal.add_argument("--reviewed-by", required=True)
    seal.add_argument("--review-note", required=True)
    args = parser.parse_args(argv)
    try:
        if args.verb == "corpus":
            if args.operation == "capture":
                from merlin.common.paths import module_source_path, schemas_dir

                from .phase0 import capture_selection

                try:
                    if args.capture_operation == "select":
                        result = capture_selection.select(
                            m2m_root=args.m2m_root,
                            workload_root=args.workload_root,
                            worker=module_source_path("merlin").parent / "targetgen/_m2m_capture_worker.py",
                            venv=args.venv,
                            schemas_root=schemas_dir(),
                            run_dir=args.run_dir,
                            output_dir=args.output,
                            dtype=args.dtype,
                            recipe=args.recipe,
                            bwrap_binary=args.bwrap,
                        )
                    else:
                        receipt = capture_selection.issue(args.selection, expected_sha256=args.expected_sha256)
                        result = {"sealed_receipt": str(receipt), "phase0_admission": "not_granted"}
                except ValueError as exc:
                    raise SpecError(str(exc)) from exc
                print(json.dumps(result, indent=2))
                return 0
            elif args.operation == "derive":
                from .phase0.requirements import capture_selection_specs, capture_selections, derive

                try:
                    result = derive(
                        _source(args.definition, args.catalog),
                        capture_selections(args.application_capture),
                        rtl_facts=args.rtl_facts,
                        output_root=args.output,
                        native_qualifications=capture_selections(args.native_qualification),
                        capture_preselections=capture_selection_specs(args.application_capture_selection),
                    )
                except ValueError as exc:
                    raise SpecError(str(exc)) from exc
                print(json.dumps(result, indent=2))
                return 0
            if args.operation == "groups":
                return group_capsules.run_from_args(args)
            if args.operation == "compare":
                from .phase0.comparison import build

                print(json.dumps(build([_source(value, args.catalog) for value in args.definitions]), indent=2))
                return 0
            if args.operation == "coverage":
                from .corpus.coverage import inspect_run

                result = inspect_run(args.run_dir, args.spec)
                print(json.dumps(result, indent=2))
                return 0
            from .corpus import release as corpus_release

            if args.operation == "prepare":
                result = corpus_release.prepare(
                    args.run_dir,
                    args.output,
                    private_baseline=args.private_baseline,
                    retirements=args.retirements,
                    generated_only=args.generated_only,
                )
            elif args.operation == "inspect":
                result = corpus_release.inspect_release(args.release)
            else:
                result = corpus_release.seal(
                    args.release,
                    expected_digest=args.expected_digest,
                    reviewed_by=args.reviewed_by,
                    review_note=args.review_note,
                )
        elif args.verb == "list":
            result = []
            for name, path in catalog(args.catalog).items():
                spec = load_spec(path)
                if spec.id != name:
                    raise SpecError(f"catalog id {name!r} differs from definition id {spec.id!r}")
                result.append(
                    {
                        "id": name,
                        "target": spec.target,
                        "phases": sorted(spec.document["phases"]),
                        "definition": str(path),
                        "description": spec.document.get("description", ""),
                        "kind": spec.document.get("kind", "experiment"),
                    }
                )
        elif args.verb == "runs":
            from .history import runs

            result = runs(root=args.root, target=args.target, experiment=args.experiment)
        elif args.verb == "status":
            result = runner.status(args.run_dir)
        elif args.verb == "lineage":
            from .history import lineage

            result = lineage(args.run_dir)
        elif args.verb == "resume":
            code = runner.resume(args.run_dir, checkpoint=args.checkpoint)
            print(json.dumps(runner.status(args.run_dir), indent=2))
            return code
        else:
            spec = load_spec(_source(args.spec, args.catalog))
            plan = runner.resolve_plan(
                spec,
                phase=args.phase,
                run_dir=args.run_dir,
                corpus_seal=args.corpus_seal,
                bundle_manifest=args.bundle_manifest,
                phase0_conformance_spec=args.phase0_conformance_spec,
                phase0_capability_contract=args.phase0_capability_contract,
                phase0_synth_profile=args.phase0_synth_profile,
                phase0_rtl_facts=args.phase0_rtl_facts,
                phase0_evidence_mode=args.phase0_evidence_mode,
                phase0_hidden_profile=args.phase0_hidden_profile,
                phase0_m2m_root=args.phase0_m2m_root,
                phase0_m2m_python=args.phase0_m2m_python,
            )
            if args.verb == "inspect":
                result = plan
            elif args.verb == "preflight":
                result = runner.preflight(plan)
                print(json.dumps(result, indent=2))
                return 0 if result["configuration_ready"] else 2
            else:
                code = runner.run(plan)
                print(json.dumps(runner.status(Path(plan["run_dir"])), indent=2))
                return code
        print(json.dumps(result, indent=2))
        return 0
    except (SpecError, OSError) as exc:
        print(f"merlin experiment: {exc}", file=sys.stderr)
        return 2
