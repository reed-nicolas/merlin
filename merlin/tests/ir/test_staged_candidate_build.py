"""A staged exact candidate may build objects without becoming an offload selection."""

from __future__ import annotations

import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from merlin.common.digest import sha256_bytes, sha256_text
from merlin.llvmlower import device_build
from merlin.llvmlower import staged_admission as staged
from merlin.llvmlower.device_build import DeviceBuild


def test_staged_candidate_build_is_byte_bound_and_unverified(tmp_path, monkeypatch):
    model, spec, contract, facts = b"model", b"spec", b"contract", b"facts"
    interface = "selected interface"
    package = tmp_path / "package"
    package.mkdir()
    (package / "source").write_bytes(b"source")
    package_sha = staged._package_sha256(package)
    abi = SimpleNamespace(symbol="target_kernel")
    resident = SimpleNamespace(m=4, n=8, k=19, dtypes=("i8", "i8", "i32"),
                               kernel_symbol="target_kernel", contract_sha256="a" * 64)
    binding = {
        "target": "target", "model_sha256": sha256_bytes(model),
        "operation_id": "mlir:" + sha256_bytes(model) + ":3",
        "interface_sha256": sha256_text(interface),
        "software_spec_sha256": sha256_bytes(spec),
        "capability_contract_sha256": sha256_bytes(contract),
        "package_sha256": package_sha,
        "kernel_abi_sha256": sha256_text(repr(abi)),
        "kernel_abi_contract_sha256": resident.contract_sha256,
        "rtl_facts_sha256": sha256_bytes(facts),
    }
    binding["binding_sha256"] = sha256_text(json.dumps(binding, sort_keys=True, separators=(",", ":")))
    report = {"exact_binding": binding, "selected_interface_mlir": interface}
    fresh = {**report, "compiler_evidence": {"status": "emitted_unverified"},
             "shim_evidence": {"status": "generated_unexecuted"}}
    monkeypatch.setattr(staged, "stage_integer_model_admission", lambda *_args, **_kw: fresh)
    monkeypatch.setattr(staged, "_tile_edge_from_exact_facts", lambda *_args: (16, None))
    monkeypatch.setattr(staged, "kernel_abi_for", lambda _target: abi)
    monkeypatch.setattr(staged, "bind_single_resident_matmul", lambda *_args, **_kw: resident)
    monkeypatch.setattr(staged, "_build_toolchain_sha256", lambda: {"clang": "c" * 64})
    seen = {}
    def inspect(kernel, shim, **kwargs):
        seen["symbol_binding"] = (kernel.name, shim.name, kwargs)
        return {"status": "object_symbol_binding_verified"}

    monkeypatch.setattr(staged, "verify_object_symbol_binding", inspect)

    def build(device, signatures, dtypes, **kwargs):
        seen.update(device=device, signatures=signatures, dtypes=dtypes, kwargs=kwargs)
        work = kwargs["workdir"]
        work.mkdir(parents=True)
        kernel = work / "merlin_staged_kernel.o"
        shim = work / "device_shim.o"
        kernel.write_bytes(b"kernel")
        shim.write_bytes(b"shim")
        (work / "merlin_staged_kernel.device.mlir").write_bytes(b"artifact")
        (work / "device_shim.c").write_bytes(b"guarded shim")
        return DeviceBuild(device, (kernel, shim), shim, {"merlin_staged_kernel": "target_kernel_0"})

    monkeypatch.setattr(staged, "build_device_objects", build)
    receipt = staged.build_staged_candidate(
        report, model=model, target="target", software_spec=spec,
        capability_contract=contract, package_dir=package, operation_id=binding["operation_id"],
        rtl_facts=facts, workdir=tmp_path / "build", cflags=("-march=rv64imafdc",),
    )
    assert seen["kwargs"]["expected_interfaces"] == {
        "merlin_staged_kernel": {"mlir": interface, "sha256": sha256_text(interface)}
    }
    assert seen["kwargs"]["tile_edge"] == 16
    assert seen["kwargs"]["cflags"] == ("-march=rv64imafdc",)
    assert receipt["status"] == "built_unverified"
    assert receipt["whole_model_offload_verified"] is False
    assert receipt["exact_binding"] == binding
    assert receipt["codegen"]["target_artifact"]["sha256"] == sha256_bytes(b"artifact")
    assert receipt["toolchain_sha256"] == {"clang": "c" * 64}
    assert receipt["symbol_binding"]["status"] == "object_symbol_binding_verified"
    assert seen["symbol_binding"] == (
        "merlin_staged_kernel.o", "device_shim.o",
        {"entry_symbol": "merlin_staged_kernel", "kernel_symbol": "target_kernel_0",
         "original_kernel_symbol": "target_kernel", "timeout": 30},
    )
    with pytest.raises(ValueError, match="stale|disagree"):
        staged.build_staged_candidate(
            report, model=model + b"changed", target="target", software_spec=spec,
            capability_contract=contract, package_dir=package, operation_id=binding["operation_id"],
            rtl_facts=facts, workdir=tmp_path / "stale", cflags=("-march=rv64imafdc",),
        )
    with pytest.raises(ValueError, match="explicit board C flags"):
        staged.build_staged_candidate(
            report, model=model, target="target", software_spec=spec,
            capability_contract=contract, package_dir=package, operation_id=binding["operation_id"],
            rtl_facts=facts, workdir=tmp_path / "implicit-flags",
        )


def test_stage_cli_writes_a_separate_unverified_build_receipt(tmp_path, monkeypatch, capsys):
    from merlin.targetgen.tool_cli import build_parser

    files = {}
    for name in ("mlir", "software-spec", "capability-contract", "rtl-facts"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        files[name] = path
    stage = {"candidate_count": 1, "review_required": True}
    monkeypatch.setattr(staged, "stage_integer_model_admission", lambda *_args, **_kw: stage)

    def build(report, **kwargs):
        assert report is stage
        assert kwargs["cflags"] == ["-march=rv64imafdc", "-ffreestanding"]
        kwargs["workdir"].mkdir()
        return {"status": "built_unverified", "whole_model_offload_verified": False}

    monkeypatch.setattr(staged, "build_staged_candidate", build)
    out = tmp_path / "stage.json"
    build_dir = tmp_path / "build"
    arguments = [
        "stage-int-mm-admission", "--target=gemmini", "--package=package",
        "--operation-id=op", f"--out={out}", f"--build-dir={build_dir}",
        "--cflag=-march=rv64imafdc", "--cflag=-ffreestanding",
    ]
    for name, path in files.items():
        arguments.append(f"--{name}={path}")
    args = build_parser().parse_args(arguments)
    assert args.func(args) == 0
    assert json.loads(out.read_text()) == stage
    assert json.loads((build_dir / "candidate-build-receipt.json").read_text()) == {
        "status": "built_unverified", "whole_model_offload_verified": False,
    }
    assert json.loads(capsys.readouterr().out)["build_status"] == "built_unverified"
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    args.build_dir = str(alias / "another-build")
    args.out = str(tmp_path / "another-stage.json")
    with pytest.raises(ValueError, match="symlink"):
        args.func(args)
    assert not (tmp_path / "another-stage.json").exists()


def test_staged_object_binding_rejects_wrong_or_unresolved_calls(tmp_path, monkeypatch):
    compiler = shutil.which("cc") or shutil.which("clang")
    if compiler is None or device_build._nm() is None:
        pytest.skip("a C compiler and nm are needed to inspect real object symbols")

    def compile_object(name, source):
        c = tmp_path / f"{name}.c"
        obj = tmp_path / f"{name}.o"
        c.write_text(source)
        subprocess.run([compiler, "-O0", "-c", str(c), "-o", str(obj)], check=True, capture_output=True)
        return obj

    kernel = compile_object("kernel", "void renamed_kernel(void) {}")
    shim = compile_object(
        "shim", "extern void renamed_kernel(void); void selected_entry(void) { renamed_kernel(); }"
    )
    kwargs = {"entry_symbol": "selected_entry", "kernel_symbol": "renamed_kernel",
              "original_kernel_symbol": "original_kernel", "timeout": 5}
    assert device_build.verify_object_symbol_binding(kernel, shim, **kwargs)["status"] == (
        "object_symbol_binding_verified"
    )

    wrong_shim = compile_object(
        "wrong_shim", "extern void other_kernel(void); void selected_entry(void) { other_kernel(); }"
    )
    with pytest.raises(ValueError, match="symbols disagree"):
        device_build.verify_object_symbol_binding(kernel, wrong_shim, **kwargs)

    unresolved_kernel = compile_object(
        "unresolved_kernel", "extern void runtime_helper(void); "
        "void renamed_kernel(void) { runtime_helper(); }"
    )
    with pytest.raises(ValueError, match="unresolved references"):
        device_build.verify_object_symbol_binding(unresolved_kernel, shim, **kwargs)

    data_kernel = compile_object("data_kernel", "int renamed_kernel = 1;")
    with pytest.raises(ValueError, match="symbols disagree"):
        device_build.verify_object_symbol_binding(data_kernel, shim, **kwargs)

    weak_kernel = compile_object("weak_kernel", "__attribute__((weak)) void renamed_kernel(void) {}")
    with pytest.raises(ValueError, match="symbols disagree"):
        device_build.verify_object_symbol_binding(weak_kernel, shim, **kwargs)

    monkeypatch.setattr(device_build, "_nm", lambda: None)
    with pytest.raises(ValueError, match="nm tool"):
        device_build.verify_object_symbol_binding(kernel, shim, **kwargs)
