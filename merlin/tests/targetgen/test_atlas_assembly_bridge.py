import importlib.util

import pytest

from merlin.common.paths import repo_root


@pytest.fixture
def bridge():
    path = repo_root() / "examples/atlas/target/rtlgraph_assembly.py"
    spec = importlib.util.spec_from_file_location("atlas_assembly_bridge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_operand_order_and_comments_survive_translation(bridge):
    original = (
        "Entry: LI x6, 0x80000800\n"
        "DMA.LOAD x6, x18, x12, 3\n"
        "VLOAD 8, x6, 0\n"
        "VMATPOP.FP8.MXU0 4, 5, 1\n"
        "CSRRW x0, 0xC10, x30 # atlas.release\n"
        "BEQ x0, x0, Entry\n"
    )
    native = bridge.translate(original, to_native=True)
    assert native == (
        "Entry:\nli x6, 0x80000800\n"
        "dma.load.ch3 x6, x18, x12\n"
        "vload m8, 0(x6)\n"
        "vmatpop.fp8.acc.mxu0 m4, acc1, e5\n"
        "csrrw x0, x30, 0xC10 # atlas.release\n"
        "beq x0, x0, Entry\n"
    )
    assert bridge.translate(native, to_native=False) == original.replace("Entry: LI", "Entry:\nLI")


@pytest.mark.parametrize(
    "original, native",
    [
        ("CSRW x30, 0xC10", "csrrw x0, x30, 0xC10"),
        ("CSRR x20, 0xC00", "csrrs x20, x0, 0xC00"),
        ("VMATPUSH.W.MXU1 1, 5", "vmatpush.weight.mxu1 w1, m5"),
        ("VFP8PACK 2, 4, 6", "vpack.bf16.fp8 m2, m4, e6"),
        ("SELD 2, x6, -4", "seld e2, -4(x6)"),
        ("DMA.WAIT 2", "dma.wait.ch2"),
    ],
)
def test_distinct_operand_forms(bridge, original, native):
    assert bridge.translate(original, to_native=True) == native + "\n"
    assert bridge.translate(bridge.translate(native, to_native=False), to_native=True) == native + "\n"


@pytest.mark.parametrize("source", [".word 0", "DELAY 2 extra", "VLOAD 1, x2", "DMA.WAIT", "CSRRWI x0, 4, 1"])
def test_unsupported_or_malformed_input_fails(bridge, source):
    with pytest.raises(ValueError, match="line 1"):
        bridge.translate(source, to_native=True)


def test_native_bytes_preserved_and_outputs_not_overwritten(bridge, tmp_path):
    source = tmp_path / "source.S"
    source.write_bytes(b"delay 30\r\necall\r\n")
    result = bridge.prepare_source(source=source, assembly="atlas-opt-native", assembler=None, output_dir=tmp_path)
    assert result["native_path"].read_bytes() == source.read_bytes()
    assert result["encoding"] == "not-checked"
    with pytest.raises(FileExistsError):
        bridge.prepare_source(source=source, assembly="atlas-opt-native", assembler=None, output_dir=tmp_path)


def test_encoding_disagreement_prevents_output(bridge, tmp_path):
    source = tmp_path / "source.S"
    source.write_text("CSRW x30, 0xC10\n")
    assembler = tmp_path / "assembler.py"
    assembler.write_text('def assemble(source):\n    return [1 if source.startswith("CSRW ") else 2]\n')
    with pytest.raises(ValueError, match="changed encoded words"):
        bridge.prepare_source(source=source, assembly="merlin-atlas", assembler=assembler, output_dir=tmp_path)
    assert not (tmp_path / "native.S").exists()


def test_merlin_dialect_requires_explicit_assembler(bridge, tmp_path):
    with pytest.raises(ValueError, match="explicitly selected assembler"):
        bridge.prepare_source(
            source=tmp_path / "missing.S", assembly="merlin-atlas", assembler=None, output_dir=tmp_path
        )


def test_wrong_declared_native_dialect_fails_before_output(bridge, tmp_path):
    source = tmp_path / "source.S"
    source.write_text("VLOAD 8, x6, 0\n")
    with pytest.raises(ValueError, match="memory instruction"):
        bridge.prepare_source(source=source, assembly="atlas-opt-native", assembler=None, output_dir=tmp_path)
    assert not (tmp_path / "native.S").exists()
