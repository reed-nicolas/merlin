"""Translate the public Atlas assembly spelling without supplying ISA encodings."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


def _rules():
    rules = {}

    def add(names, operands, native=None, order=None):
        for name in names.split():
            roles = operands.split()
            rules[name] = (native or name, roles, order or tuple(range(len(roles))))

    add("nop ecall ebreak fence", "")
    add("delay", "i")
    add("li lui auipc", "x i")
    add("add sub sll slt sltu xor srl sra or and", "x x x")
    add("addi slti sltiu xori ori andi slli srli srai jalr", "x x i")
    add("beq bne blt bge bltu bgeu", "x x target")
    add("jal", "x target")
    add("csrrw csrrs csrrc", "x i x", order=(0, 2, 1))
    add("seli", "e i")
    add("vli.all vli.row vli.col vli.one", "m i")
    add("vtrpose.xlu", "m m")
    for engine in ("mxu0", "mxu1"):
        add(f"vmatpush.w.{engine}", "w m", native=f"vmatpush.weight.{engine}")
        add(f"vmatpush.acc.fp8.{engine} vmatpush.acc.bf16.{engine}", "acc m")
        add(f"vmatpop.fp8.{engine}", "m e acc", native=f"vmatpop.fp8.acc.{engine}", order=(0, 2, 1))
        add(f"vmatpop.bf16.{engine}", "m acc", native=f"vmatpop.bf16.acc.{engine}")
        add(f"vmatmul.{engine} vmatmul.acc.{engine}", "acc m w")
    add("vadd.bf16 vsub.bf16 vmul.bf16", "m m m")
    add("vmin.bf16", "m m m", native="vminimum.bf16")
    add("vmax.bf16", "m m m", native="vmaximum.bf16")
    add("vmov vrecip.bf16 vsquare.bf16 vcube.bf16", "m m")
    for name in ("vexp", "vexp2", "vrelu", "vsin", "vcos", "vtanh", "vlog2", "vsqrt"):
        add(name, "m m", native=name + ".bf16")
    for name in ("vredsum", "vredmin", "vredmax"):
        add(name + ".bf16 " + name + ".row.bf16", "m m")
    add("vfp8pack", "m m e", native="vpack.bf16.fp8")
    add("vfp8unpack", "m m e", native="vunpack.fp8.bf16")
    return rules


RULES = _rules()
REVERSE = {native: (name, roles, order) for name, (native, roles, order) in RULES.items()}
MEMORY = {**dict.fromkeys("lb lh lw lbu lhu sb sh sw".split(), "x"), "seld": "e", "vload": "m", "vstore": "m"}
DMA = {"dma.load": 3, "dma.store": 3, "dma.config": 1, "dma.wait": 0}


def _operand(value, role, native):
    if role == "target":
        return value
    if role == "i":
        int(value, 0)
        return value
    prefix = role
    if role == "x" or native:
        if not value.lower().startswith(prefix):
            raise ValueError(f"expected {prefix} register: {value}")
        number = int(value[len(prefix) :], 0)
    else:
        number = int(value, 0)
    if number < 0:
        raise ValueError("negative register index")
    return (prefix if role == "x" or not native else "") + str(number)


def _instruction(op, args, native):
    if not native and op in {"csrw", "csrr"}:
        if len(args) != 2:
            raise ValueError(f"{op} expects two operands")
        op, args = ("csrrw", ["x0", args[1], args[0]]) if op == "csrw" else ("csrrs", [args[0], args[1], "x0"])
    dma, sep, channel = op.rpartition(".ch")
    if (native and sep and dma in DMA) or (not native and op in DMA):
        if not native:
            dma, channel, args = op, args[-1], args[:-1]
        if len(args) != DMA[dma] or int(channel, 0) < 0:
            raise ValueError("invalid DMA operands")
        args = [_operand(arg, "x", native) for arg in args]
        return (dma, [*args, str(int(channel, 0))]) if native else (f"{dma}.ch{int(channel, 0)}", args)
    if op in MEMORY:
        if native:
            if len(args) != 2:
                raise ValueError("memory instruction expects register and offset(base)")
            offset, sep, base = args[1].partition("(")
            if not sep or not base.endswith(")"):
                raise ValueError("expected offset(base)")
            args = [args[0], base[:-1], offset or "0"]
        if len(args) != 3:
            raise ValueError("memory instruction expects register, base, offset")
        reg = _operand(args[0], MEMORY[op], native)
        base, offset = _operand(args[1], "x", native), _operand(args[2], "i", native)
        return op, [reg, base, offset] if native else [reg, f"{offset}({base})"]
    rule = (REVERSE if native else RULES).get(op)
    if rule is None:
        raise ValueError(f"unsupported assembly instruction: {op}")
    name, roles, order = rule
    if len(args) != len(roles):
        raise ValueError(f"{op} expects {len(roles)} operands")
    if native:
        original = [""] * len(args)
        for index, original_index in enumerate(order):
            original[original_index] = _operand(args[index], roles[original_index], True)
        return name, original
    converted = [_operand(arg, role, False) for arg, role in zip(args, roles)]
    return name, [converted[index] for index in order]


def translate(source: str, *, to_native: bool) -> str:
    lines = []
    for number, raw in enumerate(source.splitlines(), 1):
        code, separator, comment = raw.partition("#")
        code = code.strip()
        if ":" in code:
            label, _, code = code.partition(":")
            if not label or not label.replace(".", "_").isidentifier():
                raise ValueError(f"line {number}: invalid label")
            lines.append(label + ":")
            code = code.strip()
            if not code and not separator:
                continue
        try:
            if code:
                tokens = code.replace(",", " ").split()
                op, args = _instruction(tokens[0].lower(), tokens[1:], not to_native)
                code = (op if to_native else op.upper()) + (" " + ", ".join(args) if args else "")
        except (ValueError, IndexError) as exc:
            raise ValueError(f"line {number}: {exc}") from exc
        lines.append(code + (" #" + comment if separator else ""))
    return "\n".join(lines) + "\n"


def prepare_source(*, source: Path, assembly: str, assembler: Path | None, output_dir: Path, translator=None):
    """Preserve source bytes and optionally check the declared assembler's round trip."""
    if translator is not None:
        raise ValueError("external translators are unsupported; use the explicit assembly dialect")
    if assembly not in {"atlas-opt-native", "merlin-atlas"}:
        raise ValueError(f"unsupported assembly dialect: {assembly}")
    if assembly == "merlin-atlas" and assembler is None:
        raise ValueError("merlin-atlas translation requires an explicitly selected assembler")
    raw = Path(source).read_bytes()
    native = translate(raw.decode(), to_native=True).encode() if assembly == "merlin-atlas" else raw
    baremetal = translate(native.decode(), to_native=False)
    metadata = {"assembly": assembly, "source_sha256": hashlib.sha256(raw).hexdigest(), "encoding": "not-checked"}
    if assembler is not None:
        assembler = Path(assembler).resolve(strict=True)
        assembler_raw = assembler.read_bytes()
        spec = importlib.util.spec_from_file_location("_selected_atlas_assembler", assembler)
        if spec is None or spec.loader is None:
            raise ValueError("cannot load selected assembler")
        module = importlib.util.module_from_spec(spec)
        exec(compile(assembler_raw, str(assembler), "exec"), module.__dict__)
        words = list(module.assemble(baremetal))
        if any(type(word) is not int or not 0 <= word < (1 << 32) for word in words):
            raise ValueError("assembler did not return 32-bit words")
        if assembly == "merlin-atlas" and words != list(module.assemble(raw.decode())):
            raise ValueError("assembly translation changed encoded words")
        if assembler.read_bytes() != assembler_raw:
            raise ValueError("assembler changed during encoding")
        encoded = b"".join(word.to_bytes(4, "little") for word in words)
        metadata.update(
            encoding="roundtrip-words-matched" if assembly == "merlin-atlas" else "translated-words-only",
            assembler_sha256=hashlib.sha256(assembler_raw).hexdigest(),
            words_sha256=hashlib.sha256(encoded).hexdigest(),
            word_count=len(words),
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "native.S"
    with path.open("xb") as stream:
        stream.write(native)
    return {"native_path": path, **metadata}
