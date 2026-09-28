"""Lossless generic CIRCT HW graph ingest, including typed parameter attributes."""

import subprocess
from pathlib import Path


def load_hw_graph(path: str | Path, *, circt_opt):
    """Use upstream discovery graphs without changing selected hardware bytes.

    xDSL's unregistered-attribute parser stops at ``>``; CIRCT's HW parameter
    declarations may also carry a trailing type. Preserve that type in the opaque
    attribute representation rather than deleting external modules or parameters.
    The graph is for analysis only; never serialize it as replacement hardware.
    """
    from mlc.discover.irgraph import HwGraph, to_generic
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin, UnregisteredAttr
    from xdsl.parser import Parser

    class HardwareParser(Parser):
        def _parse_dialect_type_or_attribute_body(self, attr_name, is_type, is_opaque, starting_opaque_pos):
            attribute = super()._parse_dialect_type_or_attribute_body(
                attr_name, is_type, is_opaque, starting_opaque_pos
            )
            if attr_name == "hw.param.decl" and self.parse_optional_punctuation(":") is not None:
                parameter_type = self.parse_type()
                if not isinstance(attribute, UnregisteredAttr):
                    self.raise_error("unexpected registered HW parameter declaration parser")
                return type(attribute)(
                    attr_name, is_type, is_opaque, attribute.value.data + " : " + str(parameter_type)
                )
            return attribute

    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    from .source_selection import active_selection, digest

    selected = active_selection()
    if selected is None:
        generic = to_generic(path, circt_opt=circt_opt)
    else:
        generic = Path(selected["_generic_hw_output"])
        if generic.is_symlink():
            raise ValueError("selected CIRCT genericization output may not be a symlink")
        receipt = selected.get("_genericization")
        if receipt is None:
            generic.parent.mkdir(parents=True, exist_ok=True)
            command = [str(circt_opt), "--mlir-print-op-generic", str(path), "-o", str(generic)]
            subprocess.run(command, check=True, capture_output=True)
            selected["_genericization"] = {
                "kind": "circt_generic_serialization",
                "command": command,
                "returncode": 0,
                "input": {"path": str(Path(path).resolve()), "sha256": digest(path)},
                "output": {"path": str(generic.resolve()), "sha256": digest(generic)},
                "tool": {"path": str(Path(circt_opt).resolve()), "sha256": digest(circt_opt)},
            }
        elif digest(path) != receipt["input"]["sha256"] or digest(generic) != receipt["output"]["sha256"]:
            raise ValueError("selected CIRCT genericization bytes changed during observation")
    return HwGraph(HardwareParser(context, generic.read_text()).parse_module())
