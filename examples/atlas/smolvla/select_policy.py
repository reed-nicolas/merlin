"""Bind the SmolVLA precision policy to a live capture.

Start from the rotary-host whole-model policy and change only what the flow-loss
forward needs: no TinyLlama inverse-frequency preparation scope; identity layouts;
f32 products on the selected Atlas pointwise policy (the host emitter has no
broadcast); exact host f64 products and f32/f64 casts for the timestep embedding;
host i64 minima for the expert positions; the MPFR constant-power helper for the
constant timestep frequencies; and the qualified SmolVLA host trig archive.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--constant-math-manifest", type=Path, required=True,
                        help="manifest.json written by merlin targetgen native-math-build")
    parser.add_argument("--host-unary-math", type=Path, required=True,
                        help="host-unary-math-policy.json written by host-math/scripts/write_policy.py")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    policy = json.loads((ROOT.parent / "rotary-host/whole-model-policy.json").read_text())
    policy["source_sha256"] = sha256((args.capture / "model.mlir").read_bytes()).hexdigest()
    del policy["host_preparation_scope"]
    policy["host_support"] += ["identity"]
    policy["host_float_arithmetic"] = [op for op in policy["host_float_arithmetic"] if op != "f32_mul"] + ["f64_mul"]
    policy["host_float_casts"] += ["f32_to_f64", "f64_to_f32"]
    policy["host_integer_ops"] += ["i64_min_value", "i64_first_min_index"]
    policy["source_constant_math"] = json.loads(args.constant_math_manifest.read_text())["policy"]
    policy["host_unary_math"] = json.loads(args.host_unary_math.read_text())
    with args.out.open("x") as stream:
        stream.write(json.dumps(policy, indent=2) + "\n")
    print(json.dumps({"source_sha256": policy["source_sha256"]}))


if __name__ == "__main__":
    main()
