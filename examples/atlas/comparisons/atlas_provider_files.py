"""Explicit source contracts for archived Atlas native-substitution replay."""

# Native substitution requires an archived Atlas cold-contraction producer
# root with the interface below, which current atlas-mlir no longer ships.
LEGACY_VALIDATOR_FILE = "validate_model_contraction.py"
LEGACY_HELPER_MODULES = ("atlas_tile_arithmetic", "run_model_tile")
LEGACY_PRODUCER_FILES = ("run_model_contraction.py", *(name + ".py" for name in LEGACY_HELPER_MODULES))
LEGACY_PROVIDER_REQUIREMENT = (
    "Native-substitution replay requires an archived Atlas cold-contraction producer root "
    "providing run_model_contraction.py and validate_model_contraction.py with "
    "bind/recheck/admit_contraction, which current atlas-mlir no longer ships"
)
