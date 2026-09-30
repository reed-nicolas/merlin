# ModelBlaster FireSim runner example

This package owns the historical ModelBlaster staging and queue integration.
It is **not** installed with the Merlin core wheel. Install it into the same
environment as Merlin, then select it explicitly:

```sh
python -m pip install ./examples/firesim_modelblaster
export MERLIN_FIRESIM_RUNNER=modelblaster
export MERLIN_MODELBLASTER=/absolute/path/to/ModelBlaster
```

This is a breaking migration for callers that relied on Merlin silently
importing ModelBlaster or embedding its terminal marker in every Zephyr image.

When building the image, supply
`completion_metric_prefix=runner.completion_metric_prefix` from the selected
adapter to `zephyr_model.build_app`. The adapter checks the ELF for this marker
before it submits a job; an ordinary generic Zephyr image has no vendor marker.

The ModelBlaster checkout must contain either
`src/modelblaster/validation/firesim_runner.py` or
`validation/firesim_runner.py`. `run_on_firesim` passes the ELF, FireSim paths,
timeout, and queue setting to that runner. Queue mode is the default. The
adapter returns raw UART; Merlin parses and gates it against the supplied
reference. `preflight.py` checks the installed selection and source path but
does not submit an FPGA job or establish hardware correctness.

The package is independently buildable and may be copied/installed outside the
Merlin checkout. It does not belong in the metadata-only Saturn target tree.
