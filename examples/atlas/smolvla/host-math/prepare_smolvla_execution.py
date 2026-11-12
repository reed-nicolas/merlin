"""Run the unchanged whole-model packager with the SmolVLA host math selection.

Arguments are exactly those of whole-model/unchanged_fx/prepare_live_execution.py. The packager
resolves its validation-only host math through the ``execution_host_math`` module name; this entry
point binds that name to the SmolVLA selection before the packager runs.
"""
import sys

import execution_host_math_smolvla

sys.modules['execution_host_math'] = execution_host_math_smolvla

import prepare_live_execution  # noqa: E402  (unchanged_fx on PYTHONPATH)

if __name__ == '__main__':
    prepare_live_execution.main()
