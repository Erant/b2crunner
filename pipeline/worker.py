"""Entry point run *inside* an isolated venv by SubprocessDispatcher.

Deliberately tiny and dependency-free (stdlib only) so it can be invoked by
any isolated environment's interpreter without needing this repo's full
dependency set installed there — only the specific step module's own deps
(e.g. sam-3d-body's env needs detectron2, not diffusers) plus this `pipeline`
package installed in editable mode.

Protocol: argv is [step_name, input_pickle_path, params_json_path,
output_pickle_path]. Inputs/outputs are pickled dicts of plain
numpy/str/dict data — keep step signatures to picklable types, no live GPU
tensors crossing the process boundary.
"""

from __future__ import annotations

import json
import pickle
import sys
import traceback


def main() -> None:
    step_name, input_path, params_path, output_path = sys.argv[1:5]

    from . import steps  # noqa: F401  registers all Step subclasses
    from .registry import get_step_class

    with open(input_path, "rb") as f:
        inputs = pickle.load(f)
    with open(params_path, "r") as f:
        params = json.load(f)

    try:
        step = get_step_class(step_name)()
        step.load(params)
        try:
            outputs = step.run(inputs, params)
        finally:
            step.unload()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    with open(output_path, "wb") as f:
        pickle.dump(outputs, f)


if __name__ == "__main__":
    main()
