"""Sapiens2 normal-map estimation — not in scope for the current WAN22 push.

Left as a stub; not part of fast_helical_native.yaml today.
"""

from __future__ import annotations

from typing import Any, Dict

from ..registry import register_step
from ..step import Step


@register_step("sapiens2_lite")
class Sapiens2LiteStep(Step):
    """Sapiens2 normal-map estimation (pure-PyTorch lite path — avoids the
    full mmcv/OpenMMLab install).

    inputs: {"image": np.ndarray BGR}
    params: {"resolution": [w, h]}
    outputs: {"normal_map": np.ndarray HxWx3 float32}
    """

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Sapiens2-lite native inference not yet ported")
