"""Importing this package registers every known Step (via @register_step).

Each step module must defer its own heavy/conflicting imports (torch,
diffusers, detectron2, ...) to inside Step.load()/run() rather than at module
top level. That's what lets this package import cleanly inside *any*
isolated env's worker process — pipeline.worker only ever exercises the one
step it was told to run, but it still imports this whole package to find it.
"""

from . import anchor_stub  # noqa: F401
from . import brush  # noqa: F401
from . import colmap_export  # noqa: F401
from . import crop  # noqa: F401
from . import dataset_io  # noqa: F401
from . import face_landmarks  # noqa: F401
from . import head_angle  # noqa: F401
from . import mask_splat  # noqa: F401
from . import pointmap_splat  # noqa: F401
from . import pose_refine  # noqa: F401
from . import reference_sheet  # noqa: F401
from . import render  # noqa: F401
from . import rmbg  # noqa: F401
from . import sam3d_body  # noqa: F401
from . import sapiens2  # noqa: F401
from . import seedvr2  # noqa: F401
from . import splat  # noqa: F401
from . import views  # noqa: F401
from . import wan22_vace_denoise  # noqa: F401
