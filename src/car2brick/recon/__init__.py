"""Front half: photos -> calibrated cameras -> masks -> visual hull -> car-frame interior points (+ colours, labels).

Heavy tools are optional and imported lazily: COLMAP (CLI), SAM2 (Python), and any external surface reconstructor
(2DGS / NeuS2 / TRELLIS ...) whose mesh can be passed to ``canonical.mesh_descriptor`` instead of the hull.
"""

from .cameras import Camera, read_colmap_model
from .visual_hull import carve, hull_descriptor

__all__ = ["Camera", "read_colmap_model", "carve", "hull_descriptor"]
