"""car2brick: real-car photos -> coarse shape descriptor -> official-style LEGO model.

Front half (photos -> descriptor): ``car2brick.recon`` and ``car2brick.canonical``.
Bridge (shared by both halves): ``car2brick.descriptor``.
Back half (descriptor -> LEGO): ``car2brick.sequence``, ``car2brick.dataset``, ``car2brick.decode``,
``car2brick.selftrain``, with ``car2brick.baseline`` as the non-learned reference.
"""

from .descriptor import CarDescriptor, GridSpec, descriptor_from_points

__all__ = ["CarDescriptor", "GridSpec", "descriptor_from_points"]
