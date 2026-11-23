"""hybrid: compose one car from parts of several cars (front of A, body of B, rear of C) as a high-resolution 3D model.

Stage 1 of the hybrid-car pipeline (see README.md in this folder): per-source 3D -> watertight solid + SDF in a
common car frame -> axle-anchored cut and seam blending -> coloured mesh, renders, turntable.

Car frame used everywhere: metres, x forward, y left, z up, ground at z = 0.
"""
