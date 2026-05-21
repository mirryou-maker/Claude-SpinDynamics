"""Micromag: Python interface to the C++ micromagnetic core."""

from _micromag import (
    Vec3,
    StructuredGrid,
    VectorField3D,
    write_vtk_legacy,
)

__all__ = [
    "Vec3",
    "StructuredGrid",
    "VectorField3D",
    "write_vtk_legacy",
]

__version__ = "0.1.0"
