"""Micromag: Python interface to the C++ micromagnetic core."""

from _micromag import (
    Vec3,
    StructuredGrid,
    VectorField3D,
    write_vtk_legacy,
    # Phase 1b
    Material,
    BoundaryCondition,
    IEffectiveField,
    ZeemanField,
    UniaxialAnisotropyField,
    ExchangeField,
    EffectiveFieldSum,
)

__all__ = [
    "Vec3", "StructuredGrid", "VectorField3D", "write_vtk_legacy",
    "Material", "BoundaryCondition", "IEffectiveField",
    "ZeemanField", "UniaxialAnisotropyField", "ExchangeField",
    "EffectiveFieldSum",
]

__version__ = "0.1.0"
