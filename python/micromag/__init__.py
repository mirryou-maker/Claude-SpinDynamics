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
    # Phase 1c
    gamma_0,
    llg_torque,
    RK4Integrator,
)

__all__ = [
    "Vec3", "StructuredGrid", "VectorField3D", "write_vtk_legacy",
    "Material", "BoundaryCondition", "IEffectiveField",
    "ZeemanField", "UniaxialAnisotropyField", "ExchangeField",
    "EffectiveFieldSum",
    "gamma_0", "llg_torque", "RK4Integrator",
]

__version__ = "0.1.0"
