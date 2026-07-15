"""Phase 1b demo: compute effective field for a vortex magnetization (Python)."""
import sys
from pathlib import Path

import os
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from micromag_locate import add_micromag_to_path
add_micromag_to_path()

import micromag as mm


def main():
    grid = mm.StructuredGrid(nx=32, ny=32, nz=4, dx=2e-9, dy=2e-9, dz=2e-9)
    m = mm.VectorField3D(grid)
    H = mm.VectorField3D(grid)

    m.set_vortex(cx=grid.nx * grid.dx * 0.5,
                 cy=grid.ny * grid.dy * 0.5,
                 core_radius=8e-9)

    mat = mm.Material.cobalt()

    heff = mm.EffectiveFieldSum()
    heff.add(mm.ZeemanField(mm.Vec3(0, 0, 5e4)))
    heff.add(mm.UniaxialAnisotropyField())
    heff.add(mm.ExchangeField(mm.BoundaryCondition.Neumann))

    heff.compute(m, mat, H)

    mm.write_vtk_legacy("vortex_m_py.vtk", m, "m")
    mm.write_vtk_legacy("vortex_H_py.vtk", H, "H_eff")

    print("=== Phase 1b field_demo (Python) ===")
    print(f"Grid: {grid.nx} x {grid.ny} x {grid.nz}  ({grid.size} cells)")
    print(f"Material: Ms={mat.Ms}  A={mat.A_exchange}  K={mat.K_uniaxial}\n")
    for term in heff.terms:
        print(f"  E[{term.name}] = {term.energy(m, mat):.6e} J")
    print(f"  E[total]    = {heff.total_energy(m, mat):.6e} J")


if __name__ == "__main__":
    main()
