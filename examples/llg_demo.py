"""Phase 1c demo: LLG time evolution of a vortex magnetization (Python)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build" / "windows-msvc" / "python"))

import micromag as mm


def mean_z(m, grid):
    s = 0.0
    for k in range(grid.nz):
        for j in range(grid.ny):
            for i in range(grid.nx):
                s += m.at(i, j, k).z
    return s / grid.size


def main():
    grid = mm.StructuredGrid(nx=16, ny=16, nz=2, dx=2e-9, dy=2e-9, dz=2e-9)
    m = mm.VectorField3D(grid)
    m.set_vortex(cx=16e-9, cy=16e-9, core_radius=4e-9)

    mat = mm.Material.permalloy()
    mat.alpha = 0.5

    heff = mm.EffectiveFieldSum()
    heff.add(mm.ZeemanField(mm.Vec3(0, 0, 0)))
    heff.add(mm.ExchangeField(mm.BoundaryCondition.Neumann))

    dt = 5e-13
    rk4 = mm.RK4Integrator(dt)

    print("=== Phase 1c llg_demo (Python) ===")
    print(f"{'step':>6}  {'t [ps]':>9}  {'E [J]':>14}  <m_z>")

    for step in range(2001):
        if step % 200 == 0:
            E  = heff.total_energy(m, mat)
            mz = mean_z(m, grid)
            print(f"{step:>6}  {step*dt*1e12:>9.3f}  {E:>14.6e}  {mz:+.6f}")
        if step < 2000:
            rk4.step(m, mat, heff)


if __name__ == "__main__":
    main()
