"""Phase 1a demo: create a vortex magnetization and save to VTK."""
import sys
from pathlib import Path

# Add build output to path (adjust if your build dir differs)
BUILD_PY = Path(__file__).resolve().parents[1] / "build" / "windows-msvc" / "python"
sys.path.insert(0, str(BUILD_PY))

import micromag as mm

def main():
    grid = mm.StructuredGrid(nx=32, ny=32, nz=4,
                             dx=2e-9, dy=2e-9, dz=2e-9)
    field = mm.VectorField3D(grid)

    ext_x = grid.nx * grid.dx
    ext_y = grid.ny * grid.dy
    field.set_vortex(cx=ext_x * 0.5, cy=ext_y * 0.5, core_radius=8e-9)

    mm.write_vtk_legacy("vortex_py.vtk", field, "m")
    print(f"Wrote vortex_py.vtk: {grid.nx} x {grid.ny} x {grid.nz} "
          f"cells ({grid.size} total)")

if __name__ == "__main__":
    main()
