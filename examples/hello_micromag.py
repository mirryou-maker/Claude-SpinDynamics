"""Phase 1a demo: create a vortex magnetization and save to VTK."""
import os, sys
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from micromag_locate import add_micromag_to_path
add_micromag_to_path()
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
