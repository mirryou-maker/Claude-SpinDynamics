"""Phase 1c demo: LLG time evolution of a vortex magnetization (Python)."""
import sys
from pathlib import Path

import os
from pathlib import Path

def _add_micromag_to_path():
    """Locate the micromag module + its DLLs in a release package
    (../runtime-dll + ../<variant>/python for GPU, ../python for CPU) or a source
    build (../build/<preset>/python + system CUDA). Works from any working dir."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    root = Path(__file__).resolve().parent.parent
    rtd = root / "runtime-dll"
    if rtd.is_dir():
        os.add_dll_directory(str(rtd))
        for _v in ("cuFFT-f64", "cuFFT-f32", "VkFFT-f64", "VkFFT-f32"):
            _py = root / _v / "python"
            if list(_py.glob("_micromag*.pyd")):
                sys.path.insert(0, str(_py)); return
    if list((root / "python").glob("_micromag*.pyd")):
        os.add_dll_directory(str(root / "python"))
        sys.path.insert(0, str(root / "python")); return
    _cuda = r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64"
    if os.path.isdir(_cuda):
        os.add_dll_directory(_cuda)
    for _p in ("windows-msvc-cuda", "windows-msvc"):
        _py = root / "build" / _p / "python"
        if _py.is_dir():
            sys.path.insert(0, str(_py)); return
    raise RuntimeError("micromag module not found (release package or source build).")

_add_micromag_to_path()

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
