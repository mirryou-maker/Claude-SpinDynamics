"""Phase 1a demo: create a vortex magnetization and save to VTK."""
import os, sys
from pathlib import Path

def _add_micromag_to_path():
    """Find the micromag module + its DLLs in either a downloaded release package
    (../runtime-dll + ../<variant>/python, or ../python for the CPU package) or a
    source build (../build/<preset>/python + the system CUDA toolkit)."""
    # numpy (MKL) and the bundled module both ship an OpenMP runtime; allow both
    # to load instead of aborting with "OMP: Error #15".
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    root = Path(__file__).resolve().parent.parent
    rtd = root / "runtime-dll"                          # 1) GPU release package
    if rtd.is_dir():
        os.add_dll_directory(str(rtd))
        for v in ("cuFFT-f64", "cuFFT-f32", "VkFFT-f64", "VkFFT-f32"):
            py = root / v / "python"
            if list(py.glob("_micromag*.pyd")):
                sys.path.insert(0, str(py)); return
    if list((root / "python").glob("_micromag*.pyd")):  # 2) CPU release package
        os.add_dll_directory(str(root / "python"))
        sys.path.insert(0, str(root / "python")); return
    cuda_bin = r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64"
    if os.path.isdir(cuda_bin):                          # 3) source build tree
        os.add_dll_directory(cuda_bin)
    for preset in ("windows-msvc-cuda", "windows-msvc"):
        py = root / "build" / preset / "python"
        if py.is_dir():
            sys.path.insert(0, str(py)); return
    raise RuntimeError("micromag module not found (release package or source build).")

_add_micromag_to_path()
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
