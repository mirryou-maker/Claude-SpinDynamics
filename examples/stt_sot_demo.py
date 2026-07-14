"""Phase 1d demo: STT switching and SOT dynamics (Python)."""
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
import math


def run_macrospin(label, m, mat, heff, stt_sum, dt, n_steps, out_every):
    rk4 = mm.RK4Integrator(dt)
    print(f"\n=== {label} ===")
    print(f"{'step':>6}  {'t[ps]':>8}  {'m_x':>8}  {'m_y':>8}  {'m_z':>8}")
    for step in range(n_steps + 1):
        if step % out_every == 0:
            mv = m.at(0, 0, 0)
            print(f"{step:>6}  {step*dt*1e12:>8.2f}  {mv.x:>8.4f}  {mv.y:>8.4f}  {mv.z:>8.4f}")
        if step < n_steps:
            rk4.step(m, mat, heff, stt_sum)


def main():
    g = mm.StructuredGrid(nx=1, ny=1, nz=1, dx=2e-9, dy=2e-9, dz=2e-9)

    mat = mm.Material.cobalt()
    mat.alpha = 0.02

    heff_aniso = mm.EffectiveFieldSum()
    heff_aniso.add(mm.UniaxialAnisotropyField())

    # ------------------------------------------------------------------
    # Demo 1: STT switching
    # ------------------------------------------------------------------
    m1 = mm.VectorField3D(g)
    eps = 0.02
    m1.set_uniform(mm.Vec3(eps, 0, math.sqrt(1 - eps**2)))

    stt_sum = mm.SpinTorqueSum()
    stt_sum.add(mm.SlonczewskiSTT(
        J=4e12, P=0.60, d=2e-9, p=mm.Vec3(0, 0, 1), beta=0.0))

    run_macrospin("STT P->AP switching (p=+z, J=4e12)",
                  m1, mat, heff_aniso, stt_sum,
                  dt=5e-14, n_steps=6000, out_every=500)

    # ------------------------------------------------------------------
    # Demo 2: SOT dynamics
    # ------------------------------------------------------------------
    m2 = mm.VectorField3D(g)
    m2.set_uniform(mm.Vec3(0, 0, 1))

    heff_sot = mm.EffectiveFieldSum()
    heff_sot.add(mm.UniaxialAnisotropyField())
    heff_sot.add(mm.ZeemanField(mm.Vec3(0, 0, 2e5)))

    sot_sum = mm.SpinTorqueSum()
    sot_sum.add(mm.SpinOrbitTorque(
        J_c=1e12, theta_SH=0.12, d_fm=2e-9,
        sigma=mm.Vec3(0, 1, 0), eta_DL=1.0, eta_FL=0.1))

    run_macrospin("SOT dynamics (J_c=1e12, theta_SH=0.12, sigma=y)",
                  m2, mat, heff_sot, sot_sum,
                  dt=5e-14, n_steps=4000, out_every=400)


if __name__ == "__main__":
    main()
