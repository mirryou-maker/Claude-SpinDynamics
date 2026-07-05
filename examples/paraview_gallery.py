"""Export a gallery of magnetization textures to ParaView (.vtk) and render a
preview. Demonstrates mm.save_paraview: domain wall, Neel/Bloch skyrmions,
vortex, a standard-problem state, and a 3-D skyrmion tube.

    python examples/paraview_gallery.py
-> writes paraview_demo/*.vtk (open in ParaView) and paraview_demo/gallery.png
"""
import sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build" / "windows-msvc" / "python"))
import micromag as mm

OUT = ROOT / "paraview_demo"; OUT.mkdir(exist_ok=True)
INK, GRID = "#0b0b0b", "#c3c2b7"


def _Q(m):
    from micromag.paraview import topological_charge_density
    return topological_charge_density(mm.to_numpy(m)).sum()

def build_states():
    states = []
    # 1) Neel skyrmion (radial in-plane) — Co/Pt-like disk.  cx,cy are offsets
    #    from the grid centre, so 0,0 (the default) places it dead centre.
    g = mm.StructuredGrid(120, 120, 1, 2e-9, 2e-9, 1e-9)
    m = mm.neel_skyrmion(g, 24e-9, -1, -1)
    states.append(("Neel skyrmion", g, m, f"Q = {_Q(m):+.2f}"))
    # 2) Bloch skyrmion (tangential in-plane)
    m = mm.bloch_skyrmion(g, 24e-9, -1, -1)
    states.append(("Bloch skyrmion", g, m, f"Q = {_Q(m):+.2f}"))
    # 3) Bloch domain wall in a strip
    gs = mm.StructuredGrid(200, 40, 1, 2e-9, 2e-9, 2e-9)
    m = mm.bloch_dw_np(gs, 6e-9)
    states.append(("Bloch domain wall", gs, m, "180° wall, Δ = 6 nm"))
    # 4) Head-to-head two-domain (Neel-type charged wall)
    m = mm.two_domain(gs, mm.Vec3(1, 0, 0), mm.Vec3(-1, 0, 0), "x")
    states.append(("Two-domain (SP-style)", gs, m, "head-to-head +x / -x"))
    # 5) Vortex (µMAG SP#1 ground state)
    gv = mm.StructuredGrid(80, 80, 1, 2e-9, 2e-9, 5e-9)
    m = mm.vortex_state(gv, 1, 1)
    states.append(("Vortex (SP#1)", gv, m, "curling + out-of-plane core"))
    # 6) 3-D skyrmion tube (nz = 8) — shows ParaView slicing
    g3 = mm.StructuredGrid(60, 60, 8, 2e-9, 2e-9, 2e-9)
    m = mm.neel_skyrmion(g3, 16e-9, -1, -1)
    states.append(("3-D skyrmion tube", g3, m, "60×60×8 — slice in ParaView"))
    return states


def render(states):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (title, g, m, sub) in zip(axes.flat, states):
        arr = mm.to_numpy(m)                     # (nz, ny, nx, 3)
        kz = arr.shape[0] // 2
        mz = arr[kz, :, :, 2]; mx = arr[kz, :, :, 0]; my = arr[kz, :, :, 1]
        ny, nx = mz.shape
        im = ax.imshow(mz, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1,
                       extent=[0, nx, 0, ny], aspect="equal")
        step = max(1, nx // 22)
        ys, xs = np.mgrid[0:ny:step, 0:nx:step]
        ax.quiver(xs + 0.5, ys + 0.5, mx[::step, ::step], my[::step, ::step],
                  color=INK, scale=28, width=0.004, pivot="mid")
        ax.set_title(f"{title}\n{sub}", fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color(GRID)
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02, location="right")
    cbar.set_label("$m_z$  (blue −1  →  red +1)", fontsize=11)
    fig.suptitle("Claude-SD → ParaView: magnetization textures  "
                 "(color $m_z$, arrows in-plane $m_{xy}$)", fontsize=14, y=0.98)
    out = OUT / "gallery.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print("wrote", out)


def main():
    states = build_states()
    for title, g, m, sub in states:
        fn = OUT / (title.lower().replace(" ", "_").replace("-", "").replace("(", "").replace(")", "").replace("/", "") + ".vtk")
        mm.save_paraview(m, fn)
        print(f"  {title:24s} -> {fn.name}   ({sub})")
    render(states)
    print(f"\nParaView: open {OUT}\\*.vtk  (Glyph the 'm' vector; colour by 'mz' or 'q_topo').")


if __name__ == "__main__":
    main()
