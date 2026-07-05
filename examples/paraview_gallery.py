"""Export magnetization textures to ParaView (.vtk) and render previews.

Outputs, in paraview_demo/:
  * <name>.vtk            — open in ParaView (Glyph 'm', colour 'mz'/'q_topo')
  * 2d_<name>.png         — matplotlib preview (mz colour + in-plane arrows),
                            domain-wall / two-domain cropped around the wall
  * 3d_<name>.png         — pyvista 3-D render (glyph arrows + mz=0 isosurface)
                            for the 3-D-worthy states (skyrmions, vortex, tube)
  * gallery_overview.png  — contact sheet of the 2-D previews

    python examples/paraview_gallery.py
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

try:
    import pyvista as pv
    pv.OFF_SCREEN = True
    HAVE_PV = True
except Exception:
    HAVE_PV = False


def _Q(m):
    from micromag.paraview import topological_charge_density
    return topological_charge_density(mm.to_numpy(m)).sum()


def build_states():
    """(slug, title, grid, field, subtitle, three_d, crop)  — crop = (x0,x1,y0,y1) cells or None."""
    S = []
    g = mm.StructuredGrid(120, 120, 1, 2e-9, 2e-9, 1e-9)
    m = mm.neel_skyrmion(g, 24e-9, -1, -1)
    S.append(("neel_skyrmion", "Néel skyrmion", g, m, f"Q = {_Q(m):+.2f}  (radial arrows)", True, None))
    m = mm.bloch_skyrmion(g, 24e-9, -1, -1)
    S.append(("bloch_skyrmion", "Bloch skyrmion", g, m, f"Q = {_Q(m):+.2f}  (tangential arrows)", True, None))

    gs = mm.StructuredGrid(200, 40, 1, 2e-9, 2e-9, 2e-9)
    m = mm.bloch_dw_np(gs, 6e-9)
    S.append(("domain_wall", "Bloch domain wall (zoom)", gs, m, "180° wall, Δ = 6 nm", False, (80, 120, 0, 40)))
    m = mm.two_domain(gs, mm.Vec3(1, 0, 0), mm.Vec3(-1, 0, 0), "x")
    S.append(("two_domain", "Two-domain wall (zoom)", gs, m, "head-to-head +x / −x", False, (80, 120, 0, 40)))

    gv = mm.StructuredGrid(80, 80, 1, 2e-9, 2e-9, 5e-9)
    m = mm.vortex_state(gv, 1, 1)
    S.append(("vortex", "Vortex (SP#1)", gv, m, "curling + out-of-plane core", True, None))

    g3 = mm.StructuredGrid(60, 60, 8, 2e-9, 2e-9, 2e-9)
    m = mm.neel_skyrmion(g3, 16e-9, -1, -1)
    S.append(("skyrmion_tube", "3-D skyrmion tube", g3, m, "60×60×8 cells", True, None))
    return S


def render_2d(slug, title, g, m, sub, crop):
    arr = mm.to_numpy(m)
    kz = arr.shape[0] // 2
    mz = arr[kz, :, :, 2]; mx = arr[kz, :, :, 0]; my = arr[kz, :, :, 1]
    if crop:
        x0, x1, y0, y1 = crop
        mz, mx, my = mz[y0:y1, x0:x1], mx[y0:y1, x0:x1], my[y0:y1, x0:x1]
    ny, nx = mz.shape
    fig, ax = plt.subplots(figsize=(5.2, 5.2 * ny / nx + 0.6))
    im = ax.imshow(mz, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1,
                   extent=[0, nx, 0, ny], aspect="equal")
    step = max(1, min(nx, ny) // 22)
    ys, xs = np.mgrid[0:ny:step, 0:nx:step]
    ax.quiver(xs + 0.5, ys + 0.5, mx[::step, ::step], my[::step, ::step],
              color=INK, scale=26, width=0.005, pivot="mid")
    ax.set_title(f"{title}\n{sub}", fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02); cb.set_label("$m_z$")
    fig.tight_layout()
    out = OUT / f"2d_{slug}.png"; fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig); return out


def render_3d(slug, title, vtk_path):
    if not HAVE_PV:
        return None
    mesh = pv.read(str(vtk_path))
    sp = mesh.spacing
    glyph = mesh.glyph(orient="m", scale=False, factor=max(sp) * 3.0,
                       tolerance=0.02, geom=pv.Arrow())
    p = pv.Plotter(off_screen=True, window_size=(950, 820))
    p.set_background("white")
    p.add_mesh(glyph, scalars="mz", cmap="RdBu_r", clim=[-1, 1],
               scalar_bar_args=dict(title="mz", color=INK))
    try:
        iso = mesh.contour([0.0], scalars="mz")
        if iso.n_points:
            p.add_mesh(iso, color="#7f7f7f", opacity=0.28)
    except Exception:
        pass
    p.add_text(title, font_size=13, color=INK)
    p.camera_position = "iso"; p.camera.elevation += 12; p.camera.zoom(1.3)
    out = OUT / f"3d_{slug}.png"; p.screenshot(str(out)); p.close()
    return out


def overview(states, imgs2d):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (slug, title, *_), img in zip(axes.flat, states, imgs2d):
        ax.imshow(plt.imread(img)); ax.axis("off")
    fig.suptitle("Claude-SD → ParaView magnetization textures", fontsize=15, y=0.98)
    fig.tight_layout()
    out = OUT / "gallery_overview.png"; fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig); return out


def main():
    print(f"pyvista 3-D rendering: {'ON' if HAVE_PV else 'OFF (pip install pyvista)'}")
    states = build_states()
    imgs2d = []
    for slug, title, g, m, sub, three_d, crop in states:
        vtk = OUT / f"{slug}.vtk"
        mm.save_paraview(m, vtk)
        p2 = render_2d(slug, title, g, m, sub, crop); imgs2d.append(p2)
        line = f"  {title:24s} -> {vtk.name}, {p2.name}"
        if three_d:
            p3 = render_3d(slug, title, vtk)
            if p3: line += f", {p3.name}"
        print(line)
    ov = overview(states, imgs2d)
    print(f"\noverview: {ov.name}   (all files in {OUT})")


if __name__ == "__main__":
    main()
