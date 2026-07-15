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
import os
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from micromag_locate import add_micromag_to_path
add_micromag_to_path()
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


_SBAR = dict(title="mz", color=INK, vertical=True, position_x=0.87,
             position_y=0.28, height=0.46, width=0.045, n_labels=5, title_font_size=18)


def render_3d(slug, title, vtk_path, zscale=1.0, gfactor=3.0):
    if not HAVE_PV:
        return None
    mesh = pv.read(str(vtk_path))
    sp = mesh.spacing
    glyph = mesh.glyph(orient="m", scale=False, factor=max(sp) * gfactor,
                       tolerance=0.02, geom=pv.Arrow())
    p = pv.Plotter(off_screen=True, window_size=(950, 820))
    p.set_background("white")
    p.add_mesh(glyph, scalars="mz", cmap="RdBu_r", clim=[-1, 1], scalar_bar_args=_SBAR)
    try:
        iso = mesh.contour([0.0], scalars="mz")
        if iso.n_points:
            p.add_mesh(iso, color="#7f7f7f", opacity=0.28)
    except Exception:
        pass
    p.add_text(title, font_size=13, color=INK)
    if zscale != 1.0:
        p.set_scale(zscale=zscale)
        p.add_mesh(mesh.outline(), color="#c8c8c8", line_width=1)
    p.camera_position = "iso"; p.camera.elevation += 12
    p.reset_camera(); p.camera.zoom(1.25)
    out = OUT / f"3d_{slug}.png"; p.screenshot(str(out)); p.close()
    return out


def render_tube_thickness(slug, vtk_path, arr, zfac=12):
    """Extra views that emphasize the 8-layer thickness of the skyrmion tube."""
    nz = arr.shape[0]
    outs = []
    # (C) matplotlib small-multiples: mz on every z-layer -------------------
    ncol = 4; nrow = int(np.ceil(nz / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow))
    for k in range(nrow * ncol):
        ax = axes.flat[k]
        if k < nz:
            ax.imshow(arr[k, :, :, 2], origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_title(f"z-layer {k}", fontsize=10, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        if k >= nz: ax.axis("off")
    fig.suptitle("Skyrmion tube — $m_z$ through the 8 layers (core at the same (x,y) in every layer)",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    o = OUT / f"2d_{slug}_layers.png"; fig.savefig(o, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig); outs.append(o)
    if not HAVE_PV:
        return outs
    mesh = pv.read(str(vtk_path))
    # (A) stacked z-slices, thickness exaggerated ---------------------------
    p = pv.Plotter(off_screen=True, window_size=(960, 900)); p.set_background("white")
    p.enable_parallel_projection()
    slices = mesh.slice_along_axis(n=nz, axis="z")
    p.add_mesh(slices, scalars="mz", cmap="RdBu_r", clim=[-1, 1], scalar_bar_args=_SBAR)
    p.set_scale(zscale=zfac)
    p.add_mesh(mesh.outline(), color="#c8c8c8", line_width=1)
    p.add_text(f"{nz} z-layers stacked  (z x{zfac})", font_size=13, color=INK)
    p.camera_position = "iso"; p.camera.azimuth += 20; p.camera.elevation -= 10
    p.reset_camera(); p.camera.zoom(1.1)
    o = OUT / f"3d_{slug}_slices.png"; p.screenshot(str(o)); p.close(); outs.append(o)
    # (B) core isosurface as a column, thickness exaggerated, oblique view --
    p = pv.Plotter(off_screen=True, window_size=(960, 900)); p.set_background("white")
    p.enable_parallel_projection()
    try:
        core = mesh.contour([0.0], scalars="mz")
        if core.n_points:
            p.add_mesh(core, color="#d94a3d", opacity=0.9, smooth_shading=True,
                       specular=0.4, specular_power=15)
    except Exception:
        pass
    gl = mesh.glyph(orient="m", scale=False, factor=mesh.spacing[0] * 0.65,
                    tolerance=0.07, geom=pv.Arrow())   # ~1/4 arrow size
    p.add_mesh(gl, scalars="mz", cmap="RdBu_r", clim=[-1, 1], opacity=0.45, show_scalar_bar=False)
    p.set_scale(zscale=zfac)
    p.add_mesh(mesh.outline(), color="#c8c8c8", line_width=1)
    p.add_text(f"core isosurface (mz=0) — column through {nz} layers  (z x{zfac})",
               font_size=12, color=INK)
    p.camera_position = "iso"; p.camera.azimuth += 15; p.camera.elevation -= 8
    p.reset_camera(); p.camera.zoom(0.95)
    o = OUT / f"3d_{slug}_column.png"; p.screenshot(str(o)); p.close(); outs.append(o)
    return outs


def render_fig4_warp(slug, title, vtk_path, g):
    """Fig-4-of Kammerer et al. Nat. Commun. 2:279 (2011) style: a circular
    platelet whose out-of-plane magnetization m_z is shown as both colour AND
    surface height (warp), so the skyrmion core rises as a spike out of the disc."""
    if not HAVE_PV:
        return None
    mesh = pv.read(str(vtk_path))
    cx = g.nx * g.dx * 0.5; cy = g.ny * g.dy * 0.5
    R = 0.44 * min(g.nx * g.dx, g.ny * g.dy)
    pts = mesh.points
    mesh["radius"] = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    disc = mesh.threshold([0.0, R], scalars="radius")          # circular platelet
    warped = disc.warp_by_scalar("mz", factor=0.9 * R)         # m_z -> height (spike)
    p = pv.Plotter(off_screen=True, window_size=(950, 820)); p.set_background("white")
    p.add_mesh(warped, scalars="mz", cmap="turbo", clim=[-1, 1], smooth_shading=True,
               scalar_bar_args=dict(title="mz (out-of-plane)", color=INK,
                                    vertical=True, position_x=0.86, position_y=0.28,
                                    height=0.46, width=0.045, n_labels=5))
    try:
        rim = disc.extract_feature_edges(boundary_edges=True, feature_edges=False)
        p.add_mesh(rim, color="#888888", line_width=1)          # disc rim
    except Exception:
        pass
    p.add_text(f"{title} — Fig.4-style m_z warp (core spike)", font_size=12, color=INK)
    p.camera_position = "iso"; p.camera.elevation -= 12; p.camera.zoom(1.15)
    o = OUT / f"3d_fig4_{slug}.png"; p.screenshot(str(o)); p.close()
    return o


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
            zs = 6.0 if slug == "skyrmion_tube" else 1.0
            gf = 0.75 if slug == "skyrmion_tube" else 3.0   # tube: ~1/4 arrow size
            p3 = render_3d(slug, title, vtk, zscale=zs, gfactor=gf)
            if p3: line += f", {p3.name}"
        if slug == "skyrmion_tube":
            extra = render_tube_thickness(slug, vtk, mm.to_numpy(m))
            line += "".join(f", {e.name}" for e in extra)
        if slug in ("neel_skyrmion", "bloch_skyrmion"):
            pf = render_fig4_warp(slug, title, vtk, g)
            if pf: line += f", {pf.name}"
        print(line)
    ov = overview(states, imgs2d)
    print(f"\noverview: {ov.name}   (all files in {OUT})")


if __name__ == "__main__":
    main()
