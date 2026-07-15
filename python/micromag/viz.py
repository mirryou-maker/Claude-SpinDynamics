"""Animation, DW width, skyrmion phase diagram, numpy inits, recommend_integrator

Split from the former monolithic micromag/__init__.py; public names are
re-exported by the package __init__, so `import micromag as mm` is unchanged.
"""
import math as _math
import numpy as _np

from _micromag import *          # C++ core names  # noqa: F401,F403


# ===========================================================================
# Priority 1-4: new utility functions
# ===========================================================================

# ---------------------------------------------------------------------------
# batch_to_numpy — convert a list of VectorField3D frames to a single array
# ---------------------------------------------------------------------------

def batch_to_numpy(m_list):
    """Convert a list of VectorField3D snapshots to a single numpy array.

    Parameters
    ----------
    m_list : list[VectorField3D]
        Sequence of magnetisation frames (same grid).

    Returns
    -------
    numpy.ndarray, shape (n_frames, nz, ny, nx, 3)
        All frames stacked along axis 0.  Suitable for animation or analysis.

    Example
    -------
    >>> frames = []
    >>> for _ in range(n_steps):
    ...     integ.step(mat, demag, fields)
    ...     m = mm.VectorField3D(grid); integ.download(m); frames.append(m)
    >>> arr = mm.batch_to_numpy(frames)   # (n_frames, nz, ny, nx, 3)
    """
    import numpy as _np
    if not m_list:
        return _np.empty((0,))
    frames = [_np.asarray(to_numpy(m)) for m in m_list]
    return _np.stack(frames, axis=0)


# ---------------------------------------------------------------------------
# save_animation — write VectorField3D sequence to GIF/MP4
# ---------------------------------------------------------------------------

def save_animation(
    m_list,
    filename: str,
    component: str = "z",
    colormap: str = "bwr",
    vmin: float = -1.0,
    vmax: float = 1.0,
    fps: int = 10,
    iz: int = 0,
    dpi: int = 80,
):
    """Save a list of VectorField3D snapshots as an animated GIF or MP4.

    Requires matplotlib and (for GIF) either Pillow or imageio. For MP4
    additionally requires ffmpeg on PATH.

    Parameters
    ----------
    m_list     : list[VectorField3D] — sequence of frames (same grid)
    filename   : str — output path; extension determines format (.gif / .mp4)
    component  : 'x' | 'y' | 'z' — magnetisation component to display
    colormap   : matplotlib colormap name (default 'bwr')
    vmin/vmax  : colour scale limits (default ±1)
    fps        : frames per second
    iz         : z-layer index to display (default 0)
    dpi        : figure DPI

    Example
    -------
    >>> mm.save_animation(frames, "skyrmion.gif", component="z", fps=15)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        import matplotlib.animation as _anim
    except ImportError as e:
        raise ImportError("save_animation requires matplotlib. "
                          "Install with: pip install matplotlib") from e

    comp_idx = {"x": 0, "y": 1, "z": 2}[component]
    arr = batch_to_numpy(m_list)          # (n_frames, nz, ny, nx, 3)
    data = arr[:, iz, :, :, comp_idx]    # (n_frames, ny, nx)

    fig, ax = _plt.subplots(figsize=(4, 4), dpi=dpi)
    ax.set_axis_off()
    im = ax.imshow(data[0], cmap=colormap, vmin=vmin, vmax=vmax,
                   origin="lower", interpolation="nearest")
    title = ax.set_title("Frame 0")

    def _update(frame_idx):
        im.set_data(data[frame_idx])
        title.set_text(f"Frame {frame_idx}")
        return [im, title]

    ani = _anim.FuncAnimation(fig, _update, frames=len(m_list),
                              interval=1000 // fps, blit=True)

    if filename.endswith(".gif"):
        ani.save(filename, writer="pillow", fps=fps)
    elif filename.endswith(".mp4"):
        ani.save(filename, writer="ffmpeg", fps=fps)
    else:
        ani.save(filename, fps=fps)

    _plt.close(fig)
    print(f"[save_animation] saved {len(m_list)} frames → {filename}")


# ---------------------------------------------------------------------------
# bloch_dw_width — extract Bloch DW width from 1D magnetisation profile
# ---------------------------------------------------------------------------

def bloch_dw_width(m, axis: int = 0, comp: int = 2):
    """Estimate Bloch/Neel domain-wall width from a VectorField3D.

    Uses the Lilley (1950) width definition:
        λ_Lilley = π × Δ,  where Δ = √(A/K) (exchange-anisotropy length).

    From the equilibrium tanh profile mz(x) = tanh(x/Δ):
        |dm/dx|_max = 1/Δ  →  λ_Lilley = π / |dm/dx|_max

    Parameters
    ----------
    m    : VectorField3D — 1D strip (ny=nz=1 ideal) or profile averaged over y/z
    axis : int           — propagation axis (0=x, 1=y, 2=z)
    comp : int           — component to analyse (2=mz for standard easy-z DW)

    Returns
    -------
    (lam, x0) : (float, float)
        lam — Lilley DW width λ [m]
        x0  — DW centre position [m]

    Example
    -------
    >>> lam, x0 = mm.bloch_dw_width(m, axis=0, comp=2)
    >>> import math
    >>> lam_theory = math.pi * math.sqrt(A / K)
    >>> print(f"DW width: measured={lam*1e9:.1f} nm, theory={lam_theory*1e9:.1f} nm")
    """
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2; got {axis!r}")
    if comp not in (0, 1, 2):
        raise ValueError(f"comp must be 0, 1, or 2; got {comp!r}")
    import numpy as _np
    import math as _m
    g = m.grid
    cell_sizes = [g.dx, g.dy, g.dz]
    ds = cell_sizes[axis]

    arr = _np.asarray(to_numpy(m))      # (nz, ny, nx, 3)

    # Average over the two non-propagation axes
    if axis == 0:
        profile = arr.mean(axis=(0, 1))  # (nx, 3)
    elif axis == 1:
        profile = arr.mean(axis=(0, 2))  # (ny, 3)
    else:
        profile = arr.mean(axis=(1, 2))  # (nz, 3)

    mc = profile[:, comp]
    n = len(mc)
    pos = (_np.arange(n) + 0.5) * ds

    dm_dx = _np.gradient(mc, ds)        # [1/m]
    peak_grad = _np.max(_np.abs(dm_dx))

    if peak_grad < 1e-20:
        return 0.0, pos[n // 2]

    # Lilley width: λ = π / |dm/dx|_max = π × Δ
    lam = _m.pi / peak_grad

    # Centre: position of steepest gradient
    x0 = pos[_np.argmax(_np.abs(dm_dx))]
    return float(lam), float(x0)


# ---------------------------------------------------------------------------
# skyrmion_phase_diagram_gpu — convenience D×K sweep on GPU
# ---------------------------------------------------------------------------

def skyrmion_phase_diagram_gpu(
    D_vals,
    K_vals,
    grid=None,
    mat=None,
    *,
    n_relax_steps: int = 200000,
    threshold: float = 5000.0,
    progress: bool = True,
):
    """Sweep DMI strength D and PMA K to map skyrmion stability.

    Each (D, K) point: initialise an inward Neel skyrmion seed → RelaxGPU
    → compute topological charge Q. Returns a list of result dicts.

    Requires GPU build (CUDA).

    Parameters
    ----------
    D_vals         : list[float] — DMI values [J/m²]
    K_vals         : list[float] — PMA K_uniaxial values [J/m³]
    grid           : StructuredGrid | None — defaults to 32×32×1 at 3nm
    mat            : Material | None — defaults to Co/Pt-like (Ms=0.58MA/m, A=15pJ/m)
    n_relax_steps  : max relaxation steps per point
    threshold      : |m×H|_max threshold [A/m] for convergence
    progress       : print progress

    Returns
    -------
    list[dict]  — keys: D, K, Q, is_skyrmion (|Q|>0.5), steps

    Example
    -------
    >>> res = mm.skyrmion_phase_diagram_gpu(
    ...     D_vals=[2e-3, 3e-3, 4e-3],
    ...     K_vals=[0.4e6, 0.6e6, 0.8e6],
    ... )
    >>> for r in res:
    ...     print(f"D={r['D']*1e3:.1f} mJ/m²  K={r['K']/1e6:.1f} MJ/m³  Q={r['Q']:.2f}")
    """
    import math as _m
    import numpy as _np

    if grid is None:
        grid = StructuredGrid(32, 32, 1, 3e-9, 3e-9, 1e-9)
    if mat is None:
        mat = Material()
        mat.Ms         = 0.58e6
        mat.A_exchange = 15e-12
        mat.alpha      = 0.5

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    N = nx * ny * nz
    cell = grid.dx
    cx = nx * cell * 0.5
    cy = ny * cell * 0.5
    pi = _m.pi

    def _init_neel_sky(K_val):
        R = max(2 * cell, _m.pi * _m.sqrt(15e-12 / max(K_val, 1e3)) * 0.5)
        m0 = VectorField3D(grid)
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    rx = (ix + 0.5) * cell - cx
                    ry = (iy + 0.5) * cell - cy
                    r = _m.sqrt(rx*rx + ry*ry)
                    cos_t = _m.cos(pi * r / (2 * R)) if r < 2 * R else -1.0
                    sin_t = _m.sqrt(max(0.0, 1.0 - cos_t**2))
                    lin = ix + nx * (iy + ny * iz)
                    if r < 1e-30:
                        m0[lin] = Vec3(0, 0, 1)
                    else:
                        # inward Neel: favoured by D>0 interfacial DMI
                        m0[lin] = Vec3(-(rx/r)*sin_t, -(ry/r)*sin_t, cos_t)
        return m0

    results = []
    total = len(D_vals) * len(K_vals)
    idx = 0

    for K in K_vals:
        for D in D_vals:
            idx += 1
            if progress:
                print(f"[skyrmion_phase_diagram_gpu] {idx}/{total}  "
                      f"D={D*1e3:.2f} mJ/m²  K={K/1e6:.3f} MJ/m³", flush=True)

            mat.K_uniaxial = K
            mat.easy_axis  = Vec3(0, 0, 1)

            m0 = _init_neel_sky(K)
            Q_init = topological_charge_Q(m0)

            demag = DemagFieldGPU(grid)
            exch  = ExchangeFieldGPU(grid)
            ani   = UniaxialAnisotropyFieldGPU(grid)
            dmi   = InterfacialDMIFieldGPU(grid, D)

            fields = FieldSumGPU()
            fields.add(exch)
            fields.add(ani)
            fields.add(dmi)

            relax = RelaxGPU(grid)
            relax.upload(m0)

            opts = RelaxGPU.Options()
            opts.threshold   = threshold
            opts.max_steps   = n_relax_steps
            opts.check_every = 2000
            opts.dt          = 5e-13
            steps = relax.run(mat, demag, fields, opts)

            m_out = VectorField3D(grid)
            relax.download(m_out)
            m_out.normalize()

            Q = topological_charge_Q(m_out)
            results.append({
                "D": D, "K": K,
                "Q_init": Q_init, "Q": Q,
                "is_skyrmion": abs(Q) > 0.5,
                "steps": steps,
            })

    return results


# ---------------------------------------------------------------------------
# Fast numpy-vectorized initializers
# Replaces triple Python loops (O(N) C++ boundary crossings) with a single
# numpy broadcast + one from_numpy() call. Typical speedup: 20-100x for
# grids > 32x32.
# ---------------------------------------------------------------------------

def neel_skyrmion_np(grid, R, cx=None, cy=None, mz_core=1.0):
    """Inward Neel skyrmion seed, vectorized over the full grid.

    Parameters
    ----------
    grid    : StructuredGrid
    R       : float — skyrmion radius [m]
    cx, cy  : float | None — core x,y position [m] (default: grid centre)
    mz_core : float — mz sign at core (+1 = core up, -1 = core down)

    Returns
    -------
    VectorField3D initialized as an inward Neel skyrmion via numpy broadcast.
    Replaces the manual triple-loop init; typically 20-100x faster.
    """
    import numpy as _np

    nx_, ny_, nz_ = grid.nx, grid.ny, grid.nz
    dx_, dy_ = grid.dx, grid.dy
    if cx is None:
        cx = nx_ * dx_ * 0.5
    if cy is None:
        cy = ny_ * dy_ * 0.5

    rx = (_np.arange(nx_, dtype=_np.float64) + 0.5) * dx_ - cx   # [nx]
    ry = (_np.arange(ny_, dtype=_np.float64) + 0.5) * dy_ - cy   # [ny]
    RX, RY = _np.meshgrid(rx, ry)                                 # [ny, nx]

    r  = _np.maximum(_np.sqrt(RX**2 + RY**2), 1e-30)

    cos_t = _np.where(r < 2.0 * R,
                      float(mz_core) * _np.cos(_np.pi * r / (2.0 * R)),
                      -float(mz_core))
    sin_t = _np.sqrt(_np.maximum(0.0, 1.0 - cos_t**2))

    mx2d = -(RX / r) * sin_t
    my2d = -(RY / r) * sin_t
    mz2d = cos_t

    arr = _np.stack([
        _np.broadcast_to(mx2d[_np.newaxis], (nz_, ny_, nx_)).copy(),
        _np.broadcast_to(my2d[_np.newaxis], (nz_, ny_, nx_)).copy(),
        _np.broadcast_to(mz2d[_np.newaxis], (nz_, ny_, nx_)).copy(),
    ], axis=-1)

    m = VectorField3D(grid)
    from_numpy(m, arr)
    return m


def bloch_dw_np(grid, Delta, axis=0):
    """Analytical Bloch DW profile, vectorized (tanh/sech via numpy).

    Initializes mz = tanh(x/Delta), my = sech(x/Delta) along axis.
    This IS the LLG equilibrium for Exchange + UniaxialAnisotropy in 1D.

    Parameters
    ----------
    grid  : StructuredGrid
    Delta : float — DW parameter Delta = sqrt(A/K) [m]
    axis  : int   — propagation axis (0=x, 1=y, 2=z)

    Returns
    -------
    VectorField3D with analytical Bloch DW profile. 20-100x faster than
    a manual Python loop over cells.
    """
    import numpy as _np

    nx_, ny_, nz_ = grid.nx, grid.ny, grid.nz
    cell_sizes = [grid.dx, grid.dy, grid.dz]
    n_ax = [nx_, ny_, nz_][axis]
    ds   = cell_sizes[axis]

    xi    = (_np.arange(n_ax, dtype=_np.float64) + 0.5 - n_ax * 0.5) * ds
    mz_1d = _np.tanh(xi / Delta)
    my_1d = 1.0 / _np.cosh(xi / Delta)
    mx_1d = _np.zeros(n_ax, dtype=_np.float64)

    if axis == 0:
        mx = _np.broadcast_to(mx_1d[_np.newaxis, _np.newaxis, :], (nz_, ny_, nx_)).copy()
        my = _np.broadcast_to(my_1d[_np.newaxis, _np.newaxis, :], (nz_, ny_, nx_)).copy()
        mz = _np.broadcast_to(mz_1d[_np.newaxis, _np.newaxis, :], (nz_, ny_, nx_)).copy()
    elif axis == 1:
        mx = _np.zeros((nz_, ny_, nx_), dtype=_np.float64)
        my = _np.broadcast_to(my_1d[_np.newaxis, :, _np.newaxis], (nz_, ny_, nx_)).copy()
        mz = _np.broadcast_to(mz_1d[_np.newaxis, :, _np.newaxis], (nz_, ny_, nx_)).copy()
    else:
        mx = _np.zeros((nz_, ny_, nx_), dtype=_np.float64)
        my = _np.broadcast_to(my_1d[:, _np.newaxis, _np.newaxis], (nz_, ny_, nx_)).copy()
        mz = _np.broadcast_to(mz_1d[:, _np.newaxis, _np.newaxis], (nz_, ny_, nx_)).copy()

    arr = _np.stack([mx, my, mz], axis=-1)

    m = VectorField3D(grid)
    from_numpy(m, arr)
    return m


# ---------------------------------------------------------------------------
# recommend_integrator -- choose the best GPU integrator for a given scenario
# ---------------------------------------------------------------------------

def recommend_integrator(
    mat,
    T_K: float = 0.0,
    goal: str = "dynamics",
    dt: float = None,
    B_eff_T: float = 0.1,
    t_end: float = None,
    verbose: bool = True,
):
    """Recommend the best GPU integrator for a given simulation scenario.

    Parameters
    ----------
    mat : Material
        Simulation material.  Uses ``mat.alpha`` for damping.
    T_K : float
        Temperature in Kelvin.  T_K > 0 mandates HeunIntegratorGPU (SLLG).
    goal : str
        ``"relax"`` -- converge to equilibrium; the trajectory does not matter.
        ``"dynamics"`` -- track the LLG trajectory accurately (default).
    dt : float, optional
        Proposed fixed timestep [s].  Used to compute Heun phase error.
    B_eff_T : float
        Approximate magnitude of the effective field [T] for the phase-error
        estimate.  Default 0.1 T (100 mT) -- typical for Permalloy at moderate
        applied field.  Ignored when T_K > 0 or goal == "relax".
    t_end : float, optional
        Simulation duration [s].  Used together with *dt* for the total
        accumulated phase-error estimate.  If omitted, the estimate is shown
        per 1 ns.
    verbose : bool
        If True (default), print a formatted recommendation report.

    Returns
    -------
    dict
        ``integrator``  : str   -- recommended class name.
        ``reason``      : str   -- primary justification.
        ``warning``     : str   -- potential pitfall, or empty string.
        ``heun_ok``     : bool  -- True when HeunIntegratorGPU is acceptable.
        ``phase_err_deg`` : float or None -- Heun phase-error estimate [deg].
        ``usage``       : str   -- minimal code snippet.

    Notes
    -----
    **Heun phase-error formula (T=0, linear precession)**::

        epsilon_phase = omega^3 * dt^2 * t_end / 6   [rad]

    where ``omega = gamma_0 * B_eff_T``.  This is the global (accumulated)
    truncation error for the 2nd-order Heun method applied to harmonic
    precession.  For typical CUDA timesteps (dt ~ 0.1-1 ps), this error is
    < 0.1 deg over 1 ns -- negligible in practice.

    RK4 (4th-order) accumulates O((omega*dt)^4 * t_end) phase error, roughly
    4 orders of magnitude smaller than Heun for the same dt.

    **Float32 build (cmake --preset windows-msvc-cuda-f32)**:
    All three integrators are 4-8× faster than the f64 build on Blackwell GPUs
    (Tensor Core FFT). This doesn't change *which* integrator is recommended
    (the physics requirements are precision-independent), but it makes
    fixed-step Heun at SP#4 reach ~0.086 ms/step vs mumax3's 0.482 ms/step.
    Use f32 when μMAG-level accuracy is sufficient (error < 2% vs f64).

    Examples
    --------
    >>> import micromag as mm
    >>> mat = mm.Material.permalloy()
    >>> rec = mm.recommend_integrator(mat, T_K=0, goal="relax", dt=5e-13)
    >>> rec = mm.recommend_integrator(mat, T_K=300, dt=1e-13)
    >>> rec = mm.recommend_integrator(mat, T_K=0, goal="dynamics",
    ...                               dt=5e-13, t_end=1e-9, B_eff_T=0.05)
    """
    import math
    import textwrap

    GAMMA0 = 1.7595e11   # rad / (T s)
    alpha  = float(mat.alpha)
    t_ref  = t_end if t_end is not None else 1e-9   # fallback: 1 ns

    # ------------------------------------------------------------------
    # Compute Heun phase-error estimate (only meaningful for dynamics)
    # ------------------------------------------------------------------
    phase_err_deg = None
    if dt is not None and goal == "dynamics" and T_K == 0.0:
        omega          = GAMMA0 * float(B_eff_T)
        eps_rad        = (omega ** 3) * (dt ** 2) * t_ref / 6.0
        phase_err_deg  = math.degrees(eps_rad)

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------
    if T_K > 0.0:
        # Rule 1: finite temperature -- Heun is the only correct SDE method
        rec     = "HeunIntegratorGPU"
        heun_ok = True
        reason  = (
            f"T_K={T_K:.1f} K > 0: the Stratonovich Heun method is the "
            "only formally correct discretisation of the SLLG equation. "
            "RK4/RK45 do not preserve the Stratonovich convention and give "
            "wrong thermal averages."
        )
        warning = ""
        usage   = (
            f"integ = mm.HeunIntegratorGPU(grid, dt, seed=42)\n"
            f"integ.upload(m0)\n"
            f"integ.step(mat, demag, fields, T_K={T_K:.1f})"
        )

    elif goal in ("relax", "converge"):
        # Rule 2: convergence to equilibrium -- path is irrelevant
        rec     = "HeunIntegratorGPU"
        heun_ok = True
        reason  = (
            "goal='relax': only the final equilibrium state matters, not "
            "the trajectory. Heun (2 field evals/step) is exactly 2x "
            "faster than RK4 (4 evals/step) with no accuracy penalty for "
            "the converged state. CUDA Graph replay is active for T_K=0."
        )
        warning = (
            "Call integ.invalidate_graph() after changing fields between "
            "steps (e.g. field sweep in a hysteresis loop)."
        )
        usage   = (
            "integ = mm.HeunIntegratorGPU(grid, dt)\n"
            "integ.upload(m0)\n"
            "mm.run_until_converged_gpu(integ, mat, demag, fields, m0)"
        )

    elif alpha >= 0.3:
        # Rule 3: overdamped -- precession decays rapidly
        rec     = "HeunIntegratorGPU"
        heun_ok = True
        reason  = (
            f"alpha={alpha:.3f} >= 0.3 (overdamped regime): precessional "
            "oscillations are quenched within ~1/alpha precession cycles. "
            "Heun's 2nd-order truncation error is negligible; the 2x "
            "step-time advantage is effectively free."
        )
        warning = ""
        usage   = (
            "integ = mm.HeunIntegratorGPU(grid, dt)\n"
            "integ.upload(m0)\n"
            "integ.step(mat, demag, fields, T_K=0.0)"
        )

    elif alpha >= 0.05:
        # Rule 4: moderate damping -- decide from phase error if available
        if phase_err_deg is not None and phase_err_deg < 1.0:
            rec     = "HeunIntegratorGPU"
            heun_ok = True
            reason  = (
                f"alpha={alpha:.3f} (moderate damping). Computed Heun "
                f"phase error at dt={dt:.1e} s over "
                f"{'1 ns (reference)' if t_end is None else f'{t_ref:.1e} s'}: "
                f"{phase_err_deg:.4f} deg -- negligible. "
                "Heun is acceptable and 2x faster than RK4."
            )
            warning = (
                "For significantly larger dt or longer t_end, the 2nd-order "
                "error grows as dt^2 * t_end. Verify with RK4 if in doubt."
            )
            usage   = (
                "integ = mm.HeunIntegratorGPU(grid, dt)\n"
                "integ.upload(m0)\n"
                "integ.step(mat, demag, fields, T_K=0.0)"
            )
        elif phase_err_deg is not None and phase_err_deg >= 1.0:
            rec     = "RK4IntegratorGPU"
            heun_ok = False
            reason  = (
                f"alpha={alpha:.3f} (moderate damping). Computed Heun "
                f"phase error at dt={dt:.1e} s over "
                f"{'1 ns (reference)' if t_end is None else f'{t_ref:.1e} s'}: "
                f"{phase_err_deg:.2f} deg -- exceeds 1 deg threshold. "
                "RK4 (4th-order) reduces this by ~(omega*dt)^2 ~ "
                f"{(GAMMA0*B_eff_T*dt)**2:.0e}x."
            )
            warning = (
                f"Heun phase error {phase_err_deg:.1f} deg. "
                "Reduce dt or switch to RK4."
            )
            usage   = (
                "integ = mm.RK4IntegratorGPU(grid, dt)\n"
                "integ.upload(m0)\n"
                "integ.step(mat, demag, fields)"
            )
        else:
            # No dt given -- conservative default
            rec     = "RK4IntegratorGPU"
            heun_ok = False
            reason  = (
                f"alpha={alpha:.3f} (moderate damping, no dt provided). "
                "RK4 (4th-order) is the safe default for trajectory "
                "accuracy. Provide dt and B_eff_T to get a quantitative "
                "Heun phase-error estimate."
            )
            warning = ""
            usage   = (
                "integ = mm.RK4IntegratorGPU(grid, dt)\n"
                "integ.upload(m0)\n"
                "integ.step(mat, demag, fields)"
            )

    else:
        # Rule 5: low damping (alpha < 0.05) -- adaptive RK45 preferred
        if phase_err_deg is not None and phase_err_deg < 0.1:
            # Fixed dt is fine for this regime too
            rec     = "RK45IntegratorGPU"
            heun_ok = False
            reason  = (
                f"alpha={alpha:.4f} (low damping). Heun phase error "
                f"{phase_err_deg:.4f} deg is negligible at the given dt, "
                "but RK45 (adaptive DOPRI5/FSAL) is still preferred: it "
                "automatically controls the error and often requires fewer "
                "total field evaluations than a fixed-step method."
            )
            warning = ""
            usage   = (
                "integ = mm.RK45IntegratorGPU(grid)\n"
                "integ.upload(m0)\n"
                "integ.step(mat, demag, fields)  # adaptive dt"
            )
        else:
            rec     = "RK45IntegratorGPU"
            heun_ok = False
            reason  = (
                f"alpha={alpha:.4f} (low damping): precessional dynamics "
                "require high phase accuracy over many cycles. Adaptive "
                "RK45 (DOPRI5/FSAL) adjusts dt to stay within a user-set "
                "tolerance -- fewest field evaluations for a given accuracy. "
                "Fixed-step Heun at this alpha would need dt ~10x smaller "
                "than RK4 to match accuracy, erasing the 2x eval advantage."
            )
            warning = (
                "For fixed-dt RK4, use dt <= 1e-12 s (1 ps) at alpha<0.05 "
                "to keep numerical precession phase drift below 1%."
            )
            usage   = (
                "integ = mm.RK45IntegratorGPU(grid)\n"
                "integ.upload(m0)\n"
                "integ.step(mat, demag, fields)  # adaptive dt"
            )

    # ------------------------------------------------------------------
    # Verbose report
    # ------------------------------------------------------------------
    if verbose:
        W = 62
        def wrap(s):
            return "\n".join(
                "  " + line for line in textwrap.wrap(s, W - 4)
            )

        print("=" * W)
        print("  Integrator Recommendation")
        print("=" * W)
        # Input summary
        print(f"  alpha  = {alpha:.4f}    T_K = {T_K:.1f} K    goal = '{goal}'")
        if dt is not None:
            t_lbl = f"{t_ref:.1e} s" + ("" if t_end is not None else " (ref)")
            print(f"  dt     = {dt:.2e} s   B_eff ~ {B_eff_T*1e3:.0f} mT   t_end = {t_lbl}")
        print()

        # Recommendation
        print(f"  => {rec}")
        print()
        print(wrap(reason))

        # Phase error
        if phase_err_deg is not None:
            t_lbl = f"{t_ref:.1e} s" + ("" if t_end is not None else " (1 ns ref)")
            lvl   = ("negligible" if phase_err_deg < 0.1 else
                     "acceptable" if phase_err_deg < 1.0 else
                     "SIGNIFICANT")
            print()
            print(f"  Heun phase error over {t_lbl}:")
            print(f"    {phase_err_deg:.5f} deg  [{lvl}]")
            print(f"    formula: omega^3 * dt^2 * t / 6")
            print(f"    omega = gamma_0 * B_eff = {GAMMA0*B_eff_T:.3e} rad/s")

        # Warning
        if warning:
            print()
            print("  [!]", end=" ")
            first = True
            for line in textwrap.wrap(warning, W - 6):
                if first:
                    print(line); first = False
                else:
                    print("      " + line)

        # Usage
        print()
        print("  Usage:")
        for line in usage.split("\n"):
            print(f"    {line}")
        print("=" * W)

    return dict(
        integrator=rec,
        reason=reason,
        warning=warning,
        heun_ok=heun_ok,
        phase_err_deg=phase_err_deg,
        usage=usage,
    )


