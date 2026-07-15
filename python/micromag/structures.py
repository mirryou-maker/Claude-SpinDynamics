"""Convergence observables, regions, hysteresis loops (CPU+GPU), layer stacks (Phases O, P, U, X, Y)

Split from the former monolithic micromag/__init__.py; public names are
re-exported by the package __init__, so `import micromag as mm` is unchanged.
"""
import math as _math
import numpy as _np

from _micromag import *          # C++ core names  # noqa: F401,F403


# ===========================================================================
# Phase O — Convergence observables + energy breakdown + B_eff
# ===========================================================================

def max_angle(m) -> float:
    """Maximum angle [degrees] between adjacent cells in the magnetization field.

    Equivalent to mumax3's MaxAngle quantity.  Used as a convergence criterion:
    a fully relaxed state typically has MaxAngle < 1 degree.

    Checks all 6 nearest-neighbour pairs (+/-x, +/-y, +/-z).

    Parameters
    ----------
    m : VectorField3D

    Returns
    -------
    float -- max inter-cell angle in degrees (0 for single-cell grids)

    Example
    -------
    >>> mm.relax(m, mat, heff)
    >>> print(f"MaxAngle = {mm.max_angle(m):.2f} deg")
    """
    m_np = _np.asarray(to_numpy(m))   # (nz, ny, nx, 3)
    nz, ny, nx = m_np.shape[:3]
    min_dot = 1.0
    if nx > 1:
        d = (m_np[:, :, :-1, :] * m_np[:, :, 1:, :]).sum(-1)
        v = float(d.min())
        if v < min_dot:
            min_dot = v
    if ny > 1:
        d = (m_np[:, :-1, :, :] * m_np[:, 1:, :, :]).sum(-1)
        v = float(d.min())
        if v < min_dot:
            min_dot = v
    if nz > 1:
        d = (m_np[:-1, :, :, :] * m_np[1:, :, :, :]).sum(-1)
        v = float(d.min())
        if v < min_dot:
            min_dot = v
    min_dot = max(-1.0, min(1.0, min_dot))
    import math as _m2
    return float(_m2.degrees(_m2.acos(min_dot)))


def B_eff(m, mat, heff):
    """Effective magnetic flux density B_eff = mu0 * H_eff [T].

    Equivalent to mumax3's B_eff quantity.  Returns the total effective
    field scaled by mu0, useful for plotting field distributions in Tesla.

    Parameters
    ----------
    m    : VectorField3D
    mat  : Material
    heff : EffectiveFieldSum

    Returns
    -------
    VectorField3D -- B_eff [T] at each cell

    Example
    -------
    >>> B = mm.B_eff(m, mat, heff)
    >>> B_np = mm.to_numpy(B)    # (nz, ny, nx, 3) in Tesla
    """
    mu0 = 4e-7 * _math.pi
    H = VectorField3D(m.grid)
    heff.compute(m, mat, H)
    H_np = _np.asarray(to_numpy(H))
    result = VectorField3D(m.grid)
    from_numpy(result, H_np * mu0)
    return result


def energy_table(m, mat, heff) -> dict:
    """Per-term energy breakdown [J] for all fields in an EffectiveFieldSum.

    Equivalent to querying mumax3's E_Zeeman, E_exch, E_demag, E_anis, ...
    individually.  Returns a dict with each term's name as key and energy [J]
    as value, plus the key 'total' for the sum.

    Parameters
    ----------
    m    : VectorField3D
    mat  : Material
    heff : EffectiveFieldSum

    Returns
    -------
    dict[str, float] -- {term_name: energy_J, ..., 'total': total_J}

    Example
    -------
    >>> E = mm.energy_table(m, mat, heff)
    >>> for k, v in E.items():
    ...     print(f"  {k:20s} = {v:.4e} J")
    """
    result = {}
    total = 0.0
    for term in heff.terms:
        name = term.name
        # handle duplicate names (e.g., two Zeeman fields)
        base = name
        idx = 1
        while name in result:
            name = f"{base}_{idx}"
            idx += 1
        e = float(term.energy(m, mat))
        result[name] = e
        total += e
    result["total"] = total
    return result


# ===========================================================================
# Phase P — FrozenSpins, DefRegion, (SurfaceAnisotropyField via C++ below)
# ===========================================================================

def def_region(region_map, region_id: int, geom_mask):
    """Assign region_id to all cells where geom_mask > 0.5.

    Equivalent to mumax3's DefRegion(id, shape).  Cells outside the shape
    are left unchanged (non-destructive: only writes into cells where mask > 0.5).

    Parameters
    ----------
    region_map : RegionMap  -- modified in place
    region_id  : int        -- region ID to assign (1..254; 0 = unassigned)
    geom_mask  : GeomMask   -- shape (cells > 0.5 get region_id)

    Example
    -------
    >>> rm = mm.RegionMap(grid)
    >>> mm.def_region(rm, 1, mm.circle(grid, 100e-9))
    >>> mm.def_region(rm, 2, mm.x_range(grid, 50e-9, 200e-9))
    """
    g = region_map.grid
    n = g.nx * g.ny * g.nz
    rid = min(max(int(region_id), 0), 255)
    for i in range(n):
        if geom_mask[i] > 0.5:
            region_map[i] = rid


def new_region_map(grid, *region_specs):
    """Build a RegionMap from (region_id, geom_mask) pairs.

    Convenience wrapper around def_region() for building a complete
    RegionMap from multiple geometry shapes in one call.

    Parameters
    ----------
    grid          : StructuredGrid
    *region_specs : (int, GeomMask) pairs -- assigned in order

    Returns
    -------
    RegionMap -- cells not covered by any spec have ID 0

    Example
    -------
    >>> rm = mm.new_region_map(
    ...     grid,
    ...     (1, mm.circle(grid, 100e-9)),
    ...     (2, mm.y_range(grid, -50e-9, 0)),
    ... )
    """
    rm = RegionMap(grid)
    for rid, geom in region_specs:
        def_region(rm, rid, geom)
    return rm


class FrozenIntegrator:
    """Integrator wrapper that holds selected cells fixed (pinned spins).

    Equivalent to mumax3's FreezeSpins(region) — after each LLG step the
    magnetization of frozen cells is restored to their initial values,
    effectively removing them from the dynamics.

    Works with any CPU integrator (RK4, RK45, Heun).  GPU integrators are
    not supported (the step executes entirely on-device; use a GeomMask on
    the effective field instead for GPU pinning).

    Parameters
    ----------
    integ       : RK4Integrator | RK45Integrator | HeunIntegrator
    freeze_mask : GeomMask -- cells with mask > 0.5 are frozen
    m_init      : VectorField3D -- initial (pinned) magnetization values

    Example
    -------
    >>> # Pin the left quarter of the strip
    >>> pin = mm.x_range(grid, -Lx/2, -Lx/4)
    >>> frozen = mm.FrozenIntegrator(integ, pin, m)
    >>> mm.run(frozen, m, mat, heff, t_total=1e-9)
    >>> # Cells outside pin evolve; cells inside pin stay fixed.
    """

    def __init__(self, integ, freeze_mask, m_init):
        self._integ = integ
        g = freeze_mask.grid
        nx, ny, nz = g.nx, g.ny, g.nz
        n = nx * ny * nz
        # Snapshot frozen cells as numpy array for fast restore.
        # Store (linear_idx, (x, y, z)) tuples.
        m_np = _np.asarray(to_numpy(m_init))   # (nz, ny, nx, 3)
        self._frozen = []   # list of (lin_idx, x, y, z)
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    lin = ix + nx*(iy + ny*iz)
                    if freeze_mask[lin] > 0.5:
                        v = m_np[iz, iy, ix, :]
                        self._frozen.append((iz, iy, ix, float(v[0]), float(v[1]), float(v[2])))

    # ------------------------------------------------------------------
    # Forward dt / set_dt / dt property to wrapped integrator
    # ------------------------------------------------------------------
    @property
    def dt(self):
        return self._integ.dt

    def set_dt(self, dt):
        self._integ.set_dt(dt)

    # ------------------------------------------------------------------
    # step — performs the LLG step then restores frozen cells
    # ------------------------------------------------------------------
    def step(self, m, mat, heff, stt=None):
        if stt is not None:
            result = self._integ.step(m, mat, heff, stt)
        else:
            result = self._integ.step(m, mat, heff)
        if self._frozen:
            # Read current state, overwrite frozen cells, write back
            m_np = _np.asarray(to_numpy(m)).copy()
            for (iz, iy, ix, vx, vy, vz) in self._frozen:
                m_np[iz, iy, ix, 0] = vx
                m_np[iz, iy, ix, 1] = vy
                m_np[iz, iy, ix, 2] = vz
            from_numpy(m, m_np)
        return result

    # Expose the underlying integrator for inspection
    @property
    def integrator(self):
        return self._integ


# ===========================================================================
# Phase U — Hysteresis loop automation
# ===========================================================================

def hysteresis_loop(m, mat, heff, integ, H_list, zee,
                    axis: str = 'x',
                    tol_deg: float = 1.0,
                    max_steps: int = 500_000,
                    check_interval: int = 200,
                    reset_m=None,
                    verbose: bool = False):
    """Sweep a ZeemanField through H_list and relax at each point.

    Equivalent to mumax3's ``for H in H_list { SetB_ext(...); RunWhile(...) }``
    pattern.  At each field value the ZeemanField ``zee`` is updated and the
    system is relaxed via ``run_until_converged``.

    Parameters
    ----------
    m              : VectorField3D -- modified in place
    mat            : Material
    heff           : EffectiveFieldSum -- must already contain ``zee``
    integ          : integrator (RK4, RK45, Heun, or FrozenIntegrator)
    H_list         : 1-D iterable of floats [A/m] OR shape-(N,3) array of Vec3
                     If 1-D, ``axis`` controls which component is set.
    zee            : ZeemanField -- its H_ext is modified at each step
    axis           : 'x', 'y', or 'z' — applied axis for scalar H_list
    tol_deg        : convergence threshold [degrees] passed to run_until_converged
    max_steps      : max LLG steps per field point
    check_interval : convergence check interval
    reset_m        : VectorField3D (optional) — if given, m is reset to this
                     state before relaxing at each field value (useful for
                     major loop starting from saturation)
    verbose        : print progress (field value + mean magnetization)

    Returns
    -------
    dict with numpy arrays, all shape (N,) where N = len(H_list):
      "H"       : applied field magnitude [A/m]  (signed, along ``axis``)
      "Hvec"    : applied field Vec3 [A/m], shape (N, 3)
      "mx", "my", "mz" : mean magnetization components
      "E_total" : total energy [J]

    Example — SP#3-style hysteresis loop
    -------------------------------------
    >>> zee = mm.ZeemanField()
    >>> heff.add(zee)
    >>> mu0 = 4e-7 * math.pi
    >>> H_list = np.linspace(0.1/mu0, -0.1/mu0, 201)  # ±100 mT
    >>> res = mm.hysteresis_loop(m, mat, heff, integ, H_list, zee,
    ...                          axis='x', tol_deg=1.0)
    >>> plt.plot(res["H"] * mu0 * 1e3, res["mx"])  # mT vs mx
    """
    from .sweep import run_until_converged  # lazy: avoids structures<->sweep cycle
    ax_idx = {'x': 0, 'y': 1, 'z': 2}.get(axis.lower(), 0)

    H_arr  = _np.asarray(H_list, dtype=float)
    is_vec = (H_arr.ndim == 2 and H_arr.shape[1] == 3)
    N      = len(H_arr) if not is_vec else H_arr.shape[0]

    H_mag_out = _np.zeros(N)
    Hvec_out  = _np.zeros((N, 3))
    mx_out    = _np.zeros(N)
    my_out    = _np.zeros(N)
    mz_out    = _np.zeros(N)
    E_out     = _np.zeros(N)

    # Snapshot reset state once (before any modifications)
    if reset_m is not None:
        reset_np = _np.asarray(to_numpy(reset_m)).copy()

    for i in range(N):
        if is_vec:
            hx, hy, hz = float(H_arr[i, 0]), float(H_arr[i, 1]), float(H_arr[i, 2])
        else:
            hx = hy = hz = 0.0
            if ax_idx == 0:   hx = float(H_arr[i])
            elif ax_idx == 1: hy = float(H_arr[i])
            else:             hz = float(H_arr[i])

        zee.H_ext = Vec3(hx, hy, hz)

        if reset_m is not None:
            from_numpy(m, reset_np)

        # Always take at least check_interval steps so a uniform-m state can
        # react to the new H before the convergence criterion is evaluated.
        for _ in range(check_interval):
            integ.step(m, mat, heff)

        run_until_converged(integ, m, mat, heff,
                            tol_deg=tol_deg,
                            max_steps=max(0, max_steps - check_interval),
                            check_interval=check_interval)

        mx_out[i], my_out[i], mz_out[i] = mean_magnetization(m)
        E_out[i]     = heff.total_energy(m, mat)
        H_mag_out[i] = float(H_arr[i]) if not is_vec else float(
            _np.sqrt(hx*hx + hy*hy + hz*hz))
        Hvec_out[i]  = [hx, hy, hz]

        if verbose:
            print(f"  [{i+1:3d}/{N}] H={H_mag_out[i]:+.3e} A/m"
                  f"  mx={mx_out[i]:+.4f}  my={my_out[i]:+.4f}"
                  f"  mz={mz_out[i]:+.4f}")

    return {
        "H":       H_mag_out,
        "Hvec":    Hvec_out,
        "mx":      mx_out,
        "my":      my_out,
        "mz":      mz_out,
        "E_total": E_out,
    }


# ===========================================================================
# Phase X — GPU convergence + GPU hysteresis loop
# ===========================================================================

def run_until_converged_gpu(integ, mat, demag, fsum, m_cpu,
                             tol_deg: float = 1.0,
                             max_steps: int = 500_000,
                             check_interval: int = 200,
                             torques=None,
                             verbose: bool = False):
    """Run LLG on GPU until max_angle(m) drops below *tol_deg*, then stop.

    The GPU integrator state is advanced in batches of *check_interval* steps.
    Convergence is checked by downloading m to CPU and calling max_angle().

    Parameters
    ----------
    integ          : RK4IntegratorGPU or RK45IntegratorGPU (already uploaded)
    mat            : Material
    demag          : IDemagGPU (DemagFieldGPU or DemagFieldPeriodicGPU)
    fsum           : FieldSumGPU (exchange, zeeman, anisotropy, etc.)
    m_cpu          : VectorField3D — scratch buffer for downloads (modified)
    tol_deg        : convergence threshold [degrees] (default 1.0)
    max_steps      : hard step limit (default 500 000)
    check_interval : steps per convergence check (default 200)
    torques        : SpinTorqueSumGPU or None
    verbose        : print progress each check

    Returns
    -------
    dict with keys: "converged", "steps", "max_angle", "t_sim"
    """
    step_count = 0
    t_sim = 0.0

    # Use GPU-side max_angle if the integrator supports it (avoids full D2H).
    _has_gpu_angle = hasattr(integ, "max_angle_gpu")

    def _check_angle():
        if _has_gpu_angle:
            return integ.max_angle_gpu()
        integ.download(m_cpu)
        return max_angle(m_cpu)

    # Always run at least check_interval steps first so a uniform-m state
    # can react to the current H before the convergence criterion fires.
    n_warmup = min(check_interval, max_steps)
    for _ in range(n_warmup):
        if torques is not None:
            integ.step(mat, demag, fsum, torques)
        else:
            integ.step(mat, demag, fsum)
    step_count = n_warmup
    try:
        t_sim = step_count * integ.dt
    except AttributeError:
        t_sim = float("nan")

    angle = _check_angle()
    converged = angle < tol_deg

    while not converged and step_count < max_steps:
        n_batch = min(check_interval, max_steps - step_count)
        for _ in range(n_batch):
            if torques is not None:
                integ.step(mat, demag, fsum, torques)
            else:
                integ.step(mat, demag, fsum)
        step_count += n_batch
        try:
            t_sim = step_count * integ.dt
        except AttributeError:
            t_sim = float("nan")

        angle = _check_angle()
        converged = angle < tol_deg

        if verbose:
            print(f"  step {step_count:7d}  max_angle = {angle:.4f} deg"
                  f"  (tol = {tol_deg:.4f} deg)")

    return {
        "converged":  converged,
        "steps":      step_count,
        "max_angle":  angle,
        "t_sim":      t_sim,
    }


def gpu_hysteresis_loop(integ, mat, demag, fsum, zee_gpu, H_list,
                         m_cpu,
                         axis: str = 'x',
                         tol_deg: float = 1.0,
                         max_steps: int = 500_000,
                         check_interval: int = 200,
                         reset_m=None,
                         torques=None,
                         verbose: bool = False):
    """GPU-accelerated hysteresis loop: sweep ZeemanFieldGPU through H_list.

    Equivalent to ``hysteresis_loop`` but uses GPU integrators for all LLG
    stepping. The GPU integrator must already have been initialised via
    ``integ.upload(m0)`` before calling this function.

    *zee_gpu must already be added to fsum* so its H_ext update takes effect
    at each field point.

    Parameters
    ----------
    integ          : RK4IntegratorGPU or RK45IntegratorGPU (already uploaded)
    mat            : Material
    demag          : IDemagGPU
    fsum           : FieldSumGPU containing exchange, zeeman, anisotropy, etc.
    zee_gpu        : ZeemanFieldGPU — H_ext is updated at each H point
    H_list         : 1D array of scalar [A/m], OR shape-(N,3) Vec3 array
    m_cpu          : VectorField3D — download scratch (modified in-place)
    axis           : 'x' | 'y' | 'z' — axis for 1D H_list (default 'x')
    tol_deg        : convergence tolerance [deg] (default 1.0)
    max_steps      : hard step limit per H point (default 500 000)
    check_interval : steps between convergence checks (default 200)
    reset_m        : VectorField3D snapshot to reset GPU state before each point
    torques        : SpinTorqueSumGPU or None
    verbose        : print per-point progress

    Returns
    -------
    dict with keys: "H", "Hvec", "mx", "my", "mz"
    """
    import numpy as _np
    import math as _m

    H_arr  = _np.asarray(H_list, dtype=float)
    is_vec = H_arr.ndim == 2 and H_arr.shape[1] == 3
    N      = len(H_arr)
    ax_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]

    reset_np = None
    if reset_m is not None:
        reset_np = _np.asarray(to_numpy(reset_m)).copy()

    mx_out   = _np.zeros(N)
    my_out   = _np.zeros(N)
    mz_out   = _np.zeros(N)
    H_out    = _np.zeros(N)
    Hvec_out = _np.zeros((N, 3))

    for i in range(N):
        if is_vec:
            hx, hy, hz = float(H_arr[i, 0]), float(H_arr[i, 1]), float(H_arr[i, 2])
        else:
            hx = hy = hz = 0.0
            if ax_idx == 0:   hx = float(H_arr[i])
            elif ax_idx == 1: hy = float(H_arr[i])
            else:             hz = float(H_arr[i])

        zee_gpu.H_ext = Vec3(hx, hy, hz)
        # The FieldSumGPU CUDA-graph path bakes H_ext into the captured launch
        # args and its staleness test omits H_ext, so a bare H_ext change would
        # replay the stale field (silent wrong physics). Force a re-capture.
        if hasattr(integ, "invalidate_graph"):
            integ.invalidate_graph()

        if reset_m is not None:
            from_numpy(m_cpu, reset_np)
            integ.upload(m_cpu)

        run_until_converged_gpu(integ, mat, demag, fsum, m_cpu,
                                tol_deg=tol_deg,
                                max_steps=max_steps,
                                check_interval=check_interval,
                                torques=torques)

        integ.download(m_cpu)
        mx_out[i], my_out[i], mz_out[i] = mean_magnetization(m_cpu)
        H_out[i]    = float(H_arr[i]) if not is_vec else float(
            _m.sqrt(hx*hx + hy*hy + hz*hz))
        Hvec_out[i] = [hx, hy, hz]

        if verbose:
            print(f"  [{i+1:3d}/{N}] H={H_out[i]:+.3e} A/m"
                  f"  mx={mx_out[i]:+.4f}  my={my_out[i]:+.4f}"
                  f"  mz={mz_out[i]:+.4f}")

    return {
        "H":    H_out,
        "Hvec": Hvec_out,
        "mx":   mx_out,
        "my":   my_out,
        "mz":   mz_out,
    }


# ===========================================================================
# Phase Y — Multi-layer material stack builders
# ===========================================================================

def _matf_set_cell(matf, idx, mat):
    """Set scalar parameters of one MaterialField3D cell from a Material."""
    matf.Ms_field[idx]    = mat.Ms
    matf.A_field[idx]     = mat.A_exchange
    matf.K_field[idx]     = mat.K_uniaxial
    matf.alpha_field[idx] = mat.alpha
    # easy_axis is a VectorField3D: set per-cell via its component ScalarFields
    # VectorField3D does not expose __setitem__; use the component trick if needed.
    # For most multilayer stacks the easy axis is identical across layers, so
    # MaterialField3D.set_uniform sets it correctly; here we skip it.


def bilayer(grid, mat_top, mat_bot, t_top, t_bot=None):
    """Create a MaterialField3D for a two-layer thin-film stack (z-stacked).

    Top layer occupies the highest z cells; bottom layer the rest.

    Parameters
    ----------
    grid    : StructuredGrid
    mat_top : Material — top layer (highest z)
    mat_bot : Material — bottom layer
    t_top   : float — top layer thickness [m]
    t_bot   : ignored (reserved)

    Returns
    -------
    MaterialField3D
    """
    from _micromag import MaterialField3D as _MF3D
    matf = _MF3D(grid, mat_bot)
    dz   = grid.dz
    nz   = grid.nz
    total_z = nz * dz
    z_top_start = total_z - t_top
    for iz in range(nz):
        z_centre = (iz + 0.5) * dz
        if z_centre >= z_top_start:
            for iy in range(grid.ny):
                for ix in range(grid.nx):
                    idx = ix + grid.nx * (iy + grid.ny * iz)
                    _matf_set_cell(matf, idx, mat_top)
    return matf


def trilayer(grid, mat_top, mat_mid, mat_bot, t_top, t_mid, t_bot=None):
    """Create a MaterialField3D for a three-layer thin-film stack.

    Layers are stacked along z (top=highest z). t_bot is implicit
    (remaining cells after top+mid).

    Parameters
    ----------
    grid    : StructuredGrid
    mat_top, mat_mid, mat_bot : Material
    t_top, t_mid : float [m]
    t_bot   : ignored (reserved)

    Returns
    -------
    MaterialField3D
    """
    from _micromag import MaterialField3D as _MF3D
    matf = _MF3D(grid, mat_bot)
    dz   = grid.dz
    nz   = grid.nz
    total_z = nz * dz
    z_mid_start = total_z - t_top - t_mid
    z_top_start = total_z - t_top
    for iz in range(nz):
        z_centre = (iz + 0.5) * dz
        if z_centre >= z_top_start:
            m_cell = mat_top
        elif z_centre >= z_mid_start:
            m_cell = mat_mid
        else:
            m_cell = mat_bot
        for iy in range(grid.ny):
            for ix in range(grid.nx):
                idx = ix + grid.nx * (iy + grid.ny * iz)
                _matf_set_cell(matf, idx, m_cell)
    return matf


def saf_stack(grid, mat_fl, mat_rl, t_fl, t_rl, t_spacer=None):
    """Create a MaterialField3D for a Synthetic Antiferromagnet (SAF) stack.

    Structure (bottom to top): RL | non-magnetic spacer | FL

    The spacer is modelled as Ms=0 cells. The spacer thickness is
    ``total_z - t_fl - t_rl`` unless *t_spacer* is given (in which case
    any remaining cells outside the three defined regions are also spacer).

    Parameters
    ----------
    grid     : StructuredGrid — total z height must >= t_fl + t_rl
    mat_fl   : Material — free layer (top)
    mat_rl   : Material — reference layer (bottom)
    t_fl     : float [m]
    t_rl     : float [m]
    t_spacer : float or None — spacer thickness (default: remainder)

    Returns
    -------
    MaterialField3D
    """
    from _micromag import MaterialField3D as _MF3D
    mat_spacer           = Material()
    mat_spacer.Ms        = 0.0
    mat_spacer.A_exchange = 0.0
    mat_spacer.alpha     = 0.0
    mat_spacer.K_uniaxial = 0.0
    matf = _MF3D(grid, mat_spacer)

    dz      = grid.dz
    nz      = grid.nz
    total_z = nz * dz
    z_rl_end   = t_rl
    z_fl_start = total_z - t_fl

    for iz in range(nz):
        z_centre = (iz + 0.5) * dz
        if z_centre < z_rl_end:
            m_cell = mat_rl
        elif z_centre >= z_fl_start:
            m_cell = mat_fl
        else:
            m_cell = mat_spacer
        for iy in range(grid.ny):
            for ix in range(grid.nx):
                idx = ix + grid.nx * (iy + grid.ny * iz)
                _matf_set_cell(matf, idx, m_cell)
    return matf

