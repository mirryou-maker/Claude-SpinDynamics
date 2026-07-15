"""Simulation drivers & scale helpers: run/run_while/steps, Table, set_geom, exchange_length

Split from the former monolithic micromag/__init__.py; public names are
re-exported by the package __init__, so `import micromag as mm` is unchanged.
"""
import math as _math
import numpy as _np

from _micromag import *          # C++ core names  # noqa: F401,F403

# ---------------------------------------------------------------------------
# A5: SP#2 grid-size utilities
# ---------------------------------------------------------------------------

_mu_0 = 4 * _math.pi * 1e-7


def exchange_length(mat) -> float:
    """Return the exchange length l_ex = sqrt(2A / mu_0 Ms^2) [m].

    This is the fundamental length scale below which exchange dominates
    over magnetostatics.  Used in mumax3 SP#2 to set grid resolution:
    dx = l_ex / factor  (factor ~ 20–30 for quantitative accuracy).

    Parameters
    ----------
    mat : Material
        Material with Ms and A_exchange set.

    Returns
    -------
    float
        Exchange length in metres.
    """
    return _math.sqrt(2.0 * mat.A_exchange / (_mu_0 * mat.Ms ** 2))


def optimal_dx(mat, cells_per_lex: float = 1.0) -> float:
    """Return recommended cell size dx = l_ex / cells_per_lex [m].

    Rule of thumb:
      cells_per_lex = 1.0  → dx = l_ex  (minimum; 1 cell per exchange length)
      cells_per_lex = 2.0  → dx = l_ex/2  (good accuracy)
      cells_per_lex = 5.0  → dx = l_ex/5  (high accuracy, slow)

    mumax3 SP#2 guidance: element size d = 30 l_ex, with dx ≈ l_ex (1:1).

    Parameters
    ----------
    mat           : Material
    cells_per_lex : float, optional
        Number of cells per exchange length (default 1 → dx = l_ex).

    Returns
    -------
    float
        Recommended cell size in metres.
    """
    return exchange_length(mat) / cells_per_lex


# ---------------------------------------------------------------------------
# Run(t) / Steps(n) — mumax3-style simulation helpers
# ---------------------------------------------------------------------------

def run(integ, m, mat, heff, t_total: float,
        stt=None, callback=None, callback_dt: float = 0.0):
    """Run the integrator for t_total seconds.

    Works with RK4Integrator (fixed dt), RK45Integrator (adaptive), and
    HeunIntegrator.  For RK4/Heun the step size is integ.dt; for RK45 the
    returned dt from step() is used.

    Parameters
    ----------
    integ      : RK4Integrator | RK45Integrator | HeunIntegrator
    m          : VectorField3D  (modified in-place)
    mat        : Material
    heff       : EffectiveFieldSum
    t_total    : float  — simulation time [s]
    stt        : SpinTorqueSum | None
    callback   : callable(t, m) | None — called periodically
    callback_dt: float — minimum interval between callback calls [s] (0 = every step)

    Returns
    -------
    float — actual simulated time
    """
    t = 0.0
    t_last_cb = -1.0
    while t < t_total:
        # Step
        result = integ.step(m, mat, heff, stt) if stt is not None else integ.step(m, mat, heff)
        # Determine dt used
        if isinstance(result, float):
            dt_used = result
        else:
            dt_used = integ.dt
        t += dt_used
        # Callback
        if callback is not None:
            if callback_dt <= 0 or (t - t_last_cb) >= callback_dt:
                callback(t, m)
                t_last_cb = t
    return t


def run_while(integ, m, mat, heff, condition, t_max: float = 10e-9,
              stt=None, callback=None, callback_dt: float = 0.0,
              check_every: int = 100):
    """Run until condition(t, m) returns False or t_max is reached.

    Mirrors mumax3's RunWhile().  The condition is checked every
    ``check_every`` steps (not every step, for performance).

    Parameters
    ----------
    integ       : RK4Integrator | RK45Integrator | HeunIntegrator
    m           : VectorField3D  (modified in-place)
    mat         : Material
    heff        : EffectiveFieldSum
    condition   : callable(t, m) -> bool  — continue while True
    t_max       : float — hard stop time [s] (default 10 ns)
    stt         : SpinTorqueSum | None
    callback    : callable(t, m) | None — called periodically
    callback_dt : float — minimum interval between callbacks [s]
    check_every : int — check condition every N steps (default 100)

    Returns
    -------
    float — simulated time at which the loop stopped

    Examples
    --------
    Run until <mx> < -0.9 (switching detection):

    >>> mm.run_while(integ, m, mat, heff,
    ...     condition=lambda t, m: mm.mean_magnetization(m)[0] > -0.9,
    ...     t_max=2e-9)

    Run until energy converges (relax-like):

    >>> prev_e = [float('inf')]
    >>> def not_converged(t, m):
    ...     e = heff.total_energy(m, mat)
    ...     conv = abs(e - prev_e[0]) / (abs(prev_e[0]) + 1e-30) < 1e-6
    ...     prev_e[0] = e
    ...     return not conv
    >>> mm.run_while(integ, m, mat, heff, not_converged, t_max=5e-9)
    """
    t = 0.0
    step_count = 0
    t_last_cb = -1.0

    while t < t_max:
        result = integ.step(m, mat, heff, stt) if stt is not None else integ.step(m, mat, heff)
        dt_used = result if isinstance(result, float) else integ.dt
        t += dt_used
        step_count += 1

        if callback is not None:
            if callback_dt <= 0 or (t - t_last_cb) >= callback_dt:
                callback(t, m)
                t_last_cb = t

        if step_count % check_every == 0 and not condition(t, m):
            break

    return t


def steps(integ, m, mat, heff, n: int, stt=None):
    """Run the integrator for exactly n steps.

    Parameters
    ----------
    integ : RK4Integrator | RK45Integrator | HeunIntegrator
    m     : VectorField3D  (modified in-place)
    mat   : Material
    heff  : EffectiveFieldSum
    n     : int — number of steps
    stt   : SpinTorqueSum | None

    Returns
    -------
    float — total simulated time (dt * n for fixed-step; sum for adaptive)
    """
    t = 0.0
    for _ in range(n):
        result = integ.step(m, mat, heff, stt) if stt is not None else integ.step(m, mat, heff)
        if isinstance(result, float):
            t += result
        else:
            t += integ.dt
    return t


def sp2_grid(mat, Lx: float, Ly: float, Lz: float,
             cells_per_lex: float = 1.0) -> "StructuredGrid":
    """Create a StructuredGrid sized by the SP#2 exchange-length criterion.

    Cell size = l_ex / cells_per_lex.

    Parameters
    ----------
    mat           : Material
    Lx, Ly, Lz   : float — element dimensions [m]
    cells_per_lex : float — cells per exchange length (default 1 → dx = l_ex)

    Returns
    -------
    StructuredGrid

    Example (SP#4 element, l_ex resolution)
    ----------------------------------------
    >>> mat = Material.permalloy()
    >>> g = sp2_grid(mat, 500e-9, 125e-9, 3e-9)
    >>> print(g.nx, f'{g.dx*1e9:.1f} nm')   # 88 cells, 5.7 nm
    >>> g2 = sp2_grid(mat, 500e-9, 125e-9, 3e-9, cells_per_lex=2)
    >>> print(g2.nx, f'{g2.dx*1e9:.1f} nm')  # 176 cells, 2.8 nm (more accurate)
    """
    dx = optimal_dx(mat, cells_per_lex)
    nx = max(1, round(Lx / dx))
    ny = max(1, round(Ly / dx))
    nz = max(1, round(Lz / dx))
    return StructuredGrid(nx, ny, nz,
                          Lx / nx, Ly / ny, Lz / nz)


# ---------------------------------------------------------------------------
# Table — mumax3-style data table (analogous to TableAdd / TableSave)
# ---------------------------------------------------------------------------

class Table:
    """Accumulate simulation data rows and save as CSV.

    Mirrors mumax3's TableAdd / TableSave workflow.

    Usage
    -----
    >>> tbl = mm.Table()
    >>> tbl.add_row(t, m, mat=mat, heff=heff)      # per-step callback
    >>> tbl.save("output/table.csv")

    Custom columns (mumax3 TableAdd analog)
    ----------------------------------------
    >>> tbl.add_column("Q", lambda t, m: mm.topological_charge_Q(m))
    >>> tbl.add_column("max_angle", lambda t, m: mm.max_angle(m))
    >>> # Now add_row() automatically evaluates and stores these columns.
    >>> tbl.add_row(t, m, mat=mat, heff=heff)

    Custom quantities via extra dict (legacy, still supported)
    -----------------------------------------------------------
    >>> tbl.add_row(t, m, extra={"Q": mm.topological_charge_Q(m)})
    """

    def __init__(self):
        self._header = None
        self._rows = []
        self._custom_cols = {}   # name -> callable(t, m)

    def add_column(self, name: str, fn):
        """Register a custom column evaluated on each add_row() call.

        Parameters
        ----------
        name : str   -- column label in the CSV header
        fn   : callable(t: float, m: VectorField3D) -> float

        Example
        -------
        >>> tbl.add_column("Q", lambda t, m: mm.topological_charge_Q(m))
        >>> tbl.add_column("max_angle", lambda t, m: mm.max_angle(m))
        """
        self._custom_cols[name] = fn

    def add_row(self, t: float, m, mat=None, heff=None, extra: dict = None):
        mx, my, mz = mean_magnetization(m)
        row = {"t": t, "mx": mx, "my": my, "mz": mz}
        if mat is not None and heff is not None:
            row["E_total"] = heff.total_energy(m, mat)
        # Evaluate registered custom columns
        for col_name, fn in self._custom_cols.items():
            try:
                row[col_name] = float(fn(t, m))
            except Exception:
                row[col_name] = float("nan")
        if extra:
            row.update(extra)
        if self._header is None:
            self._header = list(row.keys())
        self._rows.append([row.get(k, float("nan")) for k in self._header])

    def save(self, filename: str):
        with open(filename, "w") as f:
            f.write(",".join(self._header) + "\n")
            for row in self._rows:
                f.write(",".join(f"{v:.12g}" for v in row) + "\n")

    def to_numpy(self):
        """Return table data as numpy array, shape (n_rows, n_cols)."""
        import numpy as _np2
        return _np2.array(self._rows, dtype=float)

    def __len__(self):
        return len(self._rows)

    @property
    def columns(self):
        return list(self._header) if self._header else []


# ---------------------------------------------------------------------------
# set_geom — mumax3-style geometry setter
# ---------------------------------------------------------------------------

def set_geom(mask, m, exch=None):
    """Apply geometry mask to magnetization (and optionally Exchange field).

    Mirrors mumax3's SetGeom(shape): zeros m outside the shape and, if exch
    is supplied, sets the Neumann-BC mask on the ExchangeField so exchange
    stiffness is correctly blocked at the boundary.

    Parameters
    ----------
    mask : GeomMask   — 1 inside geometry, 0 outside
    m    : VectorField3D  — modified in-place (m→0 where mask<0.5)
    exch : ExchangeField | None — if given, calls exch.set_mask(mask)
    """
    m.apply_mask(mask)
    if exch is not None:
        exch.set_mask(mask)

