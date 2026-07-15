"""Adaptive relaxation + parameter / multi-GPU sweeps (Phase R + sweeps)

Split from the former monolithic micromag/__init__.py; public names are
re-exported by the package __init__, so `import micromag as mm` is unchanged.
"""
import math as _math
import numpy as _np

from _micromag import *          # C++ core names  # noqa: F401,F403


# ===========================================================================
# Phase R — Convergence-based adaptive relaxation
# ===========================================================================

def run_until_converged(integ, m, mat, heff, tol_deg: float = 1.0,
                         max_steps: int = 1_000_000,
                         check_interval: int = 100,
                         stt=None,
                         verbose: bool = False):
    """Run LLG until max_angle(m) drops below tol_deg, then stop.

    Equivalent to mumax3's ``RunWhile(MaxAngle.Get() > tol*pi/180)``.
    Checks convergence every ``check_interval`` steps to keep overhead low.

    Parameters
    ----------
    integ          : integrator (RK4, RK45, Heun, or FrozenIntegrator)
    m              : VectorField3D -- magnetization, modified in-place
    mat            : Material
    heff           : EffectiveFieldSum
    tol_deg        : float -- convergence threshold [degrees] (default 1.0)
    max_steps      : int   -- hard step limit (default 1 000 000)
    check_interval : int   -- check every N steps (default 100)
    stt            : optional spin-torque term
    verbose        : bool  -- print progress every check_interval (default False)

    Returns
    -------
    dict with keys:
      "converged"    : bool  -- True if tol_deg was reached
      "steps"        : int   -- total LLG steps taken
      "max_angle"    : float -- final MaxAngle [degrees]
      "t_sim"        : float -- total simulated time [s]

    Example (mumax3 style):
    >>> mm.run_until_converged(integ, m, mat, heff, tol_deg=0.5)
    >>> # Equivalent to mumax3: RunWhile(MaxAngle.Get() > 0.5*pi/180)
    """
    from .structures import max_angle  # lazy: avoids sweep<->structures cycle
    import math as _mc

    step_count = 0
    t_sim = 0.0
    angle = max_angle(m)
    converged = angle < tol_deg

    while not converged and step_count < max_steps:
        # Take check_interval steps before checking convergence
        n_batch = min(check_interval, max_steps - step_count)
        for _ in range(n_batch):
            if stt is not None:
                integ.step(m, mat, heff, stt)
            else:
                integ.step(m, mat, heff)
        step_count += n_batch
        try:
            t_sim = step_count * integ.dt
        except AttributeError:
            t_sim = float("nan")

        angle = max_angle(m)
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


# ---------------------------------------------------------------------------
# Phase N + Z: MFM, EdgeSmooth, Poisson-disk grains, overlap integral
# ---------------------------------------------------------------------------
from micromag._phase_n import (mfm_signal, edge_smooth,  # noqa: E402
                               poisson_disk_grains,
                               mfm_overlap_integral)


# ---------------------------------------------------------------------------
# parameter_sweep — run a simulation function over a parameter grid
# ---------------------------------------------------------------------------

def _sweep_worker(args):
    """Module-level worker for parameter_sweep multiprocessing (pickle-safe)."""
    fn, keys, combo = args
    kwargs = dict(zip(keys, combo))
    result = fn(**kwargs)
    if not isinstance(result, dict):
        result = {"result": result}
    return {**kwargs, **result}


def parameter_sweep(
    fn,
    params: dict,
    *,
    progress: bool = True,
    n_jobs: int = 1,
):
    """Run *fn* for every combination of values in *params*.

    Parameters
    ----------
    fn : callable
        ``fn(**kwargs) -> dict``  Must return a mapping whose keys will be
        collected as output columns.  Extra keys are forwarded as-is.
    params : dict[str, list]
        Mapping from parameter name to the list of values to sweep over.
        All combinations (Cartesian product) are evaluated.
    progress : bool
        Print a progress counter to stdout (default ``True``).
    n_jobs : int
        Number of parallel worker processes (default ``1`` = sequential).
        Uses :mod:`multiprocessing.Pool` when *n_jobs* > 1.  Each worker
        inherits the calling process's CUDA_VISIBLE_DEVICES setting (Approach A
        for multi-GPU: set ``CUDA_VISIBLE_DEVICES=<N>`` per worker via the
        *env* parameter or subprocess launcher, *not* here).

    Returns
    -------
    list[dict]
        One dictionary per parameter combination.  Each dictionary contains
        all parameter key/value pairs plus whatever *fn* returned.

    Examples
    --------
    >>> def sim(H_ext, alpha):
    ...     return {"m_final": 0.9, "t_switch": 150e-12}
    >>> results = parameter_sweep(sim, {"H_ext": [0.1, 0.2], "alpha": [0.01, 0.02]})
    >>> len(results)
    4
    """
    import itertools

    keys   = list(params.keys())
    values = list(params.values())
    combos = list(itertools.product(*values))
    total  = len(combos)

    if n_jobs == 1:
        results = []
        for k, combo in enumerate(combos):
            if progress:
                print(f"[parameter_sweep] {k+1}/{total}  "
                      + "  ".join(f"{ky}={v}" for ky, v in zip(keys, combo)))
            kwargs = dict(zip(keys, combo))
            result = fn(**kwargs)
            if not isinstance(result, dict):
                result = {"result": result}
            results.append({**kwargs, **result})
    else:
        import multiprocessing as _mp
        task_args = [(fn, keys, combo) for combo in combos]
        ctx = _mp.get_context("spawn")  # spawn is pickle-safe on all platforms
        with ctx.Pool(processes=n_jobs) as pool:
            if progress:
                print(f"[parameter_sweep] n_jobs={n_jobs}, total={total} cases")
            results = pool.map(_sweep_worker, task_args)

    return results


# ---------------------------------------------------------------------------
# multi_gpu_sweep — ensemble parallelism via multiprocessing (Approach A)
# ---------------------------------------------------------------------------

def _worker_init_cuda(gpu_id: int) -> None:
    """Set CUDA_VISIBLE_DEVICES in a worker process before importing the GPU module."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def _worker_run(args):
    """Entry point for each worker process in multi_gpu_sweep."""
    gpu_id, fn, kwargs = args
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    result = fn(**kwargs)
    if not isinstance(result, dict):
        result = {"result": result}
    return {**kwargs, **result}


def multi_gpu_sweep(
    fn,
    params: dict,
    gpu_ids=None,
    *,
    progress: bool = True,
):
    """Run *fn* over all parameter combinations, distributing across multiple GPUs.

    This is **Approach A** (ensemble parallelism): each worker process is
    assigned one GPU via ``CUDA_VISIBLE_DEVICES`` and runs independently.
    No inter-GPU communication is required; C++ code is unchanged.

    Parameters
    ----------
    fn : callable
        ``fn(**kwargs) -> dict``.  The function is called in a subprocess
        with ``CUDA_VISIBLE_DEVICES`` already set to the assigned GPU.
        It must import micromag *inside* the function body so the GPU module
        is imported after the environment variable is set.
    params : dict[str, list]
        Cartesian product of parameter values to sweep over.
    gpu_ids : list[int] | None
        List of GPU device indices to use.  Defaults to ``[0]``.
        E.g. ``[0, 1, 2, 3]`` for four-GPU parallelism.
    progress : bool
        Print assignment info to stdout.

    Returns
    -------
    list[dict]
        One dict per parameter combination (same order as Cartesian product).

    Example
    -------
    >>> def sim_gpu(H_ext, alpha):
    ...     import micromag as mm          # imported inside worker
    ...     # ... build grid, integrator, run ...
    ...     return {"t_switch": 150e-12, "m_final": -0.98}
    >>> results = multi_gpu_sweep(
    ...     sim_gpu,
    ...     {"H_ext": [0.1, 0.2, 0.3, 0.4], "alpha": [0.01, 0.02]},
    ...     gpu_ids=[0, 1],
    ... )
    """
    import itertools
    import multiprocessing as _mp

    if gpu_ids is None:
        gpu_ids = [0]

    keys   = list(params.keys())
    values = list(params.values())
    combos = list(itertools.product(*values))
    total  = len(combos)

    # Round-robin assign GPU IDs to each combo
    task_args = []
    for idx, combo in enumerate(combos):
        gpu_id = gpu_ids[idx % len(gpu_ids)]
        kwargs = dict(zip(keys, combo))
        if progress:
            print(f"[multi_gpu_sweep] job {idx+1}/{total} → GPU {gpu_id}  "
                  + "  ".join(f"{k}={v}" for k, v in zip(keys, combo)))
        task_args.append((gpu_id, fn, kwargs))

    with _mp.Pool(processes=len(gpu_ids)) as pool:
        results = pool.map(_worker_run, task_args)

    return results

