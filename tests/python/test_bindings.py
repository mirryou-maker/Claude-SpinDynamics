"""Fast CPU-only tests for the Python binding surface + utility layer.

The C++ suite (232 cases) covers the physics; these cover what it cannot — the
pybind11 boundary, the NumPy bridge, file I/O round-trips, and the pure-Python
utilities in python/micromag/. Everything here must run in seconds without a
GPU (GPU-only APIs are exercised only for graceful-absence).
"""
import math
import os
import tempfile

import numpy as np
import pytest

import micromag as mm


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------
def test_grid_basics():
    g = mm.StructuredGrid(8, 4, 2, 2e-9, 3e-9, 4e-9)
    assert (g.nx, g.ny, g.nz) == (8, 4, 2)
    assert g.size == 8 * 4 * 2
    assert g.dx == pytest.approx(2e-9)


def test_vec3_and_material():
    v = mm.Vec3(1.0, 2.0, 3.0)
    assert (v.x, v.y, v.z) == (1.0, 2.0, 3.0)
    py = mm.Material.permalloy()
    assert py.Ms == pytest.approx(8.0e5, rel=0.2)
    assert py.A_exchange > 0


def test_field_set_uniform_and_indexing():
    g = mm.StructuredGrid(4, 4, 1, 5e-9, 5e-9, 5e-9)
    m = mm.VectorField3D(g)
    m.set_uniform(mm.Vec3(0, 0, 1))
    assert m[0].z == 1.0
    m[3] = mm.Vec3(1, 0, 0)          # __setitem__
    assert m[3].x == 1.0 and m[3].z == 0.0


# ---------------------------------------------------------------------------
# NumPy bridge
# ---------------------------------------------------------------------------
def test_to_from_numpy_roundtrip():
    g = mm.StructuredGrid(6, 5, 2, 5e-9, 5e-9, 5e-9)
    m = mm.VectorField3D(g)
    rng = np.random.default_rng(42)
    arr = rng.normal(size=(2, 5, 6, 3))
    arr /= np.linalg.norm(arr, axis=-1, keepdims=True)
    mm.from_numpy(m, arr)
    back = np.asarray(mm.to_numpy(m))
    assert back.shape == (2, 5, 6, 3)
    np.testing.assert_allclose(back, arr, atol=1e-14)


def test_from_numpy_rejects_bad_shape():
    g = mm.StructuredGrid(4, 4, 1, 5e-9, 5e-9, 5e-9)
    m = mm.VectorField3D(g)
    with pytest.raises(Exception):
        mm.from_numpy(m, np.zeros((4, 4, 3)))      # missing z axis


def test_normalize_field():
    g = mm.StructuredGrid(4, 4, 1, 5e-9, 5e-9, 5e-9)
    m = mm.VectorField3D(g)
    m.set_uniform(mm.Vec3(3.0, 4.0, 0.0))          # |m| = 5
    mm.normalize_field(m)
    norms = np.linalg.norm(np.asarray(mm.to_numpy(m)).reshape(-1, 3), axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Physics smoke (fast): fields, energy, one integrator step
# ---------------------------------------------------------------------------
def test_zeeman_energy_sign():
    """E_zeeman = -mu0*Ms*(m . H)*V < 0 when m || H."""
    g = mm.StructuredGrid(4, 4, 1, 5e-9, 5e-9, 5e-9)
    mat = mm.Material.permalloy()
    m = mm.VectorField3D(g)
    m.set_uniform(mm.Vec3(0, 0, 1))
    zee = mm.ZeemanField(mm.Vec3(0, 0, 1e5))
    heff = mm.EffectiveFieldSum()
    heff.add(zee)
    assert heff.total_energy(m, mat) < 0


def test_rk4_step_preserves_norm():
    g = mm.StructuredGrid(8, 8, 1, 5e-9, 5e-9, 5e-9)
    mat = mm.Material.permalloy()
    m = mm.VectorField3D(g)
    m.set_uniform(mm.Vec3(1, 0.2, 0.1))
    m.normalize()
    heff = mm.EffectiveFieldSum()
    heff.add(mm.ExchangeField())
    heff.add(mm.ZeemanField(mm.Vec3(0, 0, 8e4)))
    rk4 = mm.RK4Integrator(1e-13)
    for _ in range(20):
        rk4.step(m, mat, heff)
    norms = np.linalg.norm(np.asarray(mm.to_numpy(m)).reshape(-1, 3), axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_demag_uniform_thin_film():
    """Uniformly magnetized flat film: H_demag ~ -Ms*mz (N_zz -> 1)."""
    g = mm.StructuredGrid(32, 32, 1, 5e-9, 5e-9, 5e-9)
    mat = mm.Material.permalloy()
    m = mm.VectorField3D(g)
    m.set_uniform(mm.Vec3(0, 0, 1))
    demag = mm.DemagField(g)
    H = mm.VectorField3D(g)
    H.set_uniform(mm.Vec3(0, 0, 0))
    demag.accumulate(m, mat, H)
    hz_centre = H[g.size // 2].z
    assert hz_centre < -0.6 * mat.Ms          # mostly -Ms, edge effects allowed


def test_skyrmion_topological_charge():
    g = mm.StructuredGrid(40, 40, 1, 4e-9, 4e-9, 4e-9)
    m = mm.neel_skyrmion(g, 30e-9, charge=1, pol=-1)
    Q = mm.topological_charge_Q(m)
    assert abs(abs(Q) - 1.0) < 0.2


# ---------------------------------------------------------------------------
# I/O round-trips
# ---------------------------------------------------------------------------
def test_ovf_roundtrip(tmp_path):
    g = mm.StructuredGrid(8, 4, 2, 5e-9, 5e-9, 5e-9)
    m = mm.uniform_mag(g, mm.Vec3(0.6, 0.0, 0.8))
    path = str(tmp_path / "m.ovf")
    mm.save_ovf(path, m)
    m2 = mm.load_ovf(path)
    np.testing.assert_allclose(np.asarray(mm.to_numpy(m2)),
                               np.asarray(mm.to_numpy(m)), atol=1e-6)


def test_profile_roundtrip(tmp_path):
    g = mm.StructuredGrid(16, 4, 1, 5e-9, 5e-9, 5e-9)
    m = mm.uniform_mag(g, mm.Vec3(0, 0, 1))
    path = str(tmp_path / "prof.csv")
    mm.save_profile(m, path, component="z", axis=0)
    pos, vals = mm.load_profile(path)
    assert len(pos) == 16
    np.testing.assert_allclose(vals, 1.0, atol=1e-9)


def test_vtk_write(tmp_path):
    g = mm.StructuredGrid(4, 4, 2, 5e-9, 5e-9, 5e-9)
    m = mm.uniform_mag(g, mm.Vec3(1, 0, 0))
    path = str(tmp_path / "m.vtk")
    mm.write_vtk_legacy(path, m, "m")
    assert os.path.getsize(path) > 100


# ---------------------------------------------------------------------------
# Geometry / per-cell materials
# ---------------------------------------------------------------------------
def test_geom_mask_rect():
    g = mm.StructuredGrid(10, 10, 1, 5e-9, 5e-9, 5e-9)
    m = mm.uniform_mag(g, mm.Vec3(0, 0, 1))
    m.apply_mask(mm.rect(g, 25e-9, 50e-9))     # centered 25x50 nm region
    arr = np.asarray(mm.to_numpy(m))[0, :, :, 2]
    assert arr[:, 4:6].min() > 0.9             # centre columns: kept
    assert abs(arr[:, 0]).max() < 1e-12        # edge columns: zeroed
    assert abs(arr[:, -1]).max() < 1e-12


def test_material_field_percell():
    g = mm.StructuredGrid(4, 4, 1, 5e-9, 5e-9, 5e-9)
    matf = mm.MaterialField3D(g, mm.Material.permalloy())
    co = mm.Material.cobalt()
    matf[0] = co
    assert matf.Ms_field[0] == pytest.approx(co.Ms, rel=1e-6)
    assert matf.Ms_field[1] == pytest.approx(mm.Material.permalloy().Ms, rel=1e-6)


# ---------------------------------------------------------------------------
# Utility layer (pure Python)
# ---------------------------------------------------------------------------
def test_parameter_sweep_serial():
    res = mm.parameter_sweep(lambda D, K: D * K,
                             {"D": [1.0, 2.0], "K": [10.0, 20.0]}, n_jobs=1)
    vals = sorted(r["result"] for r in res)
    assert vals == [10.0, 20.0, 20.0, 40.0]


def test_recommend_integrator_runs():
    mat = mm.Material.permalloy(); mat.alpha = 0.5
    rec = mm.recommend_integrator(mat, T_K=0.0, goal="relax", verbose=False)
    assert isinstance(rec, dict) and "integrator" in rec


def test_bloch_dw_width():
    """Analytic tanh wall must be measured at ~ its own width."""
    g = mm.StructuredGrid(200, 1, 1, 1e-9, 1e-9, 1e-9)
    lam_true = 10e-9
    x0 = 100e-9
    m = mm.VectorField3D(g)
    arr = np.zeros((1, 1, 200, 3))
    x = (np.arange(200) + 0.5) * 1e-9
    arr[0, 0, :, 2] = np.tanh((x - x0) / lam_true)
    arr[0, 0, :, 1] = 1.0 / np.cosh((x - x0) / lam_true)
    mm.from_numpy(m, arr)
    lam, x_meas = mm.bloch_dw_width(m, axis=0, comp=2)
    # Lilley definition: lambda = pi / max|dm_z/dx| = pi * lambda_tanh
    assert lam == pytest.approx(math.pi * lam_true, rel=0.15)
    assert x_meas == pytest.approx(x0, abs=3e-9)


# ---------------------------------------------------------------------------
# GPU absence handling
# ---------------------------------------------------------------------------
def test_cuda_flag_is_bool():
    assert isinstance(mm.cuda_available(), bool)


@pytest.mark.skipif(not hasattr(mm, "DemagFieldGPU"),
                    reason="CPU-only build: GPU classes not exported")
def test_gpu_classes_present_when_cuda():
    assert mm.cuda_available()
