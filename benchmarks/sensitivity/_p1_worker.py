"""P1 worker (runs inside ONE build's micromag module).

Skyrmion topological-charge sensitivity: relax a seeded skyrmion at several D and
N repeats per D; emit all Q (and <mz>) so the driver computes mean/std + run-to-run
spread. Same Co/Pt disk as NB45 (Dc = 4 sqrt(A K)/pi = 4.41 mJ/m^2).

Usage: python _p1_worker.py <build_python_dir> <cuda_bin_dir> <cfg_json>
cfg_json = {"D_list":[2.2e-3,...], "N":12, "max_steps":20000}
"""
import os, sys, json
import numpy as np

build_dir, cuda_bin, cfg_json = sys.argv[1], sys.argv[2], sys.argv[3]
os.add_dll_directory(cuda_bin)
sys.path.insert(0, build_dir)
import micromag as mm

cfg = json.loads(cfg_json)
Ms, A, K, alpha = 800e3, 15e-12, 0.8e6, 0.3
NX, NY, NZ, DX, DZ = 100, 100, 1, 2e-9, 1e-9


def seed(g):
    m = mm.VectorField3D(g)
    for iy in range(NY):
        for ix in range(NX):
            rx = (ix - NX//2)*DX; ry = (iy - NY//2)*DX
            mz = -1.0 if (rx*rx + ry*ry) < (20e-9)**2 else 1.0
            m[ix + NX*iy] = mm.Vec3(0, 0, mz)
    return m


def relax_Q(D, max_steps):
    g = mm.StructuredGrid(NX, NY, NZ, DX, DX, DZ)
    mat = mm.Material(); mat.Ms = Ms; mat.A_exchange = A; mat.K_uniaxial = K
    mat.easy_axis = mm.Vec3(0, 0, 1); mat.alpha = alpha
    demag = mm.DemagFieldGPU(g); exch = mm.ExchangeFieldGPU(g)
    dmi = mm.InterfacialDMIFieldGPU(g, D); ani = mm.UniaxialAnisotropyFieldGPU(g)
    fs = mm.FieldSumGPU(); fs.add(exch); fs.add(dmi); fs.add(ani)
    rel = mm.RelaxGPU(g); o = mm.RelaxGPUOptions()
    o.max_steps = max_steps; o.threshold = 1e-4*Ms
    m = seed(g); rel.upload(m); rel.run(mat, demag, fs, o); rel.download(m)
    Q = float(mm.topological_charge_Q(m))
    mz = float(np.asarray(mm.to_numpy(m))[..., 2].mean())
    return Q, mz


out = {}
for D in cfg["D_list"]:
    Qs, mzs = [], []
    for _ in range(cfg["N"]):
        q, mz = relax_Q(D, cfg.get("max_steps", 20000))
        Qs.append(q); mzs.append(mz)
    out[f"{D:.4e}"] = {"Q": Qs, "mz": mzs}

print("RESULT_JSON " + json.dumps(out))
