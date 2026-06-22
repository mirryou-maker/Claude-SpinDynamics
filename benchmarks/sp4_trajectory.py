"""P2: µMAG SP#4 Field A — record <m>(t) trajectory for Fig 2 (NB41 dropped it).

Relax the S-state, apply Field A = (-24.6, 4.3, 0) mT, integrate 1 ns with RK45
(adaptive DOPRI5), logging <mx>,<my>,<mz> vs t. Writes sp4_trajectory_cs.csv.
Run with the f64 build: PYTHONPATH = build/windows-msvc-cuda/python.
"""
import os, math
import numpy as np
os.add_dll_directory(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64")
import micromag as mm

MU0 = 4e-7*math.pi
g = mm.StructuredGrid(200, 50, 1, 2.5e-9, 2.5e-9, 3e-9)
mat = mm.Material.permalloy()
demag = mm.DemagFieldGPU(g); exch = mm.ExchangeFieldGPU(g)

# 1) relax S-state (start from near-uniform +x with a slight tilt)
m0 = mm.VectorField3D(g); m0.set_uniform(mm.Vec3(1, 0.05, 0)); m0.normalize()
zee0 = mm.ZeemanFieldGPU(g, mm.Vec3(0, 0, 0))
fs0 = mm.FieldSumGPU(); fs0.add(exch); fs0.add(zee0)
rel = mm.RelaxGPU(g); o = mm.RelaxGPUOptions(); o.threshold = 1e-3*mat.Ms; o.max_steps = 60000
try: o.throw_on_max = False
except: pass
rel.upload(m0); rel.run(mat, demag, fs0, o); rel.download(m0)
print("S-state <m> =", tuple(round(x, 4) for x in mm.mean_magnetization(m0)))

# 2) apply Field A, integrate 1 ns with RK45, log <m>(t)
Bx, By = -24.6e-3, 4.3e-3
zee = mm.ZeemanFieldGPU(g, mm.Vec3(Bx/MU0, By/MU0, 0))
fs = mm.FieldSumGPU(); fs.add(exch); fs.add(zee)
rk = mm.RK45IntegratorGPU(g); rk.upload(m0)
mc = mm.VectorField3D(g)

t = 0.0; t_end = 1e-9; log_dt = 5e-12; next_log = 0.0
rows = []
while t < t_end:
    t += rk.step(mat, demag, fs)
    if t >= next_log:
        rk.download(mc)
        mx, my, mz = mm.mean_magnetization(mc)
        rows.append((t*1e9, mx, my, mz))
        next_log += log_dt
arr = np.array(rows)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sp4_trajectory_cs.csv")
np.savetxt(out, arr, delimiter=",", header="t_ns,mx,my,mz", comments="")
# t_switch = first time mx crosses 0
tsw = next((arr[i,0] for i in range(len(arr)-1) if arr[i,1] >= 0 and arr[i+1,1] < 0), float("nan"))
print(f"<mx>(1ns)={arr[-1,1]:.4f}  t_switch={tsw*1000:.0f} ps  (ref: -0.9862, ~175 ps)")
print(f"wrote {out}  ({len(arr)} points)")
