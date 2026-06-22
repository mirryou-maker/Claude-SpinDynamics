"""B: why does CS RelaxGPU land on a different metastable Q than mumax3?

After the stream-race fix, CS is deterministic but at D/Dc~0.79 gives Q~-0.06
(skyrmion nearly collapsed) while mumax3 relax() gives Q~-0.89 (skyrmion kept).
Both deterministic -> different local minima. This script decides whether CS is
(a) correctly finding a LOWER-energy state (skyrmion is metastable, CS escapes
    it) -> not a bug, or
(b) FAILING to hold a skyrmion that is the true minimum -> a relaxation issue.

Method: seed a CLEAN Neel skyrmion AND the mz=-1 disk; relax each with RelaxGPU
and MinimizeGPU; report Q and total energy E. If a clean-skyrmion seed survives
with LOWER E than the disk-relaxed collapsed state, CS can hold skyrmions and the
disk just relaxes elsewhere. If the clean skyrmion collapses too / has higher E,
investigate the relaxer.

Run with the f64 build: set PYTHONPATH to build/windows-msvc-cuda/python.
"""
import os, math, sys
import numpy as np
os.add_dll_directory(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64")
import micromag as mm

Ms, A, K, alpha = 800e3, 15e-12, 0.8e6, 0.3
NX, NY, NZ, DX, DZ = 100, 100, 1, 2e-9, 1e-9
DC = 4*math.sqrt(A*K)/math.pi


def make(D):
    g = mm.StructuredGrid(NX, NY, NZ, DX, DX, DZ)
    mat = mm.Material(); mat.Ms=Ms; mat.A_exchange=A; mat.K_uniaxial=K
    mat.easy_axis=mm.Vec3(0,0,1); mat.alpha=alpha
    demag=mm.DemagFieldGPU(g); exch=mm.ExchangeFieldGPU(g)
    dmi=mm.InterfacialDMIFieldGPU(g,D); ani=mm.UniaxialAnisotropyFieldGPU(g)
    return g, mat, demag, exch, dmi, ani


def disk_seed(g):
    m=mm.VectorField3D(g)
    for iy in range(NY):
        for ix in range(NX):
            rx=(ix-NX//2)*DX; ry=(iy-NY//2)*DX
            mz=-1.0 if (rx*rx+ry*ry)<(20e-9)**2 else 1.0
            m[ix+NX*iy]=mm.Vec3(0,0,mz)
    return m


def neel_seed(g, R=20e-9, w=8e-9):
    # Neel skyrmion: in-plane radial component, mz from +1 (far) to -1 (core)
    m=mm.VectorField3D(g)
    for iy in range(NY):
        for ix in range(NX):
            rx=(ix-NX//2)*DX; ry=(iy-NY//2)*DX
            r=math.hypot(rx,ry)
            theta=math.pi*max(0.0, min(1.0, 1.0-(r-0.0)/(2*R)))  # crude profile
            mz=math.cos(math.pi*min(1.0, r/R))*-1.0 if r<R else 1.0
            mz=max(-1.0,min(1.0,mz))
            ip=math.sqrt(max(0.0,1.0-mz*mz))
            if r>1e-12:
                mx,my=ip*rx/r, ip*ry/r   # Neel (radial)
            else:
                mx,my=0.0,0.0
            m[ix+NX*iy]=mm.Vec3(mx,my,mz)
    m.normalize(); return m


def energy(g,mat,demag,exch,dmi,ani,m):
    return (demag.energy(m,mat)+exch.energy(m,mat)+dmi.energy(m,mat)+ani.energy(m,mat))


def relax(g,mat,demag,exch,dmi,ani,m0,method,max_steps=30000):
    fs=mm.FieldSumGPU(); fs.add(exch); fs.add(dmi); fs.add(ani)
    if method=="relax":
        r=mm.RelaxGPU(g); o=mm.RelaxGPUOptions()
    else:
        r=mm.MinimizeGPU(g); o=mm.MinimizeGPUOptions()
    o.threshold=1e-4*Ms; o.max_steps=max_steps
    try:o.throw_on_max=False
    except:pass
    m=mm.VectorField3D(g)
    for i in range(g.size): m[i]=m0[i]
    r.upload(m)
    try:r.run(mat,demag,fs,o)
    except:pass
    r.download(m)
    return float(mm.topological_charge_Q(m)), energy(g,mat,demag,exch,dmi,ani,m)


if __name__=="__main__":
    for dd in [0.68, 0.79, 0.91]:
        D=dd*DC
        g,mat,demag,exch,dmi,ani=make(D)
        Edisk0=energy(g,mat,demag,exch,dmi,ani,disk_seed(g))
        Eneel0=energy(g,mat,demag,exch,dmi,ani,neel_seed(g))
        print(f"\nD/Dc={dd:.2f}  (E_seed: disk={Edisk0:.3e}  neel={Eneel0:.3e} J)")
        for seedname, seedfn in [("disk", disk_seed), ("neel", neel_seed)]:
            for meth in ["relax", "minimize"]:
                Q,E=relax(g,mat,demag,exch,dmi,ani,seedfn(g),meth)
                print(f"   {seedname:>4} + {meth:<8}: Q={Q:+.3f}  E={E:.4e} J", flush=True)
