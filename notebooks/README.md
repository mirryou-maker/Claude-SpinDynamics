# NanoSpinDynamics — Jupyter Notebooks

Python-driven simulations using the C++ backend via pybind11.

## Setup

```powershell
# Build the Python module first
cmake --preset windows-msvc
cmake --build build/windows-msvc --config Release --target _micromag

# Activate venv and install Jupyter
.\.venv\Scripts\Activate.ps1
pip install jupyter matplotlib numpy

# Launch from the project root
jupyter notebook notebooks/
```

## Notebooks

| File | Content | Run time |
|---|---|---|
| `01_sp4_dynamics.py` | SP#4 Field A switching: RK45 to 1 ns, switching curve + quiver map | ~10 s |
| `02_sp1_phase_diagram.py` | SP#1 vortex/S-state phase diagram + thickness scaling + live 120 nm case | ~20 s |
| `03_thermal_sp4.py` | SP#4 at T = 300 K: Heun integrator + ThermalField, 3 stochastic seeds | ~60 s |

## Format

Files are plain Python with `# %%` cell markers (Jupytext format):
- **VS Code**: open as "Interactive Python" → run cells with Shift+Enter
- **Jupyter**: convert with `jupytext --to notebook 01_sp4_dynamics.py`
- **Script**: run directly with `python 01_sp4_dynamics.py`

## sys.path

All notebooks assume they are run from the `notebooks/` directory and prepend
`../build/windows-msvc/python` to `sys.path`. If running from the project root,
change to `build/windows-msvc/python`.

## Key API

```python
import sys
sys.path.insert(0, '../build/windows-msvc/python')
import _micromag as mm

grid = mm.StructuredGrid(nx, ny, nz, dx, dy, dz)
mat  = mm.Material.permalloy()          # or .cobalt(), .iron()
m    = mm.VectorField3D(grid)
m.set_uniform(mm.Vec3(1, 0, 0))
m.normalize()

heff = mm.EffectiveFieldSum()
heff.add(mm.DemagField(grid))           # CPU FFT demag
heff.add(mm.ExchangeField())
heff.add(mm.ZeemanField(mm.Vec3(...)))

# Deterministic (adaptive dt)
integ = mm.RK45Integrator()
dt = integ.step(m, mat, heff)          # returns dt used

# Stochastic SLLG (fixed dt — RK45 cannot be used here!)
thermal = mm.ThermalField(grid, T_K=300.0, dt=1e-13, seed=42)
heun    = mm.HeunIntegrator(dt=1e-13)
heun.step(m, mat, heff, thermal)

# numpy bridge
m_np = mm.to_numpy(m)                  # shape (nz, ny, nx, 3)
mm.from_numpy(m, m_np)
mx, my, mz = mm.mean_magnetization(m)
```
