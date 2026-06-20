# mumax3 `.mx3` script runner

`micromag.mx3` parses and executes a practical subset of the
[mumax3](https://mumax.github.io/) scripting language on top of the micromag
engine. It picks the GPU backend when CUDA is available, otherwise CPU.

## Usage

```bash
# CLI
python -m micromag.mx3 examples/mx3/sp4.mx3 [output_dir]
```

```python
import micromag as mm
eng = mm.run_mx3("examples/mx3/sp4.mx3", outdir="out")
print(mm.mean_magnetization(eng.m))     # final <m>
```

(GPU module: call `os.add_dll_directory(r"...CUDA/vXX.Y/bin/x64")` before
`import micromag`.)

## Supported subset

| Category | Commands |
|----------|----------|
| Grid     | `SetGridSize`, `SetCellSize`, `SetPBC` |
| Material | `Msat`, `Aex`, `alpha`, `Ku1`, `anisU`, `Dind`, `Dbulk`, `B_ext`, `EnableDemag` |
| Init `m` | `uniform`, `vortex`, `random`, `twodomain` |
| Geometry | `setgeom(ellipse/circle/rect/square/cylinder/cuboid/ellipsoid(...))` — CPU backend only |
| Solver   | `MaxErr`, `MaxDt`, `MinDt`, `SetSolver` |
| Control  | `for init; cond; post { ... }`, `if cond { ... } else { ... }` |
| Run      | `relax()`, `minimize()`, `run(t)`, `steps(n)` |
| Output   | `save`, `saveas`, `tableSave`, `autosave`, `tableautosave`, `print` |
| Operators| `+ - * /`, `< > <= >= == !=`, `&& \|\| !`, `++ -- += -= *= /=` |
| Exprs    | numbers, `vector(...)`, `sqrt/sin/cos/exp/abs/pow`, `pi`, `mu0`, `name := expr` |

Control flow uses Go-style braces, e.g. a hysteresis sweep:

```
for i := 0; i < 21; i++ {
    B := -0.1 + 0.01*i
    B_ext = vector(B, 0, 0)
    minimize()
    tableSave()
}
```

Identifiers are case-insensitive (as in mumax3). Unsupported statements emit a
warning and are skipped, so a partially-covered script still runs as far as it
can. Output is written as OVF 2.0 files (`<name>NNNNNN.ovf`) and a tab-separated
table (`<basename>.txt`).

**Geometry note:** `setgeom(...)` is honoured only on the CPU backend
(`run_mx3(..., gpu=False)`). The GPU solver normalises every cell, so empty
cells in a masked shape would diverge; the CPU `normalize()` skips them.

See the `micromag.mx3` module docstring for the full reference.

## Examples

- **`sp4.mx3`** — µMAG Standard Problem 4 (field A). Relaxes to the S-state,
  then reverses under B = (−24.6, 4.3, 0) mT for 1 ns. Produces
  ⟨mₓ⟩(1 ns) ≈ −0.97 (cf. µMAG ≈ −0.98).
- **`relax_demo.mx3`** — small smoke test (16×16×1 square): `minimize()` then a
  short field run.
- **`loop_test.mx3`** — `for` / `if`-`else` / `:=` / `++` / comparison demo.
- **`disk.mx3`** — `setgeom(circle(...))` vortex in a 120 nm disk (run with
  `gpu=False`).

## Not yet supported

Regions / `defregion` / per-region material (`Msat.SetRegion(...)`),
time-dependent excitations (`B_ext` as a function of `t`), geometry on the GPU
backend, and `expectV` / user-defined script functions. These warn and skip.
