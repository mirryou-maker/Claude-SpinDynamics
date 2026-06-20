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
| Material | `Msat`, `Aex`, `alpha`, `Ku1`, `anisU`, `Dind`, `B_ext`, `EnableDemag` |
| Init `m` | `uniform`, `vortex`, `random`, `twodomain` |
| Solver   | `MaxErr`, `MaxDt`, `MinDt`, `SetSolver` |
| Run      | `relax()`, `minimize()`, `run(t)`, `steps(n)` |
| Output   | `save`, `saveas`, `tableSave`, `autosave`, `tableautosave`, `print` |
| Exprs    | numbers, `+ - * /`, `vector(...)`, `sqrt/sin/cos/exp/abs/pow`, `pi`, `mu0`, `name := expr` |

Identifiers are case-insensitive (as in mumax3). Unsupported statements emit a
warning and are skipped, so a partially-covered script still runs as far as it
can. Output is written as OVF 2.0 files (`<name>NNNNNN.ovf`) and a tab-separated
table (`<basename>.txt`).

See the `micromag.mx3` module docstring for the full reference.

## Examples

- **`sp4.mx3`** — µMAG Standard Problem 4 (field A). Relaxes to the S-state,
  then reverses under B = (−24.6, 4.3, 0) mT for 1 ns. Produces
  ⟨mₓ⟩(1 ns) ≈ −0.97 (cf. µMAG ≈ −0.98).
- **`relax_demo.mx3`** — small smoke test (16×16×1 square): `minimize()` then a
  short field run.

## Not yet supported

Regions / `defregion` / per-region material, geometry shapes (`setgeom`),
time-dependent excitations (`B_ext` as a function of `t`), bulk DMI (`Dbulk`),
`for`/`if` control flow, and `expectV`/script functions. These warn and skip.
