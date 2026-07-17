# Release checklist (binary packages + tag)

Repeatable procedure for cutting a release (e.g. v1.0.0). Items marked **[gate]**
must pass before proceeding.

## 1. Pre-flight

- [ ] CI green on `master` (both jobs: `linux-cpu`, `windows-cpu`) **[gate]**
- [ ] All four CUDA variants build & `unit_tests_gpu` green locally
      (cuFFT/VkFFT × f32/f64) **[gate]**
- [ ] **Release builds configured with `-DMICROMAG_CUDA_ARCHS=release`** — dev
      `native` builds are single-arch and crash on every other GPU generation
      (the v1.0.0 sm_120-only incident). Verify each shipped binary:
      ```powershell
      python scripts/check_fatbin_archs.py <variant>/python/_micromag*.pyd
      python scripts/check_fatbin_archs.py <variant>/bin/sp4_gpu.exe
      ```
      Both must print PASS (sm_75..120 + compute_120 PTX) **[gate]**
- [ ] `packaging/MANIFEST.md` copied to the package root (CPU/GPU binary map)
- [ ] Performance smoke vs stored baselines (same machine as the baselines):
      ```powershell
      python benchmarks/cpu_parity_bench.py --threads 8 --baseline benchmarks/_parity_windows.json
      python benchmarks/gpu_parity_bench.py --baseline benchmarks/_parity_gpu_<GPU>.json
      ```
      Both must print PASS (±15 %) **[gate]**
- [ ] `docs/RELEASE_NOTES_v<X.Y.Z>.md` updated
- [ ] Working tree clean (`git status`) — paper figures & notebook outputs are
      committed only when the paper is final; everything else either committed
      or gitignored

## 2. Package layout

Two zips, both containing **every** executable + the Python module:

```
Claude-SD-v<X.Y.Z>-win64-cpu-py313/
├── micromag_locate.py        ← REQUIRED (bundled examples resolve it via ../)
├── README.txt
├── bin/                      (all exes from build/windows-msvc)
├── python/                   (_micromag*.pyd + micromag/ package)
├── examples/                 (repo examples/*.py, unmodified)
└── notebooks/                (repo notebooks/*.py, unmodified)

Claude-SD-v<X.Y.Z>-win64-gpu-py313/
├── micromag_locate.py        ← REQUIRED
├── README.txt
├── add_dll_to_path.bat       (from packaging/gpu/)
├── runtime-dll/              (CUDA runtime: cudart, cufft, curand,
│                              nvrtc64_*, nvrtc-builtins64_*  ← VkFFT needs NVRTC)
├── cuFFT-f64/ cuFFT-f32/ VkFFT-f64/ VkFFT-f32/
│   └── bin/  python/         (per-variant exes + module)
├── examples/  notebooks/
```

**`micromag_locate.py` is mandatory since the 26adaf6 helper consolidation** —
every bundled example/notebook does
`sys.path.insert(0, parent.parent); from micromag_locate import ...`, so the file
must sit at the package root. Without it every example fails at import.

## 3. Packaging gotchas (learned the hard way)

- **CRT**: never ship Debug-CRT-linked exes. The project already forces
  `/NODEFAULTLIB:{libcmtd,msvcrtd,vcruntimed,ucrtd}` — if a new exe pulls
  `VCRUNTIME140D.dll`/`ucrtbased.dll`, that flag set regressed. **[gate]**
- **NVRTC**: VkFFT variants compile kernels at runtime — bundle
  `nvrtc64_*.dll` **and** `nvrtc-builtins64_*.dll` in `runtime-dll/`.
- **Zip extraction**: PowerShell `Expand-Archive` silently truncates >~500 MB
  archives. README.txt must tell users to extract with Explorer / 7-Zip / `tar`.
- **.bat files**: CRLF line endings (LF-only bats mis-execute).
- Shell scripts destined for Linux: LF (`sed -i 's/\r$//'` after editing on
  Windows).
- **Stale Python package in build trees**: `build/<preset>/python/micromag/` is
  a POST_BUILD **copy** of `python/micromag/` — editing the source package
  without rebuilding `_micromag` (or manually `cmake -E copy_directory`) leaves
  every build tree running the old code. Always rebuild (or re-copy) after
  touching `python/micromag/**` before packaging or testing.

## 4. Smoke test each package (fresh shell, no dev env)

- [ ] CPU pkg: `bin\sp4.exe` runs; `python examples\hello_micromag.py` works
      with only `pip install numpy`
- [ ] GPU pkg: `add_dll_to_path.bat` (option 1) → `cuFFT-f64\bin\unit_tests_gpu.exe`
      green → `python examples\llg_demo.py` **[gate]**
- [ ] Confirm no PATH/registry leftovers needed beyond the bat

## 5. Publish

- [ ] `git tag v<X.Y.Z> && git push origin v<X.Y.Z>`
- [ ] GitHub Release: attach both zips + `docs/RELEASE_NOTES_v<X.Y.Z>.md`
- [ ] (v1.0.0, after paper acceptance) repo public flip — **irreversible,
      requires explicit owner confirmation** — then Zenodo DOI
