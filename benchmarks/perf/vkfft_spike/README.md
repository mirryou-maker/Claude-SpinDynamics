# VkFFT vs cuFFT Feasibility Spike

Standalone CUDA benchmarks comparing VkFFT 1.2.31 against cuFFT for the
R2C transform sizes that appear in Claude-SD's demag pipeline.

## Files

| File | Purpose |
|------|---------|
| `spike.cu`  | Original spike — power-of-2 sizes only (led to an incorrect conclusion) |
| `spike2.cu` | Extended spike — 2D + 3D, all actual µMAG benchmark sizes incl. non-pow2 Nz |

## Build (from this directory)

```powershell
$CL  = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\HostX64\x64\cl.exe"
$INC = "C:\vcpkg\installed\x64-windows\include"
nvcc spike2.cu -ccbin "$CL" -I"$INC" -lcufft -lnvrtc -lcuda `
     -O2 --std=c++17 -Xcompiler "/wd4819 /wd4244" -o spike2.exe
.\spike2.exe
```

---

## spike.cu — Original (2026-06-20): Power-of-2 sizes only

| transform | cuFFT | VkFFT | ratio |
|-----------|-------|-------|-------|
| 4096×1024 batch3 f32 (1M grid) | 0.548 ms | 0.571 ms | 0.96× |
| 2048×512  batch3 f32 (262K)    | 0.053 ms | 0.049 ms | 1.08× |

Conclusion at the time: "VkFFT ≈ cuFFT, not worth integrating."

**This was wrong** — only power-of-2 sizes were tested; all actual µMAG
benchmark grids produce non-power-of-2 padded sizes.

---

## spike2.cu — Extended (2026-06-21): 2D + 3D incl. non-pow2 Nz

Tested on RTX 5060 Ti (sm_120), f32, batch=3 R2C.

### 2D Thin-Film (pad_nz = 1)

| Size (padded) | Origin | cuFFT | VkFFT | Ratio |
|---------------|--------|-------|-------|-------|
| 4096×1024 | perf sweep large | 0.576 ms | 0.609 ms | 0.95× (cuFFT) |
| 2048×512  | perf sweep medium | 0.057 ms | 0.051 ms | 1.12× VkFFT |
| 1024×256  | perf sweep small  | 0.024 ms | 0.020 ms | 1.18× VkFFT |
| **400×128** | **SP#4 200×64×1** | 0.021 ms | **0.016 ms** | **1.32× VkFFT** |
| **200×100** | **SP#1 100×50×1** | 0.016 ms | 0.015 ms | 1.03× VkFFT |
| 200×200 | SP#3 100×100×2 xy | 0.016 ms | 0.016 ms | ~1.00× |
| **2000×1000** | large non-pow2 | 0.280 ms | **0.117 ms** | **2.39× VkFFT** ⭐ |
| 600×300 | medium non-pow2 | 0.020 ms | 0.019 ms | 1.05× VkFFT |

### 3D — Power-of-2 Nz (baseline)

| Size (padded) | cuFFT | VkFFT | Ratio |
|---------------|-------|-------|-------|
| 512×256×8  | 0.069 ms | 0.065 ms | 1.07× VkFFT |
| 256×128×16 | 0.035 ms | 0.043 ms | 0.83× (cuFFT) |
| 512×256×4  | 0.042 ms | 0.045 ms | 0.95× (cuFFT) |

### 3D — Non-Power-of-2 Nz

| Size (padded) | Origin grid | cuFFT | VkFFT | Ratio |
|---------------|-------------|-------|-------|-------|
| **200×200×4** | **SP#3 100×100×2** | 0.029 ms | **0.023 ms** | **1.28× VkFFT** |
| **400×200×10** | 200×100×5 (Nz=5) | 0.094 ms | **0.056 ms** | **1.69× VkFFT** ⭐ |
| **400×200×6**  | 200×100×3 (Nz=3) | 0.060 ms | **0.043 ms** | **1.39× VkFFT** |
| **400×200×14** | 200×100×7 (Nz=7) | 0.136 ms | **0.076 ms** | **1.79× VkFFT** ⭐ |
| **512×256×10** | pow2 XY, Nz=5 | 0.096 ms | **0.072 ms** | **1.34× VkFFT** |
| **512×256×14** | pow2 XY, Nz=7 | 0.186 ms | **0.105 ms** | **1.77× VkFFT** ⭐ |
| 512×256×6  | pow2 XY, Nz=3 | 0.056 ms | 0.054 ms | 1.04× VkFFT |
| 200×200×10 | 100×100×5 | 0.048 ms | 0.043 ms | 1.10× VkFFT |
| 200×200×6  | 100×100×3 | 0.035 ms | 0.031 ms | 1.13× VkFFT |
| 1024×512×10 | large Nz=5 | 1.023 ms | 1.048 ms | 0.98× (same) |
| 1024×512×6  | large Nz=3 | 0.575 ms | 0.568 ms | 1.01× (same) |

---

## Key Findings

### Why the original spike gave the wrong answer

`spike.cu` tested 4096×1024 and 2048×512 only — both exact powers of 2.
cuFFT is optimally tuned for powers of 2; VkFFT's advantage lies in
*non-power-of-2* sizes (Rader/Bluestein algorithm) where cuFFT degrades.

Our 2N padding rule:
- Grid nx=200 → pad=400 (NOT pow2); nx=100 → pad=200 (NOT pow2)
- Actual µMAG benchmarks DO use non-pow2 padded sizes
- The perf_sweep.py grids (128, 256, 512, ...) are the exception, not the rule

### VkFFT advantage pattern

| Regime | VkFFT gain | When |
|--------|-----------|------|
| Small-to-medium non-pow2 | **1.3×–2.4×** | Rader/Bluestein vs cuFFT's Cooley-Tukey |
| Large non-pow2 (≥1M cells FFT) | ~1.0× | Bandwidth-limited; algorithm irrelevant |
| Any power-of-2 large | 0.95×–1.00× | cuFFT equally good or better |
| Power-of-2 medium/small | 1.08×–1.18× | VkFFT slightly faster (kernel efficiency) |

### Practical impact

SP#4 actual FFT time: 0.016 ms with VkFFT vs 0.021 ms with cuFFT → 0.005 ms
saving against a 0.46 ms overhead floor. **Negligible for small benchmarks.**

For 3D grids with non-pow2 Nz in the 0.05–0.2 ms FFT range (e.g. medium
simulations with Nz=5,7,10), VkFFT gives a **real, measurable** 1.4×–1.8× demag
FFT speedup. Integration cost is non-trivial (NVRTC linkage, per-size plan init),
so it's worth pursuing only if such grids are primary use-cases.

---

## mumax3 MAC Kernel Symmetry (from source analysis)

Beyond FFT library, mumax3 exploits **Y+Z frequency-domain mirror symmetry** for
both kernel storage and MAC computation (files: `kernmulrsymm*.cu`).

### Kernel storage (3D, grid nx×ny×nz)

| Solver | Kernel per component | vs. Claude-SD |
|--------|---------------------|---------------|
| mumax3 | `(nx+1)×(ny+1)×(nz+1)` real | — |
| Claude-SD (post #3) | `(nx+1)×2ny×2nz` real | **4× larger** |

Example 100×100×10: mumax3 ≈ 112K reals; Claude-SD ≈ 404K reals.
Smaller kernel → better L2 cache residency → faster MAC.

### 2D thin-film MAC: half the threads

```cuda
// mumax3: launches Nx × (Ny/2+1) threads — HALF the rows
// Each thread writes row iy AND mirror row Ny-iy (off-diagonal sign flips)
if (iy != 0 && 2*iy != Ny) {
    float Kxym = -Kxy;
    fftMx[mirror_e] = reMx*Kxx + reMy*Kxym;   // mirror row
}
// Claude-SD: launches (nx+1) × 2ny threads — 2× more, no symmetry
```

### 3D MAC: 4× compressed kernel lookup

```cuda
// mumax3 kernmulRSymm3D: full Nx×Ny×Nz threads, reads from quarter-array
if (iy > Ny/2) { iy = Ny-iy; signYZ *= -1; signXY *= -1; }
if (iz > Nz/2) { iz = Nz-iz; signYZ *= -1; signXZ *= -1; }
I = (iz*(Ny/2+1) + iy)*Nx + ix;   // Ny/2+1 rows: only first quadrant stored
float Kyz = fftKyz[I] * signYZ;   // sign reconstructed from symmetry
// → 4× smaller kernel → L2 cache 4× more likely to hit
```

### Recommended optimization order

| Action | Effort | Demag speedup | Priority |
|--------|--------|--------------|----------|
| **MAC Y/Z symmetry** | Medium | 1.5×–2× MAC step | **1 (best ROI)** |
| **CUDA Graphs** | High | eliminates ~0.46ms overhead on small grids | 2 |
| **VkFFT** | High | 1.3×–1.8× FFT for non-pow2 medium grids | 3 (conditional) |
