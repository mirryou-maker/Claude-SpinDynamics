# VkFFT feasibility spike (2026-06-20)

Question: is VkFFT worth integrating to speed up the demag FFT?

`spike.cu` benchmarks a R2C transform of the demag size (2N-padded, batch=3,
float32) with cuFFT vs VkFFT (CUDA backend, vcpkg `vkfft`).

Build (Korean-locale needs /wd4819):
```
nvcc spike.cu -o spike.exe -ccbin "<MSVC Hostx64/x64>" \
  -I"C:/vcpkg/installed/x64-windows/include" -lcufft -lnvrtc -lcuda \
  -O2 --std=c++17 -Xcompiler "/wd4819"
```

## Result — VkFFT does NOT help here
| transform | cuFFT | VkFFT |
|-----------|------:|------:|
| 4096x1024 (1M grid) | 0.548 ms | 0.571 ms (0.96x) |
| 2048x512  (262K)    | 0.053 ms | 0.049 ms (1.08x) |

VkFFT ≈ cuFFT (±8%). Our 2N padding gives **power-of-2** sizes, for which cuFFT
is already optimal; VkFFT's 1.3-2x edge is for *non-power-of-2* sizes (cuFFT's
weak spot), which we don't hit. VkFFT compiles + runs fine via vcpkg, but there
is no FFT speedup to gain. Its only remaining value would be convolution-mode
*fusion* of the ~32% non-FFT demag overhead (prep+MAC+extract) — a complex,
risky integration for a bounded ≤32% upside. **Conclusion: not worth pursuing.**
The residual large-grid gap vs mumax3 is mumax3 doing *less FFT work*
(symmetry/pad scheme), an algorithmic target, not a library swap.
