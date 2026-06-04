# NanoSpinDynamics — 개발 진행 요약

**최종 업데이트**: 2026-06-04  
**브랜치**: master  
**현재 커밋**: `22f70f4`  
**CPU 테스트**: **81 / 81 통과**  
**GPU 빌드**: 정상 (CUDA 13.2, RTX 계열)

---

## 프로젝트 개요

C++20 마이크로마그네틱 시뮬레이터. Python 바인딩(pybind11), MSVC/Windows, vcpkg, CMake presets. µMAG 표준 문제 검증 + 유한 온도 + GPU 가속 완성.

---

## 전체 커밋 이력

| 커밋 | 단계 | 내용 |
|------|------|------|
| `76fe06a` | 1a | Grid, VectorField3D, VTK, pybind11 스켈레톤 |
| `bddcd67` | 1b | ZeemanField, ExchangeField, UniaxialAnisotropyField |
| `4ceb448` | 1c | RK4Integrator (LLG) |
| `0a235d5` | 1d | SlonczewskiSTT, SpinOrbitTorque |
| `f3d1f96` | **Phase 2 완성** | **6D Newell demag — 8/8 테스트 통과** |
| `f116bb8` | Phase 2 정리 | diag_step1 6D 공식 갱신 |
| `b611c19` | Phase 1e | µMAG SP#4 Field A (170°) |
| `c882018` | RK45 | Dormand-Prince DOPRI5+FSAL — 3.5× 가속 |
| `f57d491` | Phase 1e-B | SP#4 Field B (190°) |
| `a7ed2ca` | **T1** | ThermalField — Langevin 노이즈 |
| `c5a2a41` | **T2** | HeunIntegrator — Stratonovich |
| `f0c135c` | **T3** | 열 평형 검증 (등분배, 이방성) |
| `e32930f` | **T4** | Néel-Brown τ₀ ∝ 1/T 검증 |
| `c797a04` | **T5** | SP#4 @ T=300K — Δt_sw ≈ −0.5% |
| `10c2b24` | **P3 Step 1** | CMake CUDA 빌드 시스템 |
| `39ae880` | **P3 Step 4** | cuFFT GPU demag 실제 구현 |
| `36e64a3` | **P3 Step 5** | Pinned 메모리 + 사전 할당 (3.0×) |
| `5a24566` | **P3 Step 6a** | cuFFT 배치 모드 (4.3×) |
| `22f70f4` | **P3 Step 6b** | 희소 업로드 PCIe 8× 감소 (~4.9×) |

---

## 완성된 기능

### 아키텍처 계층

```
types.hpp          Vec3, Real, Index, 물리 상수 (k_B, μ₀, γ₀)
grid.hpp           StructuredGrid
field.hpp          VectorField3D
material.hpp       Material (permalloy/cobalt/iron)
effective_field.hpp  IEffectiveField + EffectiveFieldSum
spin_torque.hpp    ISpinTorque + SpinTorqueSum
thermal_field.hpp  ThermalField (Langevin, Stratonovich)
integrator.hpp     RK4 + RK45(DOPRI5+FSAL) + Heun(고정 Δt SLLG)
demag.hpp          DemagField (CPU, FFTW, 6D Newell)
demag_gpu.hpp      DemagFieldGPU (GPU, cuFFT) — CUDA 빌드 전용
```

---

## Phase 2: DemagField 핵심 성과

**근본 버그 수정**: 8-코너 ±dx/2 합산 → 올바른 6D 이중-셀 Newell 적분

| 테스트 | 수정 전 | **수정 후** | 이론값 |
|--------|---------|------------|--------|
| 4×4×4 H_z | -300,000 A/m | **-266,667 A/m** | -Ms/3 (**0.0% 오차**) |
| 박막 H_z | 0 A/m | **-754,839 A/m** | ≈-Ms (94.4%) |
| N_zz(self) | 0 | **+1/3** | 1/3 |

---

## Phase T: 유한 온도 SLLG (완성)

```
ThermalField  σ = sqrt(2α k_B T / μ₀ Ms γ V Δt)
HeunIntegrator  고정 Δt 예측자-수정자 (Stratonovich)
```

**핵심 제약**: RK45는 σ ∝ 1/√Δt 때문에 SLLG와 사용 불가 — HeunIntegrator 필수.

---

## Phase 3: GPU 가속 성능 이력

빌드: `cmake --preset windows-msvc-cuda`

| 단계 | 최적화 내용 | 시간 (안정 측정) | CPU 대비 |
|------|------------|-----------------|---------|
| CPU 기준 | FFTW | ~40 s | 1× |
| Step 4 | 기본 cuFFT | 19.6 s | 2.1× |
| Step 5 | Pinned 메모리 + 사전 할당 | 13.5 s | 3.0× |
| **Step 6a** | cuFFT 배치 (9→2 exec) | **9.4 s** | **4.3×** |
| **Step 6b** | 희소 업로드 1.92→0.24MB | **~8.2 s** | **~4.9×** |

**핵심 최적화 내용**:
- `cufftPlanMany` batch=3: FFT exec 9→2회/스텝
- `pointwise_mac_all3`: 3→1 커널 런치 (3× 캐시 효율)
- `extract_all3`: 3→1 커널 + 다운로드
- Pinned host 메모리 (DMA 전송 가속)
- 영구 GPU 스크래치 버퍼 (cudaMalloc/Free 제거)
- 희소 업로드: compact 0.24MB → GPU scatter → 패딩 (PCIe 8× 절감)

**대형 격자 전망**:

| 격자 | PCIe 업로드 (기존) | PCIe 업로드 (6b) | 절감 |
|---|---|---|---|
| 200×50×1 (현재) | 1.92MB | 0.24MB | 0.14ms/step |
| 500×500×10 | 60MB | 7.5MB | ~4ms/step |
| 1000×1000×20 | 480MB | 60MB | ~35ms/step |

---

## µMAG SP#4 결과

| | Field A (170°) | Field B (190°) | T=300K |
|---|---|---|---|
| t_switch | 0.175 ns | 0.175 ns | 0.170 ns |
| `<mx>` | -0.982 (0.4%) | -0.980 (0.4%) | -0.783 |
| 적분기 | RK45 (CPU) | RK45 (CPU) | Heun (CPU) |

---

## 빌드 및 실행

```powershell
# CPU (기존)
cmake --preset windows-msvc
cmake --build build/windows-msvc --config Release
.\build\windows-msvc\bin\Release\unit_tests.exe     # 81/81 통과

# GPU (CUDA)
cmake --preset windows-msvc-cuda
cmake --build build/windows-msvc-cuda --config Release
.\build\windows-msvc-cuda\bin\Release\sp4_gpu.exe   # 벤치마크
```

---

## 다음 작업 우선순위

### ★★★ 1순위: GPU 단위 테스트

현재 `DemagFieldGPU` 정확도는 벤치마크 앱에서만 확인됨. 체계적 검증 필요:

```cpp
// tests/test_demag_gpu.cpp [demag][gpu]
TEST_CASE("DemagFieldGPU matches CPU for 4x4x4") {
    // CPU result vs GPU result → WithinRel(1e-6)
}
```

**이유**: GPU 코드 변경 시 회귀 검출 불가 — 테스트 없이는 안전하지 않음.

### ★★★ 1순위 (공동): CUDA 스트림 (Step 6c)

업로드(PCIe)와 GPU 계산(cuFFT)이 순차적임 → 비동기 중첩으로 추가 ~0.3ms/스텝 절감 가능.

### ★★☆ 2순위: 대형 격자 벤치마크

500×500×10 격자로 실제 연구 규모에서의 GPU 성능 측정.

### ★★☆ 2순위: Python 바인딩 정비

`DemagFieldGPU` + `RK45Integrator` + `HeunIntegrator` Python 노출.  
Jupyter notebook에서 SP#4 재현 시나리오 가능.

### ★☆☆ 3순위: µMAG SP#1 검증

S-state/vortex 정적 평형 — 기존 스택으로 즉시 구현 가능.
