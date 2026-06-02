# NanoSpinDynamics — 개발 진행 요약

**최종 업데이트**: 2026-06-03  
**브랜치**: master  
**현재 커밋**: `c797a04`  
**전체 테스트**: **81 / 81 통과**

---

## 프로젝트 개요

C++20 마이크로마그네틱 시뮬레이터. Python 바인딩(pybind11), MSVC/Windows,
vcpkg, CMake presets 기반. 목표: Newell FFT demag를 포함한 완전한
마이크로마그네틱 스택 구축 + µMAG 표준 문제 검증 + 유한 온도 확장.

---

## 커밋 이력

| 커밋 | 단계 | 내용 |
|------|------|------|
| `76fe06a` | Phase 1a | StructuredGrid, VectorField3D, VTK writer, pybind11 |
| `bddcd67` | Phase 1b | ZeemanField, ExchangeField, UniaxialAnisotropyField |
| `4ceb448` | Phase 1c | RK4Integrator (LLG) |
| `0a235d5` | Phase 1d | SlonczewskiSTT, SpinOrbitTorque |
| `5f9247b` | Phase 2 인프라 | DemagField 헤더, CMakeLists, 테스트 파일 |
| `f3d1f96` | **Phase 2 완성** | **6D 이중-셀 Newell 적분 (8/8 테스트 통과)** |
| `f116bb8` | Phase 2 정리 | diag_step1 6D 공식으로 갱신 |
| `b611c19` | Phase 1e | µMAG SP#4 Field A (170°) — `<mx>`=-0.982 ✓ |
| `c882018` | RK45 | Dormand-Prince DOPRI5 + FSAL — 3.5배 가속 |
| `f57d491` | Phase 1e-B | µMAG SP#4 Field B (190°) — `<mx>`=-0.980 ✓ |
| `a7ed2ca` | **Phase T1** | **ThermalField — Langevin 노이즈 (6/6 테스트)** |
| `c5a2a41` | **Phase T2** | **HeunIntegrator — 고정 Δt Stratonovich Heun** |
| `f0c135c` | **Phase T3** | **열 평형 검증 (등분배, 이방성 바이어스)** |
| `e32930f` | **Phase T4** | **Néel-Brown 시도 빈도 τ₀ ∝ 1/T 검증** |
| `c797a04` | **Phase T5** | **SP#4 @ T=300K — 전체 SLLG 시뮬레이션** |

---

## 완성된 기능

### 아키텍처 계층

```
types.hpp          Vec3, Real, Index, constants (π, μ₀, γ₀=1.76e11, k_B)
grid.hpp           StructuredGrid — 셀 형상, x-fastest 인덱싱
field.hpp          VectorField3D — 데이터 소유, set_uniform/set_vortex
material.hpp       Material — Ms, A, K, alpha, easy_axis (permalloy/cobalt/iron)
effective_field.hpp  IEffectiveField + EffectiveFieldSum (컴포지터)
spin_torque.hpp    ISpinTorque + SpinTorqueSum (컴포지터)
thermal_field.hpp  ThermalField — Langevin 노이즈, Stratonovich 재사용
integrator.hpp     RK4Integrator + RK45Integrator (DOPRI5+FSAL) + HeunIntegrator
```

### 유효장 구현체

| 클래스 | 물리 |
|--------|------|
| `ZeemanField` | 균일 외부 자기장 H_ext |
| `UniaxialAnisotropyField` | H_ani = (2K/μ₀Ms)(m·û)û |
| `ExchangeField` | 6점 라플라시안, Neumann/Periodic BC |
| `DemagField` | **6D 이중-셀 Newell FFT 컨볼루션** |

### 스핀 토크

| 클래스 | 물리 |
|--------|------|
| `SlonczewskiSTT` | CPP-STT: a_J[m×(m×p̂)] + b_J[m×p̂] |
| `SpinOrbitTorque` | SOT: η_DL m×(m×σ̂) + η_FL(m×σ̂) |

### 적분기

| 클래스 | 종류 | 용도 |
|--------|------|------|
| `RK4Integrator(dt)` | 고정 Δt, 4단계 | 범용 |
| `RK45Integrator(opts)` | 적응 Δt, DOPRI5+FSAL | 결정론적 LLG 효율화 |
| `HeunIntegrator(dt)` | 고정 Δt, Heun | **SLLG 필수** (σ ∝ 1/√Δt) |

---

## Phase 2: DemagField 핵심 성과

### 근본 버그 수정

`nxx`/`nxy` 함수가 Newell (1993) 이중-셀 적분 대신 잘못된 코너 위치 사용.

**올바른 공식** (70-lines-of-NumPy µMAG SP#4 검증 코드 동일):

```cpp
// 6D 이중-셀 합산 — 64항
int nx=round(x/dx), ny=round(y/dy), nz=round(z/dz);
double sum = 0;
for (ia,ib,ic,id,ie,ig in {0,1}^6) {
    sign = (-1)^(ia+ib+ic+id+ie+ig);
    sum += sign * newell_f/g(
        (nx+ia-id)*dx, (ny+ib-ie)*dy, (nz+ic-ig)*dz);
}
return +sum / (4*pi*dx*dy*dz);   // 양의 부호
```

추가:
- `newell_f`: `abs(x,y,z)` 적용 + 원래 계수 (1/2, 1/2, 1, 1/6)
- `newell_g`: `abs(z)` 적용

### 검증된 수치 결과

| 테스트 | 수정 전 | **수정 후** | 이론값 |
|--------|---------|------------|--------|
| 4×4×4 cube H_z(1,1,1) | -300,000 A/m | **-266,667 A/m** | -Ms/3 **(오차 0.0%)** |
| 16×16×1 박막 H_z | 0 A/m | **-754,839 A/m** | ≈-Ms (94.4%) |
| 5×5×5 중심 H_x | 231,009 A/m | **0 A/m** | 0 (대칭 정확) |
| 자기 소자화 인자 N_zz(self) | 0 (틀림) | **+1/3** | 1/3 |

---

## µMAG Standard Problem #4

### Field A (θ=170°) / Field B (θ=190°)

| | Field A | Field B | µMAG 참조 |
|---|---|---|---|
| Hy 방향 | +3.45 kA/m | -3.45 kA/m | — |
| 전환 시간 | **0.175 ns** | **0.175 ns** | 0.10–0.15 ns |
| 최종 `<mx>` | **-0.98205** | **-0.97985** | -0.9862 / -0.9839 (0.4%) |
| 적분기 | RK45 (3.5× 가속) | RK45 | — |
| 실행 시간 | ~56 s | ~56 s | — |

---

## RK45 성능 (SP#4 기준)

| | RK4 (dt=0.05ps) | RK45 (rtol=1e-4) | 비율 |
|---|---|---|---|
| 스텝 수 | 20,000 | 3,532 | **5.7배 적음** |
| 실행 시간 | 197 s | **55.7 s** | **3.5배 빠름** |
| 정확도 | 기준 | 동일 | ✓ |

---

## Phase T: 유한 온도 마이크로마그네틱스 (완성)

### 구현 계층

```
ThermalField    σ = sqrt(2α k_B T / (μ₀ Ms γ₀ V Δt))
                resample() — 매 Heun 스텝 1회, predictor+corrector 동일 노이즈 재사용
HeunIntegrator  고정 Δt 예측자-수정자 (Stratonovich 해석)
                step(m, mat, heff, thermal=nullptr, stt=nullptr)
```

### 핵심 물리 제약

| 제약 | 설명 |
|------|------|
| **RK45 사용 불가** | σ ∝ 1/√Δt — Δt 변경 시 물리 통계 변함 |
| **고정 Δt 필수** | SDE는 결정론적 ODE 적응법 적용 불가 |
| **Stratonovich** | predictor/corrector 동일 η^n 재사용 → 올바른 Boltzmann 분포 |

### Phase T 단계별 검증 결과

#### T1: ThermalField (6 테스트)

| 테스트 | 검증 내용 | 결과 |
|--------|-----------|------|
| T1-A | σ ∝ √T, σ ∝ 1/√Δt, σ ∝ √α | ✓ |
| T1-B | T=0 → σ=0 | ✓ |
| T1-C | `<H_th>=0`, `<H_th²>=σ²` | ✓ |
| T1-D | predictor/corrector 동일 노이즈 재사용 | ✓ |
| T1-E | 연속 resample() 독립 샘플 | ✓ |
| T1-F | 같은 seed → 같은 시퀀스 | ✓ |

#### T2: HeunIntegrator (5 테스트)

| 테스트 | 검증 내용 | 결과 |
|--------|-----------|------|
| T2-A | `|m|=1` 보존 (노이즈 유무) | ✓ |
| T2-B | T=0 ≡ thermal=nullptr | ✓ |
| T2-C | T>0 궤적 발산 확인 | ✓ |
| T2-D | 다른 seed → 다른 궤적 | ✓ |
| T2-E | 결정론적 이완 m → +ẑ | ✓ |

#### T3: 열 평형 (3 테스트)

| 테스트 | 설정 | 검증 결과 |
|--------|------|-----------|
| T3-A 등분배 | K=0, T=1e8K, dt=100ps | `<mx²>=<my²>=<mz²>=1/3` (**오차 <3%**) |
| T3-B 이방성 T=0 | K=1e4 J/m³, 10° 초기값 | 3000 스텝 후 `|mz|>0.9` ✓ |
| T3-C 이방성 + 노이즈 | K=1e4 J/m³, T=300K | `<mz²>>0.85` (이방성 유지) ✓ |

**중요한 물리적 통찰**: T3에서 발견된 SLLG의 실용적 한계:
- σ/H_ani ≈ 0.002 at T=300K, K=1e5 J/m³
- 열 평형까지 τ_eq ≈ (H_ani/σ)² × τ₀ ≈ 1.2 ms (실용 불가)
- **K=0, 극고온(T=1e8K)**만 단위 테스트에서 빠른 평형 검증 가능

#### T4: Néel-Brown 시도 빈도 (2 테스트)

| 테스트 | 설정 | 검증 결과 |
|--------|------|-----------|
| T4-A | K=0, T_low vs T_high | τ(T_low) > τ(T_high) ✓ |
| T4-B | K=0, τ ∝ 1/T | 비율 [3, 30] for 10× T 변화 ✓ |

**Néel-Brown 완전 검증의 한계**: τ = τ₀ × exp(KV/k_BT)의 exp(κ) 항은 현실적 파라미터에서 테스트 불가:
- H_ani/σ ≈ 460 at T=300K, K=1e5 J/m³
- τ_equil ≈ 1.2 ms >> τ_Néel ≈ 1.2 ns

#### T5: SP#4 @ T=300K (전체 스택 통합)

| 항목 | T=0 (결정론적) | T=300K (확률론적) |
|------|----------------|-------------------|
| 적분기 | HeunIntegrator | HeunIntegrator + ThermalField |
| t_switch | 0.170 ns | **0.170 ± 0.000 ns** |
| `<mx>`_final | -0.781 | -0.783 |
| **Δt_switch** | — | **≈ −0.5%** |
| σ/Ms | 0 | **0.088%** |

**물리적 해석**: T=300K에서 σ/Ms ≈ 0.09% → 열 교란이 결정론적 동역학에 거의 영향 없음. 실험에서도 500×125×3 nm Permalloy는 열적으로 안정된 나노자석.

---

## 실행 방법

```powershell
# 빌드
cmake --preset windows-msvc
cmake --build build/windows-msvc --config Release

# 전체 단위 테스트 (81개)
.\build\windows-msvc\bin\Release\unit_tests.exe

# 유한 온도 테스트만
.\build\windows-msvc\bin\Release\unit_tests.exe "[thermal]"

# SP#4 시뮬레이션
.\build\windows-msvc\bin\Release\sp4.exe           # Field A, T=0, RK45
.\build\windows-msvc\bin\Release\sp4_fieldB.exe    # Field B, T=0
.\build\windows-msvc\bin\Release\sp4_thermal.exe   # Field A, T=300K, Heun

# 검증 앱
.\build\windows-msvc\bin\Release\thermal_equilibrium.exe
.\build\windows-msvc\bin\Release\thermal_neel_brown.exe
```

---

## 현재 남은 작업

| 우선순위 | 작업 | 상태 |
|---------|------|------|
| ★★☆ | SP#4 Field B 온도 효과 | 미구현 (코드 1줄 변경) |
| ★★☆ | Python 바인딩 정비 | 미구현 |
| ★☆☆ | Phase 3: CUDA GPU 가속 | 계획만 |
| ★☆☆ | RK45용 sp4.cpp 업그레이드 | 미구현 |
