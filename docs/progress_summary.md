# NanoSpinDynamics — 개발 진행 요약

**최종 업데이트**: 2026-06-03  
**브랜치**: master  
**현재 커밋**: `b91230d`  
**전체 테스트**: 64 / 64 통과

---

## 프로젝트 개요

C++20 마이크로마그네틱 시뮬레이터. Python 바인딩(pybind11), MSVC/Windows,
vcpkg, CMake presets 기반. 목표: Newell FFT demag를 포함한 완전한
마이크로마그네틱 스택 구축, µMAG 표준 문제 검증.

---

## 커밋 이력

| 커밋 | 날짜 | 내용 |
|------|------|------|
| `76fe06a` | — | Phase 1a: StructuredGrid, VectorField3D, VTK writer, pybind11 |
| `bddcd67` | — | Phase 1b: ZeemanField, ExchangeField, UniaxialAnisotropyField |
| `4ceb448` | — | Phase 1c: RK4Integrator (LLG) |
| `0a235d5` | — | Phase 1d: SlonczewskiSTT, SpinOrbitTorque, SpinTorqueSum |
| `5f9247b` | — | Phase 2 인프라: DemagField 헤더, CMakeLists, 테스트 파일 |
| `67b6731` | — | Phase 2: newell_g atan2→atan 패리티 수정 |
| `f3d1f96` | — | **Phase 2 완성**: 6D 이중-셀 Newell 적분 (8/8 테스트 통과) |
| `f116bb8` | — | Phase 2: diag_step1 6D 공식으로 갱신 |
| `b611c19` | — | Phase 1e: µMAG SP#4 Field A (170°) |
| `c882018` | — | **RK45**: Dormand-Prince 적응 적분기 + FSAL |
| `f57d491` | — | Phase 1e: µMAG SP#4 Field B (190°) |
| `b91230d` | — | chore: .gitignore CSV 출력 파일 |

---

## 완성된 기능

### 아키텍처 계층

```
types.hpp          Vec3, Real, Index, 물리 상수 (π, μ₀, ℏ, e, γ₀)
grid.hpp           StructuredGrid — 셀 형상, x-fastest 인덱싱
field.hpp          VectorField3D — 데이터 소유, set_uniform/set_vortex
material.hpp       Material — Ms, A, K, alpha, easy_axis (permalloy/cobalt/iron)
effective_field.hpp  IEffectiveField + EffectiveFieldSum (컴포지터)
spin_torque.hpp    ISpinTorque + SpinTorqueSum (컴포지터)
integrator.hpp     RK4Integrator (고정 Δt) + RK45Integrator (적응 Δt)
```

### 유효장 구현체

| 클래스 | 파일 | 물리 |
|--------|------|------|
| `ZeemanField` | zeeman.cpp | 균일 외부 자기장 H_ext |
| `UniaxialAnisotropyField` | anisotropy.cpp | H_ani = (2K/μ₀Ms)(m·û)û |
| `ExchangeField` | exchange.cpp | 6점 라플라시안, Neumann/Periodic BC |
| `DemagField` | demag.cpp | **6D 이중-셀 Newell 적분 (FFT 컨볼루션)** |

### 스핀 토크

| 클래스 | 물리 |
|--------|------|
| `SlonczewskiSTT` | CPP-STT: a_J[m×(m×p̂)] + b_J[m×p̂] |
| `SpinOrbitTorque` | SOT: η_DL m×(m×σ̂) + η_FL(m×σ̂) |

### 적분기

| 클래스 | 종류 | 용도 |
|--------|------|------|
| `RK4Integrator(dt)` | 고정 Δt, 4단계 | 범용, stochastic LLG 필수 |
| `RK45Integrator(opts)` | 적응 Δt, DOPRI5 + FSAL | 결정론적 LLG 효율화 |

---

## Phase 2: DemagField 핵심 성과

### 발견된 버그와 수정

가장 오래 걸린 단계. 근본 원인: `nxx`/`nxy`의 8-코너 합산이  
Newell (1993) 이중-셀 적분 공식과 다른 코너 위치 사용.

**올바른 공식** (70-lines-of-NumPy µMAG SP#4 검증 코드와 동일):

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

추가 수정:
- `newell_f`: `abs(x,y,z)` 적용 + 원래 계수 (1/2, 1/2, 1, 1/6)
- `newell_g`: `abs(z)` 적용

### 최종 수치 검증

| 테스트 | 수정 전 | **수정 후** | 이론값 |
|--------|---------|------------|--------|
| 4×4×4 cube H_z(1,1,1) | -300,000 A/m | **-266,667 A/m** | -Ms/3 **(오차 0.0%)** |
| 16×16×1 박막 H_z | 0 A/m | **-754,839 A/m** | ≈-Ms (94.4%) |
| 5×5×5 중심 H_x | 231,009 A/m | **0 A/m** | 0 (대칭 정확) |
| 자기 소자화 N_zz(self) | 0 (틀림) | **+1/3** | 1/3 |

---

## µMAG Standard Problem #4

### 설정

| 항목 | 값 |
|------|-----|
| 샘플 | Permalloy 500 × 125 × 3 nm |
| 격자 | 200 × 50 × 1 셀, 2.5 × 2.5 × 3 nm |
| 재료 | Ms=800 kA/m, A=13 pJ/m, K=0, α=0.02 |
| 초기 상태 | m ≈ +x (0.5° 틸트) |
| 적분기 | RK45 (rtol=1e-4, dt_max=5 ps) |

### Field A (θ = 170°) vs Field B (θ = 190°)

| 항목 | Field A | Field B | µMAG 참조 A | µMAG 참조 B |
|------|---------|---------|-------------|-------------|
| Hy 방향 | +3.45 kA/m (+y) | -3.45 kA/m (−y) | — | — |
| 전환 시간 | **0.175 ns** | **0.175 ns** | 0.10–0.15 ns | 0.10–0.15 ns |
| 최종 `<mx>` | **-0.98205** | **-0.97985** | -0.9862 (0.4%) | -0.9839 (0.4%) |
| 최종 `<my>` | +0.025 | -0.041 | ≈0 | ≈0 |
| RK45 steps | 3532+79 | 3532+59 | — | — |
| 실행 시간 | ~56 s | ~56 s | — | — |

**물리적 차이**: Hy 부호에 따라 전환 궤적이 y 방향으로 거울 대칭.  
두 경우 모두 µMAG 참조값의 **0.5% 이내** 달성.

---

## RK45 성능 비교

| 항목 | RK4 (dt=0.05 ps) | RK45 (rtol=1e-4) | 비율 |
|------|-----------------|-----------------|------|
| 스텝 수 | 20,000 | 3,532 (+79) | **5.7배 적음** |
| 실행 시간 | 197 s | **55.7 s** | **3.5배 빠름** |
| 최종 `<mx>` | -0.98205 | -0.98204 | ✓ 동일 |
| 거절률 | — | 2.2% | 낮음 ✓ |

FSAL(First Same As Last) 최적화: 63s → 55.7s (13% 추가 개선).

---

## 빌드 및 실행

```powershell
cmake --preset windows-msvc
cmake --build build/windows-msvc --config Release

# 전체 단위 테스트 (64개)
.\build\windows-msvc\bin\Release\unit_tests.exe

# SP#4 시뮬레이션 (RK45, ~56초)
.\build\windows-msvc\bin\Release\sp4.exe          # Field A 170°
.\build\windows-msvc\bin\Release\sp4_fieldB.exe   # Field B 190°

# 성능 비교 (RK4 vs RK45)
.\build\windows-msvc\bin\Release\sp4_rk45.exe

# 특정 태그 테스트
.\build\windows-msvc\bin\Release\unit_tests.exe "[demag]"
.\build\windows-msvc\bin\Release\unit_tests.exe "[rk45]"
```

---

## 계획된 다음 단계

### 단기 (코딩 준비 완료)

| 우선순위 | 작업 | 예상 작업량 |
|---------|------|------------|
| ★★☆ | **SP#4 전체 수렴** (2–3 ns) | sp4.cpp t_end 변경 1줄 |
| ★★☆ | **Python 바인딩 정비** | DemagField + RK45 pybind11 노출 |
| ★☆☆ | **CUDA Phase 3** | cuFFT GPU demag 가속 |

### 장기 계획 — 유한 온도 마이크로마그네틱스

> **핵심 제약**: 유한 온도에서는 RK45 **사용 불가** — 고정 Δt 필수

#### 왜 RK45를 사용할 수 없나

확률론적 LLG에 추가되는 Langevin 열 노이즈:

```
H_th[i] = η × sqrt(2α k_B T / (μ₀ Ms γ V Δt))
               ↑ 가우시안 노이즈        ↑ Δt에 반비례
```

- **Δt 변경 → 노이즈 진폭 변경 → 다른 물리 시뮬레이션**
- RK45 오차 추정 = 수치 오차 + 랜덤 노이즈 → 오차 추정 불가
- 확률 미분방정식(SDE)은 결정론적 ODE 적응법 적용 불가
- → **Euler-Maruyama (1차) 또는 Heun 방법 (2차) 필수**

#### 구현 계획

**Phase T1 — `ThermalField`**:
```cpp
class ThermalField : public IEffectiveField {
    // σ = sqrt(2α k_B T / (μ₀ Ms γ V dt))
    // 매 스텝마다 N(0, σ) 독립 가우시안 벡터
    void set_temperature(Real T_K);
    void set_dt(Real dt);
};
```

**Phase T2 — `HeunIntegrator`** (고정 Δt, Stratonovich):
```
예측:   m̃ = m + Δt × f(m,  H_th^n)
수정:   m_{n+1} = m + Δt/2 × [f(m, H_th^n) + f(m̃, H_th^n)]
```

**Phase T3 — 검증**:

| 검증 항목 | 방법 |
|-----------|------|
| 열 평형 분포 | `<m_z²>` = k_BT/(μ₀MsKV) |
| Néel-Brown 이완 | τ = τ₀ exp(KV/k_BT) |
| 초상자성 | KV/k_BT ≪ 1 → `<|m|>` ≈ 0 |
| 열적 전환 | SP#4 + T=300K → 전환 확률 분포 |

**응용 분야**: 초상자성, HAMR (열 지원 자기 기록), 스핀 칼로리트로닉스
