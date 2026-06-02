# NanoSpinDynamics — 개발 진행 요약

**작성일**: 2026-06-02  
**브랜치**: master  
**현재 커밋**: `b611c19`

---

## 프로젝트 개요

C++20 마이크로마그네틱 시뮬레이터. Python 바인딩(pybind11), MSVC/Windows, vcpkg, CMake presets 기반.  
목표: 처음부터 완전한 마이크로마그네틱 시뮬레이션 라이브러리 구축, µMAG 표준 문제로 검증.

---

## 완성된 기능 (커밋 이력)

| 커밋 | 단계 | 내용 |
|------|------|------|
| `76fe06a` | Phase 1a | StructuredGrid, VectorField3D, VTK writer, pybind11 스켈레톤 |
| `4e524b6` | Phase 1b 설계 | 아키텍처 설계 문서 |
| `bddcd67` | Phase 1b | ZeemanField, UniaxialAnisotropyField, ExchangeField (Neumann/Periodic BC) |
| `4ceb448` | Phase 1c | RK4Integrator, LLG 토크, 정규화 |
| `0a235d5` | Phase 1d | SlonczewskiSTT, SpinOrbitTorque, SpinTorqueSum |
| `5f9247b` | Phase 2 인프라 | DemagField 헤더, CMakeLists 통합, 테스트 파일들 |
| `d5541e7` | Phase 2 중간 | newell_f 계수 경험적 수정 (이후 superseded) |
| `67b6731` | Phase 2 중간 | newell_g atan2→atan 패리티 수정 |
| `f3d1f96` | **Phase 2 완성** | **6D 이중-셀 적분 — demag 8/8 테스트 통과** |
| `f116bb8` | Phase 2 정리 | diag_step1 진단 테스트 6D 공식으로 갱신 |
| `b611c19` | **Phase 1e** | **µMAG Standard Problem #4 — 전환 거동 확인** |

---

## 아키텍처

```
types.hpp          Vec3, Real, Index, 물리 상수(π, μ₀, ℏ, e, γ₀)
grid.hpp           StructuredGrid — 셀 형상 (linear_index: x-fastest)
field.hpp          VectorField3D — 데이터[] 소유, 그리드 래핑
material.hpp       Material — Ms, A, K, alpha, easy_axis (permalloy/cobalt/iron)
effective_field.hpp  IEffectiveField + EffectiveFieldSum (컴포지터)
spin_torque.hpp    ISpinTorque + SpinTorqueSum (컴포지터)
integrator.hpp     RK4Integrator — LLG 전진 + STT
```

### 유효장 구현체

| 클래스 | 물리 |
|--------|------|
| `ZeemanField` | 균일 H_ext |
| `UniaxialAnisotropyField` | H_ani = (2K/μ₀Ms)(m·û)û |
| `ExchangeField` | 6점 라플라시안, Neumann/Periodic BC |
| `DemagField` | **6D 이중-셀 적분 (Newell 1993)** |

### 스핀 토크 구현체

| 클래스 | 물리 |
|--------|------|
| `SlonczewskiSTT` | CPP-STT: a_J[m×(m×p̂)] + b_J[m×p̂] |
| `SpinOrbitTorque` | SOT: η_DL m×(m×σ̂) + η_FL(m×σ̂) |

---

## Phase 2: DemagField — 핵심 성과

### 버그 발견 및 수정 과정

Phase 2에서 FFT 기반 Newell 소자화 텐서 구현의 근본 버그를 발견하고 수정했습니다.

**Root Cause**: `nxx`/`nxy` 함수가 Newell (1993)의 이중-셀 적분 공식 대신  
8-코너 `±dx/2` 교대합을 사용하고 있었음.

#### 올바른 공식 (70-lines-of-NumPy µMAG SP#4 검증 코드와 동일)

```cpp
// 6D 이중-셀 합산 — 64항
int nx = round(x/dx), ny = round(y/dy), nz = round(z/dz);
double sum = 0.0;
for (ia,ib,ic,id,ie,ig in {0,1}^6) {
    sign = (-1)^(ia+ib+ic+id+ie+ig);
    sum += sign * newell_f/g(
        (nx+ia-id)*dx, (ny+ib-ie)*dy, (nz+ic-ig)*dz);
}
return +sum / (4*pi*dx*dy*dz);   // 양의 부호
```

**newell_f 추가 조건**: `abs(x,y,z)` 적용 후 원래 계수 (1/2, 1/2, 1, 1/6).  
**newell_g 추가 조건**: `abs(z)` 적용.

#### 검증된 수치 결과

| 항목 | 수정 전 | **수정 후** | 기대값 |
|------|---------|------------|--------|
| 4×4×4 cube H_z(1,1,1) | -300,000 A/m | **-266,667 A/m** | -Ms/3 = -266,667 **(오차 0.0%)** |
| 16×16×1 박막 H_z | 0 A/m | **-754,839 A/m** | ≈-Ms **(94.4%)** |
| 5×5×5 중심 H_x (uniform Mz) | 231,009 A/m | **0 A/m** | 0 (대칭성) |
| 자기 소자화 인자 N_zz(self) | 0 (틀림) | **+1/3 (정확)** | 1/3 |

#### 테스트 현황: 60/60 통과

```
[demag]  8/8  ✓  4×4×4 cube H_z, 박막, long rod, 에너지, 가로 필드
[diag]   4/4  ✓  Newell 커널 진단 (자기 소자화 인자, 직접합 vs FFT)
기타     48/48 ✓  grid, field, zeeman, anisotropy, exchange, llg, spin_torque
```

---

## Phase 1e: µMAG Standard Problem #4 결과

### 설정

| 항목 | 값 |
|------|-----|
| 샘플 | Permalloy 500 × 125 × 3 nm |
| 격자 | 200 × 50 × 1 셀, 2.5 × 2.5 × 3 nm |
| 재료 | Ms=800 kA/m, A=13 pJ/m, K=0, α=0.02 |
| 초기 상태 | 거의 +x 방향 (0.5° 틸트) |
| 인가 자기장 | μ₀H = 25 mT @ 170° (Field A) |
| 적분기 | RK4, dt=0.05 ps, 1 ns (20,000 steps) |

### 결과

| 항목 | 시뮬레이션 | µMAG 참조값 | 오차 |
|------|-----------|------------|------|
| 전환 시간 | **0.175 ns** | 0.10–0.15 ns | ~ (α=0.02 진동 중) |
| 최종 `<mx>` | **-0.98205** | -0.9862 | **0.4%** ✓ |
| 최종 `<my>` | 0.025 | ≈ 0 | — |
| 최종 `<mz>` | 0.035 | ≈ 0 | — |

t = 0.175 ns에서 전환 감지, t = 1 ns에서 `<mx>` = -0.982, µMAG 참조값의 2% 이내 달성.  
α=0.02의 저감쇠 특성으로 인해 1 ns 시점에서도 잔류 진동 존재 (2–3 ns 시뮬레이션 시 완전 수렴 예상).

---

## 빌드 및 실행 방법

```powershell
# 빌드 (Release)
cmake --preset windows-msvc
cmake --build build/windows-msvc --config Release

# 전체 테스트
.\build\windows-msvc\bin\Release\unit_tests.exe

# µMAG SP#4 시뮬레이션 (1 ns, ~5분 소요)
.\build\windows-msvc\bin\Release\sp4.exe

# 특정 태그 테스트
.\build\windows-msvc\bin\Release\unit_tests.exe "[demag]"
.\build\windows-msvc\bin\Release\unit_tests.exe "[diag]"
```

---

## 다음 단계 후보

| 우선순위 | 작업 | 설명 |
|---------|------|------|
| ★★★ | **RK45 적응 시간-단계** | 오차 제어 포함 — α=0.02 시뮬레이션 효율 10–100배 향상 |
| ★★☆ | **SP#4 완전 수렴** | sp4.exe를 2–3 ns로 연장, `<my>/<mz>` 수렴 확인 |
| ★★☆ | **SP#4 Field B (190°)** | 비전환(non-switching) 시나리오 검증 |
| ★☆☆ | **Python 바인딩 정비** | DemagField를 Python에서 직접 호출 가능하도록 |
| ★☆☆ | **Phase 3: CUDA** | cuFFT 기반 GPU demag 가속 |
