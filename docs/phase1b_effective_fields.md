# Phase 1b: 첫 물리 — Effective Field 구현

> Phase 1a가 끝났다는 가정. 격자, 벡터장, VTK I/O, Python 바인딩, 테스트 인프라가 모두 동작하는 상태에서 시작합니다. 이번 단계에서 처음으로 물리(Zeeman, 단축 anisotropy, Heisenberg exchange)를 넣습니다.

---

## 목차

- [1. 목표 및 범위](#1-목표-및-범위)
- [2. 물리 배경](#2-물리-배경)
- [3. 설계 결정](#3-설계-결정)
- [4. 디렉터리 변화 (1a → 1b)](#4-디렉터리-변화-1a--1b)
- [5. 사전 준비](#5-사전-준비)
- [6. Claude Code 활용 워크플로우](#6-claude-code-활용-워크플로우)
- [7. 파일별 작성 가이드](#7-파일별-작성-가이드)
- [8. 빌드 및 실행](#8-빌드-및-실행)
- [9. 검증 (이번 단계의 핵심)](#9-검증-이번-단계의-핵심)
- [10. 흔한 문제 해결](#10-흔한-문제-해결)
- [11. 다음 단계 (Phase 1c: LLG integrator)](#11-다음-단계-phase-1c-llg-integrator)

---

## 1. 목표 및 범위

### 만들 것
- ✅ `Material` 구조체 (Ms, A, K, easy_axis, alpha + 표준 재료 factory)
- ✅ `IEffectiveField` 추상 인터페이스 + `EffectiveFieldSum` 합산 클래스
- ✅ `ZeemanField`: 외부 자기장
- ✅ `UniaxialAnisotropyField`: 단축 결정 이방성
- ✅ `ExchangeField`: 6-point Laplacian, Neumann/Periodic BC
- ✅ Energy 계산 (검증용)
- ✅ 물리 sanity check 단위 테스트 다수
- ✅ `field_demo` 실행파일: 자화 + H_eff를 VTK로 저장
- ✅ Python 바인딩 확장 (위 클래스 모두)

### 미루는 것
| 항목 | 시점 |
|---|---|
| LLG integrator (RK4, RK45) | Phase 1c |
| Slonczewski STT | Phase 1d |
| OOMMF OVF I/O | Phase 1e |
| µMAG standard problem 검증 | Phase 1 마무리 |
| Demag (FFTW/cuFFT) | Phase 2 |
| Cubic anisotropy, interfacial DMI | Phase 3 |
| Per-region materials | Phase 4 |

### 성공 기준
1. 새 단위 테스트가 **모두 통과** (대략 20개 이상)
2. `field_demo` 실행 → vortex 자화 + H_eff 두 개의 VTK 파일 생성
3. ParaView에서 자화와 effective field가 물리적으로 합리적
4. Python에서 `mm.ZeemanField(...)`, `mm.ExchangeField()`, `mm.EffectiveFieldSum()` 등 모두 동작
5. Phase 1a의 모든 테스트도 여전히 통과 (회귀 없음)

---

## 2. 물리 배경

### Micromagnetic energy
연속체 micromagnetics의 자유 에너지:

```
E[m] = ∫ [ A|∇m|²  −  K(m·û)²  −  μ₀ Ms m·H_ext  −  (μ₀/2) Ms m·H_demag ] dV
       (exchange)   (anisotropy)  (Zeeman)         (demag, Phase 2)
```

여기서 `m`은 단위 자화 (`|m| = 1`), `M = Ms · m`.

### Effective field
시간 발전(LLG)에서 토크를 만드는 양:

```
H_eff = − (1 / μ₀ Ms) δE/δm
```

각 항별:
- **Zeeman**: `H_zeeman = H_ext`
- **Uniaxial anisotropy**: `H_aniso = (2K / μ₀ Ms) (m·û) û`
- **Exchange**: `H_ex = (2A / μ₀ Ms) ∇²m`

각각의 단위는 모두 [A/m] (SI).

### Exchange의 이산화 (6-point stencil)
구조 격자에서:

```
∇²m_ijk ≈   (m_{i+1,j,k} − 2 m_ijk + m_{i−1,j,k}) / dx²
          + (m_{i,j+1,k} − 2 m_ijk + m_{i,j−1,k}) / dy²
          + (m_{i,j,k+1} − 2 m_ijk + m_{i,j,k−1}) / dz²
```

또는 등가 형태:

```
∇²m_ijk ≈ Σ_neighbors (m_neighbor − m_ijk) / h²_direction
```

각 방향마다 ± 두 이웃, 총 6개. 경계에서 누락된 이웃은 **경계조건**으로 처리:

- **Neumann (free)**: `m_ghost = m_cell` → 기여 = 0. 자유 표면. 기본값.
- **Periodic (PBC)**: `m_ghost = m_opposite_side`. 반복 격자.

### Energy density (검증용)

| 항 | density [J/m³] | 균일 m에서 |
|---|---|---|
| Zeeman | `−μ₀ Ms (m·H_ext)` | `−μ₀ Ms (m·H_ext) V_total` |
| Anisotropy | `−K (m·û)²` | `m∥û`: `−K V_total`; `m⊥û`: `0` |
| Exchange | `A \|∇m\|²` | 항상 `0` |

이 표가 단위 테스트의 청사진입니다.

### 표준 재료 파라미터
| 재료 | Ms [A/m] | A [J/m] | K [J/m³] | α |
|---|---|---|---|---|
| Permalloy (Py) | 8.0×10⁵ | 1.3×10⁻¹¹ | ~0 | 0.02 |
| Cobalt (hcp Co) | 1.4×10⁶ | 3.0×10⁻¹¹ | 4.5×10⁵ | 0.05 |
| Iron (bcc Fe) | 1.7×10⁶ | 2.1×10⁻¹¹ | 4.8×10⁴ | 0.02 |

---

## 3. 설계 결정

### IEffectiveField 인터페이스

```cpp
class IEffectiveField {
public:
    virtual void accumulate(const VectorField3D& m,
                            const Material& mat,
                            VectorField3D& H_out) const = 0;
    virtual Real energy(const VectorField3D& m,
                        const Material& mat) const = 0;
    virtual const char* name() const = 0;
    virtual ~IEffectiveField() = default;
};
```

**왜 accumulate인가?**: `H_out`을 매번 새로 만드는 것보다 누적이 빠르고 메모리 효율적. `EffectiveFieldSum::compute()`이 H_out을 한 번 zero-out한 뒤 모든 term을 더함.

### Material 구조체

전체 시뮬레이션에서 단일 global Material. POD 스타일.

```cpp
struct Material {
    Real Ms;
    Real A_exchange;
    Real K_uniaxial;
    Vec3 easy_axis;
    Real alpha;        // LLG에서 사용 (Phase 1c)
    
    static Material permalloy();
    static Material cobalt();
    static Material iron();
};
```

### shared_ptr 소유 모델
`EffectiveFieldSum`은 `std::shared_ptr<IEffectiveField>` 리스트를 보유. pybind11과의 통합이 깔끔.

```cpp
EffectiveFieldSum sum;
sum.add(std::make_shared<ZeemanField>(H_ext));
sum.add(std::make_shared<UniaxialAnisotropyField>());
sum.add(std::make_shared<ExchangeField>());
```

Python에서:

```python
sum = mm.EffectiveFieldSum()
sum.add(mm.ZeemanField(H_ext))
sum.add(mm.UniaxialAnisotropyField())
sum.add(mm.ExchangeField())
```

### 단위계 일관성
모든 값은 SI. 헤더 한 곳에 물리 상수를 정의:

```cpp
namespace micromag::constants {
    inline constexpr Real mu_0 = 4.0 * 3.14159265358979323846 * 1e-7;  // T·m/A
    inline constexpr Real pi   = 3.14159265358979323846;
}
```

### 경계조건
`enum class BoundaryCondition { Neumann, Periodic };`

Phase 1b: ExchangeField에만 적용. Phase 2부터 demag, Phase 3부터 DMI 등에도.

---

## 4. 디렉터리 변화 (1a → 1b)

**Phase 1a 기준:**
```
include/micromag/
├── types.hpp           # mu_0, pi 상수 추가됨
├── grid.hpp            # 변경 없음
├── field.hpp           # 변경 없음
└── vtk_writer.hpp      # 변경 없음
src/
├── grid.cpp
├── field.cpp
└── vtk_writer.cpp
```

**Phase 1b에서 추가:**
```
include/micromag/
├── material.hpp           # NEW
├── effective_field.hpp    # NEW: BoundaryCondition + IEffectiveField + EffectiveFieldSum
├── zeeman.hpp             # NEW
├── anisotropy.hpp         # NEW
└── exchange.hpp           # NEW
src/
├── material.cpp           # NEW
├── effective_field.cpp    # NEW
├── zeeman.cpp             # NEW
├── anisotropy.cpp         # NEW
└── exchange.cpp           # NEW
apps/
└── field_demo.cpp         # NEW
tests/
├── test_zeeman.cpp        # NEW
├── test_anisotropy.cpp    # NEW
└── test_exchange.cpp      # NEW
examples/
└── field_demo.py          # NEW
```

**수정되는 파일:**
- `include/micromag/types.hpp` — `constants` namespace 추가
- `CMakeLists.txt` — 새 cpp 파일들을 `micromag_core`에 추가
- `tests/CMakeLists.txt` — 새 테스트 파일들을 `unit_tests`에 추가
- `python/bindings.cpp` — 새 클래스 바인딩 추가

---

## 5. 사전 준비

**새 의존성 없음.** Phase 1a에서 설치한 것들로 충분합니다 (Catch2, pybind11, fftw3, hdf5, nlohmann-json).

다만 Phase 1a가 완전히 통과한 상태인지 한 번 확인:

```powershell
cd D:\dev\micromag
cmake --build --preset windows-msvc
ctest --preset windows-msvc
.\build\windows-msvc\bin\Release\hello_micromag.exe
```

Phase 1a의 모든 테스트가 통과해야 Phase 1b 시작. Phase 1b 작업 중 회귀가 생기면 즉시 발견 가능.

Phase 1a checkpoint commit 권장:
```powershell
git status              # clean이어야 함
git log --oneline -5    # Phase 1a commit 확인
git checkout -b phase1b # 새 브랜치 (선택)
```

---

## 6. Claude Code 활용 워크플로우

Phase 1a와 동일한 패턴, 더 신중하게 분할:

### 위임 1 — 상수 + Material
```
@docs/phase1b_effective_fields.md 의 7.1, 7.2절에 따라
include/micromag/types.hpp 에 constants namespace를 추가하고,
material.hpp 와 material.cpp 를 만들어줘.
```

### 위임 2 — Effective field 추상 인터페이스
```
@docs/phase1b_effective_fields.md 의 7.3, 7.4절에 따라
effective_field.hpp 와 effective_field.cpp 를 만들어줘.
BoundaryCondition enum 도 포함해야 해.
```

### 위임 3 — Zeeman (가장 쉬운 것부터)
```
@docs/phase1b_effective_fields.md 의 7.5, 7.6절에 따라
zeeman.hpp, zeeman.cpp 를 만들고
tests/test_zeeman.cpp 도 같이 만들어줘.
이 시점에 빌드+테스트 통과해야 해.
```

### 위임 4 — Anisotropy
```
@docs/phase1b_effective_fields.md 의 7.7, 7.8절에 따라
anisotropy.hpp, anisotropy.cpp, tests/test_anisotropy.cpp 를 만들어줘.
빌드+테스트 통과 확인.
```

### 위임 5 — Exchange (가장 복잡, 가장 위험)
```
@docs/phase1b_effective_fields.md 의 7.9, 7.10절에 따라
exchange.hpp, exchange.cpp, tests/test_exchange.cpp 를 만들어줘.
경계조건(Neumann, Periodic) 처리에 특히 주의.
빌드+테스트 통과 확인.
```

### 위임 6 — Demo 앱 + CMake 업데이트 + Python 바인딩
```
@docs/phase1b_effective_fields.md 의 7.11~7.15절에 따라
field_demo.cpp, field_demo.py 를 만들고
CMakeLists.txt, tests/CMakeLists.txt, python/bindings.cpp 를 업데이트해줘.
빌드+테스트+demo 실행+Python 예제 실행이 모두 성공해야 해.
```

**각 위임 후 git commit**:
```powershell
git add .
git commit -m "Phase 1b step N: <설명>"
```

이렇게 하면 어느 step에서 문제가 생겨도 깨끗하게 되돌아갈 수 있습니다.

---

## 7. 파일별 작성 가이드

### 7.1 `include/micromag/types.hpp` — 상수 추가

기존 파일 끝에 (또는 `Vec3` 다음에) `constants` namespace 추가:

```cpp
namespace micromag {

// (기존 Real, Index, Vec3 정의 유지)

namespace constants {
inline constexpr Real pi = 3.14159265358979323846;
inline constexpr Real mu_0 = 4.0 * pi * 1e-7;  // [T·m/A]
}  // namespace constants

}  // namespace micromag
```

### 7.2 `include/micromag/material.hpp` + `src/material.cpp`

**material.hpp**:
```cpp
#pragma once

#include "types.hpp"

namespace micromag {

// Single global material (Phase 1b). Per-region material in Phase 4.
struct Material {
    Real Ms{8e5};               // Saturation magnetization [A/m]
    Real A_exchange{1.3e-11};   // Exchange constant [J/m]
    Real K_uniaxial{0};         // Uniaxial anisotropy [J/m³]
    Vec3 easy_axis{0, 0, 1};    // Easy-axis direction (normalized at use)
    Real alpha{0.02};           // Gilbert damping (Phase 1c)

    static Material permalloy();
    static Material cobalt();
    static Material iron();
};

}  // namespace micromag
```

**material.cpp**:
```cpp
#include "micromag/material.hpp"

namespace micromag {

Material Material::permalloy() {
    Material m;
    m.Ms = 8.0e5;
    m.A_exchange = 1.3e-11;
    m.K_uniaxial = 0.0;
    m.easy_axis = {0, 0, 1};
    m.alpha = 0.02;
    return m;
}

Material Material::cobalt() {
    Material m;
    m.Ms = 1.4e6;
    m.A_exchange = 3.0e-11;
    m.K_uniaxial = 4.5e5;
    m.easy_axis = {0, 0, 1};
    m.alpha = 0.05;
    return m;
}

Material Material::iron() {
    Material m;
    m.Ms = 1.7e6;
    m.A_exchange = 2.1e-11;
    m.K_uniaxial = 4.8e4;
    m.easy_axis = {0, 0, 1};
    m.alpha = 0.02;
    return m;
}

}  // namespace micromag
```

### 7.3 `include/micromag/effective_field.hpp`

```cpp
#pragma once

#include <memory>
#include <vector>

#include "field.hpp"
#include "material.hpp"

namespace micromag {

enum class BoundaryCondition {
    Neumann,    // ∂m/∂n = 0 (free surface), default
    Periodic    // wrap-around
};

// Abstract base for any contribution to H_eff.
class IEffectiveField {
public:
    virtual ~IEffectiveField() = default;

    // Accumulate this field's contribution into H_out.
    // Caller is responsible for zeroing H_out before the first accumulate.
    virtual void accumulate(const VectorField3D& m,
                            const Material& mat,
                            VectorField3D& H_out) const = 0;

    // Total energy [J] integrated over the grid for this contribution.
    virtual Real energy(const VectorField3D& m,
                        const Material& mat) const = 0;

    virtual const char* name() const = 0;
};

// Owns a collection of IEffectiveField terms and computes their sum.
class EffectiveFieldSum {
public:
    EffectiveFieldSum() = default;

    void add(std::shared_ptr<IEffectiveField> term);

    // Compute total H_eff: zeroes H_out, then accumulates every term.
    void compute(const VectorField3D& m,
                 const Material& mat,
                 VectorField3D& H_out) const;

    // Total energy summed over all terms [J].
    Real total_energy(const VectorField3D& m, const Material& mat) const;

    std::size_t num_terms() const { return terms_.size(); }
    const std::vector<std::shared_ptr<IEffectiveField>>& terms() const { return terms_; }

private:
    std::vector<std::shared_ptr<IEffectiveField>> terms_;
};

}  // namespace micromag
```

### 7.4 `src/effective_field.cpp`

```cpp
#include "micromag/effective_field.hpp"

namespace micromag {

void EffectiveFieldSum::add(std::shared_ptr<IEffectiveField> term) {
    if (term) {
        terms_.push_back(std::move(term));
    }
}

void EffectiveFieldSum::compute(const VectorField3D& m,
                                 const Material& mat,
                                 VectorField3D& H_out) const {
    // Zero H_out
    for (Index idx = 0; idx < H_out.size(); ++idx) {
        H_out[idx] = Vec3{0, 0, 0};
    }
    for (const auto& term : terms_) {
        term->accumulate(m, mat, H_out);
    }
}

Real EffectiveFieldSum::total_energy(const VectorField3D& m, const Material& mat) const {
    Real total = 0;
    for (const auto& term : terms_) {
        total += term->energy(m, mat);
    }
    return total;
}

}  // namespace micromag
```

### 7.5 `include/micromag/zeeman.hpp` + `src/zeeman.cpp`

**zeeman.hpp**:
```cpp
#pragma once

#include "effective_field.hpp"
#include "types.hpp"

namespace micromag {

// External (applied) field. Uniform in space and constant in time (Phase 1b).
// Spatial/temporal variation: Phase 1c+.
class ZeemanField : public IEffectiveField {
public:
    explicit ZeemanField(const Vec3& H_ext = {0, 0, 0}) : H_ext_(H_ext) {}

    void accumulate(const VectorField3D& m,
                    const Material& mat,
                    VectorField3D& H_out) const override;

    Real energy(const VectorField3D& m, const Material& mat) const override;

    const char* name() const override { return "Zeeman"; }

    Vec3 H_ext() const { return H_ext_; }
    void set_H_ext(const Vec3& H) { H_ext_ = H; }

private:
    Vec3 H_ext_;
};

}  // namespace micromag
```

**zeeman.cpp**:
```cpp
#include "micromag/zeeman.hpp"

#include "micromag/grid.hpp"

namespace micromag {

void ZeemanField::accumulate(const VectorField3D& m,
                              const Material& /*mat*/,
                              VectorField3D& H_out) const {
    for (Index idx = 0; idx < m.size(); ++idx) {
        H_out[idx] += H_ext_;
    }
}

Real ZeemanField::energy(const VectorField3D& m, const Material& mat) const {
    Real sum_dot = 0;
    for (Index idx = 0; idx < m.size(); ++idx) {
        sum_dot += m[idx].dot(H_ext_);
    }
    // E = - μ₀ Ms ∫ (m · H_ext) dV
    return -constants::mu_0 * mat.Ms * sum_dot * m.grid().cell_volume();
}

}  // namespace micromag
```

### 7.6 `tests/test_zeeman.cpp`

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/material.hpp"
#include "micromag/zeeman.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("Zeeman: H_eff equals H_ext at every cell", "[zeeman]") {
    StructuredGrid g(4, 5, 6, 1e-9, 1e-9, 1e-9);
    VectorField3D m(g);
    VectorField3D H(g);
    m.set_uniform({1, 0, 0});

    Vec3 H_ext{1e5, 2e5, -3e5};
    ZeemanField z(H_ext);
    Material mat = Material::permalloy();

    z.accumulate(m, mat, H);

    for (Index idx = 0; idx < g.size(); ++idx) {
        REQUIRE_THAT(H[idx].x, WithinAbs(H_ext.x, 1e-6));
        REQUIRE_THAT(H[idx].y, WithinAbs(H_ext.y, 1e-6));
        REQUIRE_THAT(H[idx].z, WithinAbs(H_ext.z, 1e-6));
    }
}

TEST_CASE("Zeeman: accumulate adds to existing H_out", "[zeeman]") {
    StructuredGrid g(2, 2, 2, 1e-9, 1e-9, 1e-9);
    VectorField3D m(g);
    VectorField3D H(g);
    m.set_uniform({0, 0, 1});

    // Pre-fill H with a known value
    for (Index idx = 0; idx < g.size(); ++idx) {
        H[idx] = {10, 20, 30};
    }

    ZeemanField z({1, 2, 3});
    z.accumulate(m, Material::permalloy(), H);

    REQUIRE_THAT(H.at(0, 0, 0).x, WithinAbs(11.0, 1e-6));
    REQUIRE_THAT(H.at(0, 0, 0).y, WithinAbs(22.0, 1e-6));
    REQUIRE_THAT(H.at(0, 0, 0).z, WithinAbs(33.0, 1e-6));
}

TEST_CASE("Zeeman energy: aligned m gives -μ₀ Ms |H| V", "[zeeman][energy]") {
    StructuredGrid g(4, 4, 4, 2e-9, 2e-9, 2e-9);
    VectorField3D m(g);
    Vec3 H_ext{1e5, 0, 0};
    m.set_uniform({1, 0, 0});  // aligned with H_ext

    ZeemanField z(H_ext);
    Material mat = Material::permalloy();
    Real E = z.energy(m, mat);

    Real V_total = static_cast<Real>(g.size()) * g.cell_volume();
    Real E_expected = -constants::mu_0 * mat.Ms * 1e5 * V_total;
    REQUIRE_THAT(E, WithinRel(E_expected, 1e-12));
}

TEST_CASE("Zeeman energy: antiparallel m gives +μ₀ Ms |H| V", "[zeeman][energy]") {
    StructuredGrid g(3, 3, 3, 1e-9, 1e-9, 1e-9);
    VectorField3D m(g);
    Vec3 H_ext{0, 0, 1e5};
    m.set_uniform({0, 0, -1});  // antiparallel

    ZeemanField z(H_ext);
    Material mat = Material::permalloy();
    Real E = z.energy(m, mat);

    Real V_total = static_cast<Real>(g.size()) * g.cell_volume();
    Real E_expected = +constants::mu_0 * mat.Ms * 1e5 * V_total;
    REQUIRE_THAT(E, WithinRel(E_expected, 1e-12));
}

TEST_CASE("Zeeman energy: perpendicular m gives zero", "[zeeman][energy]") {
    StructuredGrid g(3, 3, 3, 1e-9, 1e-9, 1e-9);
    VectorField3D m(g);
    m.set_uniform({1, 0, 0});

    ZeemanField z({0, 0, 1e5});  // perpendicular to m
    Real E = z.energy(m, Material::permalloy());

    REQUIRE_THAT(E, WithinAbs(0.0, 1e-30));
}
```

### 7.7 `include/micromag/anisotropy.hpp` + `src/anisotropy.cpp`

**anisotropy.hpp**:
```cpp
#pragma once

#include "effective_field.hpp"

namespace micromag {

// Uniaxial anisotropy:
//   E = -K (m·û)²  per unit volume,  K > 0 → easy-axis along û
//   H = (2K / μ₀ Ms) (m·û) û
// û is taken from Material::easy_axis (normalized internally).
class UniaxialAnisotropyField : public IEffectiveField {
public:
    void accumulate(const VectorField3D& m,
                    const Material& mat,
                    VectorField3D& H_out) const override;

    Real energy(const VectorField3D& m, const Material& mat) const override;

    const char* name() const override { return "UniaxialAnisotropy"; }
};

}  // namespace micromag
```

**anisotropy.cpp**:
```cpp
#include "micromag/anisotropy.hpp"

#include "micromag/grid.hpp"
#include "micromag/types.hpp"

namespace micromag {

void UniaxialAnisotropyField::accumulate(const VectorField3D& m,
                                          const Material& mat,
                                          VectorField3D& H_out) const {
    if (mat.K_uniaxial == 0) return;

    Vec3 u = mat.easy_axis;
    Real u_norm = u.norm();
    if (u_norm < 1e-30) return;
    u = u / u_norm;

    const Real prefactor = 2.0 * mat.K_uniaxial / (constants::mu_0 * mat.Ms);

    for (Index idx = 0; idx < m.size(); ++idx) {
        Real m_dot_u = m[idx].dot(u);
        H_out[idx] += u * (prefactor * m_dot_u);
    }
}

Real UniaxialAnisotropyField::energy(const VectorField3D& m, const Material& mat) const {
    if (mat.K_uniaxial == 0) return 0;

    Vec3 u = mat.easy_axis;
    Real u_norm = u.norm();
    if (u_norm < 1e-30) return 0;
    u = u / u_norm;

    Real sum_sq = 0;
    for (Index idx = 0; idx < m.size(); ++idx) {
        Real x = m[idx].dot(u);
        sum_sq += x * x;
    }
    return -mat.K_uniaxial * sum_sq * m.grid().cell_volume();
}

}  // namespace micromag
```

### 7.8 `tests/test_anisotropy.cpp`

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/material.hpp"
#include "micromag/anisotropy.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("Anisotropy: K=0 gives zero H", "[anisotropy]") {
    StructuredGrid g(3, 3, 3, 1e-9, 1e-9, 1e-9);
    VectorField3D m(g);
    VectorField3D H(g);
    m.set_uniform({1, 0, 0});

    Material mat = Material::permalloy();  // K=0
    UniaxialAnisotropyField a;
    a.accumulate(m, mat, H);

    for (Index idx = 0; idx < g.size(); ++idx) {
        REQUIRE_THAT(H[idx].norm(), WithinAbs(0.0, 1e-30));
    }
}

TEST_CASE("Anisotropy: m∥easy-axis gives H = (2K/μ₀Ms) û", "[anisotropy]") {
    StructuredGrid g(2, 2, 2, 1e-9, 1e-9, 1e-9);
    VectorField3D m(g);
    VectorField3D H(g);
    m.set_uniform({0, 0, 1});

    Material mat = Material::cobalt();  // K > 0, easy_axis = (0,0,1)
    UniaxialAnisotropyField a;
    a.accumulate(m, mat, H);

    Real H_expected = 2.0 * mat.K_uniaxial / (constants::mu_0 * mat.Ms);
    for (Index idx = 0; idx < g.size(); ++idx) {
        REQUIRE_THAT(H[idx].z, WithinRel(H_expected, 1e-10));
        REQUIRE_THAT(H[idx].x, WithinAbs(0.0, 1e-6));
        REQUIRE_THAT(H[idx].y, WithinAbs(0.0, 1e-6));
    }
}

TEST_CASE("Anisotropy: m⊥easy-axis gives H = 0", "[anisotropy]") {
    StructuredGrid g(2, 2, 2, 1e-9, 1e-9, 1e-9);
    VectorField3D m(g);
    VectorField3D H(g);
    m.set_uniform({1, 0, 0});  // perpendicular to easy_axis=(0,0,1)

    Material mat = Material::cobalt();
    UniaxialAnisotropyField a;
    a.accumulate(m, mat, H);

    for (Index idx = 0; idx < g.size(); ++idx) {
        REQUIRE_THAT(H[idx].norm(), WithinAbs(0.0, 1e-6));
    }
}

TEST_CASE("Anisotropy energy: m∥easy-axis gives -K V_total", "[anisotropy][energy]") {
    StructuredGrid g(4, 4, 4, 2e-9, 2e-9, 2e-9);
    VectorField3D m(g);
    m.set_uniform({0, 0, 1});

    Material mat = Material::cobalt();
    UniaxialAnisotropyField a;
    Real E = a.energy(m, mat);

    Real V_total = static_cast<Real>(g.size()) * g.cell_volume();
    Real E_expected = -mat.K_uniaxial * V_total;
    REQUIRE_THAT(E, WithinRel(E_expected, 1e-12));
}

TEST_CASE("Anisotropy energy: m⊥easy-axis gives 0", "[anisotropy][energy]") {
    StructuredGrid g(3, 3, 3, 1e-9, 1e-9, 1e-9);
    VectorField3D m(g);
    m.set_uniform({1, 0, 0});

    UniaxialAnisotropyField a;
    Real E = a.energy(m, Material::cobalt());
    REQUIRE_THAT(E, WithinAbs(0.0, 1e-30));
}

TEST_CASE("Anisotropy energy: 45° gives -K V/2", "[anisotropy][energy]") {
    StructuredGrid g(2, 2, 2, 1e-9, 1e-9, 1e-9);
    VectorField3D m(g);
    Real s = 1.0 / std::sqrt(2.0);
    m.set_uniform({s, 0, s});  // 45° from z-axis

    Material mat = Material::cobalt();
    UniaxialAnisotropyField a;
    Real E = a.energy(m, mat);

    Real V_total = static_cast<Real>(g.size()) * g.cell_volume();
    Real E_expected = -mat.K_uniaxial * 0.5 * V_total;
    REQUIRE_THAT(E, WithinRel(E_expected, 1e-10));
}
```

### 7.9 `include/micromag/exchange.hpp` + `src/exchange.cpp`

**exchange.hpp**:
```cpp
#pragma once

#include "effective_field.hpp"

namespace micromag {

// Heisenberg exchange via 6-point Laplacian:
//   E = A ∫ |∇m|² dV
//   H = (2A / μ₀ Ms) ∇²m
class ExchangeField : public IEffectiveField {
public:
    explicit ExchangeField(BoundaryCondition bc = BoundaryCondition::Neumann)
        : bc_(bc) {}

    void accumulate(const VectorField3D& m,
                    const Material& mat,
                    VectorField3D& H_out) const override;

    Real energy(const VectorField3D& m, const Material& mat) const override;

    const char* name() const override { return "Exchange"; }

    BoundaryCondition boundary() const { return bc_; }
    void set_boundary(BoundaryCondition bc) { bc_ = bc; }

private:
    BoundaryCondition bc_;
};

}  // namespace micromag
```

**exchange.cpp**:
```cpp
#include "micromag/exchange.hpp"

#include "micromag/grid.hpp"
#include "micromag/types.hpp"

namespace micromag {

namespace {

// Fetch neighbor with the chosen boundary condition.
// For Neumann (free): returns the cell itself (gives zero contribution to Laplacian).
// For Periodic: wraps around.
inline Vec3 neighbor(const VectorField3D& m, const StructuredGrid& g,
                      Index i, Index j, Index k,
                      Index di, Index dj, Index dk,
                      BoundaryCondition bc) {
    Index ni = i + di;
    Index nj = j + dj;
    Index nk = k + dk;

    const bool out_x = (ni < 0) || (ni >= g.nx());
    const bool out_y = (nj < 0) || (nj >= g.ny());
    const bool out_z = (nk < 0) || (nk >= g.nz());

    if (out_x || out_y || out_z) {
        if (bc == BoundaryCondition::Neumann) {
            return m.at(i, j, k);  // ghost = self
        }
        // Periodic
        ni = (ni % g.nx() + g.nx()) % g.nx();
        nj = (nj % g.ny() + g.ny()) % g.ny();
        nk = (nk % g.nz() + g.nz()) % g.nz();
    }
    return m.at(ni, nj, nk);
}

}  // namespace

void ExchangeField::accumulate(const VectorField3D& m,
                                const Material& mat,
                                VectorField3D& H_out) const {
    if (mat.A_exchange == 0) return;

    const StructuredGrid& g = m.grid();
    const Real inv_dx2 = 1.0 / (g.dx() * g.dx());
    const Real inv_dy2 = 1.0 / (g.dy() * g.dy());
    const Real inv_dz2 = 1.0 / (g.dz() * g.dz());
    const Real prefactor = 2.0 * mat.A_exchange / (constants::mu_0 * mat.Ms);

    for (Index k = 0; k < g.nz(); ++k) {
        for (Index j = 0; j < g.ny(); ++j) {
            for (Index i = 0; i < g.nx(); ++i) {
                Vec3 mc = m.at(i, j, k);
                Vec3 lap{0, 0, 0};

                lap += (neighbor(m, g, i, j, k, +1, 0, 0, bc_) - mc) * inv_dx2;
                lap += (neighbor(m, g, i, j, k, -1, 0, 0, bc_) - mc) * inv_dx2;
                lap += (neighbor(m, g, i, j, k, 0, +1, 0, bc_) - mc) * inv_dy2;
                lap += (neighbor(m, g, i, j, k, 0, -1, 0, bc_) - mc) * inv_dy2;
                lap += (neighbor(m, g, i, j, k, 0, 0, +1, bc_) - mc) * inv_dz2;
                lap += (neighbor(m, g, i, j, k, 0, 0, -1, bc_) - mc) * inv_dz2;

                H_out.at(i, j, k) += lap * prefactor;
            }
        }
    }
}

Real ExchangeField::energy(const VectorField3D& m, const Material& mat) const {
    if (mat.A_exchange == 0) return 0;

    const StructuredGrid& g = m.grid();
    const Real V = g.cell_volume();
    const Real inv_dx2 = 1.0 / (g.dx() * g.dx());
    const Real inv_dy2 = 1.0 / (g.dy() * g.dy());
    const Real inv_dz2 = 1.0 / (g.dz() * g.dz());
    Real sum = 0;

    // Iterate only over +x, +y, +z neighbors to avoid double counting pairs.
    for (Index k = 0; k < g.nz(); ++k) {
        for (Index j = 0; j < g.ny(); ++j) {
            for (Index i = 0; i < g.nx(); ++i) {
                Vec3 mc = m.at(i, j, k);

                auto add_pair = [&](Index ni, Index nj, Index nk, Real inv_h2) {
                    Vec3 diff = m.at(ni, nj, nk) - mc;
                    sum += diff.norm_squared() * inv_h2;
                };

                // +x direction
                if (i + 1 < g.nx()) {
                    add_pair(i + 1, j, k, inv_dx2);
                } else if (bc_ == BoundaryCondition::Periodic) {
                    add_pair(0, j, k, inv_dx2);
                }
                // +y
                if (j + 1 < g.ny()) {
                    add_pair(i, j + 1, k, inv_dy2);
                } else if (bc_ == BoundaryCondition::Periodic) {
                    add_pair(i, 0, k, inv_dy2);
                }
                // +z
                if (k + 1 < g.nz()) {
                    add_pair(i, j, k + 1, inv_dz2);
                } else if (bc_ == BoundaryCondition::Periodic) {
                    add_pair(i, j, 0, inv_dz2);
                }
            }
        }
    }

    return mat.A_exchange * sum * V;
}

}  // namespace micromag
```

### 7.10 `tests/test_exchange.cpp`

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/material.hpp"
#include "micromag/exchange.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("Exchange: uniform m gives zero H (Neumann)", "[exchange][bc]") {
    StructuredGrid g(8, 8, 8, 2e-9, 2e-9, 2e-9);
    VectorField3D m(g);
    VectorField3D H(g);
    m.set_uniform({0, 0, 1});

    ExchangeField ex(BoundaryCondition::Neumann);
    Material mat = Material::permalloy();
    ex.accumulate(m, mat, H);

    for (Index idx = 0; idx < g.size(); ++idx) {
        REQUIRE_THAT(H[idx].norm(), WithinAbs(0.0, 1e-3));
    }
}

TEST_CASE("Exchange: uniform m gives zero H (Periodic)", "[exchange][bc]") {
    StructuredGrid g(8, 8, 8, 2e-9, 2e-9, 2e-9);
    VectorField3D m(g);
    VectorField3D H(g);
    m.set_uniform({0.6, 0.8, 0});

    ExchangeField ex(BoundaryCondition::Periodic);
    Material mat = Material::permalloy();
    mat.A_exchange = 1.3e-11;
    ex.accumulate(m, mat, H);

    for (Index idx = 0; idx < g.size(); ++idx) {
        REQUIRE_THAT(H[idx].norm(), WithinAbs(0.0, 1e-3));
    }
}

TEST_CASE("Exchange energy: uniform m gives zero (both BC)", "[exchange][energy]") {
    StructuredGrid g(6, 6, 6, 1e-9, 1e-9, 1e-9);
    VectorField3D m(g);
    m.set_uniform({1, 0, 0});

    Material mat = Material::permalloy();
    mat.A_exchange = 1.3e-11;

    ExchangeField ex_n(BoundaryCondition::Neumann);
    ExchangeField ex_p(BoundaryCondition::Periodic);

    REQUIRE_THAT(ex_n.energy(m, mat), WithinAbs(0.0, 1e-30));
    REQUIRE_THAT(ex_p.energy(m, mat), WithinAbs(0.0, 1e-30));
}

TEST_CASE("Exchange: A=0 gives zero H", "[exchange]") {
    StructuredGrid g(4, 4, 4, 1e-9, 1e-9, 1e-9);
    VectorField3D m(g);
    VectorField3D H(g);
    // Non-uniform magnetization
    for (Index k = 0; k < g.nz(); ++k)
        for (Index j = 0; j < g.ny(); ++j)
            for (Index i = 0; i < g.nx(); ++i)
                m.at(i, j, k) = {static_cast<Real>(i), 0, 0};

    Material mat = Material::permalloy();
    mat.A_exchange = 0;

    ExchangeField ex;
    ex.accumulate(m, mat, H);

    for (Index idx = 0; idx < g.size(); ++idx) {
        REQUIRE_THAT(H[idx].norm(), WithinAbs(0.0, 1e-30));
    }
}

TEST_CASE("Exchange: 1D cosine matches analytical Laplacian (interior)",
          "[exchange][analytical]") {
    // m_x = cos(k x), m_y = m_z = 0 (we relax |m|=1 for this analytical test)
    // ∇²m_x = -k² cos(k x) → H_x = -(2 A k² / μ₀ Ms) cos(k x)
    const Index N = 128;
    const Real dx = 1e-10;  // 0.1 nm (very fine to minimize FD error)
    const Real L = N * dx;
    const Real k_wave = 2.0 * constants::pi / L;  // one full wavelength

    StructuredGrid g(N, 1, 1, dx, dx, dx);
    VectorField3D m(g);
    VectorField3D H(g);

    for (Index i = 0; i < N; ++i) {
        Real x = (static_cast<Real>(i) + 0.5) * dx;
        m.at(i, 0, 0) = {std::cos(k_wave * x), 0, 0};
    }

    Material mat = Material::permalloy();
    mat.A_exchange = 1.3e-11;

    // Use periodic BC so the cosine is exact everywhere
    ExchangeField ex(BoundaryCondition::Periodic);
    ex.accumulate(m, mat, H);

    const Real prefactor = 2.0 * mat.A_exchange / (constants::mu_0 * mat.Ms);
    const Real expected_amp = prefactor * k_wave * k_wave;

    // Check several interior points
    for (Index i = N / 4; i < 3 * N / 4; i += 8) {
        Real x = (static_cast<Real>(i) + 0.5) * dx;
        Real H_expected = -expected_amp * std::cos(k_wave * x);
        REQUIRE_THAT(H.at(i, 0, 0).x, WithinRel(H_expected, 1e-2));
        REQUIRE_THAT(H.at(i, 0, 0).y, WithinAbs(0.0, 1e-6));
        REQUIRE_THAT(H.at(i, 0, 0).z, WithinAbs(0.0, 1e-6));
    }
}

TEST_CASE("Exchange energy: 1D linear ramp has known value", "[exchange][energy]") {
    // m_x(i) = i * δ, m_y = m_z = 0 (small δ so |m| not zero anywhere)
    const Index N = 10;
    const Real dx = 1e-9;
    const Real delta = 1e-3;

    StructuredGrid g(N, 1, 1, dx, dx, dx);
    VectorField3D m(g);

    for (Index i = 0; i < N; ++i) {
        m.at(i, 0, 0) = {static_cast<Real>(i) * delta, 0, 0};
    }

    Material mat = Material::permalloy();
    mat.A_exchange = 1.3e-11;

    ExchangeField ex(BoundaryCondition::Neumann);
    Real E = ex.energy(m, mat);

    // Continuous estimate: ∫ A |∂m/∂x|² dV = A (δ/dx)² · (L · dy · dz)
    // FD sum over (N-1) pairs each contributing |δ|² / dx² · V_cell
    Real V_cell = g.cell_volume();
    Real expected = mat.A_exchange * (N - 1) * (delta * delta) / (dx * dx) * V_cell;

    REQUIRE_THAT(E, WithinRel(expected, 1e-12));
}
```

### 7.11 `apps/field_demo.cpp`

```cpp
#include <iostream>
#include <memory>

#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/material.hpp"
#include "micromag/effective_field.hpp"
#include "micromag/zeeman.hpp"
#include "micromag/anisotropy.hpp"
#include "micromag/exchange.hpp"
#include "micromag/vtk_writer.hpp"

int main() {
    using namespace micromag;

    // 64 nm × 64 nm × 8 nm permalloy disk-like region, 2 nm cells
    StructuredGrid grid(32, 32, 4, 2e-9, 2e-9, 2e-9);
    VectorField3D m(grid);
    VectorField3D H(grid);

    Vec3 ext = grid.extent();
    m.set_vortex(ext.x * 0.5, ext.y * 0.5, 8e-9);

    // Cobalt-like material (so anisotropy is nonzero and visible)
    Material mat = Material::cobalt();

    // Build effective field: Zeeman + Anisotropy + Exchange
    EffectiveFieldSum sum;
    sum.add(std::make_shared<ZeemanField>(Vec3{0, 0, 5e4}));      // 50 kA/m along +z
    sum.add(std::make_shared<UniaxialAnisotropyField>());
    sum.add(std::make_shared<ExchangeField>(BoundaryCondition::Neumann));

    sum.compute(m, mat, H);

    // Save magnetization and effective field
    write_vtk_legacy("vortex_m.vtk", m, "m");
    write_vtk_legacy("vortex_H.vtk", H, "H_eff");

    // Report energies
    std::cout << "=== Phase 1b field_demo ===\n";
    std::cout << "Grid: " << grid.nx() << " x " << grid.ny() << " x " << grid.nz()
              << " (" << grid.size() << " cells)\n";
    std::cout << "Material: cobalt-like (Ms=" << mat.Ms
              << ", A=" << mat.A_exchange
              << ", K=" << mat.K_uniaxial << ")\n\n";

    for (const auto& term : sum.terms()) {
        std::cout << "  E[" << term->name() << "] = "
                  << term->energy(m, mat) << " J\n";
    }
    std::cout << "  E[total]              = "
              << sum.total_energy(m, mat) << " J\n\n";

    std::cout << "Wrote vortex_m.vtk (magnetization) and vortex_H.vtk (effective field).\n";
    std::cout << "Open both in ParaView, use Glyph filter, compare directions.\n";
    return 0;
}
```

### 7.12 `examples/field_demo.py`

```python
"""Phase 1b demo: compute effective field for a vortex magnetization."""
import sys
from pathlib import Path

BUILD_PY = Path(__file__).resolve().parents[1] / "build" / "windows-msvc" / "python"
sys.path.insert(0, str(BUILD_PY))

import micromag as mm


def main():
    grid = mm.StructuredGrid(nx=32, ny=32, nz=4,
                             dx=2e-9, dy=2e-9, dz=2e-9)
    m = mm.VectorField3D(grid)
    H = mm.VectorField3D(grid)

    ext_x = grid.nx * grid.dx
    ext_y = grid.ny * grid.dy
    m.set_vortex(cx=ext_x * 0.5, cy=ext_y * 0.5, core_radius=8e-9)

    mat = mm.Material.cobalt()

    sum_field = mm.EffectiveFieldSum()
    sum_field.add(mm.ZeemanField(mm.Vec3(0, 0, 5e4)))
    sum_field.add(mm.UniaxialAnisotropyField())
    sum_field.add(mm.ExchangeField(mm.BoundaryCondition.Neumann))

    sum_field.compute(m, mat, H)

    mm.write_vtk_legacy("vortex_m_py.vtk", m, "m")
    mm.write_vtk_legacy("vortex_H_py.vtk", H, "H_eff")

    print("=== Phase 1b field_demo (Python) ===")
    print(f"Grid: {grid.nx} x {grid.ny} x {grid.nz} ({grid.size} cells)")
    print(f"Material: Ms={mat.Ms}, A={mat.A_exchange}, K={mat.K_uniaxial}")
    print()

    for term in sum_field.terms:
        print(f"  E[{term.name}] = {term.energy(m, mat):.6e} J")
    print(f"  E[total] = {sum_field.total_energy(m, mat):.6e} J")


if __name__ == "__main__":
    main()
```

### 7.13 `python/bindings.cpp` — 추가할 부분

Phase 1a의 기존 바인딩은 유지하고, **PYBIND11_MODULE 블록 안에 다음을 추가**합니다:

```cpp
#include "micromag/material.hpp"
#include "micromag/effective_field.hpp"
#include "micromag/zeeman.hpp"
#include "micromag/anisotropy.hpp"
#include "micromag/exchange.hpp"

// (PYBIND11_MODULE 블록 안, 기존 코드 다음에 추가)

    // Material
    py::class_<Material>(m, "Material")
        .def(py::init<>())
        .def_readwrite("Ms", &Material::Ms)
        .def_readwrite("A_exchange", &Material::A_exchange)
        .def_readwrite("K_uniaxial", &Material::K_uniaxial)
        .def_readwrite("easy_axis", &Material::easy_axis)
        .def_readwrite("alpha", &Material::alpha)
        .def_static("permalloy", &Material::permalloy)
        .def_static("cobalt", &Material::cobalt)
        .def_static("iron", &Material::iron);

    // BoundaryCondition enum
    py::enum_<BoundaryCondition>(m, "BoundaryCondition")
        .value("Neumann", BoundaryCondition::Neumann)
        .value("Periodic", BoundaryCondition::Periodic);

    // IEffectiveField (abstract base)
    py::class_<IEffectiveField, std::shared_ptr<IEffectiveField>>(m, "IEffectiveField")
        .def("accumulate", &IEffectiveField::accumulate)
        .def("energy", &IEffectiveField::energy)
        .def_property_readonly("name", &IEffectiveField::name);

    // ZeemanField
    py::class_<ZeemanField, IEffectiveField, std::shared_ptr<ZeemanField>>(m, "ZeemanField")
        .def(py::init<const Vec3&>(), py::arg("H_ext") = Vec3{0, 0, 0})
        .def_property("H_ext", &ZeemanField::H_ext, &ZeemanField::set_H_ext);

    // UniaxialAnisotropyField
    py::class_<UniaxialAnisotropyField, IEffectiveField,
               std::shared_ptr<UniaxialAnisotropyField>>(m, "UniaxialAnisotropyField")
        .def(py::init<>());

    // ExchangeField
    py::class_<ExchangeField, IEffectiveField, std::shared_ptr<ExchangeField>>(m, "ExchangeField")
        .def(py::init<BoundaryCondition>(),
             py::arg("bc") = BoundaryCondition::Neumann)
        .def_property("boundary", &ExchangeField::boundary, &ExchangeField::set_boundary);

    // EffectiveFieldSum
    py::class_<EffectiveFieldSum>(m, "EffectiveFieldSum")
        .def(py::init<>())
        .def("add", &EffectiveFieldSum::add)
        .def("compute", &EffectiveFieldSum::compute)
        .def("total_energy", &EffectiveFieldSum::total_energy)
        .def_property_readonly("terms", &EffectiveFieldSum::terms)
        .def_property_readonly("num_terms", &EffectiveFieldSum::num_terms);
```

### 7.14 `python/micromag/__init__.py` — 추가

기존 import에 추가:

```python
from _micromag import (
    Vec3,
    StructuredGrid,
    VectorField3D,
    write_vtk_legacy,
    # New in Phase 1b:
    Material,
    BoundaryCondition,
    IEffectiveField,
    ZeemanField,
    UniaxialAnisotropyField,
    ExchangeField,
    EffectiveFieldSum,
)

__all__ = [
    "Vec3", "StructuredGrid", "VectorField3D", "write_vtk_legacy",
    "Material", "BoundaryCondition", "IEffectiveField",
    "ZeemanField", "UniaxialAnisotropyField", "ExchangeField",
    "EffectiveFieldSum",
]
```

### 7.15 `CMakeLists.txt` 업데이트

`add_library(micromag_core ...)` 블록의 source 리스트에 새 cpp 파일 추가:

```cmake
add_library(micromag_core
    src/grid.cpp
    src/field.cpp
    src/vtk_writer.cpp
    src/material.cpp
    src/effective_field.cpp
    src/zeeman.cpp
    src/anisotropy.cpp
    src/exchange.cpp
)
```

`field_demo` executable도 추가 (`hello_micromag` 다음):

```cmake
add_executable(field_demo apps/field_demo.cpp)
target_link_libraries(field_demo PRIVATE micromag_core)
```

### 7.16 `tests/CMakeLists.txt` 업데이트

```cmake
add_executable(unit_tests
    test_grid.cpp
    test_field.cpp
    test_zeeman.cpp
    test_anisotropy.cpp
    test_exchange.cpp
)
```

---

## 8. 빌드 및 실행

### 빌드

```powershell
cd D:\dev\micromag
cmake --build --preset windows-msvc
```

새 소스가 추가되었으므로 CMake가 자동으로 재구성됩니다. 만약 안 되면:

```powershell
cmake --preset windows-msvc   # configure 강제
cmake --build --preset windows-msvc
```

### Demo 실행

```powershell
.\build\windows-msvc\bin\Release\field_demo.exe
```

기대 출력:
```
=== Phase 1b field_demo ===
Grid: 32 x 32 x 4 (4096 cells)
Material: cobalt-like (Ms=1.4e+06, A=3e-11, K=450000)

  E[Zeeman]             = (음수 또는 0에 가까운 J)
  E[UniaxialAnisotropy] = (음수, m_z 성분 작아서 절댓값 작음)
  E[Exchange]           = (양수, vortex 때문에 0이 아님)
  E[total]              = ...

Wrote vortex_m.vtk (magnetization) and vortex_H.vtk (effective field).
```

두 VTK 파일을 ParaView로 열어 비교:
- `vortex_m.vtk`: 격자에 vortex 자화 패턴
- `vortex_H.vtk`: 같은 격자에 effective field. exchange 항 때문에 vortex core 근처에 큰 H가 보여야 함.

### Python 실행

```powershell
.\.venv\Scripts\Activate.ps1
python examples\field_demo.py
```

C++ 출력과 같은 energy 값들이 나와야 합니다.

---

## 9. 검증 (이번 단계의 핵심)

### 9.1 단위 테스트

```powershell
ctest --preset windows-msvc
```

또는 자세히 보고 싶다면:

```powershell
.\build\windows-msvc\tests\Release\unit_tests.exe --reporter compact
```

모든 테스트가 통과해야 합니다. Phase 1a 테스트(grid, field) + Phase 1b 테스트(zeeman, anisotropy, exchange) 합쳐서 약 20개.

### 9.2 물리 sanity check (가장 중요)

빌드가 깔끔하고 단위 테스트가 통과해도, 실제 물리값이 합리적인지 추가 확인:

**(1) Exchange field 크기 추정**
Permalloy에서 vortex core 근처 (스케일 ~ 10 nm):
- `H_ex ~ (2A / μ₀ Ms) / lex²`
- `lex = √(2A / μ₀ Ms²) = √(2 · 1.3e-11 / (4π·1e-7 · (8e5)²))` ≈ 5.7 nm

이 값과 실제 H_ex 크기가 같은 자릿수면 OK.

**(2) Zeeman energy 부호**
m_z > 0이고 H_ext_z > 0이면 E_zeeman < 0이어야 합니다. 코드의 부호 일관성 검증.

**(3) Exchange energy는 항상 ≥ 0**
non-uniform m에서 exchange energy는 양수. 음수가 나오면 부호 오류.

**(4) ParaView 시각 검증**
`vortex_m.vtk` 열고 Glyph 필터 → m vector. 중앙에서 vortex 회전, 가장자리는 부드러움.
`vortex_H.vtk` 동일하게. H_eff가 m과 어떻게 다른지 비교. Exchange가 강한 핵심부에서 H의 크기 큰지 확인.

### 9.3 두 BC 비교 (Bonus)

`field_demo.cpp` 마지막에 한 줄 추가 시도:
```cpp
ExchangeField ex_pbc(BoundaryCondition::Periodic);
VectorField3D H_pbc(grid);
EffectiveFieldSum sum_pbc;
sum_pbc.add(std::make_shared<ExchangeField>(BoundaryCondition::Periodic));
sum_pbc.compute(m, mat, H_pbc);
write_vtk_legacy("vortex_H_pbc.vtk", H_pbc, "H_eff");
```

Neumann vs Periodic에서 경계 셀의 H_eff 차이를 확인. 내부 셀은 동일해야 함.

---

## 10. 흔한 문제 해결

### "no matching function for call to make_shared"
pybind11 바인딩에서 abstract class를 만들려고 시도한 경우. ZeemanField, UniaxialAnisotropyField, ExchangeField는 구체 클래스라 문제 없음. 만약 `IEffectiveField`로 직접 인스턴스화하려 하면 오류. Python 쪽에서도 `mm.IEffectiveField()` 같은 호출 금지.

### `cannot convert from std::shared_ptr<Derived> to std::shared_ptr<Base>`
pybind11에서 polymorphism을 쓰려면 base class도 `std::shared_ptr<T>`를 holder로 명시해야 함:
```cpp
py::class_<IEffectiveField, std::shared_ptr<IEffectiveField>>(m, "IEffectiveField");
py::class_<ZeemanField, IEffectiveField, std::shared_ptr<ZeemanField>>(m, "ZeemanField");
```
세 번째 type argument로 base class 명시 필요.

### Exchange energy 부호 또는 크기가 이상함
- Material이 K=0인 permalloy인지 cobalt인지 확인
- `A_exchange`가 m²로 잘못 들어갔는지 확인 (단위는 J/m, 1e-11 정도)
- 균일 m에서 E_ex = 0이 아니라면 코드 버그 → 디버그 빌드로 단계별 추적

### `H_ex` 가 NaN
- m이 normalize되어 있는지 확인 (vortex 직후 normalize 호출)
- 격자 셀 크기 0이 아닌지 확인
- 분모 0 (`mu_0 * Ms`)이 아닌지 확인 (Ms가 0이면 발생)

### Phase 1a 테스트가 깨짐
회귀가 생긴 것. `git diff` 로 의도치 않은 변경 확인. types.hpp에 constants 추가 시 기존 Vec3, Grid 등 건드리지 않았는지.

### `ImportError: DLL load failed` (Python)
새로운 cpp 파일이 link되지 않은 경우. CMakeLists.txt의 `add_library(micromag_core ...)`에 모든 cpp 파일이 들어있는지 확인. 빌드 디렉터리 청소 후 재빌드:
```powershell
rm -r build\windows-msvc
cmake --preset windows-msvc
cmake --build --preset windows-msvc
```

### Periodic BC에서 음수 인덱스 모듈러 버그
C++의 `%` 연산자는 음수에서 구현 정의입니다 (C++11부터 trunc-toward-zero). `(-1) % nx`는 `-1`이 될 수 있음. 해결: `((x % nx) + nx) % nx` 패턴 사용 (코드에 이미 적용됨).

---

## 11. 다음 단계 (Phase 1c: LLG integrator)

Phase 1b의 모든 effective field가 동작하고 검증된 다음, Phase 1c에서는 **시간 발전**을 넣습니다:

### Phase 1c 미리보기
1. **Landau-Lifshitz-Gilbert 방정식**:
   ```
   dm/dt = -γ (m × H_eff) - (γα / (1+α²)) m × (m × H_eff)
   ```
   여기서 γ = γ₀/(1+α²), γ₀ = 1.760859630×10¹¹ rad/(T·s)

2. **RK4 explicit integrator**: 가장 간단한 시작
3. **Adaptive RK45 (Cash-Karp 또는 Dormand-Prince)**: 효율적 시뮬레이션
4. **Energy minimization mode**: damped LLG with α=1 까지 키워서 relaxation
5. **Time-series output**: 매 N step마다 VTK 저장, 평균 m 저장 (CSV)

### 처음 시뮬레이션할 문제 후보
- **Single-domain switching**: 작은 입자, easy-axis 따라 자화, 반대 방향 H 인가 → switching 시간 측정. 해석해 비교 (Stoner-Wohlfarth + LLG).
- **Spin wave dispersion**: 1D 체인에서 perturbation 가했을 때 진동수 vs 파장.
- **µMAG standard problem 1**: 격자 의존성 검증 (Phase 1 마무리).

### Phase 1c 끝나면 가능해지는 것
- 동영상 생성 (시간 발전 ParaView 애니메이션)
- 첫 publishable한 의미의 시뮬레이션 (단순 시스템)
- Phase 2 (demag)를 위한 기반 완성

---

## 부록: Phase 1b 빠른 점검 명령

```powershell
# 빌드
cmake --build --preset windows-msvc

# 모든 테스트
ctest --preset windows-msvc

# 특정 테스트만 (예: exchange만)
.\build\windows-msvc\tests\Release\unit_tests.exe "[exchange]"

# Demo 실행
.\build\windows-msvc\bin\Release\field_demo.exe

# Python 실행
python examples\field_demo.py

# Phase 1a 회귀 점검