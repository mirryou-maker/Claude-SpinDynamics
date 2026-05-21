# Phase 1a: 프로젝트 구조 + 코어 자료구조 + 첫 빌드

> Micromagnetic 시뮬레이터 개발의 첫 번째 실제 코드 단계. 물리는 아직 넣지 않고, 전체 빌드·바인딩·테스트·시각화 파이프라인이 end-to-end로 동작하는지 검증합니다.

---

## 목차

- [1. 목표 및 범위](#1-목표-및-범위)
- [2. 핵심 설계 결정](#2-핵심-설계-결정)
- [3. 최종 디렉터리 구조](#3-최종-디렉터리-구조)
- [4. 사전 준비](#4-사전-준비)
- [5. Claude Code 활용 워크플로우](#5-claude-code-활용-워크플로우)
- [6. 파일별 작성 가이드](#6-파일별-작성-가이드)
- [7. 빌드 및 실행](#7-빌드-및-실행)
- [8. 검증](#8-검증)
- [9. Git commit](#9-git-commit)
- [10. 흔한 문제 해결](#10-흔한-문제-해결)
- [11. 다음 단계 (Phase 1b 예고)](#11-다음-단계-phase-1b-예고)

---

## 1. 목표 및 범위

### 만들 것 (Phase 1a)
- ✅ `StructuredGrid`: 격자 메타데이터 (셀 수, 셀 크기, 인덱싱)
- ✅ `VectorField3D`: 격자 위 3차원 벡터장 (자화, 자기장 저장용)
- ✅ VTK 레거시 포맷 writer (`.vtk` 파일 → ParaView)
- ✅ 데모 앱 `hello_micromag.exe`: vortex 패턴 자화 생성 + 저장
- ✅ Catch2 단위 테스트
- ✅ pybind11 Python 바인딩
- ✅ Python 예제 스크립트
- ✅ CMakePresets.json 기반 빌드 시스템

### 미루는 것
| 항목 | 시점 |
|---|---|
| Exchange / Anisotropy / Zeeman field | Phase 1b |
| LLG integrator (RK4) | Phase 1c |
| Slonczewski STT | Phase 1d |
| OOMMF OVF format I/O | Phase 1e |
| µMAG standard problem 1 검증 | Phase 1 마무리 |
| Demag (FFT/cuFFT) | Phase 2 |
| CUDA backend | Phase 2 |

### 성공 기준
1. `cmake --build build` 무경고 빌드
2. `./build/bin/hello_micromag.exe` 실행 → `vortex.vtk` 생성
3. ParaView에서 `vortex.vtk` 열기 → 32×32 vortex 자화 패턴 시각화
4. `ctest` 실행 → 모든 단위 테스트 통과
5. Python에서 `import micromag; m = micromag.StructuredGrid(...)` 동작

---

## 2. 핵심 설계 결정

### Public header vs implementation 분리
- `include/micromag/*.hpp`: 외부에서 보이는 인터페이스만
- `src/*.cpp`: 구현부 (외부 헤더 의존 최소화)

장점: 헤더 변경이 적어 컴파일 재발생 최소화. 나중에 shared library로 빌드할 때 ABI 안정.

### namespace
모든 코드는 `micromag::` 네임스페이스 안에. 서브모듈 추가될 때 `micromag::fields::`, `micromag::solvers::` 등으로 확장.

### 메모리 레이아웃
3D 격자를 1D `std::vector<Vec3>`로 저장. 인덱싱은 **i가 가장 빠르게 변하는** 순서 (Fortran/numpy/OOMMF/mumax 표준):
```
linear_index(i, j, k) = i + nx * (j + ny * k)
```
이렇게 하면 x-방향 stencil 연산에서 캐시 친화적, 그리고 후처리할 때 numpy `reshape((nz, ny, nx))`로 자연스럽게 됨.

### 단위계
SI 단위 일관 사용:
- 길이: meter (m)
- 자화: A/m (Ms ~ 10^5–10^6)
- 자기장: A/m
- 시간: second
- 에너지 밀도: J/m³
- exchange constant A: J/m

### CMakePresets.json
vcpkg toolchain, 빌드 타입, 컴파일러를 preset으로 캡슐화. VS Code CMake Tools가 자동 인식.

### 코드 스타일
- C++20 표준
- `snake_case` 함수/변수, `PascalCase` 클래스/타입
- 들여쓰기 4-space
- `.clang-format` 파일로 자동 포맷팅

---

## 3. 최종 디렉터리 구조

```
micromag/
├── CMakeLists.txt              # 최상위 빌드 스크립트
├── CMakePresets.json           # 빌드 preset (vcpkg, 빌드타입)
├── README.md                   # 프로젝트 소개
├── .gitignore                  # (이미 있음)
├── .clang-format               # 코드 스타일
│
├── include/
│   └── micromag/
│       ├── types.hpp           # Real, Vec3, Index
│       ├── grid.hpp            # StructuredGrid
│       ├── field.hpp           # VectorField3D
│       └── vtk_writer.hpp      # VTK 출력
│
├── src/
│   ├── grid.cpp
│   ├── field.cpp
│   └── vtk_writer.cpp
│
├── apps/
│   └── hello_micromag.cpp      # 데모 실행파일
│
├── python/
│   ├── bindings.cpp            # pybind11 진입점
│   └── micromag/
│       └── __init__.py         # Python 패키지 init
│
├── tests/
│   ├── CMakeLists.txt
│   ├── test_grid.cpp
│   └── test_field.cpp
│
├── examples/
│   └── hello_micromag.py       # Python 예제
│
└── docs/
    ├── phase1_environment_setup.md  # (이미 있음)
    └── phase1a_project_structure.md # (이 문서)
```

---

## 4. 사전 준비

### Catch2 v3 설치 (vcpkg)

기존 vcpkg에 Catch2 v3 추가:

```powershell
cd C:\vcpkg
.\vcpkg install catch2:x64-windows
```

설치 후 확인:
```powershell
.\vcpkg list | Select-String catch2
```

### 빠른 sanity check
이 문서 작업 전에 환경이 살아있는지 한 번 확인:

```powershell
cd D:\dev\micromag
.\.venv\Scripts\Activate.ps1
python --version
cmake --version
git status
```

가상환경이 활성화되고 (`(.venv)` 표시) 위 명령이 다 정상 출력되면 시작 OK.

---

## 5. Claude Code 활용 워크플로우

이 문서는 그 자체로 **Claude Code에 줄 수 있는 spec**입니다. 권장 흐름:

### 옵션 A: 단계별로 파일 작성 위임

VS Code에서 Claude Code 사이드바를 열고 한 파일씩:

```
@docs/phase1a_project_structure.md 를 참고해서 
include/micromag/types.hpp 파일을 6.1절의 명세대로 만들어줘.
```

각 파일을 만들 때마다 diff를 확인하고 승인. 이게 가장 안전하고 학습 효과 큼.

### 옵션 B: 한 번에 전체 구조 생성

```
@docs/phase1a_project_structure.md 의 6절에 있는 모든 파일을
명세대로 생성해줘. 각 파일을 만들 때마다 diff를 보여주고
확인을 받아.
```

빠르지만 한꺼번에 변화량이 크니 신중하게 검토.

### 옵션 C: 자율 실행 (Plan mode)

Claude Code의 Plan mode를 활성화하고:

```
@docs/phase1a_project_structure.md 를 읽고 Phase 1a 전체를 
end-to-end로 구현해줘. 빌드까지 성공해야 하고, 
ctest로 모든 테스트가 통과해야 해. 
빌드 또는 테스트 실패 시 자동으로 수정 시도해.
```

가장 강력하지만, 결과 검토에 시간 들어감. Claude Code가 어떤 결정을 내렸는지 commit log로 추적 가능하니 git을 자주 commit.

### Claude Code 프롬프트 팁
- **파일 참조는 `@경로`로** — 자동으로 파일 내용을 context에 넣어줌
- **"검증"을 명시** — "빌드 성공해야 함", "테스트 통과해야 함" 처럼 success criteria 명시
- **점진적 진행** — 큰 변화는 작은 commit으로 나눠서

---

## 6. 파일별 작성 가이드

각 파일의 **목적, 위치, 전체 내용**을 명세합니다. 그대로 복사해도 되고, Claude Code에게 위임해도 됩니다.

### 6.1 `.clang-format`

**목적**: VS Code의 C/C++ 확장이 자동 포맷팅에 사용. 팀 전체 스타일 통일.

**내용**:
```yaml
---
BasedOnStyle: LLVM
IndentWidth: 4
ColumnLimit: 100
PointerAlignment: Left
SpaceAfterTemplateKeyword: false
AccessModifierOffset: -4
NamespaceIndentation: None
AllowShortFunctionsOnASingleLine: Inline
SortIncludes: CaseSensitive
IncludeBlocks: Preserve
...
```

### 6.2 `README.md`

**목적**: GitHub 등에서 프로젝트 첫 인상.

**내용**:
```markdown
# Micromag

Micromagnetic simulator written in C++/CUDA with Python bindings.

## Status: Phase 1a (project skeleton)

Currently implemented:
- StructuredGrid, VectorField3D
- VTK output for ParaView
- Catch2 unit tests
- Python bindings via pybind11

Planned (Phase 1b–2):
- LLG dynamics with exchange, anisotropy, Zeeman, STT
- Demag field via FFTW (CPU) and cuFFT (GPU)
- OOMMF OVF format I/O
- µMAG standard problem benchmarks

## Build

See `docs/phase1_environment_setup.md` for prerequisites.

\`\`\`powershell
cmake --preset windows-msvc
cmake --build --preset windows-msvc
ctest --preset windows-msvc
\`\`\`

## Run

\`\`\`powershell
.\build\windows-msvc\bin\hello_micromag.exe
\`\`\`

Opens `vortex.vtk` in ParaView.

## License

TBD
```

### 6.3 `include/micromag/types.hpp`

**목적**: 프로젝트 전체에서 쓸 기본 타입 정의. 부동소수점 정밀도를 한 곳에서 관리.

**내용**:
```cpp
#pragma once

#include <cmath>
#include <cstddef>

namespace micromag {

// Floating-point precision used throughout.
// Change to float for single-precision builds.
using Real = double;

// Signed integer for indices and counts (supports negative offsets in stencils).
using Index = std::ptrdiff_t;

// 3D vector with basic arithmetic. POD-like, trivially copyable.
struct Vec3 {
    Real x{0}, y{0}, z{0};

    constexpr Vec3() = default;
    constexpr Vec3(Real xv, Real yv, Real zv) : x(xv), y(yv), z(zv) {}

    constexpr Vec3 operator+(const Vec3& v) const { return {x + v.x, y + v.y, z + v.z}; }
    constexpr Vec3 operator-(const Vec3& v) const { return {x - v.x, y - v.y, z - v.z}; }
    constexpr Vec3 operator*(Real s) const { return {x * s, y * s, z * s}; }
    constexpr Vec3 operator/(Real s) const { return {x / s, y / s, z / s}; }

    Vec3& operator+=(const Vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
    Vec3& operator-=(const Vec3& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
    Vec3& operator*=(Real s) { x *= s; y *= s; z *= s; return *this; }

    constexpr Real dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    constexpr Vec3 cross(const Vec3& v) const {
        return {y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x};
    }
    constexpr Real norm_squared() const { return dot(*this); }
    Real norm() const { return std::sqrt(norm_squared()); }
};

constexpr Vec3 operator*(Real s, const Vec3& v) { return v * s; }

}  // namespace micromag
```

### 6.4 `include/micromag/grid.hpp`

**목적**: 균일 구조 격자 메타데이터 + 셀 인덱싱 헬퍼.

**내용**:
```cpp
#pragma once

#include "types.hpp"

namespace micromag {

// Cell-centered uniform Cartesian grid.
// Cell (i,j,k) occupies [i*dx, (i+1)*dx] × [j*dy, (j+1)*dy] × [k*dz, (k+1)*dz].
class StructuredGrid {
public:
    StructuredGrid(Index nx, Index ny, Index nz, Real dx, Real dy, Real dz);

    Index nx() const { return nx_; }
    Index ny() const { return ny_; }
    Index nz() const { return nz_; }

    Real dx() const { return dx_; }
    Real dy() const { return dy_; }
    Real dz() const { return dz_; }

    Index size() const { return nx_ * ny_ * nz_; }
    Real cell_volume() const { return dx_ * dy_ * dz_; }

    // x is fastest (i varies first), matches numpy/Fortran convention.
    Index linear_index(Index i, Index j, Index k) const {
        return i + nx_ * (j + ny_ * k);
    }

    Vec3 cell_center(Index i, Index j, Index k) const {
        return {(static_cast<Real>(i) + Real{0.5}) * dx_,
                (static_cast<Real>(j) + Real{0.5}) * dy_,
                (static_cast<Real>(k) + Real{0.5}) * dz_};
    }

    Vec3 extent() const {
        return {static_cast<Real>(nx_) * dx_,
                static_cast<Real>(ny_) * dy_,
                static_cast<Real>(nz_) * dz_};
    }

private:
    Index nx_, ny_, nz_;
    Real dx_, dy_, dz_;
};

}  // namespace micromag
```

### 6.5 `include/micromag/field.hpp`

**목적**: 격자 위 3D 벡터장. magnetization, effective field 등 모두 이 타입 사용.

**내용**:
```cpp
#pragma once

#include <vector>
#include "types.hpp"
#include "grid.hpp"

namespace micromag {

class VectorField3D {
public:
    explicit VectorField3D(const StructuredGrid& grid)
        : grid_(&grid), data_(static_cast<std::size_t>(grid.size())) {}

    const StructuredGrid& grid() const { return *grid_; }

    Vec3& at(Index i, Index j, Index k) {
        return data_[static_cast<std::size_t>(grid_->linear_index(i, j, k))];
    }
    const Vec3& at(Index i, Index j, Index k) const {
        return data_[static_cast<std::size_t>(grid_->linear_index(i, j, k))];
    }

    Vec3& operator[](Index linear) {
        return data_[static_cast<std::size_t>(linear)];
    }
    const Vec3& operator[](Index linear) const {
        return data_[static_cast<std::size_t>(linear)];
    }

    Index size() const { return static_cast<Index>(data_.size()); }
    const std::vector<Vec3>& data() const { return data_; }
    std::vector<Vec3>& data() { return data_; }

    // Set all cells to the same vector.
    void set_uniform(const Vec3& m);

    // Normalize every vector to unit length. No-op for zero vectors.
    void normalize();

    // Set magnetization to a 2D vortex around the z-axis through (cx, cy).
    // Out-of-plane core (m_z) is +1 inside core_radius, smoothly transitioning.
    void set_vortex(Real cx, Real cy, Real core_radius);
};

}  // namespace micromag
```

### 6.6 `include/micromag/vtk_writer.hpp`

**목적**: ParaView로 결과 확인 가능하도록 VTK legacy 포맷 출력.

**내용**:
```cpp
#pragma once

#include <string>
#include "field.hpp"

namespace micromag {

// Write a VectorField3D as a VTK legacy ASCII file (.vtk).
// ParaView reads this directly. Cell centers are emitted as POINT_DATA
// at half-cell offsets (mumax/OOMMF convention).
//
// The 'field_name' becomes the VECTORS array name in ParaView.
void write_vtk_legacy(const std::string& filename,
                      const VectorField3D& field,
                      const std::string& field_name = "m");

}  // namespace micromag
```

### 6.7 `src/grid.cpp`

```cpp
#include "micromag/grid.hpp"

#include <stdexcept>

namespace micromag {

StructuredGrid::StructuredGrid(Index nx, Index ny, Index nz,
                               Real dx, Real dy, Real dz)
    : nx_(nx), ny_(ny), nz_(nz), dx_(dx), dy_(dy), dz_(dz) {
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        throw std::invalid_argument("Grid dimensions must be positive");
    }
    if (dx <= 0 || dy <= 0 || dz <= 0) {
        throw std::invalid_argument("Cell sizes must be positive");
    }
}

}  // namespace micromag
```

### 6.8 `src/field.cpp`

```cpp
#include "micromag/field.hpp"

#include <cmath>

namespace micromag {

void VectorField3D::set_uniform(const Vec3& m) {
    for (auto& v : data_) {
        v = m;
    }
}

void VectorField3D::normalize() {
    for (auto& v : data_) {
        Real n = v.norm();
        if (n > Real{1e-30}) {
            v /= n;
        }
    }
}

void VectorField3D::set_vortex(Real cx, Real cy, Real core_radius) {
    for (Index k = 0; k < grid_->nz(); ++k) {
        for (Index j = 0; j < grid_->ny(); ++j) {
            for (Index i = 0; i < grid_->nx(); ++i) {
                Vec3 c = grid_->cell_center(i, j, k);
                Real rx = c.x - cx;
                Real ry = c.y - cy;
                Real r = std::sqrt(rx * rx + ry * ry);

                if (r < Real{1e-30}) {
                    at(i, j, k) = {0, 0, 1};
                    continue;
                }

                // Smooth core: m_z drops from 1 → 0 over [0, core_radius].
                Real mz = (r < core_radius)
                              ? std::cos(Real{0.5} * Real{3.14159265358979323846} *
                                         (r / core_radius))
                              : Real{0};
                Real m_inplane = std::sqrt(std::max(Real{0}, Real{1} - mz * mz));
                Real mx = -ry / r * m_inplane;
                Real my = rx / r * m_inplane;

                at(i, j, k) = {mx, my, mz};
            }
        }
    }
}

}  // namespace micromag
```

### 6.9 `src/vtk_writer.cpp`

```cpp
#include "micromag/vtk_writer.hpp"

#include <fstream>
#include <stdexcept>

namespace micromag {

void write_vtk_legacy(const std::string& filename,
                      const VectorField3D& field,
                      const std::string& field_name) {
    std::ofstream f(filename);
    if (!f) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    const auto& g = field.grid();
    const Index n = g.size();

    f << "# vtk DataFile Version 3.0\n";
    f << "Micromag output\n";
    f << "ASCII\n";
    f << "DATASET STRUCTURED_POINTS\n";
    f << "DIMENSIONS " << g.nx() << " " << g.ny() << " " << g.nz() << "\n";
    f << "ORIGIN " << g.dx() * 0.5 << " " << g.dy() * 0.5 << " " << g.dz() * 0.5 << "\n";
    f << "SPACING " << g.dx() << " " << g.dy() << " " << g.dz() << "\n";
    f << "POINT_DATA " << n << "\n";
    f << "VECTORS " << field_name << " double\n";

    for (Index idx = 0; idx < n; ++idx) {
        const Vec3& v = field[idx];
        f << v.x << " " << v.y << " " << v.z << "\n";
    }
}

}  // namespace micromag
```

### 6.10 `apps/hello_micromag.cpp`

**목적**: end-to-end 동작 데모. 32×32×4 격자에 vortex 자화를 생성하고 VTK로 저장.

```cpp
#include <iostream>
#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/vtk_writer.hpp"

int main() {
    using namespace micromag;

    // 64 nm × 64 nm × 8 nm, 2 nm cells → 32 × 32 × 4
    StructuredGrid grid(32, 32, 4, 2e-9, 2e-9, 2e-9);
    VectorField3D m(grid);

    Vec3 ext = grid.extent();
    m.set_vortex(ext.x * 0.5, ext.y * 0.5, /*core_radius=*/8e-9);

    write_vtk_legacy("vortex.vtk", m);

    std::cout << "Wrote vortex.vtk: "
              << grid.nx() << " x " << grid.ny() << " x " << grid.nz()
              << " cells (" << grid.size() << " total)\n";
    std::cout << "Open in ParaView to visualize.\n";
    return 0;
}
```

### 6.11 `tests/CMakeLists.txt`

```cmake
find_package(Catch2 3 CONFIG REQUIRED)

add_executable(unit_tests
    test_grid.cpp
    test_field.cpp
)

target_link_libraries(unit_tests
    PRIVATE
        micromag_core
        Catch2::Catch2WithMain
)

include(Catch)
catch_discover_tests(unit_tests)
```

### 6.12 `tests/test_grid.cpp`

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "micromag/grid.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;

TEST_CASE("StructuredGrid basic dimensions", "[grid]") {
    StructuredGrid g(4, 5, 6, 1.0, 2.0, 3.0);
    REQUIRE(g.nx() == 4);
    REQUIRE(g.ny() == 5);
    REQUIRE(g.nz() == 6);
    REQUIRE(g.size() == 120);
    REQUIRE_THAT(g.cell_volume(), WithinAbs(6.0, 1e-12));
}

TEST_CASE("StructuredGrid cell centers", "[grid]") {
    StructuredGrid g(2, 2, 2, 1.0, 1.0, 1.0);
    auto c000 = g.cell_center(0, 0, 0);
    REQUIRE_THAT(c000.x, WithinAbs(0.5, 1e-12));
    REQUIRE_THAT(c000.y, WithinAbs(0.5, 1e-12));
    REQUIRE_THAT(c000.z, WithinAbs(0.5, 1e-12));

    auto c111 = g.cell_center(1, 1, 1);
    REQUIRE_THAT(c111.x, WithinAbs(1.5, 1e-12));
}

TEST_CASE("StructuredGrid linear indexing", "[grid]") {
    StructuredGrid g(3, 4, 5, 1.0, 1.0, 1.0);
    REQUIRE(g.linear_index(0, 0, 0) == 0);
    REQUIRE(g.linear_index(1, 0, 0) == 1);
    REQUIRE(g.linear_index(0, 1, 0) == 3);   // = nx
    REQUIRE(g.linear_index(0, 0, 1) == 12);  // = nx*ny
    REQUIRE(g.linear_index(2, 3, 4) == 2 + 3 * 3 + 4 * 3 * 4);
}

TEST_CASE("StructuredGrid rejects bad dimensions", "[grid]") {
    REQUIRE_THROWS_AS(StructuredGrid(0, 1, 1, 1, 1, 1), std::invalid_argument);
    REQUIRE_THROWS_AS(StructuredGrid(1, 1, 1, -1, 1, 1), std::invalid_argument);
}
```

### 6.13 `tests/test_field.cpp`

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "micromag/grid.hpp"
#include "micromag/field.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;

TEST_CASE("VectorField3D set_uniform", "[field]") {
    StructuredGrid g(3, 3, 3, 1.0, 1.0, 1.0);
    VectorField3D f(g);
    f.set_uniform({1.0, 0.0, 0.0});

    REQUIRE_THAT(f.at(0, 0, 0).x, WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(f.at(2, 2, 2).x, WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(f.at(1, 1, 1).y, WithinAbs(0.0, 1e-12));
}

TEST_CASE("VectorField3D normalize", "[field]") {
    StructuredGrid g(2, 2, 2, 1.0, 1.0, 1.0);
    VectorField3D f(g);
    f.set_uniform({3.0, 4.0, 0.0});  // length 5
    f.normalize();

    Vec3 v = f.at(0, 0, 0);
    REQUIRE_THAT(v.norm(), WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(v.x, WithinAbs(0.6, 1e-12));
    REQUIRE_THAT(v.y, WithinAbs(0.8, 1e-12));
}

TEST_CASE("VectorField3D vortex has unit length everywhere", "[field]") {
    StructuredGrid g(16, 16, 1, 1.0, 1.0, 1.0);
    VectorField3D f(g);
    f.set_vortex(8.0, 8.0, 4.0);

    for (Index idx = 0; idx < g.size(); ++idx) {
        REQUIRE_THAT(f[idx].norm(), WithinAbs(1.0, 1e-6));
    }
}
```

### 6.14 `python/bindings.cpp`

**목적**: Python에서 C++ 클래스를 그대로 사용 가능하게.

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "micromag/types.hpp"
#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/vtk_writer.hpp"

namespace py = pybind11;
using namespace micromag;

PYBIND11_MODULE(_micromag, m) {
    m.doc() = "Micromag C++ core bindings";

    py::class_<Vec3>(m, "Vec3")
        .def(py::init<>())
        .def(py::init<Real, Real, Real>(), py::arg("x"), py::arg("y"), py::arg("z"))
        .def_readwrite("x", &Vec3::x)
        .def_readwrite("y", &Vec3::y)
        .def_readwrite("z", &Vec3::z)
        .def("norm", &Vec3::norm)
        .def("dot", &Vec3::dot)
        .def("cross", &Vec3::cross)
        .def("__repr__", [](const Vec3& v) {
            return "Vec3(" + std::to_string(v.x) + ", " +
                   std::to_string(v.y) + ", " + std::to_string(v.z) + ")";
        });

    py::class_<StructuredGrid>(m, "StructuredGrid")
        .def(py::init<Index, Index, Index, Real, Real, Real>(),
             py::arg("nx"), py::arg("ny"), py::arg("nz"),
             py::arg("dx"), py::arg("dy"), py::arg("dz"))
        .def_property_readonly("nx", &StructuredGrid::nx)
        .def_property_readonly("ny", &StructuredGrid::ny)
        .def_property_readonly("nz", &StructuredGrid::nz)
        .def_property_readonly("dx", &StructuredGrid::dx)
        .def_property_readonly("dy", &StructuredGrid::dy)
        .def_property_readonly("dz", &StructuredGrid::dz)
        .def_property_readonly("size", &StructuredGrid::size)
        .def("cell_center", &StructuredGrid::cell_center);

    py::class_<VectorField3D>(m, "VectorField3D")
        .def(py::init<const StructuredGrid&>(), py::keep_alive<1, 2>())
        .def_property_readonly("grid", &VectorField3D::grid,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("size", &VectorField3D::size)
        .def("set_uniform", &VectorField3D::set_uniform)
        .def("set_vortex", &VectorField3D::set_vortex,
             py::arg("cx"), py::arg("cy"), py::arg("core_radius"))
        .def("normalize", &VectorField3D::normalize)
        .def("at", [](VectorField3D& f, Index i, Index j, Index k) {
            return f.at(i, j, k);
        });

    m.def("write_vtk_legacy", &write_vtk_legacy,
          py::arg("filename"), py::arg("field"), py::arg("field_name") = "m");
}
```

### 6.15 `python/micromag/__init__.py`

**목적**: `import micromag`이 동작하게. C++ 모듈을 re-export.

```python
"""Micromag: Python interface to the C++ micromagnetic core."""

from _micromag import (
    Vec3,
    StructuredGrid,
    VectorField3D,
    write_vtk_legacy,
)

__all__ = [
    "Vec3",
    "StructuredGrid",
    "VectorField3D",
    "write_vtk_legacy",
]

__version__ = "0.1.0"
```

### 6.16 `examples/hello_micromag.py`

**목적**: Python에서 동일 작업을 수행하는 예제.

```python
"""Phase 1a demo: create a vortex magnetization and save to VTK."""
import sys
from pathlib import Path

# Add build output to path (adjust if your build dir differs)
BUILD_PY = Path(__file__).resolve().parents[1] / "build" / "windows-msvc" / "python"
sys.path.insert(0, str(BUILD_PY))

import micromag as mm

def main():
    grid = mm.StructuredGrid(nx=32, ny=32, nz=4,
                             dx=2e-9, dy=2e-9, dz=2e-9)
    field = mm.VectorField3D(grid)

    ext_x = grid.nx * grid.dx
    ext_y = grid.ny * grid.dy
    field.set_vortex(cx=ext_x * 0.5, cy=ext_y * 0.5, core_radius=8e-9)

    mm.write_vtk_legacy("vortex_py.vtk", field, "m")
    print(f"Wrote vortex_py.vtk: {grid.nx} x {grid.ny} x {grid.nz} "
          f"cells ({grid.size} total)")

if __name__ == "__main__":
    main()
```

### 6.17 최상위 `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.24)

project(micromag
    VERSION 0.1.0
    DESCRIPTION "Micromagnetic simulator"
    LANGUAGES CXX
)

# C++ standard --------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default build type --------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# UTF-8 source encoding (avoid C4819 on Korean Windows) ---------------------
if(MSVC)
    add_compile_options(/utf-8)
endif()

# Position-independent code (needed for pybind11) ---------------------------
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Output directories --------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Options -------------------------------------------------------------------
option(MICROMAG_BUILD_PYTHON "Build Python bindings"     ON)
option(MICROMAG_BUILD_TESTS  "Build unit tests"          ON)
option(MICROMAG_USE_CUDA     "Enable CUDA backend (later phase)" OFF)

# Core library --------------------------------------------------------------
add_library(micromag_core
    src/grid.cpp
    src/field.cpp
    src/vtk_writer.cpp
)
target_include_directories(micromag_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(micromag_core PUBLIC cxx_std_20)

# Demo executable -----------------------------------------------------------
add_executable(hello_micromag apps/hello_micromag.cpp)
target_link_libraries(hello_micromag PRIVATE micromag_core)

# Python bindings -----------------------------------------------------------
if(MICROMAG_BUILD_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_micromag python/bindings.cpp)
    target_link_libraries(_micromag PRIVATE micromag_core)

    # Place the compiled module next to the Python package directory
    set_target_properties(_micromag PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
    )

    # Copy the Python package next to the compiled module
    add_custom_command(TARGET _micromag POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/python/micromag
            ${CMAKE_BINARY_DIR}/python/micromag
    )
endif()

# Tests ---------------------------------------------------------------------
if(MICROMAG_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Status summary ------------------------------------------------------------
message(STATUS "")
message(STATUS "micromag ${PROJECT_VERSION}")
message(STATUS "  Build type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:  ${CMAKE_CXX_STANDARD}")
message(STATUS "  Python:        ${MICROMAG_BUILD_PYTHON}")
message(STATUS "  Tests:         ${MICROMAG_BUILD_TESTS}")
message(STATUS "  CUDA:          ${MICROMAG_USE_CUDA}")
message(STATUS "")
```

### 6.18 `CMakePresets.json`

**목적**: vcpkg toolchain, 빌드 타입, generator를 한 곳에서 관리. VS Code CMake Tools와 명령줄 모두 인식.

```json
{
    "version": 6,
    "cmakeMinimumRequired": {
        "major": 3,
        "minor": 24,
        "patch": 0
    },
    "configurePresets": [
        {
            "name": "windows-msvc",
            "displayName": "Windows MSVC Release",
            "description": "MSVC + vcpkg, Release build",
            "generator": "Visual Studio 17 2022",
            "architecture": {
                "value": "x64",
                "strategy": "set"
            },
            "binaryDir": "${sourceDir}/build/windows-msvc",
            "cacheVariables": {
                "CMAKE_TOOLCHAIN_FILE": "C:/vcpkg/scripts/buildsystems/vcpkg.cmake",
                "VCPKG_TARGET_TRIPLET": "x64-windows",
                "CMAKE_BUILD_TYPE": "Release"
            }
        },
        {
            "name": "windows-msvc-debug",
            "inherits": "windows-msvc",
            "displayName": "Windows MSVC Debug",
            "binaryDir": "${sourceDir}/build/windows-msvc-debug",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Debug"
            }
        }
    ],
    "buildPresets": [
        {
            "name": "windows-msvc",
            "configurePreset": "windows-msvc",
            "configuration": "Release"
        },
        {
            "name": "windows-msvc-debug",
            "configurePreset": "windows-msvc-debug",
            "configuration": "Debug"
        }
    ],
    "testPresets": [
        {
            "name": "windows-msvc",
            "configurePreset": "windows-msvc",
            "configuration": "Release",
            "output": {
                "outputOnFailure": true
            }
        }
    ]
}
```

> ⚠️ vcpkg를 `C:\vcpkg`가 아닌 다른 곳에 설치했다면 `CMAKE_TOOLCHAIN_FILE` 경로를 수정하세요.

---

## 7. 빌드 및 실행

### 옵션 A: VS Code CMake Tools (권장)

1. VS Code에서 프로젝트 폴더 열기: `code D:\dev\micromag`
2. `CMakeLists.txt` 파일이 인식되면 우측 하단에 알림이 뜸. 무시 가능.
3. `Ctrl+Shift+P` → `CMake: Select Configure Preset` → **windows-msvc** 선택
4. `Ctrl+Shift+P` → `CMake: Configure` → CMake 첫 구성 실행 (vcpkg 의존성 찾고 빌드 트리 생성)
5. `Ctrl+Shift+P` → `CMake: Build` → 컴파일

상태바 아래쪽에 빌드 결과가 표시됩니다. 에러가 나면 **Output** 패널의 "CMake/Build" 채널에서 상세 로그 확인.

### 옵션 B: 명령줄 (CMakePresets 활용)

**Developer PowerShell for VS 2022**(Step 11에서 설정한 터미널)에서:

```powershell
cd D:\dev\micromag
cmake --preset windows-msvc
cmake --build --preset windows-msvc
```

성공 시 다음 위치에 결과물 생성:
- `build/windows-msvc/bin/Release/hello_micromag.exe` — 데모 실행파일
- `build/windows-msvc/python/_micromag.*.pyd` — Python 모듈
- `build/windows-msvc/tests/Release/unit_tests.exe` — 테스트 실행파일

### 실행

```powershell
.\build\windows-msvc\bin\Release\hello_micromag.exe
```

출력:
```
Wrote vortex.vtk: 32 x 32 x 4 cells (4096 total)
Open in ParaView to visualize.
```

### ParaView로 확인

1. https://www.paraview.org/download/ 에서 Windows 64-bit 다운로드 (필요하면)
2. ParaView 실행 → File → Open → `D:\dev\micromag\vortex.vtk`
3. Properties 패널에서 **Apply** 클릭
4. 상단 툴바에서 **Glyph** 필터 추가 → Glyph Type: Arrow → Scale Array: m → Apply
5. 회전·확대해서 vortex 패턴 확인

---

## 8. 검증

### 8.1 단위 테스트

```powershell
ctest --preset windows-msvc
```

또는 명령줄에서:
```powershell
cd build\windows-msvc
ctest -C Release --output-on-failure
```

기대 결과: 모든 테스트 통과 (대략 7개 정도).

### 8.2 Python 바인딩 테스트

```powershell
.\.venv\Scripts\Activate.ps1
python examples\hello_micromag.py
```

출력:
```
Wrote vortex_py.vtk: 32 x 32 x 4 cells (4096 total)
```

생성된 `vortex_py.vtk`를 ParaView로 열어 C++ 결과와 동일한지 시각 확인.

### 8.3 import 경로 트러블

`ImportError: No module named '_micromag'`가 나면:
- 빌드 디렉터리 확인: `build\windows-msvc\python\_micromag.*.pyd` 존재해야 함
- Python script 안의 `BUILD_PY` 경로가 실제 빌드 출력 위치와 일치하는지 확인

---

## 9. Git commit

Phase 1a가 동작하면 깨끗한 checkpoint commit을 남깁니다:

```powershell
git add .
git status   # 추가될 파일 확인
git commit -m "Phase 1a: project skeleton with grid, field, VTK writer, Python bindings"
```

GitHub에 원격 저장소 만들었다면:
```powershell
git remote add origin https://github.com/<username>/micromag.git
git branch -M main
git push -u origin main
```

---

## 10. 흔한 문제 해결

### `find_package(pybind11)` 실패
**증상**: CMake configure 시 "Could not find package pybind11"

**원인**: vcpkg toolchain이 안 잡힌 것.

**해결**: `CMakePresets.json`의 `CMAKE_TOOLCHAIN_FILE` 경로가 정확한지 확인. `C:/vcpkg/scripts/buildsystems/vcpkg.cmake` 파일이 실제로 존재해야 함. 백슬래시 대신 슬래시 사용.

### `find_package(Catch2 3)` 실패
**증상**: "Could not find package Catch2"

**원인**: Catch2를 vcpkg에 설치 안 함.

**해결**:
```powershell
cd C:\vcpkg
.\vcpkg install catch2:x64-windows
```
그리고 빌드 디렉터리 삭제 후 재구성:
```powershell
rm -r build
cmake --preset windows-msvc
```

### `LNK2019` 또는 `LNK1120` linker error
**증상**: 함수 정의를 찾을 수 없다는 링커 에러

**원인 후보**:
- 헤더에 선언했는데 cpp에 구현 안 함
- cpp 파일을 `CMakeLists.txt`의 `add_library(...)`에 추가 안 함

**해결**: 최상위 `CMakeLists.txt`의 `add_library(micromag_core ...)`에 모든 cpp 파일이 나열되어 있는지 확인.

### Python import에서 `DLL load failed`
**증상**: `ImportError: DLL load failed while importing _micromag`

**원인**: Python 모듈이 의존하는 다른 DLL (예: vcpkg가 빌드한 DLL)을 찾지 못함.

**해결**: 
- vcpkg 출력 디렉터리를 PATH에 추가 (`C:\vcpkg\installed\x64-windows\bin`)
- 또는 빌드 후 해당 DLL을 Python 모듈 옆에 복사

### CMakeLists.txt 변경했는데 반영 안 됨
**해결**: 빌드 디렉터리를 삭제하고 재구성:
```powershell
rm -r build\windows-msvc
cmake --preset windows-msvc
```

### UTF-8 BOM 경고
이미 `add_compile_options(/utf-8)`이 들어 있으므로 사라져야 함. 안 사라지면 해당 cpp 파일을 VS Code에서 열고 우측 하단 인코딩 클릭 → "Save with Encoding" → UTF-8 (BOM 없이).

---

## 11. 다음 단계 (Phase 1b 예고)

Phase 1a가 안정적으로 동작하면 Phase 1b에서는 **첫 물리**를 넣습니다:

1. **Material parameter 구조체**: `Ms`, `A_exchange`, `K_uniaxial`, `easy_axis`, `alpha` 등
2. **Effective field 추상 클래스**: `IEffectiveField` interface
3. **Zeeman field**: 가장 단순한 시작점 (외부 자기장)
4. **Uniaxial anisotropy field**
5. **Heisenberg exchange field**: 6-point stencil, 경계 처리 (free vs periodic)
6. **Effective field 합산 클래스**
7. **Energy 계산** (validation용)

이 시점에서 처음으로 "물리 sanity check"가 가능해집니다:
- uniform한 자화에서 exchange field = 0
- easy-axis 방향 자화에서 anisotropy field = 0
- 외부 field 방향과 자화 정렬되면 Zeeman energy 최소

이후 Phase 1c에서 LLG integrator를 넣으면 비로소 시뮬레이션이 동작합니다.

---

## 부록: 자주 쓰는 명령 모음

```powershell
# 빌드
cmake --preset windows-msvc
cmake --build --preset windows-msvc

# 청소 후 재빌드
rm -r build\windows-msvc
cmake --preset windows-msvc
cmake --build --preset windows-msvc

# 테스트
ctest --preset windows-msvc

# 데모 실행
.\build\windows-msvc\bin\Release\hello_micromag.exe

# Python 실행
.\.venv\Scripts\Activate.ps1
python examples\hello_micromag.py

# git workflow
git status
git diff
git add <file>
git commit -m "..."
```

---

## 부록: VS Code 추천 키바인딩

| 단축키 | 동작 |
|---|---|
| `Ctrl+Shift+B` | CMake Build |
| `F5` | Debug (launch.json 설정 필요) |
| `Ctrl+Shift+P` → CMake: Select Configure Preset | preset 변경 |
| `Ctrl+Shift+P` → CMake: Configure | CMake 재구성 |
| `` Ctrl+` `` | 통합 터미널 토글 |
| `Ctrl+Shift+E` | Explorer 패널 |
| `Ctrl+Shift+G` | Source Control 패널 |

Claude Code 사용 시:
- `Ctrl+L` (또는 사이드바에서 New Conversation) — 새 대화 시작
- 코드 선택 후 사이드바에서 질문하면 자동으로 selection이 context에 포함됨
- `@파일경로` — 특정 파일을 context에 추가
- Plan mode 토글 — 한 번에 큰 작업 위임 시
