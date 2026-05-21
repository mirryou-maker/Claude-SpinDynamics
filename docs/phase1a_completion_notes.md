# Phase 1a 완료 노트

> 구현 과정에서 발생한 이슈와 해결책, 스펙과의 차이점을 기록합니다.

---

## 완료 일자

2026-05-22

---

## 성공 기준 달성 여부

| 기준 | 결과 |
|---|---|
| `cmake --build` 무경고 빌드 | ✅ |
| `hello_micromag.exe` 실행 → `vortex.vtk` 생성 | ✅ |
| `ctest` → 7/7 단위 테스트 통과 | ✅ |
| Python `import micromag` → `StructuredGrid` 동작 | ✅ |
| `examples/hello_micromag.py` → `vortex_py.vtk` 생성 | ✅ |

---

## 빌드 환경

| 항목 | 값 |
|---|---|
| OS | Windows 11 Education 10.0.26200 |
| CMake generator | Visual Studio 18 2026 |
| MSVC | 19.51.36243.0 (VS 2026 Community) |
| Python | 3.13.5 (Anaconda, `D:/anaconda3`) |
| pybind11 | 3.0.1 (vcpkg) |
| Catch2 | 3.15.0 (vcpkg) |

> **주의**: VS 2022와 VS 2026이 공존하는 환경입니다.
> vcpkg는 VS 2026 (MSVC 14.51)을 기본 컴파일러로 선택하므로,
> CMakePresets.json도 `"Visual Studio 18 2026"` 제너레이터를 사용해야
> Catch2 등 vcpkg 패키지와 ABI가 일치합니다.

---

## 스펙 대비 변경 사항

### 1. `include/micromag/types.hpp` — `Vec3::operator/=` 추가

**원인**: 명세의 `Vec3`에 `operator/=`가 누락되어 있었음.
`field.cpp`의 `normalize()`에서 `v /= n` 호출 시 컴파일 에러 발생.

```cpp
// 추가된 연산자
Vec3& operator/=(Real s) { x /= s; y /= s; z /= s; return *this; }
```

### 2. `CMakePresets.json` — generator 변경

**원인**: 명세는 `"Visual Studio 17 2022"`를 사용하지만, 시스템에 VS 2026이
설치되어 있고 vcpkg도 VS 2026 툴셋으로 패키지를 컴파일함.
툴셋 불일치로 Catch2 링크 시 LNK2019 에러 발생.

```json
// 변경 전
"generator": "Visual Studio 17 2022"

// 변경 후
"generator": "Visual Studio 18 2026"
```

### 3. `CMakeLists.txt` — Python 모듈 출력 디렉터리 명시

**원인**: MSVC 멀티-config 제너레이터는 `LIBRARY_OUTPUT_DIRECTORY`에
자동으로 `Release/` 서브디렉터리를 붙임.
`_micromag.pyd`가 `python/Release/`에 생성되어 `__init__.py`가 못 찾음.

```cmake
# per-config 출력 경로 명시 추가
set_target_properties(_micromag PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY           ${CMAKE_BINARY_DIR}/python
    LIBRARY_OUTPUT_DIRECTORY_RELEASE   ${CMAKE_BINARY_DIR}/python
    LIBRARY_OUTPUT_DIRECTORY_DEBUG     ${CMAKE_BINARY_DIR}/python
    RUNTIME_OUTPUT_DIRECTORY           ${CMAKE_BINARY_DIR}/python
    RUNTIME_OUTPUT_DIRECTORY_RELEASE   ${CMAKE_BINARY_DIR}/python
    RUNTIME_OUTPUT_DIRECTORY_DEBUG     ${CMAKE_BINARY_DIR}/python
)
```

---

## 빌드 방법

```powershell
# 최초 configure (build/ 없을 때)
cmake --preset windows-msvc

# 빌드
cmake --build --preset windows-msvc

# 테스트
ctest --preset windows-msvc

# 데모 실행
.\build\windows-msvc\bin\Release\hello_micromag.exe

# Python 예제
python examples\hello_micromag.py
```

---

## 생성된 파일 구조

```
build/windows-msvc/
├── bin/Release/
│   ├── hello_micromag.exe     # 데모 실행파일
│   └── unit_tests.exe         # Catch2 테스트
├── lib/Release/
│   └── micromag_core.lib      # 코어 정적 라이브러리
└── python/
    ├── _micromag.cp313-win_amd64.pyd   # Python C 확장 모듈
    └── micromag/
        └── __init__.py
```

---

## 다음 단계: Phase 1b

- `Material` 구조체 (`Ms`, `A_exchange`, `K_uniaxial`, `alpha` 등)
- `IEffectiveField` 추상 인터페이스
- Zeeman field 구현
- Uniaxial anisotropy field 구현
- Heisenberg exchange field 구현 (6-point stencil)
- Energy 계산 (검증용)
