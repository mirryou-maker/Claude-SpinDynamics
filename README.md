# Micromag

Micromagnetic simulator written in C++/CUDA with Python bindings.

## Phase Roadmap

| Phase | 내용 | 상태 |
|---|---|---|
| 1a | 프로젝트 골격 (Grid, VectorField3D, VTK I/O, Python 바인딩) | ✅ 완료 |
| 1b | 첫 물리: Effective field (Zeeman, Anisotropy, Exchange) | 📐 설계 완료, 구현 예정 |
| 1c | LLG integrator (RK4, RK45) | ⬜ 예정 |
| 1d | Slonczewski STT | ⬜ 예정 |
| 1e | OOMMF OVF I/O, µMAG standard problem | ⬜ 예정 |
| 2 | Demag via FFTW (CPU) + cuFFT (GPU) | ⬜ 예정 |
| 3 | Cubic anisotropy, interfacial DMI | ⬜ 예정 |

## Phase 1a — 현재 구현

- `StructuredGrid` — 균일 격자, 셀 크기/볼륨/범위
- `VectorField3D` — 격자 위 3D 벡터장, uniform/vortex 초기화
- VTK legacy 포맷 I/O (ParaView 시각화)
- Catch2 단위 테스트 7개
- Python 바인딩 (pybind11): `micromag` 패키지로 C++ 클래스 전체 노출

## Phase 1b — 설계 완료

`docs/phase1b_effective_fields.md` 참조.

구현할 클래스:
- `Material` — Ms, A, K, easy_axis, alpha (표준 재료: permalloy, cobalt, iron)
- `IEffectiveField` + `EffectiveFieldSum` — effective field 추상 인터페이스
- `ZeemanField` — 균일 외부 자기장
- `UniaxialAnisotropyField` — 단축 결정 이방성
- `ExchangeField` — 6-point Laplacian, Neumann/Periodic BC

## Build

환경 설정: `docs/phase1_environment_setup.md` 참조.

```powershell
cmake --preset windows-msvc
cmake --build --preset windows-msvc
ctest --preset windows-msvc
```

## Run

```powershell
.\build\windows-msvc\bin\Release\hello_micromag.exe
```

`vortex.vtk` 을 ParaView로 열어 확인.

## Docs

| 문서 | 내용 |
|---|---|
| `docs/phase1_environment_setup.md` | 빌드 환경 설정 (vcpkg, CMake, Python) |
| `docs/phase1a_project_structure.md` | Phase 1a 파일 구조 및 설계 |
| `docs/phase1a_completion_notes.md` | Phase 1a 완료 노트 (이슈 및 해결책) |
| `docs/phase1b_effective_fields.md` | Phase 1b 설계 문서 (물리 배경, 구현 가이드) |

## License

TBD
