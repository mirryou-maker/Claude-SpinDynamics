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

```powershell
cmake --preset windows-msvc
cmake --build --preset windows-msvc
ctest --preset windows-msvc
```

## Run

```powershell
.\build\windows-msvc\bin\hello_micromag.exe
```

Opens `vortex.vtk` in ParaView.

## License

TBD
