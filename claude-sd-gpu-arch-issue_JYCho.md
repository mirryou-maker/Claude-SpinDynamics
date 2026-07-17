# [Bug] GPU 빌드가 sm_120(Blackwell) 전용으로 고정 컴파일되어 다른 GPU에서 크래시

## 요약

`Claude-SD-v1.0.0-win64-gpu-py313` 배포 패키지(cuFFT-f32/f64, VkFFT-f32/f64 4개 변형 전부)의
GPU 커널이 **단일 CUDA 아키텍처(`-arch sm_120`, Blackwell)로만 컴파일**되어 있고 PTX
forward-compat이 포함되어 있지 않습니다. 그 결과 sm_120이 아닌 GPU(Turing/Ampere/Ada/Hopper 등,
즉 RTX 20~40 시리즈, A100, H100 포함 대부분의 사용자 GPU)에서는 GPU 기능이 전부 동작하지
않습니다. 문제는 단순 미동작이 아니라, **에러 메시지 없이 조용히 죽거나(C++ exe) / CPU 파트만
실행되고 GPU 파트에서 크래시(벤치마크류)** 하는 방식이라 QA 과정에서 "일부는 되는 것 같다"는
잘못된 신호를 줄 수 있습니다.

## 재현 방법

### Python
```python
import micromag as mm
print(mm.cuda_available())      # True — 디바이스 감지는 정상
grid = mm.StructuredGrid(200, 50, 1, 2.5e-9, 2.5e-9, 3.0e-9)
demag = mm.DemagFieldGPU(grid)  # RuntimeError: CUDA: no kernel image is available for execution on the device
```

### C++ exe (`cuFFT-f64\bin\sp4_gpu.exe`)
```
=== SP#4 GPU vs CPU benchmark ===
Grid: 200x50x1  dt=0.05 ps  steps=4000

CPU (DemagField, FFTW)
  t_switch  = 0.167600 ns
  <mx>_final= -0.58815  (ref ~ -0.9862)
  wall time = 14.9 s

[GPU 섹션 헤더 없이 프로세스 즉시 종료]
```
`echo %errorlevel%` → `-1073740791` = `0xC0000409` (uncaught exception → `abort()` → Windows
fail-fast 강제 종료). GPU 진입 직후 죽기 때문에 콘솔 창을 더블클릭으로 실행하면 창이 바로
닫혀버려서 "그냥 꺼진다"는 인상만 남습니다.

## 근본 원인 (바이너리 분석으로 확인)

`_micromag.cp313-win_amd64.pyd`, `sp4_gpu.exe` 등에 `strings`를 떠보면 nvcc 컴파일 커맨드라인이
그대로 남아 있습니다.

```
-arch sm_120 -m 64
```

- `sm_120` **하나만** 타겟팅되어 있고, 다른 세대(sm_75/80/86/89/90 등)나 PTX(`compute_120` 코드
  임베딩)는 전혀 없습니다.
- 즉 `-gencode arch=compute_120,code=sm_120` 형태로만 빌드된 것으로 추정되며,
  `code=compute_120`(PTX)까지 포함되지 않아 드라이버의 forward-compatible JIT 재컴파일도
  불가능합니다.
- `mm.cuda_available()`은 `cuInit()` + 디바이스 존재 여부만 확인하므로 정상 GPU가 있으면 항상
  `True`를 반환합니다. 실제 실패는 커널 이미지를 로드하는 첫 시점(`DemagFieldGPU` 생성)에서만
  발생하기 때문에, "CUDA는 되는데 왜 안 되지"라는 혼동을 유발합니다.
- VkFFT 변형은 FFT 자체는 `--gpu-architecture=sm_%llu%llu` 형태로 실행 시점에 실제 GPU에 맞춰
  JIT 컴파일하는 코드가 확인되지만, exchange/Zeeman/RK4 integrator/cub 기반 reduction 등 나머지
  커스텀 커널은 VkFFT 빌드 안에서도 여전히 `-arch sm_120`으로 고정되어 있어 완전한 해결책은
  아닙니다.

## 이 문제를 그냥 두면 안 되는 이유 (QA/QC 혼동 위험)

1. **같은 폴더(`cuFFT-f64/bin` 등)에 CUDA를 아예 쓰지 않는 CPU 전용 exe와 GPU 전용 exe가 섞여
   있습니다.** import된 DLL을 확인한 결과:
   - CUDA 미참조(CPU 전용): `sp1.exe`, `sp3.exe`, `sp4.exe`, `hello_micromag.exe`,
     `llg_demo.exe`, `thermal_equilibrium.exe`, `unit_tests.exe`, `stt_sot_demo.exe` 등
   - CUDA 참조(GPU 경로 포함): `sp4_gpu.exe`, `sp4_gpu_1ns.exe`, `sp4_rk45_gpu.exe`,
     `sp4_full_gpu_bench.exe`, `unit_tests_gpu.exe`, `benchmark_large.exe`, `demag_profile.exe`,
     `llg_large_bench.exe`
   파일명 규칙(`_gpu` 접미사)에 의존하지 않고 폴더 위치만 보고 "GPU 빌드 폴더 안에 있으니 GPU
   테스트"라고 판단하면, 실제로는 GPU 코드를 전혀 실행하지 않은 CPU 전용 바이너리를 GPU
   검증으로 착각할 수 있습니다.
2. **크래시가 무음/무기록으로 일어납니다.** 벤치마크류(`sp4_gpu.exe`)는 CPU 파트 결과를 정상
   출력한 뒤 GPU 파트에서 uncaught exception → `abort()`로 죽는데, 이때 CUDA 에러 메시지가
   flush되지 않고 사라집니다. 로그만 보면 "CPU 결과까지는 정상"으로 보여 원인 파악이 늦어지고,
   자동화된 QA 파이프라인에서는 종료 코드만 보고 "메모리 손상/불안정" 같은 엉뚱한 원인으로
   오분류될 위험이 있습니다.
3. **`cuda_available()`이 실제 실행 가능 여부를 보장하지 않습니다.** 디바이스 존재만 확인하고
   커널 이미지 호환성은 확인하지 않으므로, 이 API를 신뢰해서 만든 자동 테스트는 sm_120이 아닌
   CI 러너에서 전부 통과된 것처럼 오탐(false positive)할 수 있습니다.

## 권장 조치

1. **멀티 아키텍처 빌드로 전환** — `CMAKE_CUDA_ARCHITECTURES`를 단일 `120`이 아니라 지원 대상
   범위(예: `75;80;86;89;90;120`)로 설정하고, 최소 최신 세대에 대해서는 PTX
   (`code=compute_XX`)도 함께 임베딩해 forward-compat을 확보합니다.
2. **`cuda_available()`을 커널 호환성까지 검사하도록 강화** — 디바이스의 compute capability를
   조회해 바이너리에 임베딩된 아키텍처 목록과 비교하고, 불일치 시 `False`를 반환하거나 명확한
   예외 메시지(`"GPU compute capability 8.6 not supported by this build (built for sm_120
   only)"`)를 던지도록 수정합니다. Python/C++ 양쪽에서 "커널 실행 직전에야 실패"하는 현재
   구조를 "임포트/초기화 시점에 즉시 실패"로 앞당기는 것이 목표입니다.
3. **C++ exe들에 CUDA 호출부 try/catch 및 stderr 출력 추가** — uncaught exception →
   `abort()`(`0xC0000409`)로 죽지 않도록 감싸고, 사람이 읽을 수 있는 에러 메시지를 stderr/로그
   파일에 남깁니다. 콘솔 자동 종료로 메시지가 사라지는 문제도 함께 해결됩니다.
4. **CPU 전용 exe와 GPU 의존 exe를 구조적으로 분리 표시** — 폴더 분리(`bin/cpu/`, `bin/gpu/`) 또는
   빌드 매니페스트(어떤 바이너리가 CUDA에 의존하는지 목록화)를 제공해, 배포/QA 담당자가
   파일명 관례에 의존하지 않고도 CPU/GPU 경로를 구분할 수 있게 합니다.
5. **다중 아키텍처 GPU에 대한 CI 매트릭스 또는 최소한 지원 compute capability 문서화** — 최소
   Turing(sm_75) ~ Blackwell(sm_120)까지 몇 개 대표 아키텍처에 대해 스모크 테스트를 돌리거나,
   README에 "이 빌드는 sm_120 전용" 같은 명시적 경고를 추가합니다.

## 참고

- 재현에 사용한 GPU: (사용자 측 `nvidia-smi --query-gpu=name,compute_cap --format=csv` 결과로
  compute capability 12.0(Blackwell)이 아님을 확인 필요 — 실행 로그상 sm_120 미스매치로 추정)
- 배포 패키지: `Claude-SD-v1.0.0-win64-gpu-py313.zip` (cuFFT-f32/f64, VkFFT-f32/f64 4개 변형 모두
  동일 증상 예상 — cuFFT-f64만 직접 검증)
