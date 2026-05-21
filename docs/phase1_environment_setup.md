# Micromagnetic 시뮬레이터 개발환경 구축 가이드 (Phase 1)

> Windows 10/11 native 환경에서 C++/CUDA + Python 기반 micromagnetic 시뮬레이터 개발을 위한 단계별 환경 구축 가이드입니다. VS Code + Claude Code 워크플로우를 전제로 작성되었습니다.

---

## 목차

- [전체 개요](#전체-개요)
- [대상 환경](#대상-환경)
- [Step 1. Visual Studio 2022 Community 설치](#step-1-visual-studio-2022-community-설치)
- [Step 2. NVIDIA GPU 드라이버 + CUDA Toolkit 설치](#step-2-nvidia-gpu-드라이버--cuda-toolkit-설치)
- [Step 3. CMake 설치](#step-3-cmake-설치)
- [Step 4. Git 설치 + GitHub 계정 연결](#step-4-git-설치--github-계정-연결)
- [Step 5. Python 정리](#step-5-python-정리)
- [Step 6. Node.js LTS 설치](#step-6-nodejs-lts-설치)
- [Step 7. Claude Code 설치 + 인증](#step-7-claude-code-설치--인증)
- [Step 8. VS Code 필수 확장 설치](#step-8-vs-code-필수-확장-설치)
- [Step 9. vcpkg + FFTW 설치 (C++ 의존성 관리)](#step-9-vcpkg--fftw-설치-c-의존성-관리)
- [Step 10. 프로젝트 폴더 + Python 가상환경](#step-10-프로젝트-폴더--python-가상환경)
- [Step 11. VS Code로 프로젝트 열고 동작 확인](#step-11-vs-code로-프로젝트-열고-동작-확인)
- [최종 체크리스트](#최종-체크리스트)
- [Troubleshooting](#troubleshooting)
- [다음 단계 예고](#다음-단계-예고)

---

## 전체 개요

큰 흐름은 다음과 같습니다:

1. **VS Code + Claude Code 확장** 설치 및 인증
2. **Git** 설치 (버전 관리, 필수)
3. **C++ 컴파일러 + CMake** 설치 (빌드 시스템)
4. **Python + 가상환경** 설정 (바인딩, 후처리)
5. **FFTW** 라이브러리 설치 (CPU demag용)
6. **CUDA Toolkit** 설치 (GPU 백엔드용)
7. **VS Code 필수 확장** 설치 (C++, CMake, Python, CUDA syntax 등)
8. **프로젝트 초기 구조** 생성 및 첫 빌드 테스트

각 단계는 "왜 필요한지 → 무엇을 설치하는지 → 어떻게 검증하는지" 순서로 설명합니다.

## 대상 환경

- **운영체제**: Windows 10 또는 11 (native)
- **GPU**: NVIDIA GPU (CUDA 개발)
- **시작 상태**: VS Code + Python 일부 도구 설치 가정

> **참고**: Windows native에서 CUDA + C++ + Python 바인딩 개발은 가능하지만, WSL2(Ubuntu) 대비 설치·디버깅이 1.5배 정도 까다롭습니다. 막히는 부분에서는 WSL2를 부분적으로 활용하는 옵션도 열어두세요.

---

## Step 1. Visual Studio 2022 Community 설치

### 왜 필요한가
Windows에서 CUDA를 컴파일하려면 **반드시 MSVC 컴파일러**가 필요합니다. MinGW, Clang으로는 nvcc가 동작하지 않습니다. VS 2022 Community는 무료이며 학술/개인 용도로 자유롭게 사용 가능합니다.

### 설치 방법
1. https://visualstudio.microsoft.com/downloads/ 에서 **Visual Studio Community 2022** 다운로드
2. 설치 관리자 실행 → **워크로드(Workloads)** 탭에서 다음 두 가지 체크:
   - ✅ **Desktop development with C++** (C++을 사용한 데스크톱 개발)
   - ✅ **Python development** (선택, 디버깅에 유용)
3. 오른쪽 "개별 구성 요소(Individual components)" 탭에서 다음 포함 여부 확인:
   - MSVC v143 - VS 2022 C++ x64/x86 build tools
   - Windows 11 SDK (최신 버전)
   - C++ CMake tools for Windows
4. 설치 (약 8-12GB, 시간 소요)

### 검증
시작 메뉴에서 **"x64 Native Tools Command Prompt for VS 2022"** 검색 → 실행 후:

```cmd
cl
```

`Microsoft (R) C/C++ Optimizing Compiler Version 19.xx.xxxxx for x64` 메시지가 나오면 성공.

> ⚠️ **중요**: 이후 CUDA 빌드는 반드시 **x64 Native Tools Command Prompt**나 VS Code에서 환경변수가 잡힌 상태로 실행해야 합니다. 일반 PowerShell에서는 `cl`이 인식되지 않습니다.

---

## Step 2. NVIDIA GPU 드라이버 + CUDA Toolkit 설치

### 왜 필요한가
CUDA Toolkit이 있어야 `nvcc` 컴파일러, cuFFT, cuBLAS 등 GPU 라이브러리를 사용할 수 있습니다. Phase 1에서 CUDA를 본격적으로 안 쓰더라도 미리 깔아두는 게 좋습니다.

### 사전 확인: GPU 드라이버 버전

PowerShell에서:

```powershell
nvidia-smi
```

- 출력 맨 위에 "Driver Version: 5xx.xx" / "CUDA Version: 12.x" 표시됨
- "CUDA Version"은 *드라이버가 지원 가능한 최대 버전*이지, 설치된 Toolkit이 아닙니다
- 명령어가 인식되지 않으면 https://www.nvidia.com/Download/index.aspx 에서 본인 GPU에 맞는 최신 Game Ready 또는 Studio 드라이버 먼저 설치

### CUDA Toolkit 설치
1. https://developer.nvidia.com/cuda-downloads 접속
2. Operating System: Windows → Architecture: x86_64 → Version: 11 또는 10 → Installer Type: **exe (local)** 권장
3. 설치 옵션은 **Express (권장)** 선택
4. 설치 중 "Visual Studio Integration" 항목이 자동으로 체크되어야 함 (Step 1을 먼저 한 이유)

### 검증
새로운 PowerShell 창을 열어서 (환경변수 갱신을 위해 **반드시 새 창**):

```powershell
nvcc --version
nvidia-smi
```

두 명령 모두 정상 출력되면 OK.

> ⚠️ **버전 매칭 중요**: `nvcc --version`의 release 버전이 `nvidia-smi`의 "CUDA Version"보다 크면 PTX JIT 컴파일 실패가 발생할 수 있습니다. [Troubleshooting > PTX 에러](#troubleshooting-7-ptx-toolchain-에러)를 참고하세요.

---

## Step 3. CMake 설치

### 왜 필요한가
복잡한 C++/CUDA/Python 혼합 프로젝트의 빌드를 관리해주는 표준 도구입니다. mumax+도 CMake를 씁니다. Visual Studio 안에 들어있는 CMake도 있지만, command line에서 직접 호출 가능한 standalone 버전이 더 편합니다.

### 설치
1. https://cmake.org/download/ → "Windows x64 Installer" (`.msi`) 다운로드
2. 설치 중 **"Add CMake to the system PATH for all users"** 옵션 반드시 체크
3. 설치 완료 후 새 PowerShell 창에서 검증

### 검증
```powershell
cmake --version
```

`cmake version 3.28.x` 이상이면 OK (최소 3.24 이상 권장, CUDA 통합 때문).

---

## Step 4. Git 설치 + GitHub 계정 연결

### 왜 필요한가
버전관리는 시뮬레이션 코드처럼 점진적으로 복잡해지는 프로젝트에서 필수입니다. 실수로 작동하던 코드를 망쳐도 되돌릴 수 있어야 합니다.

### 설치
1. https://git-scm.com/download/win → 64-bit Git for Windows Setup 다운로드
2. 설치 시 대부분 기본값 OK. 다만 다음 두 가지 확인:
   - "Adjusting your PATH environment" → **Git from the command line and also from 3rd-party software** 선택
   - "Configuring the line ending conversions" → **Checkout Windows-style, commit Unix-style** 선택
3. 설치 후 PowerShell에서 본인 정보 등록:

```powershell
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

### 검증
```powershell
git --version
```

GitHub 계정이 없다면 https://github.com 에서 무료로 가입.

---

## Step 5. Python 정리

### 왜 필요한가
Python은 시뮬레이터의 사용자 인터페이스 언어가 됩니다. Windows에는 보통 여러 Python이 깔려서 헷갈리는 경우가 많아 한 번 정리합니다.

### 현재 상태 점검

```powershell
python --version
where.exe python
python -c "import sys; print(sys.executable)"
```

- `python --version` → default Python 버전
- `where.exe python` → 설치 위치 (여러 개면 줄 단위로 출력)
- 마지막 명령 → 실제 실행 중인 Python 인터프리터 절대 경로

> ⚠️ **Microsoft Store stub 주의**: 출력 경로가 `C:\Users\...\AppData\Local\Microsoft\WindowsApps\python.exe`이면 Microsoft Store stub입니다. 진짜 Python이 아니라 Store 유도용 단축 링크라 개발에 부적합합니다.

### 권장 사항
- **Python 3.12** (3.13은 일부 과학 패키지 호환 이슈)
- python.org 정식 installer 버전 사용
- 설치 시 **"Add python.exe to PATH"** 반드시 체크

### 참고: py launcher (선택)
`py` 명령은 python.org installer 옵션 **"Install launcher for all users"** 체크 시 깔리는 도구입니다. 필수는 아니므로 없어도 진행에 지장 없습니다. [Troubleshooting > py 명령 인식 불가](#troubleshooting-1-py-명령-인식-불가)를 참고하세요.

---

## Step 6. Node.js LTS 설치

### 왜 필요한가
Claude Code CLI는 npm 패키지로 배포되어 Node.js가 필요합니다.

### 설치
1. https://nodejs.org → **LTS 버전** (현재 22.x 계열) Windows Installer 다운로드
2. 기본값으로 설치, "Automatically install the necessary tools..." 옵션은 **체크 해제** 권장 (이미 VS 2022 설치했으므로 중복 회피)

### 검증
```powershell
node --version
npm --version
```

---

## Step 7. Claude Code 설치 + 인증

### 왜 필요한가
VS Code 안에서 본격적으로 AI agent를 활용하려면 Claude Code가 필요합니다.

### (a) CLI 설치

PowerShell에서:

```powershell
npm install -g @anthropic-ai/claude-code
```

권한 오류 시 PowerShell을 관리자 권한으로 다시 열거나, npm prefix를 사용자 폴더로 변경:

```powershell
mkdir $HOME\.npm-global
npm config set prefix "$HOME\.npm-global"
# 시스템 환경변수 PATH에 %USERPROFILE%\.npm-global 추가
```

### (b) VS Code 확장 설치
1. VS Code 열기 → 좌측 사이드바 Extensions 아이콘 (`Ctrl+Shift+X`)
2. 검색창에 "Claude Code" 입력
3. **Anthropic**이 publisher인 공식 확장 설치 (비공식 확장 주의)
4. 설치 후 좌측 사이드바에 별 모양(Spark) 또는 Claude Code 아이콘 생성

### 인증
1. Claude Code 사이드바 아이콘 클릭 → 로그인 버튼
2. 브라우저로 리다이렉트 → Claude.ai 계정으로 인증 (Pro/Max/Team/Enterprise 구독 또는 API 결제 계정 필요)
3. 인증 완료되면 VS Code로 자동 복귀

### 검증
VS Code 내부 터미널 (`` Ctrl+` ``)에서:

```powershell
claude --version
```

또는 사이드바에서 직접 메시지를 보내보기.

---

## Step 8. VS Code 필수 확장 설치

### 왜 필요한가
Claude Code 외에도 C++/CUDA/CMake/Python 개발용 보조 확장들이 있어야 코드 인텔리센스, 디버깅, 빌드가 매끄럽게 동작합니다.

### 설치 목록

VS Code Extensions 검색에서 다음을 설치 (모두 Microsoft 또는 공식 publisher):

| 확장 이름 | Publisher | 식별자 |
|---|---|---|
| C/C++ | Microsoft | `ms-vscode.cpptools` |
| C/C++ Extension Pack | Microsoft | `ms-vscode.cpptools-extension-pack` |
| CMake Tools | Microsoft | `ms-vscode.cmake-tools` |
| CMake | twxs | `twxs.cmake` |
| Python | Microsoft | `ms-python.python` |
| Pylance | Microsoft | `ms-python.vscode-pylance` |
| Nsight Visual Studio Code Edition | NVIDIA | `nvidia.nsight-vscode-edition` |
| GitLens | GitKraken | `eamodio.gitlens` |

설치 후 VS Code 한 번 재시작.

### 검증

**방법 1**: Extensions 패널(`Ctrl+Shift+X`) 검색창에 `@installed` 입력 → 설치된 확장만 필터링.

**방법 2**: 터미널에서 한 번에 확인:

```powershell
code --list-extensions
```

특정 항목만 빠르게 확인:

```powershell
code --list-extensions | Select-String -Pattern "cpptools|cmake|python|pylance|nsight|gitlens"
```

**방법 3**: 각 확장이 실제 동작하는지 sanity check
- **C/C++**: `test.cpp` 파일에 `#include <iostream>` 입력 → 빨간 줄 없고 `std::` 자동완성 뜨면 정상
- **CMake Tools**: `Ctrl+Shift+P` → "CMake:" 입력 → 명령 리스트 나오면 정상
- **Python + Pylance**: `test.py`에 `import numpy` → 좌측 하단에 인터프리터 경로 표시
- **Nsight**: `test.cu` 파일에 `__global__ void hello() {}` → syntax highlighting 색상 입혀지면 정상
- **GitLens**: Git 초기화된 폴더에서 inline blame 표시되면 정상

---

## Step 9. vcpkg + FFTW 설치 (C++ 의존성 관리)

### 왜 필요한가
FFTW는 demag 계산용 핵심 라이브러리입니다. Windows에서 FFTW를 깔끔하게 쓰는 가장 좋은 방법은 Microsoft의 C++ 패키지 매니저 **vcpkg**를 쓰는 것입니다.

### vcpkg 설치

PowerShell에서:

```powershell
cd C:\
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install
```

마지막 명령이 출력하는 CMake toolchain 경로(`C:\vcpkg\scripts\buildsystems\vcpkg.cmake`)를 기억해두세요. 나중에 CMakeLists.txt에서 사용합니다.

### 의존성 설치

```powershell
.\vcpkg install fftw3:x64-windows
.\vcpkg install pybind11:x64-windows
.\vcpkg install hdf5:x64-windows
.\vcpkg install nlohmann-json:x64-windows
```

각 설치는 컴파일이라 몇 분씩 걸립니다. 한 번 설치하면 다음 프로젝트에서도 재사용됩니다.

### 검증

```powershell
.\vcpkg list
```

설치된 패키지 리스트가 보이면 OK.

---

## Step 10. 프로젝트 폴더 + Python 가상환경

적절한 위치에 프로젝트 폴더를 만듭니다. **경로에 공백이나 한글이 들어가지 않도록 주의하세요.** CMake와 일부 빌드 도구가 공백·non-ASCII 경로에서 문제를 일으킵니다.

```powershell
mkdir D:\dev\micromag
cd D:\dev\micromag

# Git 초기화
git init

# Python 가상환경
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

활성화하면 프롬프트 앞에 `(.venv)` 표시가 붙습니다. PowerShell 실행 정책 오류가 나면 한 번만:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 기본 패키지 설치

```powershell
pip install --upgrade pip
pip install numpy scipy matplotlib pybind11 pyvista
```

### .gitignore 생성

```powershell
@"
.venv/
build/
__pycache__/
*.pyc
.vscode/
*.egg-info/
out/
*.vts
*.ovf
"@ | Out-File -Encoding utf8 .gitignore
```

---

## Step 11. VS Code로 프로젝트 열고 동작 확인

```powershell
code .
```

VS Code가 열리면:

1. 우측 하단에 "Do you trust the authors..." 뜨면 "Yes, I trust"
2. `Ctrl+Shift+P` → "Python: Select Interpreter" → `.venv` 인터프리터 선택
3. (선택) `Ctrl+Shift+P` → "CMake: Select a Kit" → **Visual Studio Community 2022 Release - amd64** 선택
   - 이 명령은 폴더에 `CMakeLists.txt`가 있을 때만 활성화됩니다. 지금은 건너뛰어도 됩니다 ([Troubleshooting > CMake: Select a Kit 안 보임](#troubleshooting-3-cmake-select-a-kit-명령이-안-보임) 참고).

### 컴파일러 sanity check

VS Code 통합 터미널 (`` Ctrl+` ``)을 열고, 터미널 패널 오른쪽 위 드롭다운(`+` 옆 ⌵)에서 **"Developer PowerShell for VS 2022"** 선택. 보이지 않으면 [Troubleshooting > Developer PowerShell 안 보임](#troubleshooting-4-developer-powershell-for-vs-2022-드롭다운에-안-보임) 참고.

`hello.cpp`:
```cpp
#include <iostream>
int main() {
    std::cout << "Micromag dev environment ready!\n";
    return 0;
}
```

```powershell
cl hello.cpp
.\hello.exe
```

→ "Micromag dev environment ready!" 출력되면 정상.

### CUDA sanity check

`hello.cu` (VS Code에서 새 파일 만들어서 작성, 인코딩 UTF-8 확인):

```cpp
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d : %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return 1; \
    } \
} while(0)

__global__ void hello() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    printf("CUDA devices found: %d\n", deviceCount);

    if (deviceCount == 0) {
        printf("No CUDA device available.\n");
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device 0: %s, compute capability %d.%d\n",
           prop.name, prop.major, prop.minor);

    hello<<<1, 4>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Kernel finished.\n");
    return 0;
}
```

빌드 및 실행 (인코딩 경고 회피 + 현재 GPU 전용 컴파일):

```powershell
nvcc -arch=native -Xcompiler /utf-8 hello.cu -o hello_cuda.exe
.\hello_cuda.exe
```

4개 GPU 스레드에서 메시지 + "Kernel finished." 출력되면 CUDA 환경 완벽.

> 만약 PTX 에러가 나면 [Troubleshooting > PTX 에러](#troubleshooting-7-ptx-toolchain-에러)를 참고하세요.

---

## 최종 체크리스트

설치 후 PowerShell에서 다음 한 줄씩 모두 정상 출력되면 환경 구축 완료:

```powershell
cl                    # MSVC 컴파일러
nvcc --version        # CUDA 컴파일러
nvidia-smi            # GPU 드라이버
cmake --version       # 빌드 시스템 (>= 3.24)
git --version         # 버전관리
python --version      # Python (>= 3.11)
node --version        # Node.js (>= 18)
claude --version      # Claude Code CLI
```

추가 확인:
- [ ] VS Code 좌측 사이드바에 Claude Code 아이콘 표시, 로그인 상태
- [ ] VS Code Extensions에서 8개 확장 모두 설치 확인
- [ ] `D:\dev\micromag` 프로젝트 폴더 생성, Git 초기화 완료
- [ ] `.venv` Python 가상환경 활성화 가능
- [ ] `hello.cpp` 빌드 및 실행 성공
- [ ] `hello.cu` 빌드 및 실행 성공

---

## Troubleshooting

실제 환경 구축 중 발생할 수 있는 문제와 해결책 모음.

### Troubleshooting 1. `py` 명령 인식 불가

**증상**: `py -0` 실행 시 "인식할 수 없는 명령" 오류.

**원인**: `py.exe`는 python.org 공식 인스톨러에서 "Install launcher for all users" 옵션을 체크해야 깔리는 Python Launcher입니다. Microsoft Store 버전, Anaconda/Miniconda, 또는 launcher 옵션을 안 체크한 경우 `py`가 없습니다.

**해결**: `py`는 필수가 아닙니다. 다음 명령으로 대체:

```powershell
python --version
where.exe python
python -m pip --version
python -m pip list
```

py launcher가 꼭 필요하면 https://www.python.org/downloads/ 에서 최신 installer 받아 "Modify" → "py launcher" 체크.

---

### Troubleshooting 2. Step 8 확장 설치 확인

**증상**: 8개 확장이 모두 잘 깔렸는지 한 번에 확인하고 싶음.

**해결**:
- Extensions 패널 검색창에 `@installed` 입력 → 설치된 것만 필터링
- 또는 터미널에서:

```powershell
code --list-extensions | Select-String -Pattern "cpptools|cmake|python|pylance|nsight|gitlens"
```

각 확장 실제 동작 확인:
- C/C++: `.cpp` 파일에서 `std::` 자동완성 뜨는지
- CMake Tools: `Ctrl+Shift+P` → "CMake:" 입력 시 명령 리스트 보이는지
- Python: 좌측 하단 상태바에 인터프리터 경로 표시
- Nsight: `.cu` 파일의 `__global__` 키워드 syntax highlighting
- GitLens: Git 폴더에서 inline blame 표시

---

### Troubleshooting 3. `CMake: Select a Kit` 명령이 안 보임

**증상**: `Ctrl+Shift+P` → "CMake: Select a Kit" 명령이 검색되지 않음.

**원인**: CMake Tools 확장은 폴더에 `CMakeLists.txt` 파일이 있을 때만 명령어들이 활성화되는 lazy activation 방식. 빈 폴더에서는 `Quick Start`, `Scan for Kits` 정도만 노출됩니다.

**해결책 1 (권장)**: Kit 선택은 일단 건너뜁니다. 본격 프로젝트 `CMakeLists.txt` 작성 시 자동으로 prompt가 뜹니다.

**해결책 2**: 임시 minimal CMakeLists.txt를 만들어 확인:

```cmake
cmake_minimum_required(VERSION 3.24)
project(sanity_check LANGUAGES CXX)
add_executable(hello hello.cpp)
```

VS Code reload (`Ctrl+Shift+P` → "Developer: Reload Window") 후 `CMake: Select a Kit` 사용 가능. Kit 리스트가 비어있으면 `CMake: Scan for Kits` 먼저 실행.

---

### Troubleshooting 4. "Developer PowerShell for VS 2022" 드롭다운에 안 보임

**증상**: VS Code 터미널 드롭다운에 "Developer PowerShell for VS 2022" / "Developer Command Prompt for VS 2022" 항목 없음.

**1단계 진단**: 시작 메뉴에서 "x64 Native Tools" 검색

- **결과 있음** → VS 2022는 설치됨, VS Code 인식 문제. 2단계로.
- **결과 없음 또는 다른 버전만** → VS 2022 C++ 워크로드 미설치. 3단계로.

**2단계: VS Code 인식 시키기**

해결책 ①: VS Code 재시작
- `Ctrl+Shift+P` → `Developer: Reload Window`
- 안되면 VS Code 완전 종료 (`Ctrl+Q`) 후 다시 실행

해결책 ②: 터미널 프로필 재스캔
- `Ctrl+Shift+P` → `Terminal: Select Default Profile`
- 리스트에 "Developer PowerShell for VS 2022" 보이면 선택

해결책 ③: 수동으로 settings.json에 프로필 추가
- `Ctrl+,` → 우측 상단 `{}` 아이콘 (Open Settings JSON)
- 다음 추가:

```json
{
    "terminal.integrated.profiles.windows": {
        "Developer PowerShell for VS 2022": {
            "source": "PowerShell",
            "args": [
                "-NoExit",
                "-Command",
                "& {Import-Module 'C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\Tools\\Microsoft.VisualStudio.DevShell.dll'; Enter-VsDevShell -VsInstallPath 'C:\\Program Files\\Microsoft Visual Studio\\2022\\Community' -SkipAutomaticLocation -DevCmdArguments '-arch=x64'}"
            ],
            "icon": "terminal-cmd"
        }
    }
}
```

> Community가 아닌 Professional/Enterprise 사용 시 경로 수정 필요.

**3단계: VS 2022 C++ 워크로드 재설치**

시작 메뉴 → "Visual Studio Installer" 실행 → Modify → **Desktop development with C++** 워크로드 체크 → 설치.

**임시 우회책**: Windows 시작 메뉴에서 "x64 Native Tools Command Prompt for VS 2022" 직접 실행. 별도 창이지만 컴파일러 환경변수가 잡혀있어 즉시 사용 가능.

---

### Troubleshooting 5. C4819 인코딩 경고

**증상**: `nvcc` 컴파일 시 다음 경고:
```
warning C4819: 현재 코드 페이지(949)에서 표시할 수 없는 문자가 파일에 들어 있습니다.
```

**원인**: 한글 Windows 기본 코드 페이지(CP949)와 소스 파일 인코딩 불일치. PowerShell `echo`나 `Out-File`로 만든 파일에 UTF-16 LE BOM이 붙어 MSVC가 혼란.

**해결**:

방법 1: VS Code에서 새 파일 작성. 우측 하단 인코딩 표시 클릭 → "Save with Encoding" → **UTF-8** 선택.

방법 2: nvcc에 `/utf-8` 플래그 전달 (MSVC 호스트 컴파일러로):

```powershell
nvcc -Xcompiler /utf-8 hello.cu -o hello_cuda.exe
```

> 본격 빌드 시 CMakeLists.txt에 이 플래그를 한 번만 넣어두면 매번 안 써도 됩니다.

---

### Troubleshooting 6. CUDA 실행 시 아무 출력도 없음

**증상**: `.exe` 생성 + 정상 종료되는데 메시지 출력 없음.

**원인**: GPU 커널이 silent하게 실행 실패. `cudaDeviceSynchronize()` 반환값 미체크로 에러가 묻힘.

**해결**: 에러 체크 매크로 포함된 hello.cu 사용 ([Step 11 CUDA sanity check](#cuda-sanity-check) 참고). 에러 메시지를 보고 문제 식별:
- "no kernel image is available for execution" → architecture 문제, `-arch=native` 추가
- "PTX compiled with unsupported toolchain" → Troubleshooting 7 참고

---

### Troubleshooting 7. PTX toolchain 에러

**증상**: 
```
CUDA error: the provided PTX was compiled with an unsupported toolchain.
```

**원인**: 드라이버-Toolkit 버전 불일치.
- `nvcc`는 기본적으로 `.exe`에 PTX(중간 코드)를 넣어두고, 실행 시 드라이버가 JIT 컴파일
- **드라이버 버전이 Toolkit이 생성한 PTX를 처리하기 너무 구버전**이면 에러
- 즉 "Toolkit 최신, 드라이버 구버전" 상황

**빠른 해결 (즉시 우회)**:

PTX 의존하지 않고 현재 GPU 전용 native binary(SASS)만 생성:

```powershell
nvcc -arch=native hello.cu -o hello_cuda.exe
```

`-arch=native`는 시스템 GPU의 compute capability를 자동 감지. CUDA 12.x 이상 지원.

CUDA 11.x 이하면 GPU에 맞춰 직접 지정:

| GPU 세대 | 예시 | `-arch=` 값 |
|---|---|---|
| Pascal (GTX 10xx) | GTX 1080, 1660 | `sm_61` |
| Turing (RTX 20xx, GTX 16xx) | RTX 2080, 2070 | `sm_75` |
| Ampere (RTX 30xx, A100) | RTX 3090, 3060 | `sm_86` |
| Ada (RTX 40xx) | RTX 4090, 4070 | `sm_89` |
| Hopper (H100) | H100 | `sm_90` |
| Blackwell (RTX 50xx) | RTX 5090, 5080 | `sm_120` |

예시:
```powershell
nvcc -arch=sm_86 -code=sm_86 hello.cu -o hello_cuda.exe
```

`-code=sm_86`까지 명시하면 SASS만 생성되어 확실히 JIT 회피.

**근본 해결 (권장)**:

cuFFT 등 라이브러리 사용 시 PTX 의존 불가피한 부분이 생길 수 있으므로, 드라이버를 Toolkit과 매칭하는 것이 정공법.

1. https://www.nvidia.com/Download/index.aspx 에서 본인 GPU의 최신 Game Ready 또는 Studio Driver 다운로드
2. 설치 시 "Custom installation" → "Perform a clean installation" 체크
3. 재부팅
4. 재부팅 후 확인:

```powershell
nvidia-smi
nvcc --version
```

`nvidia-smi`의 "CUDA Version"이 `nvcc --version`의 release 버전과 같거나 크면 OK.

---

## 다음 단계 예고

환경 구축이 완료되면 **Phase 1 MVP의 프로젝트 구조와 첫 코드** 작업으로 넘어갑니다:

1. **CMakeLists.txt 최상위 구조 작성** — CUDA, FFTW, pybind11 통합
2. **디렉터리 골격 생성** — `src/core`, `src/fields`, `src/solvers`, `python/`, `tests/`
3. **기본 자료구조** — 3D vector field, structured grid 클래스
4. **단위 테스트 프레임워크** — Catch2 또는 GoogleTest (vcpkg로 설치)
5. **첫 빌드 + Python 바인딩 hello world** — pybind11로 C++ 함수를 Python에서 호출

이후 phase 진행:

- **Phase 2**: FFTW + cuFFT demag (on/off toggle), adaptive RK45, PBC, µMAG standard problem 3, 4 통과
- **Phase 3**: DMI 텐서, interfacial DMI, thermal field, SOT, multi-region, skyrmion 예제
- **Phase 4**: Antiferromagnet, magnetoelastic, polycrystalline, Voronoi tessellation
- **Phase 5**: Viewer GUI, 웹 인터페이스, FMR 분석 툴
- **Phase 6 (연구 모듈)**: FMM demag → wavelet demag → ML 가속 비교 연구

---

## 참고 자료

- [OOMMF 공식 사이트](https://math.nist.gov/oommf/)
- [Mumax3 공식 사이트](https://mumax.github.io/)
- [Mumax3 워크숍 자료](https://mumax.ugent.be/mumax3-workshop/)
- [mumax+ 논문 (arXiv:2411.18194)](https://arxiv.org/pdf/2411.18194)
- [µMAG Standard Problems](https://www.ctcms.nist.gov/~rdm/mumag.org.html)
- [Claude Code 공식 문서](https://docs.claude.com/en/docs/claude-code/overview)
- [CUDA Toolkit 문서](https://docs.nvidia.com/cuda/)
- [CMake 공식 문서](https://cmake.org/cmake/help/latest/)
- [vcpkg 공식 문서](https://learn.microsoft.com/en-us/vcpkg/)
