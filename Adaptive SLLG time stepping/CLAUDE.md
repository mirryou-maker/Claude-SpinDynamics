# CLAUDE.md — Claude-SpinDynamics / Task 1

> 저장소: github.com/mirryou-maker/Claude-SpinDynamics
> 로드맵: `CLAUDE-SD_FINITE_TEMP_ROADMAP.md`
> 현재 Task: **Task 1 — Adaptive SLLG Time Stepping**
> 트래커: `~/projects/_tracker/PROJECTS.md` P5

---

## ⚠️ 실제 스택 (2026-07-26 코드 확인 후 확정)

- **언어 스택: C++20 + CUDA + pybind11** (Python은 얇은 바인딩 계층). *Python + CuPy가 아니다.*
  - 현 유한온도 구현: `src/heun_integrator_gpu.cu` + `include/micromag/heun_integrator_gpu.hpp`
    (고정 dt, Stratonovich Heun, `GPUMagState` 소유, thermal=false일 때 CUDA Graph).
  - 새 적분기도 **C++/CUDA 클래스**로 작성하고 `python/bind_gpu.cpp`에서 노출한다.
- **빌드: CMake preset** — `windows-msvc-cuda`(로컬), `linux-gcc-cuda`(iREMB, arch `60;70`).
  GPU 모듈은 `build/<preset>/python/_micromag*.pyd|.so`.
- **테스트: Catch2** (`tests/unit_tests_gpu`, 태그 `[gpu]`) — pytest가 아니다.
  물리 검증(Arrhenius/FDT/|m|-drift)은 Catch2 케이스 + 필요 시 Python 스모크.

§9 코드 규약과 `settings.json` permission은 위 C++/CUDA·CMake·Catch2 기준으로 읽는다.

---

## 1. 프로젝트 개요

Claude-SpinDynamics는 자체 개발 micromagnetics 솔버다. 유한차분(FDM) 격자 + FFT 기반 demag을 사용한다. VAMPIRE(원자론)와는 OVF 포맷으로 연결된다.

**현재 목표는 유한온도(finite temperature) 계산의 가속이다.** 전체 로드맵은 Task 1→2→3 순서이며, 이 문서는 Task 1만 다룬다.

---

## 2. Task 1 범위

### 2.1 하는 것

유한온도 stochastic LLG(SLLG) 적분에 **adaptive time stepping**을 도입한다.

기존 통념은 "유한온도에서는 고정 dt가 필수"였으나 이는 더 이상 유효하지 않다:

- **Berkov & Gorn (2002)**: SDE에서 나타나는 drift term은 magnetization의 **길이**에만 영향을 준다.
- SLLG에서 $|\mathbf{m}|$ 은 보존량이다 → drift term이 물리적으로 무해하다.
- **Leliaert et al. (2017)**: 위 논거로 고차 solver에 adaptive stepping을 붙여 mumax3에 구현, **고정 스텝 Heun 대비 정확도 손실 없이 20× speedup**.

논문거리는 아니다. **재현이고 인프라다.** 그러나 노력 대비 효과 비가 로드맵 전 항목 중 최고이므로 최우선으로 처리한다.

### 2.2 하지 않는 것 (범위 밖 — 제안하지 말 것)

| 항목 | 이유 |
|---|---|
| Replica batching | Task 2. 단 §3의 "batch-ready" 규칙은 지금 지킨다. |
| Rare event / FFS | Task 3. Task 2 완료 전 착수 금지. |
| FFT ↔ dense GEMM, mixed precision | v2(FFT auto-tuning)로 이관됨 |
| 시간축 병렬화 (Parareal/PFASST) | **기각됨.** switching 근처 붕괴. 다시 꺼내지 말 것. |
| FEM 관련 일체 | FEM-SD(P6)는 유한온도 대상이 아님 |

---

## 3. 불변 규칙 (Invariants)

**아래는 어떤 경우에도 깨지 않는다. 위반하는 변경은 거부하고 이유를 설명할 것.**

1. **$|\mathbf{m}| = 1$ 은 기계 정밀도 수준에서 보존된다.**
   정규화(renormalize) 후처리로 때우지 않는다. 적분기 자체가 보존해야 한다.
   $10^6$ step 후 $\big||\mathbf{m}|-1\big| < 10^{-10}$.

2. **Batch-ready**: 새로 작성하는 함수는 선행 replica 차원을 받을 수 있는 시그니처로 쓴다.
   지금은 `R=1`만 쓰지만, Task 2에서 소급 개수하지 않기 위함이다.
   ```python
   # OK
   def advance(m, h_eff, dt, ...):   # m.shape == (R, nx, ny, nz, 3)
   # NG — replica 차원을 전제하지 않은 스칼라 dt
   def advance(m, h_eff, dt: float, ...):  # dt는 (R,) 배열이 될 수 있다
   ```
   **`dt`는 replica마다 달라진다.** adaptive이므로 Task 2에서 필연적으로 `dt[R]` 배열이 된다. 지금부터 스칼라로 하드코딩하지 말 것.

3. **Demag 백엔드는 교체 가능한 인터페이스 뒤에 둔다.**
   v2(FFT auto-tuning)의 접점이다. `demag/fft`만 구현하지만 인터페이스는 지금 판다.

4. **Stratonovich 해석을 유지한다.** 적분기를 바꿀 때 noise term 해석이 바뀌지 않는지 매번 확인.

5. **재현성 + counter-based RNG로 전환**: 동일 seed → 동일 하드웨어에서 bitwise 동일 결과.
   - **현 코드는 cuRAND host 생성기**(`curandCreateGenerator(CURAND_RNG_PSEUDO_DEFAULT)`,
     노이즈 버퍼 [3N] 저장)를 쓴다 — 이는 상태 저장형이라 불변규칙에 어긋난다.
   - **device-side Philox로 전환한다**: `curand_kernel.h`의 `curandStatePhilox4_32_10_t`를
     커널 안에서 counter로 초기화해 on-the-fly 생성(버퍼 저장 없음).
     Key=`(global_replica_id, sim_uid)`, Counter=`(step_index, cell_index, component)`.
   - 이 전환은 Task 1에서 미리 한다: adaptive의 step-doubling 오차추정이 full-dt와 two-half-dt에서
     **동일 Wiener 실현**을 요구하고(§5.2), Brownian additivity ΔW_full=ΔW_½1+ΔW_½2 를
     counter로 결정론적 재생성해야 하기 때문이다. cuRAND host 생성기로는 이 재현이 불가능하다.

---

## 4. 물리 / 수치 배경

### 4.1 SLLG

$$\frac{d\mathbf{m}}{dt} = -\frac{\gamma}{1+\alpha^2}\Big[\mathbf{m}\times\mathbf{H}_{\rm eff} + \alpha\,\mathbf{m}\times(\mathbf{m}\times\mathbf{H}_{\rm eff})\Big]$$

$\mathbf{H}_{\rm eff} = \mathbf{H}_{\rm ext} + \mathbf{H}_{\rm exch} + \mathbf{H}_{\rm demag} + \mathbf{H}_{\rm anis} + \mathbf{H}_{\rm therm}$

### 4.2 Thermal field

$$\mathbf{H}_{\rm therm} = \boldsymbol{\eta}\sqrt{\frac{2\mu_0\alpha k_B T}{B_{\rm sat}\gamma_{LL}\,\Delta V\,\Delta t}}$$

$\boldsymbol{\eta}$ 는 표준정규 난수 벡터, step마다 새로 뽑는다.

**$1/\sqrt{\Delta t}$ 의존성이 이 Task 전체의 핵심 위험 요인이다.** §5.1 참조.

### 4.3 Depondt–Mertens 적분기 (1순위 채택)

Rodrigues 회전 공식으로 $\mathbf{m}$ 을 회전시킨다. **회전은 정의상 $|\mathbf{m}|$ 을 정확히 보존한다.**

$$\boldsymbol{\omega} = -\frac{\gamma}{1+\alpha^2}\big(\mathbf{H}_{\rm eff} + \alpha\,\mathbf{m}\times\mathbf{H}_{\rm eff}\big)$$

$\mathbf{m}$ 을 축 $\hat{\boldsymbol{\omega}}$ 둘레로 각 $|\boldsymbol{\omega}|\Delta t$ 만큼 회전. Predictor–corrector(Heun형)로 중간점 field를 사용한다.

대안(우선순위 순): semi-implicit midpoint (Serpico–d'Aquino), SIB (Mentink et al.).

**주의**: RK45류 고차 embedded 스킴은 step 간 torque 연속성을 요구하는데 thermal field는 매 step 무작위로 바뀐다. 이 제약을 만족하지 않는 스킴은 쓸 수 없다.

---

## 5. 구현 함정 — 우선순위 순

### 5.1 dt 변경 시 thermal field 재스케일 (최우선)

$\mathbf{H}_{\rm therm} \propto 1/\sqrt{\Delta t}$ 이므로 dt가 바뀔 때마다 반드시 재계산해야 한다.

**이걸 빠뜨리면 실효 온도가 조용히 틀어진다.** 시뮬레이션은 정상적으로 돌고, 그림도 그럴듯하게 나오고, 아무 에러도 나지 않는다. Arrhenius 검증(§6.1)에서만 잡힌다.

→ 재스케일 경로를 **단일 함수로 집중**시키고, dt를 바꾸는 모든 경로가 그 함수를 통과하게 강제할 것.
→ 이 재스케일 누락을 직접 검출하는 단위 테스트를 반드시 작성할 것 (dt를 인위적으로 2배 바꾼 뒤 실효 온도 추정).

### 5.2 오차 추정 시 노이즈 고정

Embedded pair 또는 step doubling으로 오차를 추정할 때, **두 해는 반드시 동일한 thermal field 실현값을 사용해야 한다.**

다른 난수를 쓰면 측정하는 것이 "적분 오차"가 아니라 "노이즈의 크기"가 되고, controller가 무의미하게 dt를 계속 줄인다.

### 5.3 Step rejection과 Wiener path bias (이론적 위험 지점)

**여기가 이 Task의 유일한 이론적 위험 지점이다. 설계 결정을 명시적으로 내리고 문서화할 것.**

문제: 스텝을 거부하고 새 난수를 뽑으면 "큰 노이즈가 들어온 스텝"이 선택적으로 거부되어 Wiener path에 bias가 생긴다.

가능한 처리:
- (a) 이미 뽑은 increment를 **Brownian bridge로 세분화**하여 재사용 — 이론적으로 가장 안전
- (b) rejection 자체를 금지하고 dt는 다음 스텝부터만 조정 — 단순하고 bias 없음
- (c) rejection을 허용하되 빈도를 측정해 무시 가능함을 실증

**착수 시 Leliaert et al. (2017)의 실제 처리 방식을 먼저 확인할 것.** 추측으로 구현하지 말 것. 확인이 어려우면 (b)로 시작한다 — 성능은 조금 손해지만 안전하다.

### 5.4 Stratonovich vs Itô

Heun은 Stratonovich로, Euler는 Itô로 수렴한다. 둘은 drift correction만큼 다르다.

Depondt–Mertens는 회전 기반이라 구면 위에 머물며 Stratonovich와 자연스럽게 정합한다. 그러나 **적분기를 교체할 때마다 이 정합성을 재확인할 것.**

---

## 6. 검증 요건 (Definition of Done)

**아래 3개는 Task 1의 완료 조건이자, 이후 모든 변경에서 상시 실행되는 회귀 테스트다.**

> 유한온도 코드는 **조용히 틀리는 게 기본값**이다. 에러 없이 잘 도는 것은 정확성의 증거가 아니다.

### 6.1 Macrospin Arrhenius (필수)

단축 이방성 macrospin의 thermal switching rate가 Brown 해석식과 일치할 것.

$$f = \gamma_{LL}\frac{\alpha}{1+\alpha^2}\sqrt{\frac{8K_{u1}^3 V}{2\pi M_s^2 k_B T}}\,e^{-K_{u1}V/k_BT}$$

- 온도 5점 이상 스윕, Arrhenius plot의 기울기·절편 비교
- **합격선: 상대오차 15% 이내**
- 통계가 필요하므로 uncoupled macrospin을 격자에 다수 배치하는 방식으로 샘플 확보 (Task 2 전까지의 임시 수단)

### 6.2 Fluctuation–Dissipation (필수)

평형 상태에서 $\langle m_z \rangle$, $\langle m_z^2 \rangle$ 이 Boltzmann 분포와 일치할 것.
$\alpha$ 를 2배로 바꿔도 **평형 분포는 변하지 않아야 한다** (동역학만 바뀜). 이건 thermal field 정규화 오류를 잡는 가장 예민한 테스트다.

### 6.3 $|\mathbf{m}|$ drift (필수)

$10^6$ step 후 $\big||\mathbf{m}|-1\big| < 10^{-10}$. 명시적 renormalize 없이.

### 6.4 성능 기록 (필수)

고정 스텝 Heun 대비 speedup 실측값을 `benchmarks/`에 기록. 목표치를 정하지 않는다 — **정확도가 먼저이고 속도는 결과다.**

---

## 7. 세션 워크플로

### 7.1 세션 시작

1. `PROJECTS.md`의 P5 섹션을 읽고 현재 상태 / 다음 액션을 요약 보고
2. 이 문서의 §3(불변 규칙)과 §5(함정)를 확인
3. 작업 대상을 한 가지로 좁혀 제안

### 7.2 세션 종료 (또는 의미 있는 작업 완료 시)

`PROJECTS.md`에서 **P5 섹션만** 갱신한다. 다른 프로젝트 섹션은 절대 건드리지 않는다.

1. 대시보드 표의 상태·현재 Phase·갱신일
2. P5의 `다음 액션` 체크박스
3. `최근 로그`에 한 줄 추가 (`YYYY-MM-DD — 무엇을 왜 했는지`, 3줄 이내)

상세 내용은 커밋 메시지에 남긴다.

### 7.3 작업 단위

**한 번에 하나씩.** 적분기 교체와 controller 도입을 같이 하지 않는다. §6의 검증이 통과한 상태를 항상 유지하며 전진한다. 검증이 깨진 채로 다음 단계를 시작하지 않는다.

---

## 8. 그림 출력 규칙

모든 그림은 `save_fig()` 헬퍼를 통해 저장한다. 직접 `plt.savefig()` 호출 금지.

- `dpi ≤ 110`
- 긴 변 최대 1400 px
- 8-bit RGB
- 저장 위치: `./figs/`, 파일명에 타임스탬프 포함
- **저장 직후 Read tool로 실제 파일을 확인**할 것. 저장했다고 보고만 하지 말 것.

---

## 9. 코드 규약

- **세션 대화는 한국어, 코드·주석·커밋 메시지·문서 산출물은 영어.**
- 물리 상수와 단위는 SI. 단위 변환은 입출력 경계에서만.
- 물리량 변수명은 수식 기호를 따른다 (`m`, `h_eff`, `alpha`, `gamma_ll`, `k_u1`). 축약하지 않는다.
- 새 함수에는 shape 주석을 단다: `# m: (R, nx, ny, nz, 3)`
- 커밋은 원자적으로. 적분기 변경과 테스트 추가를 한 커밋에 섞지 않는다.
- 성능 주장에는 반드시 측정 근거를 붙인다. "빨라질 것으로 예상됨"은 커밋 메시지에 쓰지 않는다.

---

## 10. 참고문헌

- D. Berkov, N. Gorn, *J. Phys.: Condens. Matter* **14**, L281 (2002) — drift term이 $|\mathbf{m}|$ 에만 영향
- J. Leliaert et al., *AIP Advances* **7**, 125010 (2017) — adaptive SLLG, 20× speedup. **§5.3의 rejection 처리를 여기서 확인할 것.**
- A. Vansteenkiste et al., *AIP Advances* **4**, 107133 (2014) — mumax3 설계·검증, thermal field 정의
- P. Depondt, F. G. Mertens, *J. Phys.: Condens. Matter* **21**, 336005 (2009) — rotation 기반 적분기
- W. F. Brown, *J. Appl. Phys.* **34**, 1319 (1963) — thermal switching rate 해석식
