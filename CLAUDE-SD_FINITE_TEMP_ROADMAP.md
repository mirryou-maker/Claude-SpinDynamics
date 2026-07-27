# Claude-SpinDynamics — Finite Temperature 가속 로드맵

> 저장소: https://github.com/mirryou-maker/Claude-SpinDynamics
> 트래커: `PROJECTS.md` P5
> 작성일: 2026-07-26
> 목적: Claude Code 세션에서 참조할 작업 계획 문서

---

## 0. 배경 및 범위 결정

### 0.1 기각한 방향

**시간축 병렬화 (Parareal / PFASST / MGRIT) — 채택하지 않음.**

- Thermal noise increment $\Delta W_n$ 은 step 간·cell 간 독립이지만, **상태 $\mathbf{m}_n$ 은 독립이 아니다.** SLLG는 nonlinear SDE이므로 노이즈의 독립성이 궤적의 독립성으로 전이되지 않는다.
- 노이즈 생성은 애초에 병목이 아니다 (counter-based RNG로 on-the-fly 생성). 병목은 매 step의 demag FFT.
- Deterministic LLG에 대한 PFASST 선행연구가 이미 존재 (Kraft et al., magnum.pi).
- 결정적으로, **switching 근처에서 붕괴한다**: separatrix 부근 Lyapunov 불안정성 + 분포의 bimodality. 유한온도 계산의 목적함수는 pathwise 궤적이 아니라 확률/분포이므로 애초에 목적이 어긋난다.

### 0.2 채택한 축

| 축 | 내용 | 본 문서 |
|---|---|---|
| ① | Adaptive SLLG time stepping | Task 1 — 코어 통합 |
| ② | Replica(realization) batching | Task 2 — 코어 통합 |
| ③ | Rare event sampling (FFS 등) | Task 3 — `rare_event/` 서브모듈 |
| ④ | FFT ↔ dense GEMM crossover, mixed precision | **범위 밖** — v2(FFT auto-tuning)로 이관 |

### 0.3 FEM-SD 제외 결정

**FEM-SD는 본 로드맵의 대상이 아니다.** 사유:

- FEM의 강점(불규칙 형상, AMR)이 replica batching과 정면 충돌한다. Replica마다 메시가 적응하면 시스템 행렬 공유가 깨져 배칭 이득이 소멸한다.
- Replica 간 메시를 고정하면 FEM-BEM demag이 multiple-RHS 문제가 되는데, 이는 인수분해 재사용 + H-matrix 압축이라는 **완전히 다른 게임**이다. 규칙 격자 + batched FFT/GEMM 전략과 공유할 코드가 사실상 없다.
- 유한온도 문제에서 FEM이 FDM 대비 갖는 이점이 없다.

→ FEM-SD(P6)는 결정론적 계산 및 복잡 형상에 집중한다. 본 문서의 어떤 항목도 FEM-SD에 반영하지 않는다.

---

## Task 1 — Adaptive SLLG Time Stepping

**상태**: ☐ 미착수
**의존**: 없음 (최우선)
**성격**: 적분기 코어. 분리 불가.

### 1.1 근거

유한온도에서 고정 time step이 필수라는 통념은 **더 이상 유효하지 않다.**

- Berkov & Gorn (2002): SDE에서 나타나는 drift term은 magnetization의 **길이**에만 영향을 준다. SLLG에서 그 길이는 보존량이므로 drift term이 물리적으로 무해하다.
- Leliaert et al. (2017): 위 논거를 이용해 고차 solver에 adaptive stepping을 붙여 mumax3에 구현. **고정 스텝 Heun 대비 정확도 손실 없이 20× speedup.**

논문거리는 아니다. 재현이고 인프라다. 그러나 **노력 대비 효과 비가 전 항목 중 최고**이므로 최우선.

### 1.2 작업 항목

- [ ] 현재 고정 dt 경로 식별 및 분리 (`integrator/` 하위)
- [ ] $|\mathbf{m}|$ 보존 적분기 도입
  - 1순위: **Depondt–Mertens** (rotation 기반, $|\mathbf{m}|$ 정확히 보존)
  - 대안: semi-implicit midpoint (Serpico–d'Aquino), SIB (Mentink et al.)
- [ ] 오차 추정기 + step size controller (PI controller 권장)
- [ ] `dt` 변경 시 thermal field 재스케일 경로 확보
- [ ] Task 2 대비: controller를 **per-replica**로 확장 가능한 구조로 작성할 것 (§2.4 참조)

### 1.3 주의사항 (구현 함정)

1. **Stratonovich vs Itô 일관성.** Heun은 Stratonovich로, Euler는 Itô로 수렴한다. 적분기를 교체할 때 noise term 해석이 바뀌지 않는지 확인.
2. **$B_{\rm therm} \propto 1/\sqrt{\Delta t}$.** dt를 바꾸면 thermal field 크기를 반드시 재계산해야 한다. 이 재스케일을 빠뜨리면 실효 온도가 조용히 틀어지고, 회귀 테스트 없이는 발견되지 않는다.
3. **Step rejection의 bias.** 스텝을 거부하고 다시 뽑으면 Wiener path에 bias가 들어갈 수 있다. Berkov–Gorn 논거가 성립하는 범위 내에서만 rejection을 허용할 것. 여기가 이 Task의 유일한 이론적 위험 지점.
4. **Torque 연속성.** RK45류 고차 스킴은 step 간 torque 연속성을 요구하는데 thermal field는 매 step 무작위로 바뀐다. 이 제약을 만족하는 스킴만 사용.

### 1.4 Definition of Done

- [ ] **Macrospin Arrhenius 검증**: 단축 이방성 macrospin의 thermal switching rate가 Brown 해석식과 일치. 온도 5점 이상 스윕, 상대오차 15% 이내.
  $$f = \gamma_{LL}\frac{\alpha}{1+\alpha^2}\sqrt{\frac{8K_{u1}^3 V}{2\pi M_s^2 k_BT}} e^{-K_{u1}V/k_BT}$$
- [ ] **Fluctuation–dissipation 검증**: 평형 상태에서 $\langle m_z \rangle$, $\langle m_z^2 \rangle$ 이 Boltzmann 분포와 일치.
- [ ] **$|\mathbf{m}|$ drift**: $10^6$ step 후 $||\mathbf{m}|-1| < 10^{-10}$.
- [ ] 고정 스텝 대비 speedup 실측값을 `benchmarks/`에 기록.
- [ ] 위 3개 검증을 **회귀 테스트로 CI에 등록**. Task 2 이후 모든 변경에서 상시 실행.

---

## Task 2 — Replica Batching

**상태**: ☐ 미착수
**의존**: Task 1
**성격**: Cross-cutting concern. **지금 하지 않으면 비용이 계속 오른다.**

### 2.1 근거

- MTJ free layer 규모(60 nm 직경 / 2 nm cell → $10^2$–$10^3$ cells)는 GPU 점유율이 처참하다. 커널 launch overhead가 계산보다 크다.
- 반면 유한온도 계산의 산출물은 항상 통계량(switching probability, WER, lifetime)이므로 **어차피 수천~수억 개의 독립 궤적이 필요하다.**
- 즉 "독립성을 병렬화에 쓴다"는 원래 직관은 시간축이 아니라 **여기서** 실현된다.
- 방법론적으로 AutoDock Vina의 **inter-ligand batching**과 동일한 패턴. VAMPIRE 개선안(P4)의 ② 항목과도 동일. 세 프로젝트가 같은 구조를 공유한다.

### 2.2 데이터 레이아웃 — 최우선 결정 사항

**모든 필드 배열에 replica 차원을 도입한다. 초기값 `R=1`.**

이것이 이 Task에서 가장 중요하고, 가장 되돌리기 어려운 결정이다. 커널이 늘어난 뒤 소급 적용하면 전면 개수가 된다.

```
m[R][nx][ny][nz][3]        // 자화
h_eff[R][nx][ny][nz][3]    // 유효장
h_therm[R][nx][ny][nz][3]  // (저장하지 않고 on-the-fly 생성 권장)
```

레이아웃 선택 근거:

- **`[R]` outermost 채택.** cuFFT batched mode가 batch stride(`idist`/`odist`)를 요구하므로 이 레이아웃이 가장 단순하다.
- Replica-innermost(`[...][3][R]`)가 pointwise 커널의 coalescing에는 유리하나, FFT마다 transpose가 필요해진다. **초기에는 outermost로 가고, 프로파일링 후 재검토.** 이 재검토 자체가 Task ④(v2)의 입력이 된다.

### 2.3 Demag 처리

- **Replica 간 demag kernel tensor는 완전히 동일하다** (동일 geometry 가정). 한 번 계산해 공유.
- Batched FFT → kernel 곱을 replica 축으로 broadcast → batched inverse FFT.
- Demag 백엔드를 **교체 가능한 인터페이스**로 유지할 것 (`demag/fft`, `demag/dense`, `demag/direct`). 이 인터페이스가 Task ④(v2)를 받아들이는 접점이다. 지금은 `fft` 하나만 구현하면 되지만 **인터페이스는 지금 파둔다.**

### 2.4 Retire / Refill — Task 3이 요구하는 필수 요건

**⚠️ 고정 길이 배치로 설계하지 말 것.**

단순 앙상블 배칭은 모든 replica가 같은 스텝 수를 돈다. 그러나 Task 3(FFS)의 trial은 **길이가 제각각이다** — 어떤 것은 다음 interface에 즉시 닿고, 어떤 것은 A basin으로 되돌아가 조기 종료된다. Task 3을 붙일 때 배칭 계층을 다시 쓰지 않으려면 처음부터 비동기 은퇴/재충전을 전제해야 한다.

Replica별로 유지할 상태:

```
active_mask[R]       // 활성 여부
step_counter[R]      // 개별 스텝 수 (adaptive dt이므로 물리 시각도 별도)
sim_time[R]          // 개별 물리 시각
dt[R]                // Task 1의 controller가 replica마다 독립
rng_stream_id[R]     // 전역 고유 ID
stop_reason[R]       // 종료 사유 코드 (미종료 / 시간초과 / 조건도달 / 발산)
```

필요한 연산:

- `retire(slot)` — 해당 slot 비활성화, 결과 회수
- `refill(slot, new_initial_state, new_rng_stream)` — 새 작업 주입
- 비활성 slot이 일정 비율(예: 25%)을 넘으면 compaction 또는 일괄 refill

이 스케줄링 문제는 Vina inter-ligand 배칭에서 리간드별 BFGS 수렴 스텝 수가 달라 발생했던 divergence 문제와 **구조적으로 동일하다.** 그 경험을 재사용할 것. 그리고 이 스케줄링 자체가 Task 3 논문의 기술적 핵심이 된다.

### 2.5 RNG 설계

- **Counter-based (Philox 4x32-10) 사용. 상태 저장 금지.**
- Key = `(global_replica_id, simulation_uid)`, Counter = `(step_index, cell_index, component)`.
- 요건:
  - Replica 간 스트림이 통계적으로 독립일 것
  - Refill 시 새 `global_replica_id`를 부여하여 재사용된 slot이 이전 스트림을 반복하지 않을 것
  - **완전 재현 가능할 것** — Task 3의 디버깅과 검증에 필수. 특정 trial을 ID만으로 단독 재현할 수 있어야 한다.

### 2.6 Definition of Done

- [ ] **회귀**: `R=1`일 때 Task 1 완료 시점의 모든 검증이 tolerance 내 통과.
- [ ] **Throughput 스윕**: `R ∈ {1, 8, 64, 256, 1024}`에 대해 trajectory·ns/s 곡선 측정. 소형계($N\sim10^3$)에서 `R=1` 대비 **최소 20× throughput**.
- [ ] **통계적 독립성**: replica 간 $m_z(t)$ 시계열의 교차상관이 노이즈 수준. 최소 $R=256$, lag 0–100.
- [ ] **Retire/refill 정확성**: 인위적으로 종료 조건을 흩뜨린 상태에서 refill을 반복해도 각 궤적의 통계가 단독 실행과 일치.
- [ ] **재현성**: 동일 seed 집합으로 두 번 실행 시 bitwise 동일 (동일 하드웨어 기준).
- [ ] VRAM 사용량 vs `R` 곡선 기록 (RTX 5060 Ti 8GB 상한 확인).

---

## Task 1/2 — 코드 근거 실행 계획 (2026-07-26 병합)

로드맵 확정 후 실제 코드베이스를 확인해 구체화한 실행 계획. **현 유한온도 구현 =
`src/heun_integrator_gpu.cu` + `include/micromag/heun_integrator_gpu.hpp`** (고정 dt,
Stratonovich Heun, `GPUMagState` 소유, cuRAND host 생성기, thermal=false일 때 CUDA Graph).

### 선결 — 문서 가정 vs 실제 (2건, 착수 전 수정 완료)

1. **스택**: `Adaptive SLLG time stepping/CLAUDE.md`는 Python+CuPy를 가정했으나 실제는
   **C++20 + CUDA + pybind11**. 새 적분기는 C++/CUDA 클래스 + `bind_gpu.cpp` 바인딩,
   빌드는 CMake preset(`windows-msvc-cuda`/`linux-gcc-cuda`), 테스트는 Catch2(`[gpu]`).
   (해당 CLAUDE.md 수정 완료.)
2. **RNG**: 현 코드는 **cuRAND host 생성기 + 노이즈 버퍼[3N] 저장**(상태 저장형) → 불변규칙 #5
   위반. **device-side Philox(`curandStatePhilox4_32_10_t`)로 전환**을 Task 1에서 선반영.
   step-doubling 오차추정이 full-dt와 two-half-dt에서 동일 Wiener 실현을 요구하고
   (ΔW_full=ΔW_½1+ΔW_½2, Brownian additivity), counter로만 결정론적 재생성 가능하기 때문.

### Task 1 — 파일 단위 계획

- **1-A 적분기 코어** — 신설 `src/depondt_integrator_gpu.cu` +
  `include/micromag/depondt_integrator_gpu.hpp`. Depondt–Mertens(Rodrigues 회전)
  predictor–corrector → `|m|=1` 정확 보존(후처리 renormalize 금지). 기존
  `HeunIntegratorGPU`는 **baseline(속도 비교·폴백)으로 보존**. RK45류 embedded 금지
  (thermal field가 매 step 무작위 → torque 연속성 위배).
- **1-B RNG(Philox)** — device kernel에서 on-the-fly, Key=(replica_id,sim_uid),
  Counter=(step,cell,component). 노이즈 버퍼 미저장.
- **1-C 오차추정+controller** — step-doubling(embedded 불가) + PI controller. `dt`는
  처음부터 **배열-호환 시그니처**(불변규칙 #2, Task 2에서 `dt[R]`). rejection 정책 기본
  **(b) 거부 없이 다음 step부터 dt 조정**(Wiener bias 없음); Leliaert 2017 확인 후 (a)
  Brownian-bridge 재사용으로 승급.
- **1-D thermal 재스케일 단일화** — σ(dt)=√(2α k_B T/(μ₀ Ms γ V dt)) 계산을 **단일 함수**로
  집중, dt 변경 경로가 모두 통과. dt×2 시 실효온도 불변 단위테스트 필수(§1.3-2/§5.1).
- **1-E 검증(→CI 회귀)** — Arrhenius(≤15%, 5점) · FDT(⟨m_z⟩·⟨m_z²⟩ Boltzmann, α-불변) ·
  |m|-drift(<1e-10 @1e6) · speedup 기록. Catch2 `[gpu][thermal]` + CI 상시.

### ⚠️ 발견 — 유한온도 σ 정규화 (Heun baseline 공통, 2026-07-26)

Task 1-A/B/C 구현 중 FDT 검증에서 드러난 **코드베이스 공통 이슈**. 반드시 먼저 해결해야
Arrhenius/FDT DoD가 성립한다.

- **증상**: 장(場)-결합 Langevin 검증에서 ξ=μ₀MsVH/k_BT=3(기대 ⟨mz⟩=L(3)=0.672)일 때
  **Heun·Depondt 모두 ⟨mz⟩=1.0000** — 장 안에서 열교란이 ~1/μ₀(≈10⁶배 분산) 약함.
- **기존 테스트가 놓친 이유**: `tests/test_thermal.cpp`의 등분배 테스트는 **장 없는 자유 스핀**의
  구면 균일(1/3)만 검사(주석도 "유한 T FDT는 비현실적"이라 회피). 장-결합 정규화는 미검증.
- **원인(유도)**: Brown σ_B(Tesla)=√(2α k_BT/(γ Ms V dt)). 코드가 H(A/m)에 더하므로
  H_th=B_th/μ₀ → **σ_H = √(2α k_BT/(μ₀² Ms γ₀ V dt))**. 현 코드(`heun_integrator_gpu.cu`,
  `depondt_integrator_gpu.cu::therm_sigma`)는 분모가 `μ₀ Ms γ V dt`로 **μ₀ 하나 누락** →
  σ가 √μ₀≈892배 작음.
- **영향**: Heun 유한온도 전반, `test_thermal.cpp` 등분배 테스트(극단 T=1e8/dt=1e-10가 이
  작은 σ에 맞춰 튜닝됨 — 수정 시 재설계 필요), **논문 NB30 STT switching** 등 유한온도 결과.
- **해결 완료 (2026-07-26, 통일된 단일 수정)**:
  - **두 적분기 공통 = 표준 Brown식 + μ₀ 단위수정**:
    **σ = √(2α k_BT/(μ₀² Ms γ₀ V dt))**. 유일한 변경은 분모 **μ₀ → μ₀²**(A/m 단위:
    B=μ₀H → H_th=B_th/μ₀에서 분산에 1/μ₀²). `heun_integrator_gpu.cu`(2곳)·`thermal_field.cpp`·
    `depondt_integrator_gpu.cu` 모두 적용.
  - **검증**: 장-결합 Langevin ⟨m_z⟩=L(ξ)를 ξ=1,3,6에서 일치. Heun CPU 0.672, Depondt GPU
    0.673 (기대 0.6716). 회귀: `[thermal]` "Langevin <mz> in a field (absolute sigma)"(CPU,
    클라우드 CI 실행) + `[depondt][gpu]` "FDT: Langevin <mz> in a field"(GPU).
  - **한때 '스킴별 factor-2' 로 보였던 것은 Depondt의 d_H 스트림 레이스**였다(필드 커널을
    integrator 스트림에 안 묶어 비결정론). `demag.set_stream/extra_fields.set_stream`로 수정하니
    같은 seed에서 결정론 회복 + 표준 2α로 두 적분기 일치. → 스킴 의존성 없음.
  - **등분배 테스트 재설계(b)**: 새 σ(892배 큼)로 Néel-Brown 자유확산·"anisotropy+noise"가
    깨져 **물리적 온도(100~1000K, T=5K 강confine)**로 재설계. 등분배(K=0,극단T)는 거대회전이
    여전히 등방→1/3이라 통과(불변). CPU 236 / GPU 125 케이스 전부 green.
- **미해결 (후속)**:
  - **논문 NB30(STT switching)은 수정 전 Heun 사용** → 열교란이 ~1/μ₀ 약했음. **NB30 P_sw
    곡선 전면 재계산·재검증 필요**(열-assist가 사실상 빠진 상태였을 가능성). = 다음 작업(3).

### Task 2 — 파일 단위 계획 (의존: Task 1)

- **2-A** GPU 필드에 선행 replica 차원, **R outermost** (`m[R][3N]`), 초기 R=1.
  `GPUMagState`/커널 시그니처를 R 차원 수용형으로. cuFFT batched(idist/odist).
- **2-B** demag kernel tensor replica 공유(동일 geometry), broadcast 곱. demag 백엔드
  **교체 인터페이스**(`fft`/`dense`/`direct`) 지금 파둠(`fft`만 구현, v2 접점).
- **2-C** retire/refill: per-replica `active_mask·step_counter·sim_time·dt·rng_stream_id·
  stop_reason`, 비활성≥25% compaction, refill 시 새 global_replica_id(§2.4, Task 3 요건 선반영).
- **2-D** 검증: R=1 회귀 · throughput R∈{1,8,64,256,1024} ≥20×(N~1e3) · 교차상관 노이즈수준 ·
  retire/refill 정확성 · bitwise 재현 · VRAM vs R(로컬 8GB 상한).

#### Task 2 — Phase 진행 현황 (2026-07-27)

- ✅ **Phase 2.0** `BatchedMacrospinGPU`: R=1 bitwise 회귀, STT/FDT 검증, 158× throughput.
- ✅ **Phase 2.1** `BatchedLLGGPU`: N-셀 exchange+uniaxial+Zeeman 배칭, R=1 vs Depondt 5.9e-15, 59.8×.
- ✅ **Phase 2.4** retire/refill: per-replica active/reason/step_counter/rng_stream_id,
  은퇴·재충전(새 Philox 스트림), Néel-Brown switch-time 분포. front-compaction은 후속 최적화.
- ✅ **Phase 2.2** batched demag(`BatchedDemagGPU`, cuFFT batch=3R, 공유 Newell 커널):
  R=1 vs DemagFieldGPU 1.3e-10, 64 replica 55.4×. `BatchedLLGGPU.enable_demag()`로
  exchange+uniaxial+Zeeman+demag+STT+thermal 물리적 완비.
- ✅ **Phase 2.3** — Philox replica 스트림 2.1/2.4/2.2에 반영 완료.
- ✅ **NB30 배칭 재작성** `30_thermal_stt_batched_gpu.py`: Part B 0.56s(vs ~2.5h), 전이 0.97.
- ✅ **Phase 2.5** API 문서화(USER_GUIDE §8.10) + 물리완비 MTJ 데모 `31_mtj_batched_switching_gpu.py`:
  공간분해 CoFeB PMA 자유층 P_sw(J), 1248 replica; 다중셀 전이 1.30 vs coherent 1.10(own Jc0).
- ⏳ front-compaction(은퇴>25% warp효율) — 후속 최적화(정확성 경로 완성).

#### Task 2 — Phase 실행 순서 (2026-07-27 구체화)

동기: NB30 단일-셀 매크로스핀이 **GPU 26%**(launch/Python 오버헤드 병목)로 실측됨 → 유한온도
통계(수천 궤적)를 **replica 차원 R**로 한 step()에 병렬 전진. 이식 대상은 `DepondtMertensGPU`
(Task 1에서 batch-ready 시그니처 + device Philox 보유).

- **Phase 2.0 — 스캐폴딩(R=1부터, MVP)** ← *최우선, 되돌리기 어려운 레이아웃 결정*
  - 첫 배칭 대상은 **단일-셀 매크로스핀**(N=1, NB30/MTJ 워크로드) — GPUMagState/필드 클래스
    깊은 리팩터 없이 자립형 `BatchedMacrospinGPU`로 R-차원·throughput·retire-ready 구조를 먼저 증명.
  - 레이아웃 **replica-outermost** `m[r*3 + c]`(N=1) → 일반화 시 `m[r*3N + c*N + i]`.
  - Depondt 회전(omega/rotate/avg) + inline 단축이방성·Zeeman·Slonczewski STT + **batched Philox**
    (key=global_replica_id). `step_all()` 1회 = R replica 전진. Python `upload/download [R,3]`, `mz[R]`.
  - **불변**: R=1이 `DepondtMertensGPU` 매크로스핀 결과를 재현.
- **Phase 2.1 — 로컬 필드 배칭**: Exchange/Uniaxial/Zeeman/ZeroDemag 커널을 `R·N` 스레드로(자명).
  다중-셀(N>1)로 일반화, `GPUMagState` 버퍼 `[R·3N]` 승격.
- **Phase 2.2 — batched demag(핵심 난이도)**: cuFFT batch=R, kernel tensor 1회 공유·broadcast.
  `IDemagGPU` 배칭 변형(`fft` 우선).
- **Phase 2.3 — Philox replica 확장**: subsequence=global_replica_id·N+cell, refill 시 새 id.
- **Phase 2.4 — retire/refill**: per-replica 상태배열 + compaction(§2.4, Task 3 선반영).
- **Phase 2.5 — Python API + throughput**: `mm.BatchedDepondtGPU`, NB30 배칭 재작성 → R 스윕 실측.

---

## Task 3 — `rare_event/` 모듈

**상태**: ☐ 미착수
**의존**: Task 2 완료 (**엄격**)
**성격**: Orchestration layer. 저장소 안의 독립 모듈로 시작.

### 3.1 근거 — brute force를 개선하지 말고 버린다

Switching 영역은 어떤 병렬화로도 직접 적분으로는 도달할 수 없다. STT-MRAM에서 요구되는 $10^{-9}$ 수준의 WER를 stochastic LLGS 직접 적분으로 구하려면 $10^9$개를 훨씬 넘는 독립 궤적이 필요하다. 1000× 병렬화로도 못 메운다 — 지수적 문제에 다항적 하드웨어를 붓는 것이다.

대신 rare event method는 이미 확립되어 있고, **결정적으로 GPU 배칭과 궁합이 최고다.** 각 interface에서 발사하는 trial trajectory는 완전히 독립이며 길이가 짧다.

### 3.2 배치

```
Claude-SpinDynamics/
├── CLAUDE.md                    # 코어용
├── src/ ...
└── rare_event/
    ├── CLAUDE.md                # 자체 컨텍스트
    ├── backend/
    │   ├── interface.py|.h      # 백엔드 계약 (아래 §3.3)
    │   └── claude_sd.py         # 첫 백엔드
    ├── ffs/
    ├── neb/
    └── tests/                   # 코어 테스트에 의존하지 않음
```

**초기에 별도 저장소로 만들지 않는다.** 백엔드가 하나뿐일 때 추상화하면 거의 확실히 잘못 설계된다. 둘일 때 추출하면 맞는다.

### 3.3 백엔드 계약 (backend interface)

솔버에 요구하는 것은 좁다. Demag을 FFT로 하든 BEM으로 하든 무관하다.

```
class SpinBackend:
    def n_replicas() -> int
    def load_state(slots, states)        # per-replica 상태 주입
    def save_state(slots) -> states      # per-replica 상태 회수
    def set_rng_stream(slots, ids)
    def advance(n_steps)                 # 배치 전체 전진
    def order_parameter(slots) -> array  # λ(m), FFS interface 판정용
    def active_mask() -> array
    def retire(slots)
    def capabilities() -> dict           # 지원 기능 질의
```

**설계 원칙**: 이 인터페이스에 Claude-SD 전용 타입이 단 하나도 들어가지 않게 한다. 배열은 numpy/표준 버퍼, 상태는 OVF 또는 그에 준하는 중립 포맷. 이것이 §3.5 분리 트리거 2번의 판정 기준이다.

### 3.4 개발 단계

**Phase 1 — MEP 확보**
- [ ] GNEB (geodesic NEB) 구현. 초기·최종 상태를 잇는 minimum energy path.
- [ ] MEP를 따라 order parameter $\lambda$ 및 interface 집합 $\{\lambda_0, ..., \lambda_n\}$ 정의.
- [ ] 검증: macrospin에서 해석적 saddle과 일치.

**Phase 2 — FFS 구현 + 해석해 대조**
- [ ] Rosenbluth-style FFS. 각 interface에서 배치로 trial 발사, retire/refill로 채움.
- [ ] Interface 간 조건부 확률 $P(\lambda_{i+1}|\lambda_i)$ 및 초기 flux 추정.
- [ ] **검증**: macrospin lifetime이 Brown 해석식과 일치. 이것이 이 Phase의 유일한 성공 판정 기준.
- [ ] 직접 적분이 가능한 낮은 장벽 영역($\Delta E \sim 10\,k_BT$)에서 brute force와 교차 검증.

**Phase 3 — 다중 셀 계로 확장**
- [ ] Incoherent reversal이 일어나는 실제 micromagnetic 계.
- [ ] Interface 정의가 macrospin에서처럼 자명하지 않음 — order parameter 선택이 결과에 미치는 영향을 감도 분석할 것. FFS의 알려진 약점.
- [ ] 목표 산출물: 장벽 $\Delta E$, attempt frequency $\tau_0$, WER vs pulse 곡선.

**Phase 4 — mumax3 백엔드** (§3.5 트리거 1)
- [ ] mumax3의 `RunWhile()` + OVF `save`/`load` + `RandSeed()`로 백엔드 구현.
- [ ] 위치: **느리지만 신뢰받는 검증 백엔드.** 프로덕션용이 아니다. 프로세스 launch overhead 때문에 짧은 FFS trial에는 부적합하나, 검증에는 충분하다.
- [ ] 목적: **자체 솔버의 신뢰도 리스크를 방법론에서 분리하는 것.** 이게 없으면 리뷰어가 Claude-SD를 의심하는 순간 rare event 결과 전체가 같이 무너진다.

### 3.5 독립 저장소 분리 트리거

**아래 4개를 모두(AND) 충족하는 시점에 분리한다.**

| # | 조건 | 판정 방법 |
|---|---|---|
| 1 | **두 번째 백엔드 검증 완료** | mumax3 백엔드로 동일 계의 FFS lifetime을 계산했을 때 Claude-SD 백엔드 결과와 신뢰구간이 겹침 |
| 2 | **의존성 역전 완료** | `rare_event/backend/interface`에 Claude-SD 전용 타입·헤더·import가 0개 |
| 3 | **테스트 독립** | `rare_event/tests/`가 코어 테스트 스위트 없이 단독 통과 |
| 4 | **발표 정체성 확보** | 별도 논문의 outline(제목·타깃 저널·figure 목록)이 문서로 존재 |

→ 충족 시: `spin-rare-event` (가칭)로 분리, `PROJECTS.md`에 **P9 신설**.
→ 그 전까지: **P5 하위 항목으로만 관리. 별도 트래커 항목을 만들지 않는다.**

**Anti-trigger — 아래 중 하나라도 해당하면 분리하지 않는다:**

- 백엔드가 아직 Claude-SD 하나뿐 (조기 추상화 위험)
- 백엔드 인터페이스가 최근 3회 이상 breaking change 중 (설계 미확정)
- Task 2가 미완 (구동 대상 부재)
- 단순히 "정리되어 보여서" — 이건 이유가 아니다

### 3.6 `rare_event/CLAUDE.md`에 넣을 문안

```markdown
# rare_event 모듈

## 범위
Claude-SD 코어 위에 올라타는 rare event sampling 계층.
FFS, GNEB, importance splitting. 솔버 내부 구현에 의존하지 않는다.

## 불변 규칙
- `backend/interface`에 Claude-SD 전용 타입을 절대 도입하지 말 것.
  솔버별 코드는 `backend/<name>.py`에만 존재한다.
- 이 모듈의 테스트는 코어 테스트에 의존하지 않는다. 단독 실행 가능해야 한다.
- 모든 확률 추정치는 통계적 불확실도와 함께 보고한다. 점추정만 내지 말 것.
- 검증 기준은 항상 해석해(Brown 공식) 또는 brute force 대조다.
  "그럴듯한 값"은 통과 기준이 아니다.

## 분리 트리거
`../CLAUDE-SD_FINITE_TEMP_ROADMAP.md` §3.5 참조.
4개 조건을 모두 충족하기 전에는 별도 저장소로 분리하지 않는다.
```

---

## Task 4 — 범위 밖 (v2로 이관)

**FFT ↔ dense GEMM crossover 및 mixed precision demag은 본 저장소의 작업 항목이 아니다.**

이관 대상: **v2 연구주제 "Hardware- and accuracy-aware FFT auto-tuning framework"** 의 application case.

### 4.1 이관 사유

이 항목의 본질은 micromagnetics 기능이 아니라 **하드웨어·정밀도 자동 튜닝 프레임워크의 적용 사례**다. 여기에 micromagnetics 물리를 직접 박아넣으면 v2 프레임워크가 범용성을 잃는다.

### 4.2 Claude-SD가 제공해야 할 것 (이것만)

- [ ] Demag 백엔드를 **교체 가능한 인터페이스**로 유지 (§2.3). `fft` / `dense-gemm` / `direct` 를 런타임에 선택 가능하게.
- [ ] 정밀도를 백엔드 파라미터로 노출 (fp32 / tf32 / bf16).
- [ ] Task 1의 Arrhenius 회귀 테스트를 **정밀도 감도 측정 도구로 재사용 가능하게** 유지.

### 4.3 v2로 넘길 핵심 아이디어 — 오차 예산의 재정의

v2의 error budget 설계에 다음을 반영할 것:

> **유한온도 micromagnetics에서 오차 예산은 `dm/dt` 잔차가 아니라 $\Delta E_{\rm barrier} / k_BT$ 단위로 정의해야 한다.**

근거:

- Demag의 저정밀 계산은 결정론적 계산에서는 단순 오차지만, 유한온도에서는 **systematic bias**로 작용해 에너지 장벽 $\Delta E$ 를 이동시킨다.
- Switching rate는 $\propto e^{-\Delta E/k_BT}$ 이므로 **장벽 오차가 지수적으로 증폭된다.**
- 정량: retention 10년 기준 $\Delta E \sim 60\,k_BT$. 여기서 **1% 장벽 오차 = rate 1.8× 오차.**
- 결론: 정밀도 저하가 결정론적 계산보다 유한온도 계산에서 **훨씬 위험하다.** 두 경우에 같은 오차 허용치를 쓰면 안 된다.

이 기준은 v2 프레임워크에서 **교체 가능한 error metric 플러그인의 사례 하나**로 넣는다. 프레임워크 코어에 넣지 않는다.

### 4.3 참고 — 왜 crossover가 존재하는가 (v2 입력용 메모)

- Demag 연산은 $3N \times 3N$ dense matrix–vector. $R$개 replica를 모으면 $(3N \times 3N)\cdot(3N \times R)$ tall-skinny GEMM이 된다.
- $R \gtrsim 100$ 이면 행렬 read가 amortize되어 **memory-bound FFT가 compute-bound GEMM으로 전환**되고 tensor core 사용이 가능해진다.
- $N \sim 10^3$ 에서 행렬은 약 38 MB (fp32) — L2에 거의 들어간다. bf16이면 19 MB.
- **단 $O(N^2)$ 이므로 $N$이 커지면 즉시 무너진다. Crossover 지점 측정 자체가 결과물이다.**

---

## 5. 실행 순서

```
Task 1 (adaptive stepping)
   ↓  [DoD 통과 + CI 등록]
Task 2 (replica batching, retire/refill 포함)
   ↓  [DoD 통과]
Task 3 Phase 1–3 (rare_event/, Claude-SD 백엔드)
   ↓
Task 3 Phase 4 (mumax3 백엔드)
   ↓  [§3.5 트리거 4개 충족]
분리 → P9 신설
```

**엄격한 제약:**

- **Task 3은 Task 2 완료 전 착수 금지.** 구동할 대상이 없다.
- **Task 2의 retire/refill 설계는 Task 3의 요구사항을 반영해야 한다** (§2.4). 순서는 2→3이지만 **설계는 함께** 한다. 이걸 어기면 배칭 계층을 두 번 쓰게 된다.
- Task 1의 검증 3종은 이후 모든 변경에서 상시 실행. 유한온도 코드는 조용히 틀리는 게 기본값이다.

---

## 6. `PROJECTS.md` P5 섹션 반영 문안 (복붙용)

```markdown
### P5. Claude-SpinDynamics

- **저장소**: github.com/mirryou-maker/Claude-SpinDynamics
- **P4와의 관계**: OVF 브리지로 연결
- **로드맵 문서**: `CLAUDE-SD_FINITE_TEMP_ROADMAP.md`
- **현재 Phase**: 유한온도 가속 (Task 1)
- **다음 액션**:
  - [ ] Task 1 — Adaptive SLLG stepping (Berkov–Gorn / Depondt–Mertens)
  - [ ] Task 1 — Macrospin Arrhenius 회귀 테스트 CI 등록
  - [ ] Task 2 — 필드 배열에 replica 차원 도입 (R=1부터)
  - [ ] Task 2 — retire/refill 큐 설계 (Task 3 요구사항 반영)
  - [ ] Task 2 — demag 백엔드 인터페이스 분리 (v2 접점)
  - [ ] Task 3 — `rare_event/` 스켈레톤 + 자체 CLAUDE.md
- **범위 제외**:
  - 시간축 병렬화(Parareal/PFASST) — switching 근처 붕괴로 기각
  - FEM-SD 반영 — FEM은 유한온도에서 이점 없음, AMR과 배칭 충돌
  - FFT↔GEMM crossover — v2(FFT auto-tuning) application case로 이관
- **블로커**: 없음
- **최근 로그**
  - `2026-07-26` — 유한온도 가속 로드맵 확정. Task 1→2→3 순서 및 rare_event 분리 트리거 정의.
```

---

## 6b. 장기 계획 — 다중 벤더 GPU 지원 (Intel / AMD, 2026-07-27 조사)

**목표**: NVIDIA(CUDA) 외에 AMD(ROCm)·Intel(oneAPI/Level-Zero) GPU에서도 GPU 백엔드를
돌린다. 현재 GPU 코드가 이식에 얼마나 유리한지 실측 조사한 결과를 근거로 단계 계획을 둔다.

### 현 CUDA 결합도 (실측, 21개 `.cu`, 92 커널 런치)
- **이식을 막는 난해 구조가 없다** (가장 중요): warp shuffle/ballot **0건**, texture memory **0건**.
  `__shared__` 3파일(단순 block-reduction), `atomicAdd` 2파일 — 전부 HIP/SYCL/OpenCL로 자명 이식.
- CUDA 런타임 API ~38종(`cudaMalloc/Memcpy/Stream/Event/Graph`) — 전부 HIP에 1:1 대응(`hipMalloc`…).
- **cuFFT** 13심볼(표준 `PlanMany`/`ExecD2Z`) — 벤더 FFT 필요: rocFFT(AMD)·oneMKL DFT(Intel)·
  **VkFFT**(범용). VkFFT는 이미 부분 통합돼 있음(단 CUDA-Graph와 비호환 — 배칭 demag 경로는 graph
  미사용이라 무방).
- **cuRAND** 4파일: host generator + device Philox(`curandStatePhilox`). 배칭 엔진의 핵심 —
  rocRAND·oneMKL RNG 모두 Philox 지원.
- `GReal`(gpu_real.hpp)이 이미 float/double + cufft 타입을 매크로로 추상화 — **백엔드 seam으로 재사용 가능**.
- CUDA Graphs(`cudaGraph*`) 사용처 있음 — HIP `hipGraph` 대응, SYCL은 command-graph(신규). 배칭
  엔진은 graph 불필요.

### 이식 경로 3안 (커버리지 × 노력)
| 경로 | 커버 | 노력 | 비고 |
|---|---|---|---|
| **HIP (ROCm)** | AMD (+NVIDIA via HIP) | **낮음** | `hipify`로 기계적 변환; `__global__`/`<<<>>>` 유지. rocFFT+rocRAND. Intel 미포함 |
| **SYCL (DPC++/AdaptiveCpp)** | **Intel+AMD+NVIDIA** | 중~높음 | 커널을 `parallel_for` 람다로 재작성(단, 난해 intrinsic 없어 기계적). oneMKL DFT/RNG. 단일 소스로 3벤더 |
| **Vulkan+VkFFT** | 범용(모바일 포함) | 높음 | 커널을 GLSL/SPIR-V 컴퓨트 셰이더로 전면 재작성. 최고 이식성/최고 비용 |

### 권장 단계 계획
- **선결 — 백엔드 seam 도입** (벤더 무관, 지금 해도 이득): `cudaMalloc/Memcpy/Stream/Launch`를
  얇은 `gpu_backend.hpp` 래퍼로 감싸고 커널 런치를 매크로화. `GReal`처럼 컴파일타임 스위치.
  현재도 유지보수성↑, 이후 어느 경로든 착지점이 된다.
- **Phase G-AMD (HIP)**: `hipify-perl`로 `.cu`→`.hip` 변환, `find_package(hip/rocfft/rocrand)` CMake
  프리셋 `linux-hip`. NVIDIA는 HIP-over-CUDA로 회귀 유지. 검증: 기존 GPU 테스트 118종 + 배칭 R=1 회귀.
- **Phase G-INTEL/범용 (SYCL)**: 커널을 SYCL로 재작성(seam 위에서). **AdaptiveCpp**(오픈소스, CUDA/HIP/
  OpenMP/Level-Zero 백엔드)로 단일 소스 3벤더. FFT는 oneMKL DFT 또는 VkFFT. Intel Arc/Data Center
  Max + NVIDIA/AMD 동시 커버.
- **FFT 공통화**: 벤더 FFT 3종을 `IDemagFFT` 인터페이스로 추상화(이미 `IDemagGPU` 존재). VkFFT를
  범용 폴백으로.

### 제약·미확인 (사용자 보고 필요)
- **로컬 하드웨어 부재**: 개발 PC는 NVIDIA(RTX 5060 Ti)뿐, iREMB도 P100/V100(NVIDIA)뿐 →
  **AMD/Intel GPU 실기 검증 수단이 현재 없음**. CI는 벤더 GPU 러너 또는 클라우드(AMD MI-계열,
  Intel Max) 필요 — 전역 규칙상 **클라우드는 사전 요청 대상**.
- SYCL 재작성은 커널 92개 런치 전면 수정 — 기능 회귀 위험. 백엔드 seam + R=1 bitwise 회귀로 방어.
- **우선순위 판단**: 실사용 타깃 하드웨어가 정해지면(예: iREMB 차기 도입 GPU, 특정 클러스터) 그 벤더
  경로부터. 미정이면 seam 도입까지만 진행하고 실기 확보 시 착수 권장.

---

## 7. 참고문헌

**Adaptive stepping (Task 1)**
- D. Berkov, N. Gorn, *J. Phys.: Condens. Matter* **14**, L281 (2002) — drift term이 $|\mathbf{m}|$ 에만 영향
- J. Leliaert et al., *AIP Advances* **7**, 125010 (2017) — mumax3 adaptive SLLG, 20× speedup
- A. Vansteenkiste et al., *AIP Advances* **4**, 107133 (2014) — mumax3 설계·검증
- P. Depondt, F. G. Mertens, *J. Phys.: Condens. Matter* **21**, 336005 (2009) — rotation 기반 적분기
- W. F. Brown, *J. Appl. Phys.* **34**, 1319 (1963) — thermal switching rate 해석식

**Rare event (Task 3)**
- C. Vogler, F. Bruckner, B. Bergmair, T. Huber, D. Suess, C. Dellago, *Phys. Rev. B* **88**, 134409 (2013) — FFS + NEB for micromagnetics
- L. Desplat, C. Vogler, J.-V. Kim, R. L. Stamps, D. Suess, *Phys. Rev. B* **101**, 060403 (2020) — skyrmion lifetime, FFS vs Kramers
- arXiv:1412.5057 — thermal stability + attempt frequency, free-parameter 없이
- arXiv:1603.08512 — STT-RAM WER, importance splitting ($10^{-9}$ WER에 $10^9$+ 궤적 필요)

**기각한 방향 (참고용)**
- R. Kraft, S. Koraltan, M. Gattringer, F. Bruckner, D. Suess, C. Abert, arXiv:2310.11819 / *JMMM* (2024) — PFASST for deterministic LLG
- K. Pentland, M. Tamborrino, D. Samaddar, L. C. Appel, arXiv:2106.10139 — Stochastic Parareal
- I. Bossuyt, S. Vandewalle, G. Samaey, *SIAM J. Sci. Comput.* (2024) — micro-macro Parareal, bimodal 처리의 어려움
