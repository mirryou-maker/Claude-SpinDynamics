# mumax3 vs mumax+ 기술 비교 분석

**작성일:** 2026-06-21  
**목적:** Claude-SpinDynamics 벤치마크 비교 대상 선정을 위한 사전 분석

---

## 1. 개발 현황

| 항목 | mumax3 | mumax+ |
|------|--------|--------|
| 최신 버전 | 3.11.1 (2023) | v1.2.1 (2026-06) |
| 유지 상태 | 유지보수 모드 (주요 개발 중단) | 활발히 개발 중 |
| 주요 언어 | Go + CUDA C | C++23 + CUDA + Python |
| 개발팀 | Ghent University DyNaMat | Ghent University DyNaMat (동일팀) |
| 라이선스 | GPLv3 | Open source |
| 논문 | AIP Advances 4, 107133 (2014) | npj Comput. Mater. (2026-02) |
| GitHub 활동 | 최소 (버그 수정 위주) | 활발 (1,564 commits, 지속 PR) |

mumax+는 mumax3의 **공식 후속작**으로, 동일 개발팀이 설계 한계를 극복하기 위해 처음부터 재작성한 코드입니다.

---

## 2. 기술 아키텍처

### FFT 백엔드

| | mumax3 | mumax+ |
|--|--------|--------|
| 라이브러리 | NVIDIA cuFFT | NVIDIA cuFFT |
| VkFFT | 없음 | 없음 |
| 최적 그리드 크기 | 7-smooth (2,3,5,7의 배수) | 동일 (cuFFT 제약 그대로) |
| 비최적 크기 성능 | ~10× 저하 | 동일 제약 |

**Claude-SD와 비교**: Claude-SD는 cuFFT와 VkFFT(혼합기저)를 모두 지원합니다. 비최적 그리드(예: 400×20, 소인수가 크면)에서 Claude-SD VkFFT 빌드가 mumax3/mumax+ 대비 유리합니다.

### 정밀도

| | mumax3 | mumax+ |
|--|--------|--------|
| 단정밀도 (f32) | 기본 | 지원 |
| 배정밀도 (f64) | 지원 | 지원 |
| 혼합 정밀도 | 없음 | 없음 |

### 다중 GPU

- **mumax3**: 단일 GPU만
- **mumax+**: 공식 다중 GPU 지원 미문서화 (사실상 단일 GPU)
- **Claude-SD**: 단일 GPU (현재)

---

## 3. 물리 범위

### 공통 지원

- LLG 동역학 (Gilbert 감쇠)
- 교환 상호작용 (Heisenberg)
- 정자기장 (demag tensor, FFT 기반)
- 일축/입방 자기이방성
- 균일/공간변조 Zeeman 필드
- DMI (Dzyaloshinskii-Moriya)
- 스핀전달토크 (Zhang-Li CIP-STT, Slonczewski CPP-STT)
- 열적 요동 (확률적 LLG, Stratonovich Heun)
- 주기적 경계조건

### mumax3의 한계

- **반강자성(AFM), 훼리자성(FiM) 미지원** — 단일 격자 강자성 전용
- SOT: Slonczewski 공식 우회 사용 (native SOT API 없음)
- DMI: 스칼라 형태만 (텐서 일반화 불가)
- 공간적으로 다른 재료: 이산 region 정의 방식 (제약 있음)

### mumax+의 추가 기능

| 기능 | 설명 |
|------|------|
| **반강자성/훼리자성** | 다중 자기 격자, AFM/FiM 완전 지원 |
| **완전 DMI 텐서** | 9개 독립 성분 (mumax3의 스칼라 대비 일반적) |
| **자기탄성 결합** | 탄성동역학 방정식 + 변형률 구동 효과 (입방 결정) |
| **고급 적분기** | RK12(Heun), RK23, Cash-Karp, RKF45, RK45 적응형 |
| **Python 우선 API** | NumPy 배열로 직접 파라미터 설정, Colab 지원 |
| **다중 자석 인스턴스** | 한 시뮬레이션 내 여러 독립 자석 객체 공존 |

---

## 4. 성능 특성

### 알려진 성능 지표

**mumax3:**
- CPU 대비 ~100× 속도향상 (게이밍 GPU 기준)
- SP#4 µMAG 1 ns: ~5 초 (RTX 5060 Ti 측정, 이번 NB41 결과)
- 최적화된 고정 dt Heun 또는 RK4 루프

**mumax+:**
- 공식 mumax3 대비 벤치마크 **미발표** (논문에서도 명시적 성능 비교 없음)
- 논문 설명: *"versatility and extensibility, combined with user-friendliness, in favor of performance"* — 성능보다 유연성·사용성 우선을 명시
- Python 오버헤드: 각 스텝마다 Python 콜백 없이 CUDA 루프로 동작하지만, 설정 단계의 Python 오버헤드 존재

### Claude-SpinDynamics 대비 예상

| 비교 | 예상 |
|------|------|
| Claude-SD cuFFT_f64 vs mumax3 | NB41 결과: CS 1.32× 빠름 (SP#4 기준) |
| Claude-SD vs mumax+ | 미측정; mumax3보다 느리거나 유사할 가능성 |
| VkFFT 장점 | 비최적 그리드에서 Claude-SD VkFFT가 두 mumax 모두에 우세 예상 |

---

## 5. API 및 사용성

### mumax3 스크립트 방식

```
// mumax3 방식 (도메인 특화 언어)
SetGridSize(200, 50, 1)
SetCellSize(2.5e-9, 2.5e-9, 3e-9)
Msat = 860e3
Jc := 1e12
run(1e-9)
```
- 커스텀 DSL (Go 기반 인터프리터)
- 복잡한 파라미터 스윕에 제약
- Python 생태계와 통합 어려움

### mumax+ Python 방식

```python
# mumax+ 방식 (Python-first)
import mumax

m = mumax.Magnet(nx=200, ny=50, nz=1, dx=2.5e-9, dy=2.5e-9, dz=3e-9)
m.msat = 860e3
m.aex  = 13e-12
m.alpha = 0.02
m.minimize()
m.run(1e-9)
mx = m.m.numpy()[..., 0]  # NumPy 배열로 직접 접근
```
- NumPy/SciPy 직접 연동
- 공간 변화 파라미터를 배열 인덱싱으로 설정
- Google Colab 지원, pip install 가능
- Sphinx 문서화

---

## 6. Claude-SpinDynamics 벤치마크 포함 권장사항

### 추가해야 하는 이유

1. **물리 검증 강화**: mumax+는 AFM/FiM을 지원하므로, 향후 CS가 반강자성 기능을 추가할 때 참조 기준이 됨
2. **DMI 텐서 검증**: 현재 CS의 InterfacialDMIFieldGPU 정확도를 mumax+의 완전 DMI 텐서와 비교 가능
3. **최신 참조**: mumax+는 2026년 논문으로 현재 분야의 기준점
4. **Python 통합 용이**: mumax+가 Python API를 제공하므로 bench_utils.py에서 직접 호출 가능

### 구현 방법

mumax+는 pip 설치 가능:
```bash
pip install mumax   # GPU 빌드 (CUDA 12+ 필요)
```

`bench_utils.py`에 mumax+ 실행자 추가:
```python
def run_mumaxplus(scenario_fn, timeout_s=300):
    """scenario_fn: mumax+ Python 코드를 실행하는 callable"""
    try:
        import mumax
        t0 = time.perf_counter()
        result = scenario_fn(mumax)
        wall_ms = (time.perf_counter() - t0) * 1e3
        return {"ok": True, "wall_ms": wall_ms, **result}
    except Exception as e:
        return {"ok": False, "error": str(e)}
```

### 예상 제약사항

- **설치 복잡도**: CUDA 12+, GPU cc≥8.0 필요 (RTX 5060 Ti는 cc12.0이므로 호환)
- **성능 벤치마크 없음**: mumax+의 속도는 직접 측정해야 함 (문서 부재)
- **스크립트 재작성**: mx3 파일을 mumax+ Python으로 변환 필요 (STT/SOT API가 다름)
- **버전 안정성**: v1.2.1이지만 여전히 활발히 변경 중 — API 변경 가능성

---

## 7. 요약

| 기준 | mumax3 | mumax+ | Claude-SD |
|------|--------|--------|-----------|
| 개발 상태 | 유지보수 | 활발 | 활발 |
| FFT | cuFFT | cuFFT | cuFFT + VkFFT |
| 정밀도 | f32/f64 | f32/f64 | f32/f64 (별도 빌드) |
| 강자성 | ✓ | ✓ | ✓ |
| AFM/FiM | ✗ | ✓ | ✗ |
| SOT (native) | ✗ | ✓ | ✓ |
| DMI 텐서 | 스칼라 | 완전 | Interfacial |
| 자기탄성 | ✗ | ✓ | ✓ |
| Python API | ✗ | ✓ | ✓ (pybind11) |
| 적응형 적분기 | ✗ | ✓ (다수) | ✓ (DOPRI5) |
| 성능 기준 | 검증됨 | 미발표 | 1.18~2.55× mumax3 |

**권장**: 현재 NB41~45 벤치마크에 mumax+ 항목을 **추가로 신설**. 강자성 시나리오(SP#4, STT, DW 이동)에서 mumax3와 mumax+를 모두 비교하면, CS가 얼마나 발전된 코드베이스 대비에서도 경쟁력이 있는지 보여줄 수 있음.

---

*참고문헌*  
- Vansteenkiste et al., *AIP Advances* 4, 107133 (2014) — mumax3 원논문  
- mumax+ 논문: arXiv:2411.18194 → *npj Computational Materials* (2026-02)  
- GitHub: https://github.com/mumax/3 · https://github.com/mumax/plus
