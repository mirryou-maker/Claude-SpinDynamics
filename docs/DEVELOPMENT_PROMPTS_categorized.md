# Claude-SD 개발 프롬프트 — 주제별 분류

281개 프롬프트를 개발 단계/주제별로 분류. 번호는 [DEVELOPMENT_PROMPTS.md](DEVELOPMENT_PROMPTS.md)의 프롬프트 번호.
기간: 2026-05-25 ~ 06-22 (NanoSpinDynamics → Claude-SpinDynamics).

분포 요약:

| # | 주제 (theme) | 프롬프트 | 개수 |
|---|---|---|---|
| 1 | 코어 물리 구현 (LLG·STT·SOT) | 1–11 | 11 |
| 2 | Demag 버그 디버깅 (토큰 제약) | 12–56 | ~40 |
| 3 | 적분기 (RK45) + 유한온도 Phase T | 57–72 | 16 |
| 4 | Phase 3 CUDA/GPU 구현 | 73–86 | 14 |
| 5 | µMAG SP + Python 바인딩 + 노트북 | 87–92 | 6 |
| 6 | GPU full LLG (G1–G9) | 93–111 | 19 |
| 7 | mumax 예제·로드맵·CLAUDE.md | 112–124 | 13 |
| 8 | Phase A/B/C (지오메트리·MFM·per-cell) | 125–136 | 12 |
| 9 | 프로젝트 리네임 → Claude-SpinDynamics | 137–146 | 10 |
| 10 | mumax3 API 커버리지 구현 | 147–179 | 33 |
| 11 | 성능 최적화 · 3rd-party lib · float32 | 180–214 | 35 |
| 12 | VkFFT vs cuFFT 분석 | 211–220 | (겹침) |
| 13 | 4-solver 벤치마크 캠페인 | 196–259 | (겹침) |
| 14 | 출시 전 버그 점검 + GitHub 공개 + 매뉴얼 | 246–268 | (겹침) |
| 15 | 논문 (저널·본문·그림·PPT) | 269–281 | 13 |

> 메타 패턴: "step by step으로 나눠서", "다음 작업을 추천해줘", "결과를 저장/커밋하고 요약해줘"가 반복 — 토큰 한도 안에서 분할 실행하고 매 단계 검증·기록하는 워크플로우.

---

## 1. 코어 물리 구현 — LLG, STT/SOT (1–11)
Phase 1b/1c/1d 순차 구현. LLG 적분기, STT와 SOT 동시 구현 요청, Demag를 1순위로.
- [4] STT 구현 시 SOT도 함께 — [9] Demag Field 1순위 구현 — [10] "중간에 묻지말고 진행"

## 2. Demag 버그 디버깅 — 토큰 제약 하의 장기 디버깅 (12–56)
개발 전체에서 가장 긴 난관. FFTW/Newell 텐서 demag 버그가 반복 실패. 핵심은 **토큰 한도** 때문에 "한꺼번에 못 고치니 step by step 전략"을 반복 요구.
- [13][16] MAX_OUTPUT_TOKENS 조정 — [38] "기존 Demag 코드 완전 제거 후 재작성, 단 토큰 초과 안 되게 단계 계획"
- [40][43] "토큰 소모 아끼는 전략 제시" — [45] term 3 계수 1/3 수정만 — [48] newell_g 참조 구현과 비교
- [53] "A1부터 한단계씩, 끝나면 분석 후 전략 재수립"

## 3. 적분기 + 유한온도 (57–72)
RK45 단계 구현. 유한온도(SLLG)는 adaptive step 불가 → fixed step 필요함을 사용자가 지적, Phase T로 계획.
- [58][59] "finite 온도는 RK45 못 쓰고 fixed step" (계획만 추가) — [62] Phase T 계획 먼저 — [63–70] T1–T5 순차

## 4. Phase 3 CUDA/GPU (73–86)
CUDA 단계별 구현, 대형 격자 벤치마크, GPU 커널 사전계산, 스트림.
- [73] CUDA step by step 계획 — [83] 500×500×10 벤치 — [85] GPU 커널 사전계산

## 5. µMAG 표준문제 + 바인딩 + 노트북 (87–92)
- [88] SP#1 — [90] SP#1 위상 다이어그램 — [92] Jupyter 노트북

## 6. GPU full LLG — G1–G9 (93–111)
전체 LLG를 GPU에서. 사용량 reset 타이밍에 맞춰 G1씩 분할 구현하는 패턴.
- [94] "사용량 reset 되는 40분 후 G1만 먼저" — [98] G4 토크 커널+RK4 스테이지 — [104] HeunIntegratorGPU

## 7. mumax 예제·로드맵·문서 (112–124)
- [112] periodic BC demag — [115] mumax.github.io 예제 실행+노트북 — [119] Rotating Cheese/Spinning hard disk 최하위 우선순위 — [121] 논문 형식 문서화

## 8. Phase A/B/C — 지오메트리·MFM·per-cell (125–136)
- [125] Phase A step by step — [127] Phase B 목록화 후 B1-1부터 — [134] B2 MFM Imaging 계획

## 9. 프로젝트 리네임 (137–146)
NanoSpinDynamics → Claude-SpinDynamics, 폴더 이동, 기존 폴더 보관.
- [137] 이름 변경 + 폴더 이동 권한 — [138] "기존 폴더는 삭제 말고 보관"

## 10. mumax3 API 커버리지 (147–179)
mumax3 API 중 미구현 함수 조사·구현. "우선순위대로 수행 후 다음 추천"의 강한 반복.
- [148] 구현/미구현 함수 정리 — [163] 난이도 낮음·중간 구현 — [171] 멀티 GPU 가능? — [172] 멀티 GPU 방식 A

## 11. 성능 최적화 · 3rd-party · float32 (180–214)
- [180] "3rd party library 사용 검토" — [181] float32 자세한 설명 — [184] P9/P12/P11/P10/P13/P14 순차 — [214] P4 구현 + VkFFT 검증

## 12. VkFFT vs cuFFT (211–220)
- [211] (영문 상세) non-2의 거듭제곱 셀 크기에서 VkFFT vs cuFFT 처리량 비교, mumax→cuFFT 데이터 흐름 분석
- [219] non-2^N도 동일한가 — [220] CS가 mumax 대비 우세한 경우는?

## 13. 4-solver 벤치마크 캠페인 (196–259)
Claude-SD(cuFFT/VkFFT × f32/f64) vs mumax3 / mumax+ / MuMax-CO / OOMMF. 공정 비교 설계가 핵심.
- [196] 4종(mumax3/oommf/CS f32/f64) 비교 계획 — [197] MuMax-CO 포함, SP#4 고해상도, 고정스텝 — [199] "솔버 편차 먼저 조사·해결 후 벤치, 앱 이름 Claude-SD" — [221] CS/mumax 각각 유리한 시나리오 공정 설계 — [222] adaptive RK45 비교 추가 — [251] 논문용 표/그림 포함 벤치 계획 — [254] SP#2 문제가 CS만인가 타 솔버도인가

## 14. 출시 전 점검 · GitHub · 매뉴얼 (246–268)
- [248] "개발 거의 마무리, 마지막 개선점 점검" — [249] Critical 4건 수정+4빌드 테스트 — [267] GitHub 등록 — [268] 사용자 매뉴얼

## 15. 논문 — 저널·본문·그림·PPT (269–281)
- [269] 저널 추천 — [271] npj Comput Mater 전략·목차 먼저 — [272] 추가작업 P1–P4 — [273] (Q non-determinism) "1번으로 해결 확인" — [274] A/B/P2/P4 후 전략 재정리 — [275] F1/F2/F5/F6 + T3 + Zenodo + scaling + OOMMF — [276] 본문+Supplementary+cover letter — [278] PPT — [280] 캡션+speaker notes — [281] 프롬프트 정리

---

### 관찰된 개발 스타일 (논문 Methods/SI에 인용 가능)
1. **단계 분할 강제**: 토큰 한도를 명시적 제약으로 다루며 "step by step", "잘게 쪼개서"를 반복 — 큰 변경을 작은 검증 가능 단위로.
2. **검증 루프**: 매 단계 "결과를 분석/보고하고 다음을 추천" — 사람이 게이트키퍼.
3. **상태 영속화**: "메모리에 저장", "커밋", "문서화"를 자주 — 세션 경계를 넘는 연속성.
4. **물리 도메인 지식 주입**: [58] SLLG는 fixed step, [222] 실사용 솔버는 RK45, [254] 솔버 편차 교차검증 — 사용자의 전문성이 방향을 교정.
5. **공정성 설계**: 벤치마크에서 동일 integrator/정밀도/스텝 조건을 반복 강조.
