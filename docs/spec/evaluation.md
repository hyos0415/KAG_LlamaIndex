# Evaluation contract 초안

> 상태: **골격**. comparison §5 블로킹 5번.
> 지표 정의는 **실행 전에 확정하고 실행 후 바꾸지 않는다** (CONTEXT §7.4).

## 1. 기존 저장소가 ablation 을 못 한 이유 (코드 사실)

새 프로젝트가 같은 배선을 반복하지 않기 위한 기록이다
(`docs/audit/claude-architecture-audit.md` §5.2):

- `graph_flow.py:53` — `need_graph=True` 고정. 조건부 엣지가 선언돼 있으나 분기하지 않는다.
  **그래프를 끄는 경로가 코드에 존재하지 않았다**
- `solver.py:69` 의 `use_graph` 플래그는 경로 A 에만 있고 LangGraph 경로에는 없다
- `evaluator.py:66` 은 `NewsLangChainSolver.solve` 만 호출한다 —
  **평가 코드가 그래프 경로를 한 번도 지나지 않았다**

→ **신규 저장소의 요구사항**: 그래프 on/off 스위치가 평가 진입점에서 접근 가능해야 한다.

## 2. Factorial evaluation matrix (미정)

handoff §10 은 "Graph extraction 오류와 Verifier 오류를 분리한다"고 쓰지만 측정
설계가 없다 (Codex B13). 최소 2×2 가 필요하다.

|  | oracle checker | 실제 checker |
|---|---|---|
| **gold graph** | `미정` — upper bound | `미정` |
| **extracted graph** | `미정` — extractor 오류 격리 | `미정` — realized performance |

추가 통제군 (미정):

- `no graph + verifier` — handoff §10 의 A (Verifier Only)
- `oracle evidence + verifier` — **선택이 아니라 필수 통제군**
  (comparison §3 M2: 기존 저장소는 `LLMRerank(top_n = top_k//2)` 로 검색 문서의
   절반을 버렸다. "증거 리콜 부족"은 새 가설이 아니라 코드로 확인된 실패 양상이다)
- `deterministic checklist + verifier` — Graph 없이 결정론적 체크리스트만
  (Codex 반대 가설 4)

## 3. Graph-only 의 지위 (미정)

handoff §10 은 Graph Only 를 Verifier Only 와 나란히 3-way 비교한다. 그러나
handoff §4 는 Graph 가 verifier 를 대체하지 않는다고 말한다. Graph 가
`missing required condition → UNSUPPORTED 후보`만 낼 수 있으면 SUPPORTED /
INSUFFICIENT 를 판정할 수 없고, 3-way Macro F1 비교가 성립하지 않는다 (Codex B12).

Codex 제안 (결정 아님):

```
Graph-only 를 classifier 가 아니라 rule trigger / abstaining detector 로 정의
지표: trigger precision · condition_omission recall within coverage
      coverage · false reject among supported claims
```

## 4. 지표 사전 등록 (미정)

handoff §10 의 지표 후보 8종을 그대로 쓰기 전에 각각에 대해 정할 것:

| 지표 | 정의 | 유효 구간 | 방향 |
|---|---|---|---|
| Condition Omission Recall | `미정` | `미정` | 높을수록 |
| False Accept Rate | `미정` | `미정` | 낮을수록 |
| UNSUPPORTED Recall | `미정` | `미정` | 높을수록 |
| Macro F1 | `미정` | **§3 미해결 시 Graph-only 에 무효** | 높을수록 |
| False Reject | `미정` | `미정` | 낮을수록 |
| Additional Latency | `미정` | `미정` | 낮을수록 |
| Graph Rule Application Rate | `미정` | `미정` | — |
| Schema / Extraction Valid Rate | `미정` | **처치의 정의일 위험** | — |

> **마지막 줄 경고 (comparison §8.2 방법론 자산)**: 기존 저장소는
> `entities_per_doc` 을 v1→v2 구간에서 **무효 지표로 사전 선언**했다 —
> 병합률이 곧 처치이므로 순환논증이 되기 때문이다. "Extraction Valid Rate" 도
> 추출 유효율이 곧 처치이면 성과 지표가 아니다. 유효 구간을 실행 전에 선언할 것.

## 5. 데이터셋 구축 프로토콜 (미정)

handoff 에 없는 항목 (Codex B14):

- blind annotation — slice 를 Graph 설계자가 고르면 선택 편향
- inter-annotator agreement
- held-out 상품
- false reject 측정용 negative 사례의 출처와 크기
- sample size justification / 검정력

## 6. 재현성 요구 (미정 — comparison Q1 결정 대기)

기존 저장소에서 두 번 재현성이 깨졌다. 신규 저장소 초기 커밋의 수용 기준 후보:

- 모든 LLM 호출의 **실제 응답 모델 ID** 기록 (CONTEXT 발견 11 — 모델이 폐기되면
  과거 측정값이 재현 불가가 된다)
- 분석 코드를 산출물과 **함께 커밋** (발견 20 — 대화형 세션의 계산은 결과만 남고
  방법이 사라진다)
- 생성 → strip → metric → fixture 검증이 **하나의 재현 가능한 명령**으로 고정
  (Codex A4 의 반증 조건)
