# 공통 project spec (초안) — Finance Rule / Constraint Graph

> 상태: **골격**. 아래 12개 항목은 handoff §15 의 최소 요구 항목이다.
> 전부 미정이며, 채우는 순서는 `docs/audit/00-comparison.md` §5 를 따른다.
> 이 문서는 신규 저장소로 옮길 초안이다 (handoff §13).

## 0. 이 문서의 지위

tool-neutral 공통 spec 이다. Claude 전용 지시는 별도 `CLAUDE.md` 에 두되,
**핵심 문제 정의와 결정은 특정 Agent 에 종속된 문서에만 두지 않는다** (handoff §15).

---

## 1. Problem

`미정`

> 채울 때 확인할 것: handoff §3.1 은 `condition_omission` 을 핵심 failure 로 둔다.
> 그러나 감사 결과 이것이 Graph 로 풀 문제라는 전제가 미검증이다
> (`docs/audit/codex-adversarial-review.md` B7, 반대 가설 4개).
> **Problem 을 "condition_omission 을 Graph 로 해결한다"로 쓰면 해결책을 문제 정의에
> 미리 넣는 것이 된다.** 문제는 failure 자체로, 해결 수단은 §Hypothesis 로 분리한다.

## 2. Hypothesis

`미정`

> handoff §4 의 가설을 그대로 옮기기 전에, Graph 필요성 반증 게이트
> (comparison §5 블로킹 1번)를 통과 조건으로 명시할지 결정해야 한다.

## 3. Scope

`미정`

> handoff §5.1 기준선: 은행권 정기예금 · 금융상품 한눈에 Open API ·
> finance_verifier 의 snapshot / canonical data.

## 4. Non-goals

`미정`

> handoff §9 의 초기 MVP 제외 목록 7개가 출발점.
> 감사에서 추가된 후보: **범용 그래프 순회 / PageRank 계열은 이 저장소에서
> "평가 없이 사용된 Graph algorithm"으로 이미 폐기 판정을 받았다**
> (handoff §12 B, 두 감사 모두 동의).

## 5. Data

`미정`

> 확인 필요: finance_verifier 의 `condition_omission` 사례가 실제 몇 건이고 어떤
> 분포인지. 두 감사 모두 원자료 접근이 없어 검증하지 못했다
> (`codex-adversarial-review.md` 검토 범위 "읽지 않은 것" 3번).
> **데이터셋 크기와 검정력 계획이 handoff 에 없다** (같은 문서 B14).

## 6. Schema

`미정` → `docs/spec/schema.md` 로 분리

> **최우선 블로킹 항목.** 두 감사가 독립적으로 같은 요구에 도달했다
> (comparison §1 C3, §3 M1): "노드로 둘 것 / 리터럴 속성으로 둘 것 /
> 논리식 내부 값으로 둘 것" 구분표가 MVP 착수 전에 필요하다.

## 7. Constraint

`미정`

> 감사에서 도출된 확정 제약 후보 (comparison §5 블로킹 항목):
> - 원문 필드와 LLM 파생 필드의 강제 분리를 **스키마 불변식**으로 선언
>   (기존 저장소에서 LLM 생성 category 가 그래프 노드 `정치/경제` 를 만들었고,
>    그 노드가 가짜 다리의 중간 노드였다 — `claude-architecture-audit.md` §3.3)
> - operator enum → expression grammar 로 재설계
>   (`codex-adversarial-review.md` B11)

## 8. Evaluation contract

`미정` → `docs/spec/evaluation.md` 로 분리

## 9. Metrics

`미정` → `docs/spec/evaluation.md` 로 분리

## 10. Current decisions

`없음`

> 결정이 생길 때마다 `docs/spec/decisions/NNNN-*.md` 를 추가하고 여기에 한 줄로 색인한다.

## 11. Rejected alternatives

기존 저장소에서 **이미 실측으로 기각된 것** — 새 프로젝트에서 다시 시도하지 않는다.

| 기각된 접근 | 근거 | 출처 |
|---|---|---|
| 코퍼스 규모 확대로 그래프 조각화 해소 | 문서 8배에 관계 타입 6.3배. entities_per_doc −2.0% | CONTEXT §4 |
| 관계 활용형 정규화 | 460→444종(3.5%). 컴포넌트 구조에 영향 없음이 수학적 필연 | CONTEXT §4, v2b |
| substring 기반 엔티티 별칭 병합 | 표본 40쌍에서 유의미한 별칭 0건. 재등장 엔티티가 925개 중 42개(4.5%) | CONTEXT 발견 15·18 |
| 클러스터 크기 상한으로 오병합 차단 | 크기 2짜리 오병합이 상한을 통과. 문제는 크기가 아니라 substring 매칭 | CONTEXT v2a-mid 폐기 |
| 후처리 정규화로 타입 폭발 회수 | 4.3%만 회수. 스키마 제약 추출기로의 교체(재추출)를 요구 | CONTEXT §4 |

> **주의 (comparison §2 D2):** handoff §2.1 은 "정규화 규칙이 없었다"를 개선 가능한
> 결함으로 나열한다. 그러나 위 표대로 이 저장소는 정규화를 시도해 **기각**했다.
> 새 프로젝트가 "이번엔 정규화를 제대로 하자"로 읽으면 이미 기각된 경로를 다시 걷는다.
> handoff §2.1 정정 여부는 열린 결정이다 (`decisions/README.md` Q2).

## 12. Known limitations

`미정`

> 이관 시점에 최소 포함할 것: 두 감사가 확인하지 못한 항목
> (`claude-architecture-audit.md` §9 5건, `codex-adversarial-review.md` 검토 범위 3건)
