# Schema 초안 — 노드 / 리터럴 / 논리식 구분

> 상태: **골격**. comparison §5 의 **블로킹 2번**이며 MVP 착수 전 최우선 산출물.
> 두 감사가 독립적으로 같은 요구에 도달한 항목이다 (comparison §1 C3, §3 M1).

## 1. 왜 이 표가 먼저인가 — 실측 근거

기존 저장소는 노드 타입 계약 없이 그래프를 만들었고, 그 결과가 측정됐다
(`docs/audit/extraction_stats.json`, v1 = 40문서/86청크):

```
리터럴 형태 엔티티 노드        152 / 925 = 16.4%
목적어가 리터럴인 트리플        183 / 773 = 23.7%   ← 트리플 1/4이 속성을 간선으로 표현
교차문서 브릿지 중 리터럴          6 / 42  = 14.3%   ← 다리의 1/7이 날짜·퍼센트
관계 타입                        460종 / 773관계 (단발성 70.9%)
```

추출기 원시 출력 예 (`experiments/v1/raw_completions/call_0001.json`):

```
(코스닥, 마감, 1064.41)        ← 1064.41 은 다른 무엇과도 연결될 이유가 없다
(영업점, 축소율, 21%)          ← '축소율' 은 술어가 아니라 속성명
(감소 기간, is, 5년)
```

**handoff §6 의 노드 후보 10종 중 `Rate`·`Term`·수치성 `Condition` 을 노드로 두면,
기존 프로젝트가 트리플의 23.7%에서 겪은 실패를 스키마에 명시적으로 설계해 넣는 것이 된다.**

## 2. 채울 표 (미정)

금융상품 한눈에 Open API 필드별로 세 칸 중 하나를 사전 결정한다.

| 필드 | 성격 | 노드 | 리터럴 속성 | 논리식 내부 값 | 근거 |
|---|---|---|---|---|---|
| `fin_prdt_nm` | 상품명 | `미정` | | | |
| `kor_co_nm` | 기관명 | `미정` | | | |
| `save_trm` | 저축 기간 | | `미정` | | Codex B10 은 term literal 제안 |
| `intr_rate` | 기본금리 | | `미정` | | Codex B10 은 product-option literal 제안 |
| `intr_rate2` | 최고우대금리 | | `미정` | | |
| `spcl_cnd` | 우대조건 (자연어) | | | `미정` | Codex B10 은 logic expression 제안 |
| `mtrt_int` | 만기후 이율 (자연어) | `미정` | `미정` | `미정` | |
| `etc_note` | 기타 (자연어) | | `미정` | | Codex B10 은 coverage/scope note 제안 |
| `join_way` | 가입 채널 | `미정` | | | handoff §6 `Channel` |
| `join_member` | 가입 대상 | `미정` | | | handoff §6 `Eligibility` |
| `join_deny` | 가입 제한 | `미정` | | | |

> 위 "제안" 칸은 Codex 리뷰의 제안이며 **결정이 아니다**. 실제 API 응답 샘플을 보고
> 채워야 하고, 두 감사 모두 금융 API 샘플에 접근하지 못했다.

## 3. 판정 규칙 (미정)

무엇을 노드로 둘지 사례별로 정하면 계약이 안 된다. 규칙이 필요하다.

```
미정 — 후보 규칙의 형태 예시 (결정 아님)

R1. 다른 상품·기관과 공유될 수 있는 개체만 노드
R2. 값이 비교 연산의 대상이면 리터럴 (금액·기간·비율·날짜)
R3. 논리 결합의 항이면 논리식 내부 값
R4. R1~R3 이 충돌하면 ?
```

**반증 조건** (Codex A2): 새 스키마에서 `Condition`·`Rate`·`Term`·날짜·금액·기간이
노드가 되어도 무의미한 연결·오병합·false reject 가 증가하지 않는다는 ablation 이
나오면 이 우려는 틀린다. 그 ablation 을 설계할지 여부도 열린 결정이다.

## 4. Expression grammar (미정)

handoff §7 의 operator 7종은 같은 층위가 아니다 (Codex B11):

```
ALL_OF / ANY_OF / NOT          논리 결합자
THRESHOLD / TEMPORAL           predicate 의 비교 제약
EXCEPTION                      scope override
MUTUALLY_EXCLUSIVE             조건 간 consistency relation
```

Codex 제안 (결정 아님):

```
Expr      = All | Any | Not | Predicate | Exception
Predicate = { comparator, value, unit, period }
MUTUALLY_EXCLUSIVE → operator 가 아니라 validation constraint 로 분리
```

## 5. 추출 계약 (미정)

기존 저장소의 추출 계약이 비어 있던 7개 칸
(`docs/audit/claude-architecture-audit.md` §3.4). 새 스키마는 전부 채워야 한다.

| 계약 항목 | 기존 | 신규 |
|---|---|---|
| 허용 노드 타입 | 없음 (`entity` 단일) | `미정` |
| 허용 관계 타입 | 없음 (460종 자유 생성) | `미정` |
| 출력 언어 | 없음 (영문 라벨 83종 혼입) | `미정` |
| 리터럴 vs 개체 구분 | 없음 | `미정` (§2) |
| 엔티티 경계 | 없음 (이름 52%가 복합구) | `미정` |
| 파싱 실패 처리 | 조용히 버림, **계측 없음** | `미정` |
| 스키마 검증 | 없음 | `미정` |

> 마지막 두 칸이 특히 중요하다. CONTEXT §6 은 "버려진 트리플 비율을 로그로 남기세요.
> 20% 초과면 온톨로지가 너무 좁습니다"를 요구했으나 구현되지 않았다
> (`build_index.py:189` 의 `except ValueError` 가 카운터를 올리지 않는다).
> handoff §10 의 "Schema / Extraction Valid Rate" 가 같은 요구라면,
> **요구가 계측 누락으로 이어진 전례가 있다**는 것을 명시해야 한다.
