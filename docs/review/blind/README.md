# blind 모드 입력 — 감사 결론 없음

> 이 파일만 읽고 판단하라. 같은 저장소의 `docs/audit/` 은 **읽지 마라.**
> 다른 도구들이 이미 낸 결론이 거기 있고, 그것을 읽으면 독립 검증이 성립하지 않는다.

## 과제

1. 아래 handoff 문서를 읽는다
2. 아래 원자료를 직접 본다
3. **이 프로젝트가 좌초할 가장 큰 이유 5개**를 스스로 지목한다
4. 각각에 반증 조건(무엇이 관측되면 내 지적이 틀린가)을 붙인다

## 입력 1 — handoff

`docs/handoff/finance_rule_graph_project_handoff.md`

신규 금융 Rule / Constraint Graph 프로젝트의 공통 맥락 문서.

## 입력 2 — 기존 저장소 코드

```
app/graph/jit_builder.py          트리플 추출 + JIT 그래프
app/graph/knowledge_graph.py      Neo4j 경로, Cypher 생성, 지표 산출
app/etl/enricher.py               메타데이터 추출 + 색인
app/etl/storage.py                저장소 인터페이스
app/rag/graph_flow.py             LangGraph 오케스트레이션
app/rag/solver.py                 검색 + 리랭킹
app/rag/evaluator.py              RAGAS 평가
scripts/build_index.py            측정용 2단계 그래프 구축
```

## 입력 3 — 실측 산출물 (원본 데이터)

```
experiments/v0prime/graph_public.json     5문서 그래프 (엔티티 118 / 관계 94)
experiments/v1/graph_public.json          40문서 그래프 (엔티티 925 / 관계 773)
experiments/v1/raw_completions/*.json     추출 LLM 원시 응답 73건
tests/fixtures/baseline_v*.json           측정 지표
tests/fixtures/entity_hapax.json          엔티티 문서 출현 분포
```

## 입력 4 — 기존 진단 기록

`docs/CONTEXT.md` — 이 저장소의 진단 진실 원본. 발견 1~20.

> 주의: 이것도 이미 내려진 결론이다. **완전한 blind 를 원하면 §3(발견 목록)과 §4(소거법)를
> 건너뛰고 코드·데이터만 보라.** 어디까지 가릴지는 선택이다 — 선택한 범위를 결과에 적어라.

## 산출물

`docs/review/blind-findings.md` 로 저장한다. 그 뒤에야 `docs/audit/` 을 열고
`docs/review/agenda.md` 로 넘어간다.
