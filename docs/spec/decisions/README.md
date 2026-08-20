# Architecture decisions

`NNNN-<slug>.md` 형식. 템플릿은 `0000-template.md`.

## 확정된 결정

없음.

## 열린 결정 — 사람이 정할 것

`docs/audit/00-comparison.md` §6 에서 온 것들이다.

| # | 항목 | 상태 |
|---|---|---|
| Q1 | `verify_metrics.py` / serialization 을 이관 자산으로 볼 것인가 | 두 감사가 반대로 판정. 조정안은 comparison §2 D1 |
| Q2 | handoff §2.1 "정규화 부재" 비판을 정정할 것인가 | 정정 시 기존 프로젝트 한계 서사가 바뀐다. 포트폴리오 서사와 연동 |
| Q3 | CONTEXT 발견 1의 "의도적 표본 추출" 서술 정정 여부 | 진단 문서의 원인 귀속이 코드 사실과 충돌 (`enricher.py:117`) |
| Q4 | 감사 브랜치를 main 에 병합할 것인가 | 현재 5개 브랜치 분리 유지 |

## MVP 착수 전 블로킹 항목 — comparison §5

| # | 항목 | 담당 문서 |
|---|---|---|
| 1 | Graph 필요성 반증 게이트 (baseline 4개) | `evaluation.md` §2 |
| 2 | 노드 / 리터럴 / 논리식 구분표 | `schema.md` §2 |
| 3 | operator enum → expression grammar | `schema.md` §4 |
| 4 | 집합 차분을 ALL_OF MVP 의 한 연산으로 격하 | `project_spec.md` §7 |
| 5 | factorial evaluation matrix + extractor gold 평가 | `evaluation.md` §2 |
