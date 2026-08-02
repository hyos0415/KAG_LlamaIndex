# 데이터 및 재현성

## 저장소에 포함하지 않은 것

- 기사 원문 — 매일경제(mk.co.kr) 저작물
- 임베딩 벡터 — 이번 측정의 어떤 지표에도 사용되지 않는 부산물
- LLM 생성 요약 — 기사 파생 저작물

원본 그래프 파일에는 위 셋이 포함돼 있어 그대로 공개하지 않고,
`scripts/strip_graph.py` 로 제거한 축약본(`graph_public.json`)을 커밋한다.
제거 내역은 각 파일의 `_stripped` 필드와 스크립트 코드로 확인할 수 있다.

## 포함한 것

- `experiments/*/graph_public.json` — 엔티티, 관계, 트리플, provenance
- `tests/fixtures/baseline_v*.json` — 측정 지점별 지표
- `experiments/*/raw_completions/` — LLM 원시 응답
- `scripts/` — 파이프라인 및 검증 스크립트

## 검증

```
python scripts/verify_metrics.py --graph experiments/v0prime/graph_public.json \
    --baseline tests/fixtures/baseline_v0prime.json --section _root
python scripts/verify_metrics.py --graph experiments/v1/graph_public.json \
    --baseline tests/fixtures/baseline_v1.json --section full_40
```

수치 검증: `python scripts/verify_metrics.py`
8개 지표 중 7개가 baseline 과 완전 일치한다. cross_doc_path_ratio 만 0.45% 오차가 있으며, 원 계산 스크립트가 커밋되지 않아 발생한 재현 한계다(CONTEXT.md 발견 20). 이 오차는 어떤 판정 구간도 넘지 않는다.

## 재현 범위

지표 재계산은 저장소 데이터만으로 가능하다.
추출 단계부터의 재현은 원문이 필요하며 포함되어 있지 않다.
원본 기사는 각 노드의 `news_id` 로 매일경제에서 조회 가능하다(2026-01-26 ~ 02-03 수집 40건).
