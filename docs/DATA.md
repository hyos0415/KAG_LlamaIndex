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

`docs/design-review.md` 의 모든 수치를 `graph_public.json` 에서 재계산하고
`baseline_v*.json` 과 대조한다. `cross_doc_path_ratio`/`cross_doc_pairs` 한 항목만
0.45% 오차 범위 내에서 근사 일치한다 — 원 계산 스크립트가 한 번도 커밋되지 않아
정확한 재현이 불가능하기 때문이며, 이 지표는 CONTEXT.md §6에서 이미 "조합론적
인플레이션으로 규모 효과 지표로 부적합"이라고 정정된 항목이라 §4 결론에 영향을
주지 않는다. 그 외 모든 지표(엔티티/관계/컴포넌트 수, `entities_per_doc`,
`components_per_doc`, `rel_type_ratio`, `isolated_doc_ratio` 등)는 완전히 일치한다.

## 재현 범위

지표 재계산은 저장소 데이터만으로 가능하다.
추출 단계부터의 재현은 원문이 필요하며 포함되어 있지 않다.
원본 기사는 각 노드의 `news_id` 로 매일경제에서 조회 가능하다(2026-01-26 ~ 02-03 수집 40건).
