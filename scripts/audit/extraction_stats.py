"""감사용 추출 산출물 통계 (audit/claude-architecture 브랜치 전용).

docs/CONTEXT.md 발견 20의 교훈("분석 코드는 산출물과 함께 커밋되어야 한다")에 따라
docs/audit/claude-architecture-audit.md 에 인용된 수치를 재현 가능하게 만드는 스크립트다.

측정 대상: experiments/{v0prime,v1}/graph_public.json
읽기 전용. LLM 호출 없음.

사용:
    python scripts/audit/extraction_stats.py > docs/audit/extraction_stats.json
"""

import json
import re
import sys
from collections import Counter

LITERAL_PATTERNS = [
    ("percent", re.compile(r"^[+\-]?[\d,.]+\s*%$")),
    ("pure_number", re.compile(r"^[+\-]?[\d,]+(\.\d+)?$")),
    ("date_or_time", re.compile(r"^\d{1,4}\s*(년|월|일|시|분|주|분기|반기)(\s*\d{1,2}\s*(월|일|시|분))*$")),
    ("year_month_day", re.compile(r"^\d{4}[.\-/년]\s*\d{1,2}([.\-/월]\s*\d{1,2}[일]?)?$")),
    ("money_or_measure", re.compile(r"^[+\-]?[\d,.]+\s*(원|달러|엔|위안|억|조|만|천|백만|억원|조원|만원|억달러|조달러|%p|포인트|p|bp|명|건|개|배|년|개월|주|평|㎡|km|kg|t|톤|회|층|세|억주|주)$")),
    ("count_with_unit", re.compile(r"^[\d,.]+\s*\S{1,4}$")),
]

ENG_RE = re.compile(r"^[A-Za-z][A-Za-z\s\-'./]*$")
HAS_SPACE = re.compile(r"\s")


def classify_literal(name):
    s = name.strip()
    for kind, pat in LITERAL_PATTERNS:
        if pat.match(s):
            return kind
    return None


def analyze(path):
    with open(path, encoding="utf-8") as f:
        g = json.load(f)

    entities = {k: v for k, v in g["nodes"].items() if v.get("label") == "entity"}
    chunks = {k: v for k, v in g["nodes"].items() if v.get("label") == "text_chunk"}
    relations = g["relations"]
    triplets = g["triplets"]

    # --- 1. 노드 공간 오염: 리터럴이 엔티티 노드로 등록된 비율 (발견 19 정량화)
    literal_kinds = Counter()
    literal_names = []
    for name in entities:
        kind = classify_literal(name)
        if kind:
            literal_kinds[kind] += 1
            literal_names.append(name)

    # --- 2. 엔티티 이름 형태: 서술구 여부 (발견 17 정량화)
    lengths = [len(n) for n in entities]
    multiword = [n for n in entities if HAS_SPACE.search(n)]
    long_phrase = [n for n in entities if len(n) >= 10]

    # --- 3. 관계 라벨 분류
    rel_labels = Counter(r["label"] for r in relations.values())
    eng_labels = sorted(l for l in rel_labels if ENG_RE.match(l))
    hapax_labels = sorted(l for l, c in rel_labels.items() if c == 1)

    # --- 4. 트리플 형태: 목적어가 리터럴인 비율
    #     (엔티티, 속성명, 리터럴) 은 그래프 간선이 아니라 노드의 속성이어야 하는 것
    obj_literal = 0
    subj_literal = 0
    both_entity = 0
    for s, r, o in triplets:
        so = classify_literal(str(s)) is not None
        oo = classify_literal(str(o)) is not None
        subj_literal += so
        obj_literal += oo
        if not so and not oo:
            both_entity += 1

    # --- 5. provenance 커버리지
    with_src = sum(1 for r in relations.values()
                   if r.get("properties", {}).get("triplet_source_id"))
    with_newsid = sum(1 for r in relations.values()
                      if r.get("properties", {}).get("news_id"))
    # provenance 가 청크 단위인가 문장 단위인가
    src_granularity = "chunk" if any(
        "-chunk" in str(r.get("properties", {}).get("triplet_source_id", ""))
        for r in relations.values()
    ) else "unknown"

    # --- 6. 관계 프로퍼티가 문서 메타데이터를 그대로 복제하는지 (저장 비용/오염)
    sample_props = next(iter(relations.values()))["properties"] if relations else {}

    n_ent = len(entities) or 1
    n_tri = len(triplets) or 1
    return {
        "source": path,
        "counts": {
            "entity_nodes": len(entities),
            "text_chunk_nodes": len(chunks),
            "relations": len(relations),
            "triplets": len(triplets),
            "distinct_relation_labels": len(rel_labels),
        },
        "node_space_pollution": {
            "literal_entity_nodes": sum(literal_kinds.values()),
            "literal_ratio": round(sum(literal_kinds.values()) / n_ent, 4),
            "by_kind": dict(literal_kinds),
            "examples": literal_names[:40],
        },
        "entity_name_shape": {
            "mean_len": round(sum(lengths) / n_ent, 2),
            "max_len": max(lengths) if lengths else 0,
            "len_ge_10_count": len(long_phrase),
            "len_ge_10_ratio": round(len(long_phrase) / n_ent, 4),
            "multiword_count": len(multiword),
            "multiword_ratio": round(len(multiword) / n_ent, 4),
            "longest_examples": sorted(entities, key=len, reverse=True)[:15],
        },
        "relation_labels": {
            "distinct": len(rel_labels),
            "type_ratio": round(len(rel_labels) / (len(relations) or 1), 4),
            "hapax_count": len(hapax_labels),
            "hapax_ratio": round(len(hapax_labels) / (len(rel_labels) or 1), 4),
            "english_label_count": len(eng_labels),
            "english_labels": eng_labels[:40],
            "top20": rel_labels.most_common(20),
            "top20_coverage": round(
                sum(c for _, c in rel_labels.most_common(20)) / (len(relations) or 1), 4
            ),
        },
        "triple_shape": {
            "object_is_literal": obj_literal,
            "object_is_literal_ratio": round(obj_literal / n_tri, 4),
            "subject_is_literal": subj_literal,
            "subject_is_literal_ratio": round(subj_literal / n_tri, 4),
            "both_sides_entity": both_entity,
            "both_sides_entity_ratio": round(both_entity / n_tri, 4),
        },
        "provenance": {
            "relations_with_triplet_source_id": with_src,
            "relations_with_news_id": with_newsid,
            "coverage_ratio": round(with_src / (len(relations) or 1), 4),
            "granularity": src_granularity,
            "relation_property_keys": sorted(sample_props.keys()),
        },
    }


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "experiments/v0prime/graph_public.json",
        "experiments/v1/graph_public.json",
    ]
    out = {p: analyze(p) for p in paths}
    print(json.dumps(out, ensure_ascii=False, indent=2))
