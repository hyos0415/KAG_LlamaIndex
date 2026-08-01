"""
정규화 규칙 구현 (§8 미정 1·2 확정, docs/CONTEXT.md §6 "정규화 작업 내용" 참고).

엔티티 병합 규칙 (v2a/v2c 대상) — 쌍 단위만, 전이적 클러스터링 금지:
  1. 쌍 단위만 적용
  2. 짧은 쪽 길이 3자 이상
  3. 차이 부분이 알려진 수식어(직함/조사)와 정확히 일치
  4. 차이 부분 길이 5자 이하
  5. 차이 부분 또는 긴 쪽 문자열에 B유형(사건) 키워드 포함 시 제외

관계 정규화 (v2b/v2c 대상) — 활용형 어미/조사 제거만. 온톨로지 매핑 아님.

이 스크립트는 규칙 함수와 드라이런 리포트만 제공한다. 실제 v2a/v2b/v2c 그래프
재구축(alias_map/stem_map을 실제 트리플에 적용해 persist)은 이 스크립트 실행만으로는
하지 않는다 — 승인 후 별도 실행.

사용 예 (드라이런, 그래프를 만들지 않고 예상 건수만 출력):
    python scripts/normalize.py --graph experiments/v1/property_graph_store.json
"""

import argparse
import json

TITLE_WORDS = ["대통령", "의원", "장관", "대표", "회장", "위원장", "시장", "청장"]
PARTICLES = ["은", "는", "이", "가", "의", "을", "를"]
B_KEYWORDS = ["사퇴", "제명", "논란", "중단", "원인", "발생", "사고", "혐의", "조사"]

KNOWN_MODIFIERS = set(TITLE_WORDS) | set(PARTICLES)

VERB_ENDINGS = [
    "했습니다", "되었다", "됐다", "하였다", "였다", "했다", "한다", "된다", "되다", "하다", "이다",
    "했음", "됨", "함", "임",
]
REL_PARTICLES = [
    "으로는", "에서는", "까지는", "으로", "에서", "에게", "부터", "까지",
    "이는", "는", "은", "이", "가", "을", "를", "의", "로", "과", "와", "도", "만",
]


def _diff(short, long_):
    """short가 long_의 접두 또는 접미일 때 그 차이(공백 제거)를 반환. 아니면 None."""
    if short == long_ or len(short) < 3:
        return None
    if long_.startswith(short):
        return long_[len(short):].strip()
    if long_.endswith(short):
        return long_[: len(long_) - len(short)].strip()
    return None


def find_entity_merge_pairs(entities):
    """쌍 단위 보수적 병합 규칙. 전이적 클러스터링 없음 (union-find 사용 안 함)."""
    names = sorted(set(entities))
    pairs = []
    for i, a in enumerate(names):
        for b in names:
            if a == b:
                continue
            diff = _diff(a, b)
            if diff is None:
                continue
            if len(diff) == 0 or len(diff) > 5:
                continue
            if diff not in KNOWN_MODIFIERS:
                continue
            if any(kw in diff for kw in B_KEYWORDS) or any(kw in b for kw in B_KEYWORDS) or any(kw in a for kw in B_KEYWORDS):
                continue
            short, long_ = (a, b) if len(a) < len(b) else (b, a)
            pairs.append((short, long_))
    # 중복 제거 (양방향 스캔으로 같은 쌍이 두 번 잡힐 수 있음)
    return sorted(set(pairs))


def stem_relation_type(t):
    s = t
    changed = True
    while changed:
        changed = False
        for e in VERB_ENDINGS:
            if s.endswith(e) and len(s) > len(e):
                s = s[: -len(e)]
                changed = True
                break
        else:
            for p in REL_PARTICLES:
                if s.endswith(p) and len(s) > len(p) + 1:
                    s = s[: -len(p)]
                    changed = True
                    break
    return s


def build_relation_stem_map(relation_types):
    """활용형/조사 제거만 수행. 온톨로지(의미) 매핑 아님."""
    return {t: stem_relation_type(t) for t in relation_types}


def load_graph(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    nodes = d["nodes"]
    rels = list(d["relations"].values())
    entities = [nid for nid, v in nodes.items() if v.get("label") == "entity"]
    rel_types = [r.get("label") for r in rels]
    return entities, rel_types


def dry_run_report(graph_path):
    entities, rel_types = load_graph(graph_path)

    # 병합 방향: long_(수식어 붙은 변형) -> short(짧은 canonical 형태)
    merge_pairs = find_entity_merge_pairs(entities)
    absorbed_long_forms = set(long_ for _, long_ in merge_pairs)  # 병합되어 사라질 엔티티
    canonical_short_forms = set(short for short, _ in merge_pairs)

    stem_map = build_relation_stem_map(rel_types)
    unique_types_before = len(set(rel_types))
    unique_stems_after = len(set(stem_map.values()))

    report = {
        "graph_path": graph_path,
        "entity_v2a": {
            "total_entities": len(set(entities)),
            "merge_pairs_found": len(merge_pairs),
            "canonical_short_forms": len(canonical_short_forms),
            "absorbed_long_forms": len(absorbed_long_forms),
            "expected_reduction": len(absorbed_long_forms),
            "expected_reduction_pct_of_total": round(len(absorbed_long_forms) / len(set(entities)) * 100, 2),
            "sample_pairs": merge_pairs[:30],
        },
        "relation_v2b": {
            "total_types": unique_types_before,
            "estimated_after_stemming": unique_stems_after,
            "reduction_count": unique_types_before - unique_stems_after,
            "reduction_pct": round((unique_types_before - unique_stems_after) / unique_types_before * 100, 1),
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, help="property_graph_store.json 경로")
    parser.add_argument("--dry-run", action="store_true", default=True, help="기본값. 실제 그래프를 만들지 않고 예상 건수만 출력")
    args = parser.parse_args()

    report = dry_run_report(args.graph)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
