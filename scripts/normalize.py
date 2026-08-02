"""
정규화 규칙 구현 (§8 미정 1·2 확정, docs/CONTEXT.md §6 "정규화 작업 내용" 참고).

엔티티 병합 — v2a-lo(하한, --mode lo)와 v2a-hi(상한, --mode hi)로 분리 (§6 "v2a 상하한 분할"):

  v2a-lo: 쌍 단위만, 전이적 클러스터링 금지. 오병합 0 목표.
    1. 쌍 단위만 적용
    2. 짧은 쪽 길이 3자 이상
    3. 차이 부분이 알려진 수식어(직함/조사)와 정확히 일치
    4. 차이 부분 길이 5자 이하
    5. 차이 부분 또는 긴 쪽 문자열에 B유형(사건) 키워드 포함 시 제외

  v2a-hi: B유형 키워드로 사건 노드만 제외하고 나머지는 substring 기준 전이적
    클러스터링. 의도적 과다 병합(오탐 포함, 고치지 않음) — 정규화 효과의 상한.

관계 정규화 (v2b/v2c 대상) — 활용형 어미/조사 제거만. 온톨로지 매핑 아님.

이 스크립트는 규칙 함수와 드라이런 리포트만 제공한다. 실제 v2a-lo/v2a-hi/v2b/v2c
그래프 재구축(alias_map/stem_map을 실제 트리플에 적용해 persist)은 이 스크립트
실행만으로는 하지 않는다 — 승인 후 별도 실행.

사용 예 (드라이런, 그래프를 만들지 않고 예상 건수만 출력):
    python scripts/normalize.py --graph experiments/v1/property_graph_store.json --mode lo
    python scripts/normalize.py --graph experiments/v1/property_graph_store.json --mode hi
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


EVENT_KEYWORDS_HI = [
    "사퇴", "논란", "화재", "원인", "중단", "재고", "사건", "의혹", "확산", "발생", "위생", "인상", "인하",
    "도입", "추진", "결정", "발표", "개최", "체결", "합병", "매각", "인수", "상승", "하락", "폭등", "폭락",
    "소송", "기소", "구속", "처벌", "선고", "판결", "조사", "수사", "발령", "지시", "요청", "요구", "촉구",
    "반발", "항의", "시위", "집회", "회의", "총회", "협상", "계약", "투자", "매입", "확정", "승인", "거부",
    "반대", "찬성", "지지", "비판", "논쟁", "갈등", "대립", "충돌", "부상", "사망", "피해", "복구", "철회",
    "재개", "착수", "완료", "발효", "폐지", "시행", "연기", "취소", "종료", "시작",
]


def find_entity_clusters_hi(entities):
    """의도적 과다 병합(정규화 효과의 상한). substring 전이적 클러스터링.
    오탐(서울/경찰/경찰청 등)을 그대로 둔다 — 고치지 않는다. §6 "v2a 상하한 분할" 참고.
    B유형(키워드 포함) 엔티티만 사전 제외하고, 나머지는 전이적으로 묶는다.
    """
    def has_event_keyword(s):
        return any(kw in s for kw in EVENT_KEYWORDS_HI)

    names_all = sorted(set(entities))
    type_b = [e for e in names_all if has_event_keyword(e) and len(e) > 3]
    remaining = sorted(set(names_all) - set(type_b))

    parent = {x: x for x in remaining}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    n = len(remaining)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = remaining[i], remaining[j]
            if len(a) >= 2 and len(b) >= 2 and (a in b or b in a):
                union(a, b)

    clusters = {}
    for x in remaining:
        clusters.setdefault(find(x), []).append(x)
    multi_clusters = {r: members for r, members in clusters.items() if len(members) > 1}

    return type_b, multi_clusters


def find_entity_clusters_mid(entities, max_cluster_size=5):
    """사후 설계 (v2 결과 확인 후 추가, 2026-08-02). 사전 등록된 v2a-hi 판정을
    변경하는 것이 아니라, 무효화된 v2a-hi(거대 오병합 허브)를 대체할 현실적
    추정치를 구하기 위한 후속 조사다. §6 "v2a-mid" 참고.

    v2a-hi와 동일한 전이적 substring 클러스터링을 쓰되, 클러스터 크기가
    max_cluster_size를 넘으면 병합을 거부한다. 크기 상한 5는 "이재명/이재명
    대통령/대통령" 같은 정당한 다원 병합은 허용하되 34개짜리 "경제" 허브
    (서울/경찰/경찰청/삼성전자 등을 한 노드로 묶은 오병합) 같은 거대 허브는
    차단하기 위함이다. 상한 값은 클러스터 크기 분포를 보고 조정될 수 있다
    (§6 참고 — 사람이 결정). 구현만 하고 실행하지 않는다.
    """
    def has_event_keyword(s):
        return any(kw in s for kw in EVENT_KEYWORDS_HI)

    names_all = sorted(set(entities))
    type_b = [e for e in names_all if has_event_keyword(e) and len(e) > 3]
    remaining = sorted(set(names_all) - set(type_b))

    parent = {x: x for x in remaining}
    size = {x: 1 for x in remaining}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return True
        if size[rx] + size[ry] > max_cluster_size:
            return False  # 상한 초과 시 병합 거부 (거대 허브 차단)
        parent[rx] = ry
        size[ry] += size[rx]
        return True

    n = len(remaining)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = remaining[i], remaining[j]
            if len(a) >= 2 and len(b) >= 2 and (a in b or b in a):
                union(a, b)

    clusters = {}
    for x in remaining:
        clusters.setdefault(find(x), []).append(x)
    multi_clusters = {r: members for r, members in clusters.items() if len(members) > 1}

    return type_b, multi_clusters


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


def dry_run_report(graph_path, mode="lo", max_cluster_size=5):
    entities, rel_types = load_graph(graph_path)
    total_entities = len(set(entities))

    if mode == "lo":
        # 병합 방향: long_(수식어 붙은 변형) -> short(짧은 canonical 형태)
        merge_pairs = find_entity_merge_pairs(entities)
        absorbed = set(long_ for _, long_ in merge_pairs)
        canonical = set(short for short, _ in merge_pairs)
        entity_report = {
            "mode": "lo",
            "total_entities": total_entities,
            "merge_pairs_found": len(merge_pairs),
            "canonical_short_forms": len(canonical),
            "absorbed_long_forms": len(absorbed),
            "expected_reduction": len(absorbed),
            "expected_reduction_pct_of_total": round(len(absorbed) / total_entities * 100, 2),
            "sample_pairs": merge_pairs[:30],
        }
    elif mode == "hi":
        type_b, multi_clusters = find_entity_clusters_hi(entities)
        entities_in_clusters = sum(len(m) for m in multi_clusters.values())
        reduction = entities_in_clusters - len(multi_clusters)
        entity_report = {
            "mode": "hi",
            "total_entities": total_entities,
            "type_b_excluded": len(type_b),
            "cluster_count": len(multi_clusters),
            "entities_in_clusters": entities_in_clusters,
            "expected_reduction": reduction,
            "expected_reduction_pct_of_total": round(reduction / total_entities * 100, 2),
            "sample_clusters": [sorted(m, key=len) for m in list(multi_clusters.values())[:15]],
            "warning": "의도적 과다 병합 — 오탐(예: 서울/경찰/경찰청) 포함, 고치지 않음. 정규화 효과의 상한 측정용.",
        }
    elif mode == "mid":
        type_b, multi_clusters = find_entity_clusters_mid(entities, max_cluster_size=max_cluster_size)
        entities_in_clusters = sum(len(m) for m in multi_clusters.values())
        reduction = entities_in_clusters - len(multi_clusters)
        entity_report = {
            "mode": "mid",
            "max_cluster_size": max_cluster_size,
            "total_entities": total_entities,
            "type_b_excluded": len(type_b),
            "cluster_count": len(multi_clusters),
            "entities_in_clusters": entities_in_clusters,
            "expected_reduction": reduction,
            "expected_reduction_pct_of_total": round(reduction / total_entities * 100, 2),
            "sample_clusters": [sorted(m, key=len) for m in list(multi_clusters.values())[:15]],
            "note": "사후 설계 (v2 결과 확인 후 추가). 판정은 components_per_doc이 아니라 isolated_doc_ratio 기준 — §6 'v2a-mid' 참고. 구현만, 아직 미실행.",
        }
    else:
        raise ValueError(f"unknown mode: {mode}")

    stem_map = build_relation_stem_map(rel_types)
    unique_types_before = len(set(rel_types))
    unique_stems_after = len(set(stem_map.values()))

    report = {
        "graph_path": graph_path,
        f"entity_v2a_{mode}": entity_report,
        "relation_v2b": {
            "total_types": unique_types_before,
            "estimated_after_stemming": unique_stems_after,
            "reduction_count": unique_types_before - unique_stems_after,
            "reduction_pct": round((unique_types_before - unique_stems_after) / unique_types_before * 100, 1),
        },
    }
    return report


def load_judged_pairs_alias_map(pairs_file):
    """사람이 판정한 쌍 목록을 읽어 '동일 대상'만 병합하는 alias map을 만든다.
    전이적 클러스터링을 쓰지 않는다 — 판정된 쌍끼리만 직접 연결(짧은 쪽을 canonical로).
    판정 파일 형식: [{"a": "...", "b": "...", "judgment": "동일 대상"|"다른 대상"}, ...]
    """
    with open(pairs_file, encoding="utf-8") as f:
        judged = json.load(f)

    valid_pairs = [p for p in judged if p.get("judgment") == "동일 대상"]
    alias = {}
    for p in valid_pairs:
        a, b = p["a"], p["b"]
        short, long_ = (a, b) if len(a) <= len(b) else (b, a)
        alias[long_] = short
    return alias, len(judged), len(valid_pairs)


def dry_run_report_from_pairs_file(graph_path, pairs_file):
    entities, rel_types = load_graph(graph_path)
    total_entities = len(set(entities))

    alias, total_judged, valid_count = load_judged_pairs_alias_map(pairs_file)
    absorbed = set(alias.keys())
    canonical = set(alias.values())

    report = {
        "graph_path": graph_path,
        "pairs_file": pairs_file,
        "entity_v2a_judged": {
            "mode": "judged_pairs",
            "total_entities": total_entities,
            "total_pairs_in_file": total_judged,
            "valid_pairs": valid_count,
            "canonical_short_forms": len(canonical),
            "absorbed_long_forms": len(absorbed),
            "expected_reduction": len(absorbed),
            "expected_reduction_pct_of_total": round(len(absorbed) / total_entities * 100, 2) if total_entities else 0,
            "note": "전이적 클러스터링 미사용 — 판정된 쌍만 직접 병합. 드라이런만, 그래프 재구축은 하지 않음.",
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, help="property_graph_store.json 경로")
    parser.add_argument("--mode", choices=["lo", "hi", "mid"], default="lo", help="lo=보수 규칙(하한), hi=substring 전이적 클러스터링(상한, 의도적 과다병합), mid=전이적 클러스터링+크기 상한(사후 설계, 아직 미실행)")
    parser.add_argument("--max-cluster-size", type=int, default=5, help="mid 모드 전용 클러스터 크기 상한 (기본 5, 클러스터 분포를 보고 조정 가능)")
    parser.add_argument("--pairs-file", default=None, help="사람이 판정한 쌍 목록(JSON). 지정 시 --mode 무시하고 판정된 '동일 대상' 쌍만 직접 병합(전이 클러스터링 없음)")
    parser.add_argument("--dry-run", action="store_true", default=True, help="기본값. 실제 그래프를 만들지 않고 예상 건수만 출력")
    args = parser.parse_args()

    if args.pairs_file:
        report = dry_run_report_from_pairs_file(args.graph, args.pairs_file)
    else:
        report = dry_run_report(args.graph, mode=args.mode, max_cluster_size=args.max_cluster_size)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
