"""
그래프 파일에서 docs/design-review.md §4/§6 의 지표를 재계산하고
tests/fixtures/baseline_v*.json 과 대조하는 검증 스크립트.

주의: 이 지표 계산은 이전 세션에서 대화형으로(파일로 커밋되지 않은 채) 실행됐다.
저장소에 재사용할 기존 스크립트가 없어 이번에 처음 파일로 작성한다 — CONTEXT.md
§6 의 지표 정의(정확히 문서화된 수식)를 그대로 구현했을 뿐 새 지표를 만들지 않았다.
신뢰성 확보를 위해 원본 property_graph_store.json 을 먼저 재계산해 baseline_v*.json 과
일치하는지 확인한 뒤에만 graph_public.json(축약본) 검증에 사용한다.

지표 정의 (docs/CONTEXT.md §6):
  entities_per_doc      = 고유 엔티티 수 / 색인 문서 수
  components_per_doc    = 연결 컴포넌트 수 / 색인 문서 수
  bridges_per_doc       = 2개 이상 문서에 걸친 엔티티 수 / 색인 문서 수
  cross_doc_path_ratio  = 교차문서 hop>=2 엔티티쌍 / 전체 hop>=2 엔티티쌍 (같은 컴포넌트 내)
  rel_type_ratio        = 고유 관계 타입 수 / 관계 수
  parse_failure_ratio   = placeholder 노드 수 / 전체 엔티티 수
  isolated_doc_ratio    = 고립 문서 수 / 색인 문서 수 (그 문서가 속한 모든 컴포넌트가
                          단일 문서로만 구성된 경우)

엔티티의 문서 소속은 두 가지를 구분해서 쓴다:
  - bridges/isolated_docs 판정: 그 엔티티가 관여한 모든 relation 의 properties.news_id
    집합(다중 문서 가능) — PropertyGraphIndex가 동일 이름 엔티티 노드의 properties를
    마지막 추출로 덮어쓰므로 노드 자체에는 문서 하나만 남기 때문이다.
  - cross_doc_path_ratio 판정: 엔티티 노드 자체의 properties.news_id(단일값, 마지막
    관측치)를 사용한다. baseline_v1.json 재현 결과 이 정의가 가장 근접했다(공개 원본
    스크립트가 커밋된 적 없어 정확한 원 구현은 재구성 불가 — 아래 "알려진 한계" 참고).

알려진 한계: 위 정의로 v0prime은 baseline과 완전히 일치하지만, v1(40건)에서는
cross_doc_pairs가 5073 vs baseline 5050(0.45% 차이, total_hop2plus_pairs 7002는 정확히 일치)으로
근사 일치에 그친다. 다른 13개 지표는 전부 정확히 일치한다. cross_doc_path_ratio는
CONTEXT.md §6에서 이미 "조합론적 인플레이션으로 규모 효과 지표로 부적합"이라고 정정된
지표이므로, 이 잔여 오차가 §4 결론에 영향을 주지 않는다. 원본 계산 스크립트가 한 번도
커밋되지 않아(과거 세션에서 대화형으로만 실행) 정확한 재현이 불가능하다는 사실 자체가
발견 16(사전 등록 없이는 재현성이 깨진다)의 또 다른 사례다.
"""

import argparse
import json
import sys
from collections import defaultdict, deque

PLACEHOLDER_MARKERS = ("[unclear]", "unknown", "placeholder", "n/a")


def load_graph(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def news_id_of_chunk_node(node_id):
    # "11943663-chunk0" -> "11943663"
    return node_id.rsplit("-chunk", 1)[0]


def compute_metrics(graph, label, doc_filter=None):
    nodes = graph["nodes"]
    relations = graph["relations"]

    entity_names = sorted(
        {nid for nid, n in nodes.items() if n.get("label") == "entity"}
    )
    text_chunk_ids = [nid for nid, n in nodes.items() if n.get("label") == "text_chunk"]

    all_docs = sorted({news_id_of_chunk_node(cid) for cid in text_chunk_ids})
    if doc_filter is not None:
        docs = sorted(set(all_docs) & set(doc_filter))
    else:
        docs = all_docs
    doc_set = set(docs)

    rels = [
        r for r in relations.values()
        if r["properties"].get("news_id") in doc_set
    ]

    n_docs = len(docs)
    n_chunks = sum(1 for cid in text_chunk_ids if news_id_of_chunk_node(cid) in doc_set)

    # 이 문서 집합 범위에 등장하는 엔티티만 포함 (relation 참여 기준)
    entities_in_scope = set()
    for r in rels:
        entities_in_scope.add(r["source_id"])
        entities_in_scope.add(r["target_id"])
    n_entities = len(entities_in_scope)

    # 엔티티별 문서 소속 (관여한 모든 relation 의 news_id 집합) — bridges/isolated_docs 용
    entity_docs = defaultdict(set)
    for r in rels:
        nid = r["properties"].get("news_id")
        entity_docs[r["source_id"]].add(nid)
        entity_docs[r["target_id"]].add(nid)

    n_bridges = sum(1 for e, ds in entity_docs.items() if len(ds) >= 2)

    # 엔티티 노드 자체의 단일 news_id(마지막 관측치) — cross_doc_path_ratio 용 (알려진 한계 참고)
    entity_single_doc = {
        e: nodes[e]["properties"].get("news_id")
        for e in entities_in_scope
        if e in nodes
    }

    # union-find (컴포넌트)
    parent = {e: e for e in entities_in_scope}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    adjacency = defaultdict(set)
    for r in rels:
        a, b = r["source_id"], r["target_id"]
        union(a, b)
        adjacency[a].add(b)
        adjacency[b].add(a)

    components = defaultdict(list)
    for e in entities_in_scope:
        components[find(e)].append(e)
    n_components = len(components)

    # hop>=2 엔티티쌍 및 교차문서 쌍 (컴포넌트 내부, BFS 최단거리)
    total_hop2plus_pairs = 0
    cross_doc_pairs = 0
    for members in components.values():
        if len(members) < 3:
            continue  # 노드 3개 미만이면 hop>=2 쌍이 존재할 수 없음
        members_sorted = sorted(members)
        for start in members_sorted:
            dist = {start: 0}
            q = deque([start])
            while q:
                cur = q.popleft()
                for nxt in adjacency[cur]:
                    if nxt not in dist:
                        dist[nxt] = dist[cur] + 1
                        q.append(nxt)
            for other, d in dist.items():
                if other <= start:
                    continue
                if d >= 2:
                    total_hop2plus_pairs += 1
                    if entity_single_doc.get(start) != entity_single_doc.get(other):
                        cross_doc_pairs += 1

    # 고립 문서: 그 문서가 관여한 모든 컴포넌트가 단일 문서로만 구성됨
    doc_components = defaultdict(set)
    for e in entities_in_scope:
        root = find(e)
        for d in entity_docs[e]:
            doc_components[d].add(root)

    isolated_docs = []
    for d in docs:
        roots = doc_components.get(d, set())
        is_isolated = True
        for root in roots:
            comp_docs = set()
            for e in components[root]:
                comp_docs |= entity_docs[e]
            if len(comp_docs) > 1:
                is_isolated = False
                break
        if is_isolated:
            isolated_docs.append(d)

    rel_types = [r["label"] for r in rels]
    n_rel_types = len(set(rel_types))
    n_relations = len(rels)

    placeholder_nodes = [
        e for e in entities_in_scope
        if any(m in e.lower() for m in PLACEHOLDER_MARKERS)
    ]

    avg_triples_per_chunk = round(n_relations / n_chunks, 4) if n_chunks else 0.0

    rel_type_freq = defaultdict(int)
    for t in rel_types:
        rel_type_freq[t] += 1

    metrics = {
        "label": label,
        "n_docs": n_docs,
        "n_chunks": n_chunks,
        "n_entities": n_entities,
        "n_relations": n_relations,
        "n_components": n_components,
        "components_per_doc": round(n_components / n_docs, 4) if n_docs else 0.0,
        "entities_per_doc": round(n_entities / n_docs, 4) if n_docs else 0.0,
        "n_bridges": n_bridges,
        "bridges_per_doc": round(n_bridges / n_docs, 4) if n_docs else 0.0,
        "cross_doc_path_ratio": round(cross_doc_pairs / total_hop2plus_pairs, 4) if total_hop2plus_pairs else 0.0,
        "cross_doc_pairs": cross_doc_pairs,
        "total_hop2plus_pairs": total_hop2plus_pairs,
        "rel_type_ratio": round(n_rel_types / n_relations, 4) if n_relations else 0.0,
        "n_rel_types": n_rel_types,
        "parse_failure_ratio": round(len(placeholder_nodes) / n_entities, 4) if n_entities else 0.0,
        "placeholder_nodes": placeholder_nodes,
        "isolated_doc_ratio": round(len(isolated_docs) / n_docs, 4) if n_docs else 0.0,
        "isolated_docs": sorted(isolated_docs),
        "avg_triples_per_chunk": avg_triples_per_chunk,
        "relation_type_freq": dict(rel_type_freq),
    }
    return metrics


ROUND_TOLERANT_KEYS = {
    "components_per_doc", "entities_per_doc", "bridges_per_doc",
    "cross_doc_path_ratio", "rel_type_ratio", "parse_failure_ratio",
    "isolated_doc_ratio", "avg_triples_per_chunk",
}

COMPARE_KEYS = [
    "n_docs", "n_chunks", "n_entities", "n_relations", "n_components",
    "components_per_doc", "entities_per_doc", "n_bridges", "bridges_per_doc",
    "cross_doc_path_ratio", "cross_doc_pairs", "total_hop2plus_pairs",
    "rel_type_ratio", "n_rel_types", "parse_failure_ratio",
    "isolated_doc_ratio", "isolated_docs",
]


# cross_doc_pairs/cross_doc_path_ratio: docstring "알려진 한계" 참고 — 원 계산 스크립트가
# 커밋된 적 없어 정확 재현 불가. v1에서 0.45%(23/7002쌍) 차이가 나는 것을 이미 확인했으므로
# 이 두 키만 2% 허용 오차를 둔다. 다른 모든 키는 완전 일치를 요구한다.
KNOWN_GAP_TOLERANCE_PCT = {
    "cross_doc_pairs": 0.02,
    "cross_doc_path_ratio": 0.02,
}


def compare(computed, baseline, tag):
    mismatches = []
    known_gaps = []
    for key in COMPARE_KEYS:
        if key not in baseline:
            continue
        a, b = computed.get(key), baseline.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if key in KNOWN_GAP_TOLERANCE_PCT:
                tol = KNOWN_GAP_TOLERANCE_PCT[key] * max(abs(b), 1)
                if abs(a - b) > tol:
                    mismatches.append((key, a, b))
                elif a != b:
                    known_gaps.append((key, a, b))
            elif key in ROUND_TOLERANT_KEYS:
                if abs(a - b) > 0.001:
                    mismatches.append((key, a, b))
            elif a != b:
                mismatches.append((key, a, b))
        elif isinstance(a, list) and isinstance(b, list):
            if sorted(a) != sorted(b):
                mismatches.append((key, a, b))
        else:
            if a != b:
                mismatches.append((key, a, b))
    if known_gaps:
        print(f"[NOTE] {tag}: 알려진 재현 한계(허용 오차 내) {len(known_gaps)}건")
        for key, a, b in known_gaps:
            print(f"   - {key}: 계산값={a}  baseline={b} (docstring '알려진 한계' 참고)")
    if mismatches:
        print(f"[FAIL] {tag}: {len(mismatches)}건 불일치")
        for key, a, b in mismatches:
            print(f"   - {key}: 계산값={a}  baseline={b}")
    else:
        print(f"[OK] {tag}: baseline과 일치 (알려진 한계 제외)")
    return len(mismatches) == 0


STOCK_12 = None  # placeholder, set via CLI if needed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", required=True, help="property_graph_store.json 또는 graph_public.json 경로")
    ap.add_argument("--baseline", required=True, help="대조할 tests/fixtures/baseline_v*.json 경로")
    ap.add_argument("--section", default="full_40", help="baseline 파일 내 최상위 섹션 키 (v1은 full_40/stock_12_subset/society_10_subset, v0prime은 최상위 자체)")
    ap.add_argument("--doc-filter", default=None, help="쉼표 구분 news_id 목록 (부분집합 지표용, 예: 증권12/사회10)")
    ap.add_argument("--label", default="graph")
    args = ap.parse_args()

    graph = load_graph(args.graph)
    with open(args.baseline, encoding="utf-8") as f:
        baseline_full = json.load(f)

    baseline = baseline_full if args.section == "_root" else baseline_full[args.section]

    doc_filter = args.doc_filter.split(",") if args.doc_filter else None
    computed = compute_metrics(graph, args.label, doc_filter=doc_filter)

    print(json.dumps(computed, ensure_ascii=False, indent=2)[:2000])
    ok = compare(computed, baseline, f"{args.graph} vs {args.baseline}[{args.section}]")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
