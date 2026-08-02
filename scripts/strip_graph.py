"""
property_graph_store.json 에서 기사 본문·임베딩·LLM 생성 요약을 제거한 공개용
축약본(graph_public.json)을 만든다. 원본은 수정하지 않는다.

배경: property_graph_store.json 은 "엔티티/관계만 있는 파일"이 아니다.
text_chunk 라벨 노드 86개(v1)/10개(v0prime)에 기사 본문 전체(text 필드)와
1536차원 임베딩 벡터(embedding 필드)가 그대로 들어 있고, entity/relation/
text_chunk 노드 전부의 properties에 LLM이 생성한 요약(summary, 기사 파생
콘텐츠)이 중복 저장돼 있다. text_chunk 노드의 properties._node_content 는
llama-index 내부 직렬화 필드로 metadata(summary 포함)를 다시 한번 품고 있어
같이 제거해야 한다(내부 text 필드는 이미 빈 문자열이라 본문 중복은 아니다).

제거: text_chunk.text, text_chunk.embedding, text_chunk.properties._node_content,
      모든 노드/관계 properties.summary
보존: text_chunk 노드 자체(id, news_id, pub_date, category 등 메타데이터 —
      triplet_source_id 참조와 provenance 추적에 필요), entity 노드 전부,
      relations/triplets 전부, keywords 등 summary 외 메타데이터

사용:
    python scripts/strip_graph.py --input experiments/v1/property_graph_store.json \
        --output experiments/v1/graph_public.json
"""

import argparse
import copy
import datetime
import json
import os


def strip_graph(data):
    removed_fields = [
        "nodes[label=text_chunk].text",
        "nodes[label=text_chunk].embedding",
        "nodes[label=text_chunk].properties._node_content",
        "nodes[*].properties.summary",
        "relations[*].properties.summary",
    ]

    stripped = copy.deepcopy(data)

    for node in stripped.get("nodes", {}).values():
        if node.get("label") == "text_chunk":
            node["text"] = ""
            node["embedding"] = None
            node.get("properties", {}).pop("_node_content", None)
        node.get("properties", {}).pop("summary", None)

    for rel in stripped.get("relations", {}).values():
        rel.get("properties", {}).pop("summary", None)

    return stripped, removed_fields


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    original_size = os.path.getsize(args.input)

    stripped, removed_fields = strip_graph(data)

    stripped["_stripped"] = {
        "removed_fields": removed_fields,
        "original_size_bytes": original_size,
        "stripped_at": datetime.datetime.now().isoformat(),
        "script": "scripts/strip_graph.py",
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(stripped, f, ensure_ascii=False, indent=2)

    new_size = os.path.getsize(args.output)

    # 보존 검증: 노드/관계/트리플 개수가 정확히 같아야 한다
    orig_nodes = len(data.get("nodes", {}))
    orig_rels = len(data.get("relations", {}))
    orig_triplets = len(data.get("triplets", []))
    new_nodes = len(stripped.get("nodes", {})) - 0  # _stripped는 nodes 안에 없음
    new_rels = len(stripped.get("relations", {}))
    new_triplets = len(stripped.get("triplets", []))

    print(f"입력: {args.input} ({original_size:,} bytes)")
    print(f"출력: {args.output} ({new_size:,} bytes, {new_size/original_size*100:.1f}%)")
    print(f"노드: {orig_nodes} -> {new_nodes} ({'OK' if orig_nodes == new_nodes else 'MISMATCH'})")
    print(f"관계: {orig_rels} -> {new_rels} ({'OK' if orig_rels == new_rels else 'MISMATCH'})")
    print(f"트리플: {orig_triplets} -> {new_triplets} ({'OK' if orig_triplets == new_triplets else 'MISMATCH'})")

    if (orig_nodes, orig_rels, orig_triplets) != (new_nodes, new_rels, new_triplets):
        raise SystemExit("노드/관계/트리플 개수 불일치 — 축약 과정에서 데이터가 손실됐다. 중단.")


if __name__ == "__main__":
    main()
