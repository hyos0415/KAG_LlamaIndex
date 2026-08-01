"""
2단계 그래프 구축 스크립트 (docs/CONTEXT.md §6 "추출 코드 경로" 참고).

Stage 1: 청킹 + 메타데이터 추출 (40건 전체, 1회만) -> tests/fixtures/chunks_40.json
Stage 2: Stage 1 산출물을 읽어 트리플 추출 + PropertyGraphIndex 구축 + persist

Stage 2는 jit_builder.py(JITGraphAnalyzer.build_and_analyze)의 그래프 구축 단계만
재사용한다 (문서화 -> SimpleLLMPathExtractor -> PropertyGraphIndex). 육각형 리포트
쿼리 단계는 포함하지 않는다.

사용 예:
    python scripts/build_index.py stage1 \
        --input-dir result/airflow --output tests/fixtures/chunks_40.json

    python scripts/build_index.py stage2 \
        --chunks tests/fixtures/chunks_40.json \
        --news-ids 11952015,11952014,11952002,11951999,11951990 \
        --output-dir experiments/v0prime
"""

import argparse
import asyncio
import glob
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from llama_index.core import Settings
from llama_index.core.schema import TextNode, MetadataMode
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.indices.property_graph import PropertyGraphIndex, SimpleLLMPathExtractor
from llama_index.core.graph_stores.types import EntityNode, Relation, KG_NODES_KEY, KG_RELATIONS_KEY
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.openai import OpenAIEmbedding

from app.etl.enricher import NewsMetadata

MODEL_ID = "claude-sonnet-4-5-20250929"
TRIPLE_CACHE_PATH = "experiments/shared/triple_cache.json"


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

def load_unique_articles(input_dir):
    """result/airflow/*.json 을 읽어 news_id 기준 중복 제거, 정렬된 리스트 반환."""
    articles = {}
    for fp in sorted(glob.glob(os.path.join(input_dir, "*.json"))):
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        for a in data:
            aid = a["id"]
            if aid not in articles:
                articles[aid] = a
    return [articles[k] for k in sorted(articles.keys(), key=int)]


async def extract_metadata(llm, content):
    prompt_template_str = (
        "다음 뉴스 기사 내용을 분석하여 지정된 형식의 메타데이터를 추출해주세요.\n"
        "기사 내용: {content}\n"
    )
    program = LLMTextCompletionProgram.from_defaults(
        output_cls=NewsMetadata,
        prompt_template_str=prompt_template_str,
        llm=llm,
        verbose=False,
    )
    return program(content=content)


async def run_stage1(input_dir, output_path):
    articles = load_unique_articles(input_dir)
    print(f"고유 기사 {len(articles)}건 로드 (from {input_dir})")

    llm = Anthropic(model=MODEL_ID, timeout=300.0)
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm = llm
    Settings.embed_model = embed_model

    node_parser = SemanticSplitterNodeParser(
        buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
    )

    chunk_records = []
    per_article_counts = {}

    for article in articles:
        news_id = article["id"]
        content = article.get("content", "")
        if not content:
            print(f"  ! news_id={news_id}: content 없음, 건너뜀")
            continue

        print(f"  - news_id={news_id} 메타데이터 추출 중...")
        metadata_obj = await extract_metadata(llm, content)

        from llama_index.core import Document
        doc = Document(
            text=content,
            metadata={
                "title": article.get("title", "N/A"),
                "url": article.get("url", "N/A"),
                "pub_date": article.get("pub_date", "N/A"),
                "news_id": news_id,
                "category": metadata_obj.category,
                "sentiment": metadata_obj.sentiment,
                "keywords": ", ".join(metadata_obj.keywords),
                "summary": metadata_obj.summary,
            },
        )

        nodes = node_parser.get_nodes_from_documents([doc])
        per_article_counts[news_id] = len(nodes)

        for idx, node in enumerate(nodes):
            node_id = f"{news_id}-chunk{idx}"
            chunk_records.append(
                {
                    "node_id": node_id,
                    "news_id": news_id,
                    "chunk_index": idx,
                    "text": node.get_content(),
                    "metadata": dict(node.metadata),
                }
            )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunk_records, f, ensure_ascii=False, indent=2)

    print()
    print(f"총 청크 수: {len(chunk_records)}")
    print(f"기사당 청크 수 분포: {sorted(per_article_counts.values())}")
    print(f"저장 위치: {output_path}")

    return {
        "article_count": len(articles),
        "chunk_count": len(chunk_records),
        "per_article_chunk_counts": per_article_counts,
        "output_path": output_path,
    }


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

class CachingLLMPathExtractor(SimpleLLMPathExtractor):
    """SimpleLLMPathExtractor 와 완전히 동일한 트리플 파싱 로직을 쓰되,
    node_id 가 캐시에 있으면 LLM 호출 없이 캐시된 (subj, rel, obj) 튜플을 재사용한다.

    docs/CONTEXT.md §6 "2단계 분리" 트리플 캐싱 설계 참고.
    """

    def __init__(self, *args, triple_cache=None, cache_hits=None, cache_misses=None, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_triple_cache", triple_cache if triple_cache is not None else {})
        object.__setattr__(self, "_cache_hits", cache_hits if cache_hits is not None else [])
        object.__setattr__(self, "_cache_misses", cache_misses if cache_misses is not None else [])

    async def _aextract(self, node):
        cache = self._triple_cache

        if node.id_ in cache:
            self._cache_hits.append(node.id_)
            entry = cache[node.id_]
            # 구버전 캐시(순수 트리플 리스트) 호환: provenance 없이 읽힌 값은 그대로 사용
            triples = entry["triples"] if isinstance(entry, dict) else entry
        else:
            self._cache_misses.append(node.id_)
            text = node.get_content(metadata_mode=MetadataMode.LLM)
            try:
                llm_response = await self.llm.apredict(
                    self.extract_prompt,
                    text=text,
                    max_knowledge_triplets=self.max_paths_per_chunk,
                )
                triples = self.parse_fn(llm_response)
            except ValueError:
                triples = []
            import datetime
            cache[node.id_] = {
                # self.llm.model은 요청 모델 문자열이다. 핀 고정 스냅샷은
                # 실제 API 응답 모델과 항상 일치함을 별도로 확인했다(§3 발견 11 검증 절차 참고).
                "model_id": self.llm.model,
                "extracted_at": datetime.datetime.now().isoformat(),
                "provenance": "measured",
                "triples": [list(t) for t in triples],
            }

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])

        metadata = node.metadata.copy()
        for subj, rel, obj in triples:
            subj_node = EntityNode(name=subj, properties=metadata)
            obj_node = EntityNode(name=obj, properties=metadata)
            rel_node = Relation(
                label=rel,
                source_id=subj_node.id,
                target_id=obj_node.id,
                properties=metadata,
            )
            existing_nodes.extend([subj_node, obj_node])
            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node


def instrument_anthropic_client(llm, raw_dir):
    """llm._aclient.messages.create 를 감싸 모든 호출의 원시 응답을 raw_dir 에 저장.

    1순위: response.stop_reason 으로 절단 여부 판정
    2순위: stop_reason 이 없으면 output_tokens >= 510 으로 대체 추정
    3순위: 순위와 무관하게 모든 호출의 원시 응답 텍스트를 항상 저장
    """
    os.makedirs(raw_dir, exist_ok=True)
    original_create = llm._aclient.messages.create
    call_log = []

    async def wrapped_create(*args, **kwargs):
        response = await original_create(*args, **kwargs)

        stop_reason = getattr(response, "stop_reason", None)
        usage = getattr(response, "usage", None)
        output_tokens = getattr(usage, "output_tokens", None) if usage else None
        input_tokens = getattr(usage, "input_tokens", None) if usage else None
        model_id = getattr(response, "model", None)
        content_text = "".join(
            getattr(block, "text", "") for block in getattr(response, "content", [])
        )

        idx = len(call_log) + 1
        record = {
            "call_index": idx,
            "model": model_id,
            "stop_reason": stop_reason,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "content_text": content_text,
        }
        call_log.append(record)

        with open(os.path.join(raw_dir, f"call_{idx:04d}.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        return response

    llm._aclient.messages.create = wrapped_create
    return call_log


def summarize_calls(call_log):
    total = len(call_log)
    if total == 0:
        return {"num_calls": 0}

    has_stop_reason = all(c["stop_reason"] is not None for c in call_log)
    if has_stop_reason:
        method = "stop_reason"
        truncated = sum(1 for c in call_log if c["stop_reason"] == "max_tokens")
    else:
        method = "output_tokens_ge_510"
        truncated = sum(
            1 for c in call_log if (c["output_tokens"] or 0) >= 510
        )

    out_tokens = [c["output_tokens"] for c in call_log if c["output_tokens"] is not None]
    model_ids_seen = sorted({c["model"] for c in call_log if c["model"]})

    token_stats = {}
    if out_tokens:
        sorted_tokens = sorted(out_tokens)
        token_stats = {
            "min": min(sorted_tokens),
            "max": max(sorted_tokens),
            "mean": round(statistics.mean(sorted_tokens), 1),
            "median": statistics.median(sorted_tokens),
            "p90": sorted_tokens[int(0.9 * (len(sorted_tokens) - 1))],
        }

    return {
        "num_calls": total,
        "model_ids_seen": model_ids_seen,
        "truncation_method": method,
        "truncated_count": truncated,
        "truncated_ratio": round(truncated / total, 4),
        "output_tokens": token_stats,
    }


def load_triple_cache(use_cache):
    if not use_cache:
        return {}
    if os.path.exists(TRIPLE_CACHE_PATH):
        with open(TRIPLE_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_triple_cache(cache, use_cache):
    if not use_cache:
        return
    os.makedirs(os.path.dirname(TRIPLE_CACHE_PATH), exist_ok=True)
    with open(TRIPLE_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


async def run_stage2(chunks_path, news_ids, output_dir, use_cache=True):
    with open(chunks_path, encoding="utf-8") as f:
        all_chunks = json.load(f)

    if news_ids:
        wanted = set(news_ids)
        selected = [c for c in all_chunks if c["news_id"] in wanted]
        missing = wanted - {c["news_id"] for c in selected}
        if missing:
            print(f"경고: chunks 파일에 없는 news_id: {sorted(missing)}")
    else:
        selected = all_chunks

    print(f"선택된 청크 수: {len(selected)} (news_id {len(set(c['news_id'] for c in selected))}건)")
    print(f"캐시 사용: {use_cache}")

    nodes = [
        TextNode(id_=c["node_id"], text=c["text"], metadata=c["metadata"])
        for c in selected
    ]

    llm = Anthropic(model=MODEL_ID, timeout=300.0)
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm = llm
    Settings.embed_model = embed_model

    raw_dir = os.path.join(output_dir, "raw_completions")
    call_log = instrument_anthropic_client(llm, raw_dir)

    triple_cache = load_triple_cache(use_cache)
    cache_hits, cache_misses = [], []
    extractor = CachingLLMPathExtractor(
        llm=llm,
        num_workers=2,
        triple_cache=triple_cache,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
    )

    start_time = time.monotonic()
    index = PropertyGraphIndex(
        nodes=nodes,
        kg_extractors=[extractor],
        show_progress=True,
    )
    elapsed_seconds = round(time.monotonic() - start_time, 1)

    save_triple_cache(triple_cache, use_cache)

    os.makedirs(output_dir, exist_ok=True)
    index.storage_context.persist(persist_dir=output_dir)

    run_metrics = {
        "chunk_count_in": len(nodes),
        "news_id_count": len(set(c["news_id"] for c in selected)),
        "cache_enabled": use_cache,
        "cache_hits": len(cache_hits),
        "cache_misses": len(cache_misses),
        "elapsed_seconds": elapsed_seconds,
        **summarize_calls(call_log),
    }
    with open(os.path.join(output_dir, "run_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(run_metrics, f, ensure_ascii=False, indent=2)

    print()
    print(json.dumps(run_metrics, ensure_ascii=False, indent=2))
    print(f"저장 위치: {output_dir}")

    return run_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    p1 = sub.add_parser("stage1", help="청킹 + 메타데이터 추출 (40건 전체)")
    p1.add_argument("--input-dir", default="result/airflow")
    p1.add_argument("--output", default="tests/fixtures/chunks_40.json")

    p2 = sub.add_parser("stage2", help="트리플 추출 + 그래프 구축 + persist")
    p2.add_argument("--chunks", default="tests/fixtures/chunks_40.json")
    p2.add_argument("--news-ids", default=None, help="쉼표로 구분된 news_id 목록. 생략 시 전체")
    p2.add_argument("--output-dir", required=True)
    p2.add_argument("--no-cache", action="store_true", help="트리플 캐시를 쓰지 않고 전량 재추출 (검증용)")

    args = parser.parse_args()

    if args.stage == "stage1":
        asyncio.run(run_stage1(args.input_dir, args.output))
    elif args.stage == "stage2":
        news_ids = args.news_ids.split(",") if args.news_ids else None
        asyncio.run(run_stage2(args.chunks, news_ids, args.output_dir, use_cache=not args.no_cache))


if __name__ == "__main__":
    main()
