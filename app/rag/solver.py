import asyncio
from typing import List, Optional
from llama_index.core import Settings, StorageContext, VectorStoreIndex, QueryBundle
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.openai import OpenAIEmbedding

from app.etl.storage import StorageManager
from app.graph.jit_builder import JITGraphAnalyzer
from app.rag.decomposer import QueryDecomposer

class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever: VectorIndexRetriever, bm25_retriever: BM25Retriever):
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle):
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        bm25_nodes = self._bm25_retriever.retrieve(query_bundle)

        # 단순 합집합 및 중복 제거
        all_nodes_dict = {n.node.node_id: n for n in vector_nodes}
        for n in bm25_nodes:
            if n.node.node_id not in all_nodes_dict:
                all_nodes_dict[n.node.node_id] = n
        
        return list(all_nodes_dict.values())

class NewsRAGSolver:
    def __init__(self, model_name: str = "claude-sonnet-4-0", db_url: Optional[str] = None, chroma_path: Optional[str] = None):
        self.llm = Anthropic(model=model_name, timeout=300.0)
        self.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        self.storage_manager = StorageManager(db_url=db_url, chroma_path=chroma_path)
        self.graph_analyzer = JITGraphAnalyzer(model_name=model_name)
        self.decomposer = QueryDecomposer(model_name=model_name)
        
    async def query(self, query_str: str, top_k: int = 4, use_graph: bool = True):
        """
        Elasticsearch 기반 하이브리드 검색 후 지식 그래프 딥다이브 분석 수행
        """
        # 1. 인덱스 로드 (Elasticsearch)
        vector_index = VectorStoreIndex.from_vector_store(
            self.storage_manager.es_store
        )
        
        # 2. Elasticsearch 하이브리드 리트리버 설정
        retriever = vector_index.as_retriever(
            similarity_top_k=top_k,
            vector_store_query_mode="hybrid",
            alpha=0.5
        )
        
        # 3. 결과 추출 및 리랭킹
        retrieved_nodes = retriever.retrieve(query_str)
        reranker = LLMRerank(choice_batch_size=5, top_n=top_k // 2 if top_k > 2 else top_k, llm=self.llm)
        final_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle=QueryBundle(query_str))
        
        if use_graph and final_nodes:
            # 4. JIT 지식 그래프 분석
            print(f"🧠 {len(final_nodes)}개 핵심 문서를 추출하여 지식 그래프 분석을 수행합니다...")
            response, _ = self.graph_analyzer.build_and_analyze(final_nodes, query_str)
            return response
        else:
            query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                node_postprocessors=[reranker]
            )
            return query_engine.query(query_str)

    async def recommend_similar_articles(self, article_content: str, top_k: int = 5):
        """
        입력된 기사 원문을 분석하여 다차원 검색을 수행하고 유사한 뉴스를 추천합니다.
        """
        # 1. 기사 분석 및 페이셋 도출
        analysis_result = await self.decomposer.decompose_article(article_content)
        
        # 2. 각 페이셋별로 하이브리드 검색 수행 (병렬 처리)
        search_tasks = []
        for facet in analysis_result.facets:
            search_tasks.append(self.query(facet.facet_query, top_k=3, use_graph=False))
        
        search_results = await asyncio.gather(*search_tasks)
        
        # 3. 결과 통합 (단순 합집합 + 리랭킹 유도)
        unique_nodes = {}
        for res in search_results:
            # RetrieverQueryEngine의 결과는 Response 객체이며, source_nodes에 노드들이 있음
            if hasattr(res, 'source_nodes'):
                for node_with_score in res.source_nodes:
                    node_id = node_with_score.node.node_id
                    if node_id not in unique_nodes:
                        unique_nodes[node_id] = node_with_score
        
        final_nodes = list(unique_nodes.values())
        
        # 4. 최종 추천 리스트 및 사유 생성
        context_docs = "\n\n".join([
            f"[추천 기사: {n.node.metadata.get('title')}]\n내용 요약: {n.node.metadata.get('summary', '요약 없음')}"
            for n in final_nodes[:top_k]
        ])
        
        recommendation_prompt = (
            f"사용자가 작성한 기사의 핵심 주제는 다음과 같습니다: {analysis_result.core_summary}\n\n"
            f"다음은 데이터베이스에서 검색된 관련성 높은 기사들입니다:\n{context_docs}\n\n"
            f"위 기사들 중에서 사용자의 기사와 가장 유사하거나 상호 보완적인 뉴스 3개를 선정하고, "
            f"각각 왜 추천하는지 '의미적 유사성' 관점에서 설명해줘."
        )
        
        final_response = self.llm.complete(recommendation_prompt)
        return final_response
