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
    def __init__(self, model_name: str = "claude-sonnet-4-0"):
        self.llm = Anthropic(model=model_name, timeout=300.0)
        self.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        self.storage_manager = StorageManager()
        self.graph_analyzer = JITGraphAnalyzer(model_name=model_name)
        
    def query(self, query_str: str, top_k: int = 3, use_graph: bool = True):
        """
        하이브리드 검색 후 필요시 지식 그래프 딥다이브 분석 수행
        """
        # 1. 하이브리드 리트리버 설정
        storage_context = self.storage_manager.get_storage_context()
        vector_index = VectorStoreIndex.from_vector_store(
            self.storage_manager.vector_store, storage_context=storage_context
        )
        
        vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=top_k)
        
        # BM25용 노드 확보 (지속적인 BM25 인덱스 관리는 별도 필요, 여기선 임시)
        # 실제로는 Ingestion 단계에서 BM25 인덱스를 만들어두는 것이 정석입니다.
        nodes = vector_index.as_retriever(similarity_top_k=50).retrieve("all nodes")
        bm25_retriever = BM25Retriever.from_defaults(nodes=[n.node for n in nodes], similarity_top_k=top_k)
        
        hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)
        
        # 2. 결과 추출 및 리랭킹
        retrieved_nodes = hybrid_retriever.retrieve(query_str)
        reranker = LLMRerank(choice_batch_size=5, top_n=top_k, llm=self.llm)
        final_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle=QueryBundle(query_str))
        
        if use_graph and final_nodes:
            # 3. JIT 지식 그래프 분석 (선별된 문서 대상)
            response, _ = self.graph_analyzer.build_and_analyze(final_nodes, query_str)
            return response
        else:
            # 일반 RAG 합성
            query_engine = RetrieverQueryEngine.from_args(
                retriever=hybrid_retriever,
                node_postprocessors=[reranker]
            )
            return query_engine.query(query_str)
