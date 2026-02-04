from typing import List, Dict, Any
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document as LCDocument
from langchain_core.retrievers import BaseRetriever

class NewsHybridRetriever(BaseRetriever):
    sparse_retriever: BaseRetriever
    dense_retriever: BaseRetriever
    k: int = 5
    c: int = 60  # RRF constant

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[LCDocument]:
        # 1. invoke를 사용하여 검색 수행
        sparse_docs = self.sparse_retriever.invoke(query)
        dense_docs = self.dense_retriever.invoke(query)

        # 2. RRF (Reciprocal Rank Fusion) 알고리즘 적용
        doc_scores = {}
        
        # Sparse 결과 점수화
        for rank, doc in enumerate(sparse_docs):
            doc_id = doc.metadata.get("news_id", doc.page_content[:50])
            score = 1.0 / (rank + self.c)
            doc_scores[doc_id] = {"score": score, "doc": doc}

        # Dense 결과 점수화 및 통합
        for rank, doc in enumerate(dense_docs):
            doc_id = doc.metadata.get("news_id", doc.page_content[:50])
            score = 1.0 / (rank + self.c)
            if doc_id in doc_scores:
                doc_scores[doc_id]["score"] += score
            else:
                doc_scores[doc_id] = {"score": score, "doc": doc}

        # 3. 점수 순으로 정렬 및 k개 반환
        sorted_results = sorted(
            doc_scores.values(), key=lambda x: x["score"], reverse=True
        )
        
        return [res["doc"] for res in sorted_results[:self.k]]
