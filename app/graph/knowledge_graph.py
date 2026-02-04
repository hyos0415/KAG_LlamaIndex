import os
from typing import List, Optional, Tuple
from llama_index.core import Document, PropertyGraphIndex, Settings, StorageContext
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.core.schema import NodeWithScore
from llama_index.llms.anthropic import Anthropic
from app.etl.storage import StorageManager

class KnowledgeGraphManager:
    def __init__(self, model_name: str = "claude-sonnet-4-0"):
        self.llm = Anthropic(model=model_name, timeout=300.0)
        Settings.llm = self.llm
        self.storage_manager = StorageManager()

    def sync_to_neo4j(self, retrieved_nodes: List[NodeWithScore]):
        """
        검색된 노드들로부터 지식을 추출하여 Neo4j에 영구 저장합니다.
        """
        print(f"🔗 {len(retrieved_nodes)}개 문서의 지식을 Neo4j로 동기화 중...")
        documents = [Document(text=n.node.get_content(), metadata=n.node.metadata) for n in retrieved_nodes]
        
        # Neo4j 스토리지 준비
        graph_store = self.storage_manager.get_neo4j_graph_store()
        
        # 추출기 설정
        extractor = SimpleLLMPathExtractor(llm=self.llm, num_workers=2)
        
        # Neo4j에 인덱스 생성 및 저장
        index = PropertyGraphIndex.from_documents(
            documents,
            property_graph_store=graph_store,
            kg_extractors=[extractor],
            show_progress=True
        )
        return index

    async def analyze_with_cypher(self, query_str: str):
        """
        [Direct Cypher Engine]
        라이브러리 추상화 계층을 우회하여, LLM이 직접 스키마를 파악하고 
        Cypher를 생성/실행하는 직관적인 분석을 수행합니다.
        """
        print(f"🧠 [Direct Analysis] Cypher 추론 중: {query_str}")
        
        # 1. Neo4j 스토어 및 스키마 확보
        graph_store = self.storage_manager.get_neo4j_graph_store()
        # 몽키패치된 aget_schema_str 사용
        schema_str = await graph_store.aget_schema_str()
        
        # 2. Cypher 생성 프롬프트 구성
        prompt = (
            "당신은 Neo4j Cypher 전문가입니다. 다음 [스키마 정보]를 바탕으로 [사용자 질문]에 답할 수 있는 Cypher 쿼리를 작성하세요.\n"
            "반드시 **Cypher 쿼리문만** 출력하고, 부연 설명이나 코드 블록(```)은 생략하세요.\n\n"
            f"### [스키마 정보]\n{schema_str}\n\n"
            f"### [사용자 질문]\n{query_str}\n\n"
            "Cypher 쿼리:"
        )
        
        # 3. LLM에게 쿼리 생성 요청
        llm_response = self.llm.complete(prompt)
        cypher_query = str(llm_response).strip().replace("```cypher", "").replace("```", "")
        
        print(f"📟 생성된 Cypher: {cypher_query}")
        
        # 4. Neo4j에 직접 쿼리 실행
        try:
            results = graph_store.query(cypher_query)
            
            if not results:
                return "🔍 조회 결과가 비어 있습니다. 그래프에 관련 정보가 아직 부족하거나 쿼리 조건이 너무 엄격할 수 있습니다."

            # 5. 결과 해석 요청 (LLM)
            interpret_prompt = (
                "당신은 뉴스 전문 분석가입니다. [조회 결과]로 제공된 그래프 데이터를 바탕으로 [사용자 질문]에 대한 심층적인 분석 답변을 작성하세요.\n"
                "데이터에 기반하여 인물, 기업 간의 연결 고리와 그 의미를 구체적으로 설명해 주세요.\n\n"
                f"### [사용자 질문]\n{query_str}\n\n"
                f"### [조회 결과]\n{results}\n\n"
                "분석 답변:"
            )
            final_answer = self.llm.complete(interpret_prompt)
            return final_answer
        except Exception as e:
            return f"❌ Cypher 실행 중 오류 발생: {str(e)}\n쿼리: {cypher_query}"
