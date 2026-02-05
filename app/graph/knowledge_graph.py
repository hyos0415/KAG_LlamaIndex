import os
from typing import List, Optional, Tuple
from llama_index.core import Document, PropertyGraphIndex, Settings, StorageContext
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.core.schema import NodeWithScore
from llama_index.llms.anthropic import Anthropic
from app.etl.storage import StorageManager
import re

class KnowledgeGraphManager:
    def __init__(self, model_name: str = "claude-sonnet-4-0"):
        self.llm = Anthropic(model=model_name, timeout=300.0)
        Settings.llm = self.llm
        self.storage_manager = StorageManager()

    def sync_to_neo4j(self, retrieved_nodes: List[NodeWithScore], label: str = "Article"):
        """
        검색된 노드들로부터 지식을 추출하여 Neo4j에 저장합니다.
        라벨을 통해 '검증된 기사(Article)'와 '사용자 초안(Draft)'을 구분합니다.
        """
        print(f"🔗 {len(retrieved_nodes)}개 문서의 지식을 Neo4j({label})로 동기화 중...")
        documents = [Document(text=n.node.get_content(), metadata={**n.node.metadata, "type": label}) for n in retrieved_nodes]
        
        # Neo4j 스토리지 준비
        graph_store = self.storage_manager.get_neo4j_graph_store()
        
        # 추출기 설정 (노드에 라벨 부여를 위한 프롬프트 가이드 포함 가능 - 여기선 메타데이터 활용)
        extractor = SimpleLLMPathExtractor(llm=self.llm, num_workers=2)
        
        # Neo4j에 저장
        index = PropertyGraphIndex.from_documents(
            documents,
            property_graph_store=graph_store,
            kg_extractors=[extractor],
            show_progress=True
        )
        return index

    async def calculate_hexagonal_metrics(self, query_str: str, results: list) -> dict:
        """
        Neo4j 데이터를 정량적으로 분석하여 6가지 육각형 지표를 산출합니다.
        """
        print("📊 [Metric Calculation] 육각형 지표 산출 중...")
        graph_store = self.storage_manager.get_neo4j_graph_store()
        
        metrics = {
            "connectivity": 0,
            "factuality": 0,
            "depth": 0,
            "originality": 0,
            "density": 0,
            "insight": 0
        }

        try:
            # 1. 연결성 (Connectivity): 결과 노드들의 평균 차수(Degree)
            # 결과 노드 ID 추출 (결과가 노드 리스트라고 가정하거나 쿼리 결과에서 추출)
            connectivity_query = "MATCH (n)-[r]-() RETURN count(r) as connections, count(distinct n) as nodes"
            conn_res = graph_store.query(connectivity_query)
            if conn_res and conn_res[0]['nodes'] > 0:
                # 전체 그래프의 평균 연결 밀도를 기반으로 정규화 (0~100)
                avg_conn = conn_res[0]['connections'] / conn_res[0]['nodes']
                metrics["connectivity"] = min(100, int(avg_conn * 10))

            # 2. 사실성 (Factuality): LLM이 결과의 근거가 얼마나 명확한지 평가
            fact_prompt = (
                "제공된 [데이터]가 얼마나 구체적이고 상호 검증 가능한 사실(엔티티 간 관계)을 포함하고 있는지 0~100점으로 평가하세요.\n"
                "숫자만 출력하세요.\n"
                f"데이터: {results}"
            )
            metrics["factuality"] = int(str(self.llm.complete(fact_prompt)).strip() or 0)

            # 3. 심층성 (Depth): 2-hop 이상의 관계 존재 여부
            depth_query = "MATCH path=(n)-[*2..3]-(m) RETURN count(path) as paths LIMIT 1"
            depth_res = graph_store.query(depth_query)
            metrics["depth"] = min(100, int(depth_res[0]['paths'] * 5)) if depth_res else 0

            # 4. 독창성 (Originality): 기존 지식 대비 신규 정보의 가치 평가 (LLM)
            originality_prompt = (
                "제공된 [결과]가 일반적인 상식이나 기존 보도 내용을 넘어 얼마나 고유하고 구체적인 새로운 정보(신규 인물, 특정 수치, 미시적 사건 등)를 담고 있는지 0~100점으로 평가하세요.\n"
                "숫자만 출력하세요.\n"
                f"결과: {results}"
            )
            metrics["originality"] = int(str(self.llm.complete(originality_prompt)).strip() or 0)

            # 5. 정보 밀도 (Density): 결과 내 관계 수 / 노드 수
            if results and isinstance(results, list):
                rel_count = len(results)
                node_ids = set()
                for r in results:
                    if isinstance(r, dict):
                        node_ids.update(r.values())
                metrics["density"] = min(100, int((rel_count / max(1, len(node_ids))) * 20))

            # 6. 주제 통찰 (Insight): 질문과의 관련성 평가 (LLM)
            insight_prompt = (
                "사용자의 질문에 대해 제공된 결과가 얼마나 핵심적인 통찰을 주는지 0~100점으로 평가하세요.\n"
                "숫자만 출력하세요.\n"
                f"질문: {query_str}\n"
                f"결과: {results}"
            )
            metrics["insight"] = int(str(self.llm.complete(insight_prompt)).strip() or 0)

        except Exception as e:
            print(f"⚠️ 지표 산출 중 오류: {e}")
            
        return metrics

    async def analyze_with_cypher(self, query_str: str):
        """
        [Direct Cypher Engine + Hexagonal Analysis 2.0]
        """
        print(f"🧠 [Direct Analysis] Cypher 추론 중: {query_str}")
        
        graph_store = self.storage_manager.get_neo4j_graph_store()
        schema_str = await graph_store.aget_schema_str()
        
        prompt = (
            "당신은 Neo4j Cypher 전문가입니다. 다음 [스키마 정보]를 바탕으로 [사용자 질문]에 답할 수 있는 Cypher 쿼리를 작성하세요.\n"
            "**주의**: 현재 데이터셋이 작으므로 `count(*) > 1`과 같은 엄격한 필터링은 피하고, 최대한 많은 관계를 보여줄 수 있도록 작성하세요.\n"
            "반드시 **Cypher 쿼리문만** 출력하고, 부연 설명이나 코드 블록(```)은 생략하세요.\n\n"
            f"### [스키마 정보]\n{schema_str}\n\n"
            f"### [사용자 질문]\n{query_str}\n\n"
            "Cypher 쿼리:"
        )
        
        llm_response = self.llm.complete(prompt)
        cypher_query = str(llm_response).strip().replace("```cypher", "").replace("```", "")
        print(f"📟 생성된 Cypher: {cypher_query}")
        
        try:
            results = graph_store.query(cypher_query)
            
            # [핵심] 육각형 지표 산출 (결과가 없더라도 그래프 전체 통계로 대체 가능하도록 로직 내에서 처리)
            metrics = await self.calculate_hexagonal_metrics(query_str, results or [])

            if not results:
                answer = "🔍 특정 조건에 맞는 결과는 없으나, 전체 지식 그래프의 통계적 수치를 기반으로 분석해 드립니다."
            else:
                # 결과 해석 요청 (LLM)
                interpret_prompt = (
                    "당신은 뉴스 전문 분석가입니다. [조회 결과]와 [육각형 분석 수치]를 바탕으로 심층 리포트를 작성하세요.\n"
                    "인물/기업 간의 관계를 중심으로 설명하고, 마지막에는 분석 수치에 대한 근거를 덧붙여주세요.\n\n"
                    f"### [사용자 질문]\n{query_str}\n"
                    f"### [조회 결과]\n{results}\n"
                    f"### [육각형 분석 수치]\n{metrics}\n\n"
                    "분석 답변:"
                )
                answer = self.llm.complete(interpret_prompt)

            return {
                "answer": answer,
                "metrics": metrics,
                "cypher": cypher_query
            }
        except Exception as e:
            return f"❌ 오류 발생: {str(e)}"


    async def validate_user_article(self, user_article_text: str, context_nodes: List[NodeWithScore]):
        """
        사용자 기사와 검색된 유사 기사들(Context)을 대조하여 
        사실성(Factuality)과 독창성(Originality)을 검증합니다.
        """
        print("🔍 [Validation] 사용자 기사 vs 검색 지식 대조 분석 시작...")
        
        # 1. 기존 데이터 정리 (깨끗한 대조를 위해)
        graph_store = self.storage_manager.get_neo4j_graph_store()
        graph_store.query("MATCH (n) DETACH DELETE n")
        
        # 2. Context 기사들의 지식을 Neo4j에 로드 (VerifiedSource 라벨 부여)
        self.sync_to_neo4j(context_nodes, label="VerifiedSource")
        
        # [핵심] 로드된 지식을 명시적으로 확인
        existing_knowledge = graph_store.query(
            "MATCH (n)-[r]->(m) RETURN n.id as source, type(r) as relation, m.id as target LIMIT 100"
        )
        
        # 3. 사용자 기사로부터 트리플 추출 (메모리 상에서만 수행하여 격리 유지)
        extractor = SimpleLLMPathExtractor(llm=self.llm)
        temp_doc = Document(text=user_article_text)
        user_triplets = await extractor.acall([temp_doc])
        
        # 4. 사실성 및 독창성 교차 검증 (LLM)
        fact_check_prompt = (
            "당신은 뉴스 팩트체크 전문가입니다. [사용자 주장]들이 [검증된 기존 지식]과 일치하는지, "
            "모순되는지, 아니면 새로운 정보인지 판별하세요.\n\n"
            f"### [검증된 기존 지식]\n{existing_knowledge}\n\n"
            f"### [사용자 주장]\n{user_triplets}\n\n"
            "일치하면 'Factual', 모순되면 'Contradiction', 없으면 'New Information'으로 분류하고 "
            "그 근거를 관계 중심으로 설명하세요.\n"
            "판별 결과:"
        )
        validation_res = self.llm.complete(fact_check_prompt)
        
        # 5. 육각형 지표 산출
        metrics = await self.calculate_hexagonal_metrics(user_article_text, user_triplets)
        
        return {
            "validation_report": validation_res,
            "existing_knowledge": existing_knowledge,
            "user_triplets": user_triplets,
            "metrics": metrics
        }
