from typing import List
from llama_index.core import Document, PropertyGraphIndex, Settings
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.core.schema import NodeWithScore
from llama_index.llms.anthropic import Anthropic

class JITGraphAnalyzer:
    def __init__(self, model_name: str = "claude-sonnet-4-0"):
        self.llm = Anthropic(model=model_name, timeout=300.0)
        Settings.llm = self.llm

    def build_and_analyze(self, retrieved_nodes: List[NodeWithScore], query_str: str, use_hexagon_report: bool = True):
        """
        검색된 노드들로부터 즉석에서 지식 그래프를 구축하고 육각형 분석 리포트를 생성합니다.
        """
        print(f"🏗️ 선별된 {len(retrieved_nodes)}개 문서로부터 지식 그래프 구축 시작...")
        
        # 1. 문서화
        documents = [Document(text=n.node.get_content(), metadata=n.node.metadata) for n in retrieved_nodes]
        
        # 2. 추출기 설정
        extractor = SimpleLLMPathExtractor(
            llm=self.llm,
            num_workers=2
        )
        
        # 3. JIT 지식 그래프 생성 (메모리 상에 구축)
        index = PropertyGraphIndex.from_documents(
            documents,
            kg_extractors=[extractor],
            show_progress=True
        )
        
        print("🔍 지식 그래프 기반 심층 분석 및 육각형 리포트 생성 중...")
        # 4. 분석 프롬프트 설정
        query_engine = index.as_query_engine(include_text=True)
        
        if use_hexagon_report:
            from llama_index.core import PromptTemplate
            hexagon_prompt = (
                "당신은 '그래프 지능 전문가'입니다. 제공된 [지식 그래프 정보]를 구조적으로 분석하여 답변하고, "
                "아래의 **육각형 객관적 계산 로직**에 따라 분석 품질을 점수화하세요.\n\n"
                "### [육각형 점수 계산 로직]\n"
                "1. 사실성 (Factuality) = (매칭 엔티티 수 / 답변 내 전체 엔티티 수) * 100\n"
                "2. 독창성 (Originality) = 30 + (Max 경로 깊이 * 20) + (고유 관계 수 * 15) [최대 100]\n"
                "3. 연결성 (Connectivity) = (연결된 서로 다른 노드 도메인 수 / 3) * 100 [최대 100]\n"
                "4. 정보 밀도 (Density) = (사용된 Triplet 수 / 문장 수) * 33 [최대 100]\n"
                "5. 주제 집중도 (Relevance) = (핵심 주제 관련 Triplet 수 / 전체 사용 Triplet 수) * 100\n"
                "6. 논리 정합성 (Consistency) = 100 - (상충 건수 * 25) [최소 0]\n\n"
                "질문에 대해 다음 형식을 **반드시** 지커 답변하세요:\n"
                "---답변---\n"
                "[지식 그래프 기반 핵심 분석 내용]\n\n"
                "---육각형 분석 리포트---\n"
                "- 사실성: [결과]/100\n- 독창성: [결과]/100\n- 연결성: [결과]/100\n"
                "- 정보 밀도: [결과]/100\n- 주제 집중도: [결과]/100\n- 논리 정합성: [결과]/100\n"
                "- 종합 평점: [위 6개 점수의 평균]\n\n"
                "질문: {query_str}\n"
                "지식 그래프 정보: {context_str}\n\n"
                "답변:"
            )
            query_engine.update_prompts({
                "response_synthesizer:text_qa_template": PromptTemplate(hexagon_prompt)
            })
            
        response = query_engine.query(query_str)
        return response, index

    def get_graph_triples(self, index: PropertyGraphIndex):
        """
        구축된 그래프에서 추출된 트리플들을 시각화용 데이터 등으로 추출
        """
        # 현재 LlamaIndex API를 통해 추출된 모든 트리플 반환
        return index.property_graph_store.get_triplets()
