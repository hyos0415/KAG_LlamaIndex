from typing import List, Optional
from llama_index.core import Settings
from llama_index.llms.anthropic import Anthropic
from llama_index.core.program import LLMTextCompletionProgram
from pydantic import BaseModel, Field

class SearchFacet(BaseModel):
    """
    기사 분석을 통해 도출된 개별 검색 관점(Facet)
    """
    facet_query: str = Field(description="특정 관점에서 유사 뉴스를 찾기 위한 검색어")
    dimension: str = Field(description="분석 차원 (예: 주요 인물, 기업 전략, 시장 반응, 기술 동향 등)")

class ArticleAnalysisResult(BaseModel):
    """
    기사 전문 분석 결과 모델
    """
    facets: List[SearchFacet] = Field(description="기사에서 추출한 다차원 검색 관점 목록 (최대 5개)")
    core_summary: Optional[str] = Field(default="요약 생성 실패", description="입력 기사의 핵심 맥락 요약")
    primary_entities: Optional[List[str]] = Field(default_factory=list, description="기사 내 주요 엔티티(인물, 조직, 장소) 목록")

class QueryDecomposer:
    def __init__(self, model_name: str = "claude-sonnet-4-0"):
        self.llm = Anthropic(model=model_name, timeout=300.0)
        Settings.llm = self.llm

    async def decompose_article(self, article_content: str) -> ArticleAnalysisResult:
        """
        입력된 기사 원문을 분석하여 다차원의 검색 관점(Facet)을 도출합니다.
        """
        print("📰 입력 기사 원문 심층 분석 중...")
        
        prompt = (
            "아래 뉴스를 분석하여 유사 뉴스 검색을 위한 3~5개의 검색 관점(facets)을 도출하세요.\n"
            f"기사: {article_content[:3000]}\n\n"
            "반드시 아래 JSON 형식을 지키고, 다른 설명 없이 JSON만 답변하세요.\n"
            '{"facets": [{"facet_query": "유사 뉴스 검색용 쿼리", "dimension": "분석 차원"}]}'
        )
        
        response = await self.llm.acomplete(prompt)
        text = str(response)
        
        # JSON 추출 시도
        import json
        import re
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                result = ArticleAnalysisResult(facets=[SearchFacet(**f) for f in data.get('facets', [])])
            else:
                raise ValueError("JSON not found")
        except Exception as e:
            print(f"⚠️ JSON 파싱 실패, 기본값으로 응답합니다: {e}")
            result = ArticleAnalysisResult(facets=[SearchFacet(facet_query=article_content[:50], dimension="핵심 주제")])
            
        print(f"✅ 분석 완료")
        for idx, facet in enumerate(result.facets, 1):
            print(f"  {idx}. [{facet.dimension}] -> {facet.facet_query}")
            
        return result
