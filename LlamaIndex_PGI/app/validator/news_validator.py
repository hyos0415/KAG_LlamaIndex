import os
from dotenv import load_dotenv
from llama_index.llms.anthropic import Anthropic
from llama_index.core import StorageContext, load_index_from_storage, Settings, PromptTemplate
from llama_index.embeddings.openai import OpenAIEmbedding

class ClaudeNewsValidator:
    def __init__(self, model_name: str = "claude-sonnet-4-0", storage_dir: str = "./storage_claude"):
        """
        사용자 기사 사실성 및 독창성 검증 엔진 (Validator)
        """
        self.model_name = model_name
        self.storage_dir = storage_dir
        
        load_dotenv()
        
        self.llm = Anthropic(model=self.model_name, timeout=300.0, max_tokens=2048)
        self.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        if not os.path.exists(self.storage_dir):
            raise ValueError(f"🚨 저장된 인덱스가 없습니다: {self.storage_dir}")
            
        storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
        self.index = load_index_from_storage(storage_context)

    def validate_article(self, article_text: str):
        """
        사용자 기사를 지식 그래프와 대조하여 검증 리포트 생성
        """
        # 리트리버를 사용하여 관련 지식 그래프 정보 추출
        retriever = self.index.as_retriever(similarity_top_k=10)
        nodes = retriever.retrieve(article_text)
        
        # 지식 그래프 컨텍스트 구성
        context_str = "\n".join([node.node.get_content() for node in nodes])
        
        # 검증 전용 프롬프트 (육각형 객관적 점수 로직 포함)
        validation_prompt_str = (
            "당신은 '뉴스 미디어 팩트체커'입니다. 아래 [사용자 기사]를 [지식 그래프 정보]와 대조하여 "
            "6가지 육각형 지표를 기반으로 정밀 검증하세요.\n\n"
            
            "### [육각형 점수 계산 로직]\n"
            "1. 사실성 (Factuality) = (매칭 엔티티 수 / 기사 내 전체 엔티티 수) * 100\n"
            "2. 독창성 (Originality) = 30 + (Max 경로 깊이 * 20) + (고유 관계 수 * 15) [최대 100]\n"
            "3. 연결성 (Connectivity) = (연결된 서로 다른 노드 도메인 수 / 3) * 100 [최대 100]\n"
            "4. 정보 밀도 (Density) = (사용된 Triplet 수 / 문장 수) * 33 [최대 100]\n"
            "5. 주제 집중도 (Relevance) = (핵심 주제 관련 Triplet 수 / 전체 사용 Triplet 수) * 100\n"
            "6. 논리 정합성 (Consistency) = 100 - (상충 건수 * 25) [최소 0]\n\n"
            
            "### [검증 리포트 형식]\n"
            "1. 검증 요약: [기사의 전체적인 신뢰도 및 가치 평가]\n"
            "2. 사실 상충 항목 (Conflicts): [구체적인 팩트 오류 기술]\n"
            "3. 육각형 분석 결과:\n"
            "   - 사실성: [수식] = [결과]/100\n"
            "   - 독창성: [수식] = [결과]/100\n"
            "   - 연결성: [수식] = [결과]/100\n"
            "   - 정보 밀도: [수식] = [결과]/100\n"
            "   - 주제 집중도: [수식] = [결과]/100\n"
            "   - 논리 정합성: [수식] = [결과]/100\n"
            "   - 종합 평점: [6개 점수의 산술 평균]\n\n"
            
            "[사용자 기사]\n"
            "{article_text}\n\n"
            "[지식 그래프 정보]\n"
            "{context_str}\n\n"
            "검증 리포트:"
        )
        
        prompt = PromptTemplate(validation_prompt_str)
        formatted_prompt = prompt.format(article_text=article_text, context_str=context_str)
        
        print("🔍 기사 검증 중...")
        response = self.llm.complete(formatted_prompt)
        
        return response

if __name__ == "__main__":
    # 테스트 코드
    dummy_article = (
        "삼성전자 이재용 회장이 오는 2월 1일, 미국 뉴욕을 방문하여 "
        "북미 정보기술(IT) 기업들과 '차세대 반도체 공급망 동맹'을 논의할 예정입니다. "
        "코닝사 경영진과 만나 반도체 유리기판 협력을 논의할 것으로 보입니다. "
        "하지만 이번 행사의 진정한 의미는 삼성의 '문화보국' 정신이 어떻게 북미 비즈니스 파트너들과의 "
        "'소프트파워 공급망'으로 변모하는지를 보여주는 데 있습니다."
    )
    
    validator = ClaudeNewsValidator()
    report = validator.validate_article(dummy_article)
    
    print("\n" + "="*50)
    print("📢 기사 검증 리포트")
    print("="*50)
    print(report)
    print("="*50)
