import os
from dotenv import load_dotenv
from llama_index.llms.anthropic import Anthropic
from llama_index.core import StorageContext, load_index_from_storage, Settings, PromptTemplate
from llama_index.embeddings.openai import OpenAIEmbedding

class ClaudeNewsSolver:
    def __init__(self, model_name: str = "claude-sonnet-4-0", storage_dir: str = "./storage_claude"):
        """
        Claude 기반 뉴스 지능형 추론 및 답변 엔진(Solver)
        """
        self.model_name = model_name
        self.storage_dir = storage_dir
        
        # 환경 변수 로드
        load_dotenv()
        
        # LLM 및 임베딩 설정 (2026년 기준 Claude 4.0 사용)
        self.llm = Anthropic(model=self.model_name, timeout=300.0, max_tokens=2048)
        self.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        
        # LlamaIndex 전역 설정 적용
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        # 저장된 인덱스 로드
        if not os.path.exists(self.storage_dir):
            raise ValueError(f"🚨 저장된 인덱스가 없습니다: {self.storage_dir}. Builder를 먼저 실행하세요.")
            
        print(f"📦 '{self.storage_dir}'에서 지식 그래프를 로드하는 중...")
        storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
        self.index = load_index_from_storage(storage_context)
        print("✅ 인덱스 로드 완료.")

    def solve(self, query_str: str, use_reasoning: bool = True):
        """
        지식 그래프 기반 질의응답 및 추론 수행
        (사실성 검증 및 독창성 확보 로직 포함)
        """
        # 1. 쿼리 엔진 생성
        query_engine = self.index.as_query_engine(
            llm=self.llm,
            similarity_top_k=5,
            include_text=True
        )
        
        # 2. 객관적 지식 그래프 지표 기반 점수화 커스텀 프롬프트
        if use_reasoning:
            custom_prompt_str = (
                "당신은 '그래프 지능 전문가'입니다. 제공된 [지식 그래프 정보]를 구조적으로 분석하여 답변하고, "
                "아래의 **객관적 계산 로직**에 따라 사실성과 독창성을 점수화하세요.\n\n"
                
                "### [점수 계산 로직]\n"
                "1. 사실성 (Factuality) = (그래프 매칭 엔티티 수 / 답변 내 전체 엔티티 수) * 100\n"
                "   - 답변에 언급된 고유명사가 [지식 그래프 정보]의 Subject 혹은 Object와 일치하는 비율을 계산하세요.\n"
                "2. 독창성 (Originality) = 30 + (Multi-hop 경로 깊이 * 20) + (고유 관계 수 * 15)\n"
                "   - Multi-hop: 서로 다른 노드를 2개 이상 연결하여 결론을 도출했는지 확인 (최대 100점).\n\n"
                
                "### [답변 형식]\n"
                "---답변---\n"
                "[지식 그래프 기반 핵심 분석 (최대 300자)]\n\n"
                "---객관적 평가 지표---\n"
                "- 사용된 그래프 Triplet: [S-P-O 리스트]\n"
                "- 사실성 점수: [수식 및 결과]/100\n"
                "- 독창성 점수: [수식 및 결과]/100\n\n"
                
                "질문: {query_str}\n"
                "지식 그래프 정보: {context_str}\n\n"
                "답변:"
            )
            text_qa_template = PromptTemplate(custom_prompt_str)
            
            query_engine.update_prompts({
                "response_synthesizer:text_qa_template": text_qa_template
            })
        
        print(f"🔍 분석 중: {query_str}")
        response = query_engine.query(query_str)
        
        return response

if __name__ == "__main__":
    # 테스트 실행
    solver = ClaudeNewsSolver()
    
    # 예시 질문 1 (사실성 확인)
    print("\n[Test 1: 사실성 검증]")
    res1 = solver.solve("이건희 컬렉션 전시가 열리는 정확한 장소와 전시 제목은?")
    print(f"A: {res1}")
    
    # 예시 질문 2 (독창적 추론 확인)
    print("\n[Test 2: 독창성/추론 검증]")
    res2 = solver.solve("이재용 회장의 이번 행보가 향후 한미 경제 협력 관계에 어떤 상징적 의미를 가질 수 있을까?")
    print(f"A: {res2}")
