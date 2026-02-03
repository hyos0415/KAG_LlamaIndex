import os
import json
import asyncio
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Document, PropertyGraphIndex, Settings, StorageContext, load_index_from_storage
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.program import LLMTextCompletionProgram

from app.builder.storage_manager import StorageManager
import nest_asyncio
nest_asyncio.apply()

class NewsMetadata(BaseModel):
    category: str = Field(description="뉴스 기사의 주요 카테고리 (경제, IT, 사회, 정치 등)")
    sentiment: str = Field(description="뉴스 기사의 전반적인 감성 (긍정, 중립, 부정)")
    keywords: List[str] = Field(description="기사에서 추출한 주요 키워드 5개")
    summary: str = Field(description="기사의 내용을 한 문장으로 요약")

class ClaudeNewsBuilder:
    def __init__(self, model_name: str = "claude-sonnet-4-0", storage_dir: str = "/opt/airflow/storage_claude"):
        """
        Claude 기반 뉴스 지능형 빌더 초기화
        """
        self.model_name = model_name
        self.storage_dir = storage_dir
        
        # LLM 및 임베딩 설정
        self.llm = Anthropic(model=self.model_name, timeout=300.0)
        self.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        
        # LlamaIndex 전역 설정 적용
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        # StorageManager 초기화
        self.storage_manager = StorageManager()
        
        # Semantic Splitter 설정
        self.node_parser = SemanticSplitterNodeParser(
            buffer_size=1, breakpoint_percentile_threshold=95, embed_model=self.embed_model
        )
        
        self.index = None

    async def _extract_metadata(self, content: str) -> NewsMetadata:
        """
        LLM을 사용하여 기사의 메타데이터를 추출합니다.
        """
        prompt_template_str = (
            "다음 뉴스 기사 내용을 분석하여 지정된 형식의 메타데이터를 추출해주세요.\n"
            "기사 내용: {content}\n"
        )
        program = LLMTextCompletionProgram.from_defaults(
            output_cls=NewsMetadata,
            prompt_template_str=prompt_template_str,
            llm=self.llm,
            verbose=True
        )
        return program(content=content)

    async def load_news_documents(self, file_path: str, limit: Optional[int] = None) -> List[Document]:
        """
        JSON 뉴스 파일에서 문서를 로드하고 T (Transform) 과정을 거칩니다.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            news_data = json.load(f)
        
        if limit:
            news_data = news_data[:limit]
            
        documents = []
        for item in news_data:
            content = item.get('content')
            if content:
                # 1. T (Transform): 메타데이터 추출
                print(f"🔍 '{item.get('title')}' 기사에서 메타데이터 추출 중...")
                metadata = await self._extract_metadata(content)
                
                # 2. L (Load): RDBMS 저장
                article_record = {
                    "news_id": item.get('id', 'N/A'),
                    "title": item.get('title', 'N/A'),
                    "content": content,
                    "url": item.get('url', 'N/A'),
                    "pub_date": item.get('pub_date', 'N/A'),
                    "category": metadata.category,
                    "sentiment": metadata.sentiment,
                    "summary": metadata.summary,
                    "keywords": metadata.keywords
                }
                self.storage_manager.save_article_metadata(article_record)
                
                # 3. LlamaIndex Document 생성 (메타데이터 포함)
                doc = Document(
                    text=content,
                    metadata={
                        "title": article_record["title"],
                        "url": article_record["url"],
                        "pub_date": article_record["pub_date"],
                        "news_id": article_record["news_id"],
                        "category": metadata.category,
                        "sentiment": metadata.sentiment,
                        "keywords": ", ".join(metadata.keywords),
                        "summary": metadata.summary
                    }
                )
                documents.append(doc)
        
        print(f"✅ {len(documents)}개의 뉴스를 성공적으로 로드 및 변환했습니다.")
        return documents

    def build_graph(self, documents: List[Document], persist: bool = True):
        """
        뉴스 문서들을 지식 그래프로 변환합니다. (L: Vector DB 및 Graph 저장)
        """
        print(f"🚀 '{self.model_name}' 엔진을 사용하여 지식 추출 및 그래프 구축을 시작합니다...")
        
        # 시맨틱 청킹 적용
        nodes = self.node_parser.get_nodes_from_documents(documents)
        
        # 추출기 설정
        extractor = SimpleLLMPathExtractor(
            llm=self.llm,
            num_workers=2
        )
        
        # 하이브리드 저장소 컨텍스트 가져오기
        storage_context = self.storage_manager.get_storage_context()
        
        # 기존 저장소 삭제 (새로 구축 시)
        if persist and os.path.exists(self.storage_dir):
            import shutil
            shutil.rmtree(self.storage_dir)
        
        self.index = PropertyGraphIndex(
            nodes=nodes,
            kg_extractors=[extractor],
            storage_context=storage_context,
            show_progress=True
        )
        
        if persist:
            # PropertyGraphIndex 자체 메타데이터 저장을 위한 디렉토리
            os.makedirs(self.storage_dir, exist_ok=True)
            self.index.storage_context.persist(persist_dir=self.storage_dir)
            print(f"💾 지식 그래프 메타데이터가 '{self.storage_dir}'에 저장되었습니다.")
            
        return self.index

    def query(self, query_str: str):
        """
        구축된 그래프를 기반으로 질문을 수행합니다.
        """
        if not self.index:
            if os.path.exists(self.storage_dir):
                storage_context = self.storage_manager.get_storage_context()
                # 저장된 인덱스 로드 (ChromaDB와 연결된 컨텍스트 사용)
                # PropertyGraphIndex.from_storage 가 현재 버전에서 공식 지원되는지 확인 필요
                # 일반적인 경우 load_index_from_storage 등을 사용하거나 인출 방식 확인
                try:
                    self.index = PropertyGraphIndex.from_storage(storage_context)
                except:
                    # 대체 로드 방식 (버전에 따라 다를 수 있음)
                    self.index = load_index_from_storage(storage_context)
            else:
                raise ValueError("구축된 인덱스가 없습니다. build_graph를 먼저 실행하세요.")
                
        query_engine = self.index.as_query_engine(include_text=True)
        return query_engine.query(query_str)

async def run_pgi_pipeline(data_file: Optional[str] = None):
    """
    Airflow 작업 등으로 실행될 전체 PGI 파이프라인 프로세스
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    # 데이터 파일이 지정되지 않은 경우 최신 파일 찾기
    if not data_file:
        base_dir = "/opt/airflow/result/airflow"
        if os.path.exists(base_dir):
            files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith(".json")]
            if files:
                data_file = max(files, key=os.path.getmtime)
                print(f"📂 최신 데이터 파일을 자동으로 선택했습니다: {data_file}")
    
    if not data_file or not os.path.exists(data_file):
        print("❌ 처리할 데이터 파일을 찾을 수 없습니다.")
        return

    builder = ClaudeNewsBuilder()
    
    try:
        # 데이터 로드 및 고도화 (T & L: RDBMS)
        docs = await builder.load_news_documents(data_file)
        
        # 지식 그래프 구축 (L: Vector DB & Graph)
        builder.build_graph(docs)
        
        print("✅ PGI 파이프라인 처리가 완료되었습니다.")
        
    except Exception as e:
        print(f"❌ 파이프라인 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        # 에어플로우에서 실패로 인식하도록 예외 재발생
        raise e

if __name__ == "__main__":
    import sys
    # 인자로 파일 경로를 넘겨받을 수 있도록 함
    target_file = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(run_pgi_pipeline(target_file))
