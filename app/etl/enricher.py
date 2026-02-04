import os
import json
import asyncio
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.program import LLMTextCompletionProgram

from app.etl.storage import StorageManager
import nest_asyncio
nest_asyncio.apply()

class NewsMetadata(BaseModel):
    category: str = Field(description="뉴스 기사의 주요 카테고리 (경제, IT, 사회, 정치 등)")
    sentiment: str = Field(description="뉴스 기사의 전반적인 감성 (긍정, 중립, 부정)")
    keywords: List[str] = Field(description="기사에서 추출한 주요 키워드 5개")
    summary: str = Field(description="기사의 내용을 한 문장으로 요약")

class NewsEnricher:
    def __init__(self, model_name: str = "claude-sonnet-4-0"):
        """
        뉴스 고도화 및 적재 엔진 초기화
        """
        self.model_name = model_name
        
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

    async def process_and_load(self, file_path: str, limit: Optional[int] = None):
        """
        JSON 뉴스 파일에서 문서를 로드하고 T (Transform) & L (Load) 과정을 수행합니다.
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
                print(f"🔍 '{item.get('title')}' 기사에서 메타데이터 추출 중...")
                metadata = await self._extract_metadata(content)
                
                # 1. RDBMS 저장
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
                
                # 2. Document 생성
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
        
        # 3. Vector DB 적재 (Elasticsearch)
        if documents:
            print(f"🚀 {len(documents)}개의 문서를 Elasticsearch에 적재합니다...")
            nodes = self.node_parser.get_nodes_from_documents(documents)
            
            # Elasticsearch에 적재
            es_storage_context = self.storage_manager.get_storage_context(store_type="elasticsearch")
            VectorStoreIndex(nodes, storage_context=es_storage_context)
            
            # (옵션) 기존 사용자를 위해 ChromaDB에도 병행 유지하고 싶다면 아래 주석 해제 가능
            # chroma_storage_context = self.storage_manager.get_storage_context(store_type="chroma")
            # VectorStoreIndex(nodes, storage_context=chroma_storage_context)
            
            print("✅ Elasticsearch 적재 완료.")
        
        return documents

async def run_etl_pipeline(data_file: Optional[str] = None):
    """
    Airflow 작업 등으로 실행될 전체 ETL 파이프라인 프로세스
    """
    from dotenv import load_dotenv
    load_dotenv()
    
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

    enricher = NewsEnricher()
    try:
        await enricher.process_and_load(data_file)
        print("✅ ETL 파이프라인 처리가 완료되었습니다.")
    except Exception as e:
        print(f"❌ 파이프라인 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    import sys
    target_file = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(run_etl_pipeline(target_file))
