import os
import json
from typing import List, Optional
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Document, PropertyGraphIndex, Settings, StorageContext, load_index_from_storage
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor

class ClaudeNewsBuilder:
    def __init__(self, model_name: str = "claude-sonnet-4-0", storage_dir: str = "./storage_claude"):
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
        
        self.index = None

    def load_news_documents(self, file_path: str, limit: Optional[int] = None) -> List[Document]:
        """
        JSON 뉴스 파일에서 문서를 로드합니다.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            news_data = json.load(f)
        
        # 리밋 설정 (공모전 크레딧 절약용)
        if limit:
            news_data = news_data[:limit]
            
        documents = []
        for item in news_data:
            if item.get('content'):
                doc = Document(
                    text=item['content'],
                    metadata={
                        "title": item.get('title', 'N/A'),
                        "url": item.get('url', 'N/A'),
                        "pub_date": item.get('pub_date', 'N/A'),
                        "news_id": item.get('id', 'N/A')
                    }
                )
                documents.append(doc)
        
        print(f"✅ {len(documents)}개의 뉴스를 성공적으로 로드했습니다.")
        return documents

    def build_graph(self, documents: List[Document], persist: bool = True):
        """
        뉴스 문서들을 지식 그래프로 변환합니다.
        """
        print(f"🚀 '{self.model_name}' 엔진을 사용하여 지식 추출을 시작합니다...")
        
        # 추출기 설정 (자유도 높은 지식 추출을 위해 SimpleLLMPathExtractor 사용)
        extractor = SimpleLLMPathExtractor(
            llm=self.llm,
            num_workers=2
        )
        
        # 기존 저장소 삭제 (새로 구축 시)
        if persist and os.path.exists(self.storage_dir):
            import shutil
            shutil.rmtree(self.storage_dir)
        
        self.index = PropertyGraphIndex.from_documents(
            documents,
            kg_extractors=[extractor],
            show_progress=True
        )
        
        if persist:
            os.makedirs(self.storage_dir, exist_ok=True)
            self.index.storage_context.persist(persist_dir=self.storage_dir)
            print(f"💾 지식 그래프가 '{self.storage_dir}'에 저장되었습니다.")
            
        return self.index

    def query(self, query_str: str):
        """
        구축된 그래프를 기반으로 질문을 수행합니다.
        """
        if not self.index:
            # 저장된 인덱스 로드 시도
            if os.path.exists(self.storage_dir):
                storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
                self.index = load_index_from_storage(storage_context)
            else:
                raise ValueError("구축된 인덱스가 없습니다. build_graph를 먼저 실행하세요.")
                
        query_engine = self.index.as_query_engine(include_text=True)
        return query_engine.query(query_str)

if __name__ == "__main__":
    # 테스트 실행 로직
    from dotenv import load_dotenv
    load_dotenv()
    
    builder = ClaudeNewsBuilder()
    
    # 1. 뉴스 데이터 딱 1개만 로드 및 구축 테스트
    data_file = "../result/airflow/mk_news_20260126_1000.json"
    try:
        docs = builder.load_news_documents(data_file, limit=1)
        builder.build_graph(docs)
        
        # 2. 간단한 검증 질문
        res = builder.query("이재용 회장의 방문 목적은?")
        print(f"\n📢 질의 결과: {res}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
