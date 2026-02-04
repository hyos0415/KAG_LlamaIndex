import os
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# LlamaIndex Imports
from llama_index.vector_stores.elasticsearch import ElasticsearchStore as LlamaESStore
from llama_index.core import StorageContext

# LangChain Imports
from langchain_elasticsearch import ElasticsearchStore as LangChainESStore
from langchain_openai import OpenAIEmbeddings
import chromadb
from langchain_community.vectorstores import Chroma as LangChainChroma

Base = declarative_base()

class NewsArticleModel(Base):
    __tablename__ = 'news_articles'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    news_id = Column(String(50), unique=True, nullable=False)
    title = Column(String(500))
    content = Column(Text)
    url = Column(String(500))
    pub_date = Column(String(100))
    category = Column(String(100))
    sentiment = Column(String(50))
    summary = Column(Text)
    keywords = Column(JSON)

class StorageManager:
    def __init__(self, db_url: Optional[str] = None):
        # 1. RDBMS 설정 (Source of Truth)
        db_url = db_url or os.getenv("DB_URL", "sqlite:///news_arena.db")
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # 2. Elasticsearch 기본 설정
        self.es_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
        self.index_name = "news_index"

    # --- LlamaIndex 전용 인터페이스 ---
    def get_llama_es_store(self):
        from llama_index.vector_stores.elasticsearch import AsyncDenseVectorStrategy
        return LlamaESStore(
            index_name=self.index_name,
            es_url=self.es_url,
            retrieval_strategy=AsyncDenseVectorStrategy(hybrid=True, text_field="content", rrf=False)
        )

    def get_llama_storage_context(self) -> StorageContext:
        return StorageContext.from_defaults(vector_store=self.get_llama_es_store())

    # --- LangChain 전용 인터페이스 ---
    def get_langchain_sparse_retriever(self, k: int = 10):
        """키워드 검색용 ES 리트리버"""
        return LangChainESStore(
            index_name=self.index_name,
            es_url=self.es_url,
            strategy=LangChainESStore.BM25RetrievalStrategy() # 순수 BM25 설정
        ).as_retriever(search_kwargs={"k": k})

    def get_langchain_dense_retriever(self, k: int = 10):
        """벡터 검색용 Chroma 리트리버"""
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        chroma_path = os.getenv("CHROMA_PATH", "./chroma_db")
        vectorstore = LangChainChroma(
            persist_directory=chroma_path,
            embedding_function=embeddings,
            collection_name="news_collection"
        )
        return vectorstore.as_retriever(search_kwargs={"k": k})

    def get_hybrid_retriever(self, k: int = 5):
        """커스텀 하이브리드 리트리버 결합"""
        from app.rag.hybrid_retriever import NewsHybridRetriever
        return NewsHybridRetriever(
            sparse_retriever=self.get_langchain_sparse_retriever(k=k*2),
            dense_retriever=self.get_langchain_dense_retriever(k=k*2),
            k=k
        )

    # --- 데이터 관리 로직 ---
    def save_article_metadata(self, article_data: Dict[str, Any]):
        session = self.Session()
        try:
            existing = session.query(NewsArticleModel).filter_by(news_id=article_data['news_id']).first()
            if existing:
                for key, value in article_data.items():
                    setattr(existing, key, value)
            else:
                new_article = NewsArticleModel(**article_data)
                session.add(new_article)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_all_articles(self) -> List[NewsArticleModel]:
        session = self.Session()
        articles = session.query(NewsArticleModel).all()
        session.close()
        return articles
