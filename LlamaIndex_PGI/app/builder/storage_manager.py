import os
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

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
    keywords = Column(JSON)  # 리스트 형태로 저장

class StorageManager:
    def __init__(self, db_url: str = "sqlite:////opt/airflow/news_arena.db", chroma_path: str = "/opt/airflow/chroma_db"):
        # RDBMS 설정
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # ChromaDB 설정
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.chroma_collection = self.chroma_client.get_or_create_collection("news_collection")
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        
    def save_article_metadata(self, article_data: Dict[str, Any]):
        """
        RDBMS에 뉴스 메타데이터 및 원문을 저장합니다.
        """
        session = self.Session()
        try:
            # 기존 기사가 있는지 확인
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

    def get_storage_context(self) -> StorageContext:
        """
        LlamaIndex에서 사용할 StorageContext를 반환합니다.
        """
        return StorageContext.from_defaults(vector_store=self.vector_store)

    def get_article_by_id(self, news_id: str) -> Optional[Dict[str, Any]]:
        session = self.Session()
        article = session.query(NewsArticleModel).filter_by(news_id=news_id).first()
        if article:
            data = {c.name: getattr(article, c.name) for c in article.__table__.columns}
            session.close()
            return data
        session.close()
        return None
