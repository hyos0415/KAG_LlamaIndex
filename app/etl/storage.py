import os
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# LlamaIndex Imports
from llama_index.vector_stores.elasticsearch import ElasticsearchStore as LlamaESStore
from llama_index.graph_stores.neo4j import Neo4jGraphStore
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

    def get_neo4j_graph_store(self):
        """Neo4j 그래프 스토리지 인스턴스 반환"""
        store = Neo4jGraphStore(
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
            url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        )
        # Compatibility patch for llama-index-core 0.14+
        if not hasattr(store, "supports_vector_queries"):
            store.supports_vector_queries = False
        if not hasattr(store, "supports_structured_queries"):
            store.supports_structured_queries = True
        
        # Mock class for LabelledNode
        class MockNode:
            def __init__(self, id):
                self.id = id
            def __repr__(self):
                return f"MockNode(id={self.id})"

        # Monkeypatch 'get' to support 'ids' argument and return objects with .id
        orig_get = store.get
        def monkey_get(subj=None, ids=None, **kwargs):
            if ids is not None:
                return [MockNode(node_id) for node_id in ids]
            # subj가 있는 경우 기존 로직 수행 (트리플렛 리스트 반환)
            return orig_get(subj)
        store.get = monkey_get

        # Add missing async methods
        if not hasattr(store, "aget_schema_str"):
            async def aget_schema_str():
                if not store.schema:
                    store.refresh_schema()
                return str(store.schema)
            store.aget_schema_str = aget_schema_str
            
        if not hasattr(store, "arun_query"):
            async def arun_query(query, params=None):
                return store.query(query, params)
            store.arun_query = arun_query

        if not hasattr(store, "astructured_query"):
            async def astructured_query(query, params=None):
                return store.query(query, params)
            store.astructured_query = astructured_query

        # Add missing Llama node management methods (stubbed for compatibility)
        if not hasattr(store, "get_llama_nodes"):
            def get_llama_nodes(ids=None, **kwargs):
                return [] # 기존에 저장된 노드가 없다고 가정
            store.get_llama_nodes = get_llama_nodes

        if not hasattr(store, "upsert_llama_nodes"):
            def upsert_llama_nodes(nodes, **kwargs):
                pass
            store.upsert_llama_nodes = upsert_llama_nodes

        if not hasattr(store, "upsert_nodes"):
            store.upsert_nodes = lambda nodes, **kwargs: None

        if not hasattr(store, "upsert_relations"):
            def upsert_relations(relations, **kwargs):
                for rel in relations:
                    try:
                        s = str(getattr(rel, 'source_id', ''))
                        r = str(getattr(rel, 'label', 'RELATED'))
                        o = str(getattr(rel, 'target_id', ''))
                        if s and o:
                            store.upsert_triplet(s, r, o)
                    except:
                        pass
            store.upsert_relations = upsert_relations

        return store

    def get_llama_graph_storage_context(self) -> StorageContext:
        """그래프용 스토리지 컨텍스트 반환"""
        return StorageContext.from_defaults(graph_store=self.get_neo4j_graph_store())

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
