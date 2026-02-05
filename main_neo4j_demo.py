import asyncio
import os
from dotenv import load_dotenv

# MUST apply nest_asyncio to avoid loop conflicts between LangGraph and LlamaIndex/Neo4j
import nest_asyncio
nest_asyncio.apply()

load_dotenv()

from app.graph.knowledge_graph import KnowledgeGraphManager
from app.etl.storage import StorageManager

async def main():
    print("🚀 뉴스 아레나 Neo4j + Text-to-Cypher 데모 시작")
    
    kg_manager = KnowledgeGraphManager()
    storage_manager = StorageManager()
    
    # 1. 문서 샘플링 (DB에서 최신 뉴스 3개 가져오기)
    print("\n[단계 1] DB에서 분석할 뉴스 샘플링 중...")
    articles = storage_manager.get_all_articles()[:3]
    
    if not articles:
        print("❌ DB에 뉴스가 없습니다. 먼저 ETL을 실행해주세요.")
        return

    # 2. Neo4j에 동기화 (지식 추출 및 저장)
    print("\n[단계 2] 추출된 지식을 Neo4j에 저장 중...")
    from llama_index.core.schema import NodeWithScore, TextNode
    
    nodes = [
        NodeWithScore(
            node=TextNode(
                text=art.content, 
                metadata={"title": art.title, "news_id": art.news_id}
            ),
            score=1.0
        ) for art in articles
    ]
    
    # Neo4j로 데이터 push
    kg_manager.sync_to_neo4j(nodes)
    print("✅ Neo4j 동기화 완료 (브라우저 http://localhost:7474 에서 확인 가능)")

    # 3. Text-to-Cypher 분석
    print("\n[단계 3] Text-to-Cypher 심층 분석 실행")
    # 질문 예시: "기사에 언급된 주요 기업들과 그들의 관계를 분석해줘."
    query = "뉴스 기사들에 공통적으로 등장하거나 연관된 주요 인물과 기업들의 관계망을 설명해줘."
    
    result = await kg_manager.analyze_with_cypher(query)
    
    print("\n" + "="*50)
    print("📊 [육각형 분석 2.0] 그래프 분석 결과 보고서")
    print("="*50)
    
    if isinstance(result, dict):
        print(f"📄 분석 답변:\n{result['answer']}")
        print("\n" + "-"*50)
        print("🕸️ 실행된 Cypher 쿼리:")
        print(f"{result['cypher']}")
        print("\n" + "-"*50)
        print("📈 정량 분석 지표 (Hexagonal Metrics):")
        m = result['metrics']
        for key, val in m.items():
            bar = "█" * (val // 5)
            print(f"{key.capitalize():<15} | {val:>3} pts {bar}")
    else:
        print(result)
        
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
