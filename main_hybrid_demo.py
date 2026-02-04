import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

import nest_asyncio
nest_asyncio.apply()

from app.rag.graph_flow import NewsAppGraph
from app.etl.storage import StorageManager

async def main():
    print("🚀 뉴스 아레나 하이브리드(LangChain + LlamaIndex) 분석 엔진 기동...")
    
    # 1. 테스트용 기사 선정 (RDBMS에서 하나 가져오기)
    storage = StorageManager(db_url="sqlite:///news_arena.db")
    articles = storage.get_all_articles()
    
    if not articles:
        print("❌ 분석할 기사가 DB에 없습니다. 먼저 데이터를 적재해주세요.")
        return
        
    test_article = articles[4] # 샘플 하나 선정 (예: 하이닉스 관련)
    print(f"\n[대상 기사]: {test_article.title}")
    
    # 2. LangGraph 워크플로우 실행
    app_graph = NewsAppGraph()
    result = await app_graph.run(test_article.content)
    
    # 3. 결과 출력
    print("\n" + "="*50)
    print("📊 최종 분석 리포트")
    print("="*50)
    print(result["final_report"])
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
