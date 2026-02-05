import asyncio
import os
from dotenv import load_dotenv
import nest_asyncio

# MUST apply nest_asyncio to avoid loop conflicts
nest_asyncio.apply()
load_dotenv()

from app.rag.solver import NewsRAGSolver
from app.graph.knowledge_graph import KnowledgeGraphManager

async def main():
    print("🚀 [실전 데모] 사용자 기사 검증 파이프라인 가동")
    
    solver = NewsRAGSolver()
    kg_manager = KnowledgeGraphManager()

    # 1. 사용자 작성 기사 (테스트 케이스: 사실 왜곡 포함)
    # 실제 데이터는 '징역 3년'이나, 사용자가 '징역 5년'으로 작성한 상황을 가정
    user_article = """
    [속보] 강영권 전 에디슨모터스 회장, 1심서 '징역 5년' 중형 선고
    
    쌍용차 인수 과정에서 허위 정보를 유포해 주가를 조작한 혐의로 기소된 강영권 전 에디슨모터스 회장이 
    오늘 서울남부지법에서 열린 1심 판결에서 징역 5년을 선고받았다. 재판부는 "금융 시장의 신뢰를 
    심각하게 훼손한 죄질이 무겁다"고 양형 이유를 밝혔다. 
    또한 이번 판결에는 과거 언급되지 않았던 '신규 협력사 A사'의 가담 여부도 새롭게 적시되었다.
    """
    
    print("\n[단계 1] 사용자 기사 분석 및 유사 지식 검색 중...")
    # 사용자 기사의 핵심 키워드로 Hybrid RAG 실행
    search_query = "에디슨모터스 강영권 회장 주가조작 판결 결과"
    context_nodes = await solver.retrieve_similar_nodes(search_query, top_k=3)
    
    print(f"✅ 관련 뉴스 {len(context_nodes)}건 발견. 그래프 대조 분석을 시작합니다.")

    # 2. 통합 검증 파이프라인 실행
    # (검색된 노드들을 Neo4j에 로드 -> 사용자 기사와 대조 -> 보고서 생성)
    report = await kg_manager.validate_user_article(user_article, context_nodes)

    # 3. 결과 출력
    print("\n" + "="*60)
    print("📊 [검증 리포트] 사용자 기사 신뢰성 분석 결과")
    print("="*60)
    
    print(f"🔍 사실 정합성 판별:\n{report['validation_report']}")
    print("\n" + "-"*60)
    
    print("📈 기사 품질 육각형 지표:")
    m = report['metrics']
    for key, val in m.items():
        bar = "█" * (val // 5)
        print(f"{key.capitalize():<15} | {val:>3} pts {bar}")
        
    print("\n" + "-"*60)
    print("🕸️ 추출된 지식 트리플 (사용자 기사):")
    for triplet in report['user_triplets'][:5]:
        print(f" - {triplet}")
    
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
