import asyncio
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from typing import List, Dict, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from app.rag.langchain_solver import NewsLangChainSolver
from app.etl.storage import StorageManager, NewsArticleModel

class RAGEvaluator:
    def __init__(self, model_name: str = "claude-sonnet-4-0"):
        # 로설 실행을 위해 DB 경로를 현재 디렉토리로 설정
        db_path = "sqlite:///news_arena.db"
        self.storage = StorageManager(db_url=db_path)
        self.solver = NewsLangChainSolver(model_name=model_name)
        self.llm = self.solver.llm
        
    async def generate_gold_dataset(self, num_samples: int = 2) -> List[Dict[str, Any]]:
        """
        DB의 실제 뉴스를 기반으로 합성 질문(시험 문제) 및 정답셋을 생성합니다.
        """
        print(f"🧪 {num_samples}개의 뉴스를 샘플링하여 골든 데이터셋을 생성합니다...")
        session = self.storage.Session()
        # 최신 뉴스 중 일부를 샘플링
        articles = session.query(NewsArticleModel).order_by(NewsArticleModel.id.desc()).limit(num_samples).all()
        
        gold_dataset = []
        for art in articles:
            prompt = (
                f"당신은 RAG 시스템 평가를 위한 시험 출제 위원입니다.\n"
                f"아래 뉴스 기사를 읽고, 이 뉴스를 찾기 위해 사용자가 입력했을 법한 '구체적인 검색 질문' 하나만 생성해줘.\n"
                f"출력은 오직 생성된 질문 텍스트만 하세요.\n\n"
                f"[대상 뉴스]\n제목: {art.title}\n내용: {art.content[:1000]}\n"
            )
            # LangChain Chat 모델 호출
            response = await self.llm.ainvoke(prompt)
            query = response.content.strip()
            
            gold_dataset.append({
                "question": query,
                "ground_truth": art.content,
                "reference_title": art.title
            })
            print(f"  - Q: {query[:50]}...")
            
        session.close()
        return gold_dataset

    async def run_evaluation(self, gold_dataset: List[Dict[str, Any]]):
        """
        생성된 골든 데이터셋을 사용하여 RAGAS 평가를 수행합니다.
        """
        print("\n🚀 RAG 로직 실행 및 데이터 수집 중...")
        eval_records = []
        
        for item in gold_dataset:
            # 1. RAG 실행
            result = await self.solver.solve(item["question"])
            
            # 2. RAGAS 포맷에 맞게 결과 추출
            contexts = [doc.page_content for doc in result["docs"]]
            answer = result["analysis"]
            
            eval_records.append({
                "question": item["question"],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": item["ground_truth"]
            })
            print(f"  - 답변 생성 완료: {item['question'][:30]}...")

        # 3. RAGAS 평가 수행
        dataset = Dataset.from_list(eval_records)
        print("\n📊 RAGAS 지표 계산 중...")
        
        # LangChain의 ChatAnthropic과 OpenAIEmbeddings를 직접 전달
        # RAGAS 0.1.x 버전에서는 llm과 embeddings 인자를 직접 받을 수 있음
        results = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=self.llm,
            embeddings=self.solver.storage_manager.get_hybrid_retriever().dense_retriever.vectorstore.embeddings
        )
        
        return results

if __name__ == "__main__":
    async def main():
        evaluator = RAGEvaluator()
        gold_data = await evaluator.generate_gold_dataset(num_samples=2)
        eval_results = await evaluator.run_evaluation(gold_data)
        
        print("\n" + "="*50)
        print("📊 [RAG Baseline] 정량 평가 결과 요약")
        print("="*50)
        print(eval_results)
        
        print("\n" + "-"*50)
        print("💡 [분석 가이드]")
        print("1. 현재 평가는 '그래프 추론(PGI)' 이전의 '순수 RAG' 검색 성능 테스트입니다.")
        print("2. 적합한 문서를 잘 찾았는지 보려면 'context_precision'이 가장 중요합니다.")
        print("3. 나머지 지표(faithfulness 등)는 지식 그래프를 거친 최종 리포트 단계에서 더 큰 의미를 갖습니다.")
        print("="*50)
        
        # 결과 저장
        df = eval_results.to_pandas()
        df.to_csv("rag_eval_results.csv", index=False)
        print(f"\n✅ 데이터셋 상세 결과가 rag_eval_results.csv에 저장되었습니다.")

    asyncio.run(main())
