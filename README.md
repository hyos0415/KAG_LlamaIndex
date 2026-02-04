# 뉴스 아레나: AI 기반 지능형 뉴스 RAG & PGI 파이프라인

이 프로젝트는 매일경제 RSS 피드에서 뉴스를 수집하여 **AI(Claude)를 활용한 지능형 분석(Transform)**을 수행하고, **하이브리드 RAG와 Property Graph Index(PGI)를 결합**하여 심층적인 통찰을 제공하는 차세대 뉴스 지능 엔진입니다.

## 🚀 하이브리드 RAG & PGI 아키텍처

본 프로젝트는 단순 검색을 넘어, 정보의 발견과 관계의 추론을 동시에 수행하는 고도화된 아키텍처를 지향합니다.

### 1. 지능형 워크플로우 (LangGraph)
- **분석적 루프**: `LangGraph`를 사용하여 [기사 분석 -> 검색 관점 도출 -> 하이브리드 검색 -> 그래프 기반 추론 -> 최종 리포트]로 이어지는 결정적 워크플로우를 관리합니다.
- **다차원 분석(Decomposition)**: 입력 기사를 인물, 기업 전략, 시장 반응 등 다각도의 페이셋(Facets)으로 분해하여 보다 정밀한 검색을 수행합니다.

### 2. 커스텀 하이브리드 리트리버 (LangChain)
- **Ensemble Retrieval**: Elasticsearch의 **Sparse(BM25)** 검색과 ChromaDB의 **Dense(Vector)** 검색을 파이썬 레이어에서 결합합니다.
- **RRF (Reciprocal Rank Fusion)**: ES 라이선스 제약 없는 순수 알고리즘 구현을 통해 검색 결과의 상호 보완성을 극대화하여 검색 정밀도(Context Precision) 1.0을 실현했습니다.

### 3. Just-in-Time 지식 그래프 (LlamaIndex PGI)
- **동적 그래프 구축**: 검색된 우량 문서들로부터 즉석에서 엔티티와 관계를 추출하여 `PropertyGraphIndex`를 구축합니다.
- **육각형 분석 리포트**: 구축된 그래프 상에서 사실성, 연결성, 독창성 등 6개 지표를 기반으로 심층적인 관계 분석 리포트를 생성합니다.

## 📊 정량적 성능 평가 (RAGAS)
- **평가 자동화**: `faithfulness`, `context_precision` 등 RAGAS 지표를 사용하여 하이브리드 리트리버의 성능을 정기적으로 측정합니다.
- **골든 데이터셋**: 실제 뉴스 데이터를 기반으로 LLM이 합성 질문과 정답을 자동 생성하여 평가의 객관성을 확보합니다.

## 🛠 필수 요구 사항
- **[Docker Desktop](https://www.docker.com/products/docker-desktop/)**
- **python 3.10+ & uv**
- **API Keys**: Anthropic 및 OpenAI API 키가 필요합니다 (`.env` 파일에 기록).

## 📂 프로젝트 구조
- **`app/etl/`**: 데이터 수집, 메타데이터 추출, 하이브리드 동기화(ES/Chroma)
- **`app/rag/`**: 하이브리드 리트리버, LangGraph 워크플로우, RAGAS 평가 모듈
- **`app/graph/`**: JIT 지식 그래프 빌더 및 그래프 리포트 분석기
- **`main_hybrid_demo.py`**: 통합 분석 엔진 작동 시현 데모

## 🏃 실행 방법

### 1. 서비스 가동
```bash
docker-compose up -d
```

### 2. 하이브리드 분석 데모 실행
```bash
PYTHONPATH=. uv run python3 main_hybrid_demo.py
```

### 3. RAG 성능 평가 실행
```bash
PYTHONPATH=. uv run python3 app/rag/evaluator.py
```
