# 뉴스 아레나: AI 기반 지능형 뉴스 ETL 파이프라인

이 프로젝트는 매일경제 RSS 피드에서 뉴스를 수집하여 **AI(Claude)를 활용한 지능형 분석(Transform)**을 수행하고, **RDBMS와 Vector DB에 하이브리드로 적재(Load)**하는 차세대 뉴스 분석 시스템입니다.

## 🚀 고도화된 주요 기능

### 1. 지능형 변환 (T - Transform)
- **AI 메타데이터 추출**: Claude 3.x/4.x 모델을 활용하여 기사 로드 시 **카테고리, 감성, 핵심 키워드, 한 줄 요약**을 자동으로 생성합니다.
- **시맨틱 청킹 (Semantic Chunking)**: 단순 길이 기반 분할이 아닌, 의미적 문맥을 보존하는 방식으로 텍스트를 파쇄하여 검색 정확도를 높였습니다.

### 2. 하이브리드 적재 (L - Load)
- **RDBMS (SQLite)**: 원본 기사와 구조화된 메타데이터를 저장하여 정교한 필터링과 이력 관리를 지원합니다.
- **Vector DB (ChromaDB)**: 고도화된 시맨틱 검색을 위한 벡터 임베딩을 관리합니다.
- **Property Graph Index**: 인물, 기업 등 지식 간의 유기적인 연결을 위한 그래프 구조를 구축합니다.

### 3. 실무 지향적 자동화
- **Airflow Full Pipeline**: 수집 -> 분석 -> 하이브리드 적재로 이어지는 전 과정을 자동화했습니다.
- **확장형 아키텍처**: SQLAlchemy(ORM)를 사용하여 추후 MySQL, PostgreSQL 등 운영용 DB로의 전환이 즉시 가능합니다.

## 🛠 필수 요구 사항
- **[Docker Desktop](https://www.docker.com/products/docker-desktop/)**
- **API Keys**: Anthropic 및 OpenAI API 키가 필요합니다 (`.env` 파일에 기록).

## 🏃 실행 방법

### 1. 환경 설정
프로젝트 루트의 `.env` 파일에 필요한 API 키를 입력하세요:
```env
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

### 2. 시스템 시작
```bash
docker-compose up -d
```

### 3. 에어플로우 접속 및 로그인
- **URL**: [http://localhost:8080](http://localhost:8080)
- **ID/PW**: `airflow` / `airflow`

### 4. 파이프라인 활성화
DAG 리스트에서 `mk_news_full_pipeline`을 활성화(Unpause)하면 한 시간마다 인공지능이 뉴스를 분석하고 저장하기 시작합니다.

## 📂 폴더 구조
- **`LlamaIndex_PGI/`**: AI 분석 및 하이브리드 저장 핵심 로직 (`builder`, `schema`, `solver`)
- **`ETL_expr/`**: 뉴스 원문 수집 엔진
- **`dags/`**: 통합 파이프라인 정의서
- **`news_arena.db`**: 구조화된 메타데이터 저장소 (SQLite)
- **`chroma_db/`**: 벡터 데이터 저장소

## 📝 참고 사항
- **PGI 모듈**: `LlamaIndex`의 `PropertyGraphIndex`를 사용하여 복합적인 지식 추출을 수행합니다.
- **안정성**: `nest_asyncio`를 적용하여 에어플로우 컨테이너 내의 비동기 충돌 문제를 해결했습니다.