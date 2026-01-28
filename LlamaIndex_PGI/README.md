# LlamaIndex News PGI (Property Graph Index) System

이 프로젝트는 매일경제 뉴스 데이터를 활용하여 지식 그래프(Knowledge Graph)를 구축하고, 이를 기반으로 심층적인 추론 및 사용자 기사 검증을 수행하는 지능형 시스템입니다.

## 🚀 주요 기능

### 1. News Knowledge Graph Builder
- **Claude 4.0**과 **SimpleLLMPathExtractor**를 사용하여 뉴스 본문에서 엔티티와 관계를 정밀하게 추출합니다.
- 추출된 데이터는 한국어 깨짐 현상 없이 LlamaIndex의 Property Graph Index로 구축됩니다.

### 2. Intelligent News Solver
- 지식 그래프의 관계망을 이용해 사실에 근거한(Fact-grounded) 답변과 심층적인 비즈니스 인사이트를 도출합니다.
- **객관적 점수화**: 답변의 사실성과 독창성을 지식 그래프 지표(Triplet matching, Multi-hop depth)를 기반으로 자동 산출합니다.

### 3. Article Validator
- 사용자가 작성한 기사의 초안을 기존 지식 그래프와 대조하여 교차 검증합니다.
- 상충되는 팩트(날짜, 장소 등)를 찾아내고 검증 리포트를 생성합니다.

## 🛠 설치 및 실행 방법

### 1. 가상환경 설정 (uv 사용)
```bash
# 가상환경 생성 및 패키지 설치
uv sync
```

### 2. 환경 변수 설정
`.env.example` 파일을 복사하여 `.env` 파일을 생성하고 API 키를 입력하세요.
```bash
cp .env.example .env
```

### 3. 시스템 실행
`main.py`를 통해 빌더, 솔버, 검증기를 통합 실행할 수 있습니다.
```bash
uv run main.py
```

## 📂 폴더 구조
- `app/builder/`: 지식 그래프 구축 모듈
- `app/solver/`: 지식 그래프 기반 추론/질의 모듈
- `app/validator/`: 사용자 기사 검증 모듈
- `app/schema/`: 엔티티 및 관계 정의 스키마
- `storage_claude/`: 영구 저장된 지식 그래프 데이터 (Git 제외)

## 📊 점수 산출 공식
- **사실성 (Factuality)**: `(그래프 매칭 엔티티 수 / 전체 엔티티 수) * 100`
- **독창성 (Originality)**: `30 + (Multi-hop 경로 깊이 * 20) + (고유 관계 수 * 15)`
