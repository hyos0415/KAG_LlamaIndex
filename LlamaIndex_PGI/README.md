# LlamaIndex News PGI (Property Graph Index) System

이 프로젝트는 매일경제 뉴스 데이터를 활용하여 지식 그래프(Knowledge Graph)를 구축하고, 이를 기반으로 심층적인 추론 및 사용자 기사 검증을 수행하는 지능형 시스템입니다. 최근 고도화된 **육각형 분석 지표(Hexagonal Metrics)**를 도입하여 분석의 객관성을 극대화했습니다.

## 🚀 주요 기능

### 1. News Knowledge Graph Builder
- **Claude 4.0**과 **SimpleLLMPathExtractor**를 사용하여 뉴스 본문에서 엔티티와 관계를 정밀하게 추출합니다.
- 추출된 데이터는 한국어 깨짐 현상 없이 LlamaIndex의 **Property Graph Index**로 구축됩니다.

### 2. Intelligent News Solver (with Hexagonal Scoring)
- 지식 그래프의 관계망을 이용해 사실에 근거한(Fact-grounded) 답변과 심층적인 비즈니스 인사이트를 도출합니다.
- 모든 답변에 대해 아래 6가지 객관적 지표를 기반으로 한 **육각형 분석 결과**를 제공합니다.

### 3. News Article Validator
- 사용자가 작성한 기사의 초안을 구축된 지식 그래프와 교차 대조하여 검증합니다.
- 날짜, 장소, 사건 목적 등의 **상충(Conflict)** 여부를 식별하고 종합적인 신뢰도 리포트를 생성합니다.

## 📊 육각형 객관적 점수 체계
AI의 주관적 평가를 배제하고, 지식 그래프의 구조적 데이터(Node, Triplet, Path)를 활용한 수식을 적용했습니다.

| 지표 | 산출 공식 | 의미 |
| :--- | :--- | :--- |
| **사실성** | `(매칭 엔티티 / 전체 엔티티) * 100` | 팩트의 정확도 |
| **독창성** | `30 + (Max 경로 깊이 * 20) + (고유 관계 수 * 15)` | 추론의 깊이 (Multi-hop) |
| **연결성** | `(연결된 노드 도메인 수 / 3) * 100` | 이종 분야 융합 분석력 |
| **정보 밀도** | `(Triplet 수 / 문장 수) * 33` | 문장당 지식 압축도 |
| **주제 집중도**| `(핵심 Triplet / 전체 Triplet) * 100` | 핵심 주제 결집도 |
| **논리 정합성**| `100 - (상충 건수 * 25)` | 지식간 충돌 안정성 |

## 🛠 설치 및 실행 방법

### 1. 가상환경 설정 (uv 사용)
```bash
uv sync
```

### 2. 환경 변수 설정
`.env.example` 파일을 복사하여 `.env` 파일을 생성하고 Claude 및 OpenAI API 키를 입력하세요.
```bash
cp .env.example .env
```

### 3. 시스템 실행
`main.py`를 통해 통합 메뉴를 이용하거나, 개별 모듈을 실행할 수 있습니다.
```bash
# 통합 메뉴 실행 (추천)
uv run main.py

# 개별 검증기 실행
uv run app/validator/news_validator.py
```

## 📂 폴더 구조
- `app/builder/`: 뉴스 지식 그래프 구축 모듈
- `app/solver/`: 지식 지표 기반 질의/추론 모듈
- `app/validator/`: 사용자 기사 검증 및 리포트 생성 모듈
- `app/schema/`: 지식 엔티티 및 관계 정의 스키마
- `storage_claude/`: 영구 저장된 지식 그래프 데이터 (Git 제외)

---
본 프로젝트는 데이터 기반의 객관적인 뉴스 분석 및 사실 검증 환경을 제공하는 것을 목표로 합니다.
