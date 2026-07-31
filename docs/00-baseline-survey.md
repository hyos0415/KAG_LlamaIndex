# 00. 기준 현황 조사 (Baseline Survey)

Claude Code를 이용해 2026-08-01에 수행한 KAG_LlamaIndex(`news-arena`) 프로젝트 현황 조사 전문. 조사 시점 기준 코드/데이터 상태를 그대로 기록한 진단 문서이며, 이후 실험 전 "before" 스냅샷 역할을 한다. 조사 태그: `v0-pre-experiment` (`3ba9426`).

---

## Part A. 레포/인프라/코드 구조 조사

### 1. 레포 위치

홈 아래에 `KAG_LlamaIndex`라는 이름의 디렉터리는 없었으나, 실제 프로젝트가 다른 이름으로 존재.

- 최상위 레포: `/Users/janghyoseong/Desktop/mk_news` (git 초기 상태, 커밋 없음)
- 실질적 프로젝트(중첩 git 레포): `/Users/janghyoseong/Desktop/mk_news/news-arena`
  - remote가 `https://github.com/hyos0415/KAG_LlamaIndex.git` — 찾던 프로젝트와 일치

2단계 구조 (news-arena):
```
news-arena/
├── app/            (etl, graph, rag)
├── core/           (rlm_evaluator, code_sandbox, lib/rlm)
├── KAG/            (별도 중첩 프로젝트/venv, app/builder, app/solver, conf)
├── dags/           (Airflow DAG)
├── config, plugins, logs, tests
├── result/airflow/ (수집 JSON 11개)
├── storage_claude/ (LlamaIndex 스토리지 JSON)
├── neo4j_data/, neo4j_logs/, chroma_db/
├── docker-compose.yml, Dockerfile, airflow.cfg
├── .env, requirements.txt, pyproject.toml, uv.lock
└── news_arena.db
```

### 2. Git 상태

| 항목 | mk_news (최상위) | news-arena (중첩) |
|---|---|---|
| 브랜치 | main | main |
| 마지막 커밋 | 없음 (커밋 전) | `3ba9426` / 2026-02-06 01:34:23 +0900 |
| 커밋 메시지 | - | "fix : 성과 리포트 내 지표 해석 순서 일원화" |
| 미커밋 변경 | untracked 다수 | 없음 (clean) |
| 태그 | 없음 | 없음 (조사 후 `v0-pre-experiment` 생성) |
| remote | 없음 | `origin` → `github.com/hyos0415/KAG_LlamaIndex.git` |

### 3. Docker 자산

조사 시점에는 Docker 데몬 미기동 상태였음. Compose 정의 및 로컬 바인드 마운트로 대체 확인:

| 데이터 종류 | 저장 방식 | 로컬 경로 | 실측 크기 |
|---|---|---|---|
| Neo4j 데이터 | bind mount | `news-arena/neo4j_data` | 4.2M |
| Neo4j 로그 | bind mount | `news-arena/neo4j_logs` | 408K |
| Chroma | 컨테이너 밖 로컬 디렉터리(compose 미마운트) | `news-arena/chroma_db` | 31M |
| Elasticsearch | named volume(`elasticsearch_data`) | compose 선언 | 미확인(당시 데몬 미기동) |
| Postgres | named volume(`postgres_data`) 선언되었으나 어떤 서비스도 참조 안 함 | - | - |
| Redis | named volume(`redis_data`) 선언되었으나 redis 서비스 자체가 compose에 없음 | - | - |

### 4. Compose 파일

- 위치: `news-arena/docker-compose.yml`
- 서비스: `postgres`, `elasticsearch`, `neo4j`, `airflow-webserver`, `airflow-scheduler`, `airflow-init`

| 서비스 | 볼륨 마운트 | 포트 |
|---|---|---|
| postgres | (named volume 미연결) | - |
| elasticsearch | `elasticsearch_data:/usr/share/elasticsearch/data` | 9200:9200 |
| neo4j | `./neo4j_data:/data`, `./neo4j_logs:/logs` | 7474:7474, 7687:7687 |
| airflow-* | `.:/opt/airflow`, `./dags`, `./logs`, `./plugins` | webserver 8080:8080 |

### 5. 환경 변수

`.env.example` 등 비교 대상 파일 없음.

`news-arena/.env` (7개 키, 모두 값 채워짐 — 값 자체는 비공개):
`AIRFLOW_UID`, `_PIP_ADDITIONAL_REQUIREMENTS`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`

`news-arena/KAG/.env` (별도 파일, 2개 키 모두 채워짐): `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`

### 6. 로컬 데이터 (조사 시점)

| 항목 | 위치 | 크기/건수 |
|---|---|---|
| Neo4j 덤프(.dump) | - | 없음 |
| 수집 기사 원문 (JSON) | `result/airflow/*.json` | 11개 파일 |
| KG 추출 결과 | `KAG/result/kg_output.jsonl` | 5줄, 44K |
| LlamaIndex 스토리지 | `storage_claude/*.json` | docstore 10건(64K), property_graph_store 544K, graph_store 4K |
| RAG 평가 결과 | `rag_eval_results.csv` | 95행 |
| 리포트/요약 문서 | `technical_report.md`(109줄), `final_results_summary.md`(105줄) | 기존 실험 결과 문서 |
| 로컬 SQLite | `news_arena.db` | 128K |

### 7. 코드 구조

| 항목 | 내용 |
|---|---|
| 트리플 추출 코드 | `app/graph/knowledge_graph.py`, `app/graph/jit_builder.py` |
| 사용 추출기 클래스 | `SimpleLLMPathExtractor` (`llama_index.core.indices.property_graph`), 인덱스는 `PropertyGraphIndex` |
| llama-index 버전 | `llama-index-core==0.14.13` |
| 최종 리포트 생성 | `app/rag/graph_flow.py`의 `node_synthesize` (LangGraph 워크플로), `final_report` 필드로 반환 |
| 프롬프트 파일 | `core/lib/rlm/utils/prompts.py` |
| Python 환경 | `uv` 기반 (`pyproject.toml` + `uv.lock` + `.venv`). `KAG/` 하위는 별도 독립 `.venv` |
| ETL 코드 | `app/etl/extractor.py` — 매일경제 RSS 크롤러(BeautifulSoup), 트리플 추출과는 별개 |

### Part A 판정

- **(a) 코드 실행 가능 상태**: 정적으로는 가능. `uv.lock`/`.venv` 존재, 마지막 커밋이 리포트 정리 커밋인 것으로 보아 한 차례 완결된 실험 사이클을 거침. 단 compose의 `postgres_data`/`redis_data` named volume 미사용 등 정리되지 않은 부분 존재.
- **(b) 누적 그래프 데이터 잔존 가능성**: 높음. `neo4j_data`(4.2M, 실체 파일 존재), `chroma_db`(31M), `storage_claude/property_graph_store.json`(544K) 모두 로컬에 실체로 남아있음.
- **(c) 부족한 것**: `.env.example` 부재, Docker 데몬 미기동으로 볼륨/컨테이너 실체 미확인, compose 볼륨 선언 불일치, Neo4j 덤프 없음, RAGAS 전용 로그 미발견.

---

## Part B. 핵심 숫자 4개 딥다이브

Docker 데몬 기동 후 `docker compose up -d neo4j`로 Neo4j만 기동(허용된 범위), 실측 조회. 그래프 적재/ETL/DAG 트리거는 수행하지 않음.

### 1. 실제 기사 건수

| 항목 | 값 |
|---|---|
| JSON 파일 수 | 11 |
| 레코드 합계(파일 단순 합산) | 55 |
| **고유 기사 수** (id/url/title 모두 일치) | **40** |
| 중복 수집된 id 종류 | 9종 (2~4회 중복. `11951796` 4회, `11951793`/`11951782`/`11951811`/`11951806` 각 3회) |
| 날짜 분포 | 2026-01-26: 6건 / 2026-02-03: 34건 |
| 섹션 분포(url 경로 기준) | stock 12, society 10, politics 4, realestate 3, world 3, business 2, it 2, economy 2, culture 2 |
| content 비어있음 | 0/40 (평균 1,056자) |

**docstore 10건과의 차이**: 원본 문서는 정확히 5개(`ref_doc_info` 확인, news_id: 11952015/11952014/11952002/11951999/11951990)이며 각 2청크 → 10건. 이는 40개 고유 기사 중 **마지막 수집 배치(`mk_news_20260203_1400.json`) 5건과 정확히 일치**. 즉 40개 중 5개만 색인되었고 나머지 35개는 ETL 수집만 되고 색인되지 않은 상태.

### 2. Neo4j 실체

| 항목 | 값 |
|---|---|
| 전체 노드 수 | 27 |
| 전체 관계 수 | 18 |
| 노드 라벨 | `Entity` 1종뿐 (27개 전부) |
| 고립 노드(degree 0) | 0 |

관계 타입 전체(16종, 총 18건): `선고받았다`×3, `PREDICATE`×1, `전_회장`×1, `챙긴`×1, `혐의`×1, `인수_대상`×1, `피해자_수`×1, `선고`×1, `판단`×1, `피해입었다`×1, `선고했다`×1, `판단했다`×1, `조작했다`×1, `얻은_이익_없다`×1, `특징`×1, `모습`×1

**중요 발견**: 이 그래프는 강영권/에디슨모터스 주가조작 판결 관련 단일 주제로, 실제 수집된 40개 기사 어느 내용과도 무관. `neo4j_data`가 조사 이전부터 존재하던 bind-mount 데이터였다는 Part A 결과와 일치 — 이전의 별개 데모/테스트 실행 결과가 그대로 남아있는 것으로 판단.

### 3. 그래프 스키마

Neo4j(현재 로드된 데이터)와 `storage_claude/property_graph_store.json`(로컬 인메모리 저장소, 실제 코드가 설계한 스키마)을 비교.

| 항목 | Neo4j (현재 상태) | property_graph_store.json (로컬) |
|---|---|---|
| 엔티티 노드 라벨 | `Entity` | `entity`(134개) + `text_chunk`(10개, 임베딩 보유) |
| 노드 프로퍼티 | `id`만 존재 | `title, url, pub_date, news_id, category, sentiment, keywords, summary, triplet_source_id` 전부 존재 |
| 관계 프로퍼티 | 없음(`{}`) | `news_id`, `pub_date`, `triplet_source_id`(→ text_chunk 노드 id) 등 모든 관계(92/92건)에 부착 |
| 출처 문서 추적 방식 | 전혀 없음 | 별도 MENTIONS 관계/Document 노드가 아니라, **관계(edge)의 property로 news_id/triplet_source_id를 직접 복제**하는 방식 |
| 시점 정보(published_at 계열) | 없음 | 관계 property의 `pub_date`(문자열, 파싱 안 됨) — 노드가 아니라 관계에 존재 |

샘플 노드 5개 (Neo4j): `{id:"Subject"} {id:"Object"} {id:"강영권"} {id:"에디슨모터스"} {id:"징역 3년"}`

샘플 엣지 5개 (Neo4j):
```
Subject -[PREDICATE]-> Object          (props: {})
강영권 -[전_회장]-> 에디슨모터스        (props: {})
강영권 -[선고받았다]-> 징역 3년         (props: {})
강영권 -[챙긴]-> 1600억원 부당이득      (props: {})
강영권 -[혐의]-> 주가조작              (props: {})
```

**데이터 품질 이슈**: `Subject`/`Object`라는 리터럴 placeholder 노드가 양쪽 저장소 모두에서 발견됨. `SimpleLLMPathExtractor`가 트리플 파싱 실패 시 남기는 fallback 값으로 추정.

### 4. 교차 기사 다홉 경로

Neo4j는 provenance가 전혀 없어 원천적으로 분석 불가. `property_graph_store.json`(5개 기사, 92개 트리플, 134개 엔티티)로 대체 분석(순수 조회, Neo4j 미적재).

| 항목 | 값 |
|---|---|
| 연결 컴포넌트 수 | 44개 |
| 서로 다른 출처 문서 엔티티 쌍 중 hop≥2 연결 | 30쌍 |
| 경로 길이 분포 | hop2: 17 / hop3: 6 / hop4: 7 |
| 전체 hop≥2 쌍(186쌍) 중 단일문서 폐쇄 경로 | 153쌍 (82.3%) |
| 전체 hop≥2 쌍(186쌍) 중 교차문서 경로 | 33쌍 (17.7%) |

브릿지 엔티티는 단 3개뿐: `이재명 대통령`(11951990↔11952015), `이재명`(11951990↔11952015), `국민의힘`(11951990↔11951999). 5개 기사 중 3개만 서로 연결되고, 나머지 2개(구준엽 11952002, 태백축제 11952014)는 완전히 고립.

대표 경로 5개(전부 동일한 2개 브릿지 엔티티 경유):
```
[Spc삼립 시화공장](11952015) --질책했던--> [이재명 대통령] --언급(11951990)--> [부동산 이슈]
[Spc삼립 시화공장](11952015) --질책했던--> [이재명] --Is(11951990)--> [대통령]
[2026년 2월 3일](11952015) --화재 발생--> [Spc삼립 시화공장] --질책했던--> [이재명 대통령] --언급(11951990)--> [부동산 이슈]
[2026년 2월 3일](11952015) --화재 발생--> [Spc삼립 시화공장] --질책했던--> [이재명] --Is(11951990)--> [대통령]
[화재](11952015) --발생--> [Spc삼립 시화공장] --질책했던--> [이재명 대통령] --언급(11951990)--> [부동산 이슈]
```

### Part B 최종 판정

**"현재 그래프로 hop1과 traversal이 서로 다른 트리플을 반환하는 케이스를 8개 구성할 수 있는가?" → 부족**

근거:
- Neo4j에 실제 로드된 데이터 기준: 0건. 완전히 무관한 단일 주제 데이터이고 provenance가 없어 "교차 문서" 개념 자체가 성립하지 않음.
- 설계된 스키마가 정상 동작한 유일한 표본(로컬 5개 기사) 기준: hop≥2 교차문서 쌍 30개가 존재하나, 이는 단 3개의 브릿지 엔티티에 전적으로 의존. 대표 경로 5개가 전부 동일한 2개 엔티티를 경유 — 실질적으로 1~2개 경로 패턴의 변형에 불과할 가능성이 높음. 5개 기사 중 2개는 고립되어 후보군에서 배제됨.

**필요 규모 추정**: 40개 수집 기사 중 5개만 색인된 상태이므로, 나머지 35개를 색인하는 것만으로도 브릿지 후보가 늘어날 가능성이 큼(단, 실행하지 않고 추정만 함). 현재 관측된 브릿지 발생률(기사 5개당 브릿지 엔티티 3개, 재발성 인물/정당명 위주)을 감안하면, 서로 다른 브릿지 엔티티 8개 이상을 확보하려면 **최소 30~50건 규모**의, 인물·기관명이 반복 등장하는 기사를 색인해야 통계적으로 안정적인 8개 케이스가 나올 것으로 추정.

---

## 부록: 조사 중 수행한 상태 변경 작업

- `git tag v0-pre-experiment` (news-arena, HEAD `3ba9426`)
- `docker compose up -d neo4j` → 조회 후 `docker compose stop neo4j` (데이터/컨테이너 삭제 없음, 정지만)
- `storage_claude.backup-20260801/`, `result_airflow.backup-20260801/` 생성 (원본 무변경, 복사본만 추가)

ETL 실행, DAG 트리거, 그래프 적재, 코드 수정은 수행하지 않음.
