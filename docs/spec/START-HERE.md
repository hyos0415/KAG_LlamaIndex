# Claude Cowork 작업 공간 — 공통 spec 집필

> 브랜치 `spec/cowork` · 기준 `audit/comparison` (39c2515)

## 이 워크트리의 성격

**신규 저장소의 공통 spec 초안을 집필하는 스테이징 영역이다.**

handoff §13 은 새 프로젝트를 신규 저장소에서 시작하고 이 저장소의 git history 를
잇지 않기로 정했다. 따라서 `docs/spec/` 아래 문서는 **여기서 완성한 뒤 신규 저장소로
옮길 초안**이며, 이 저장소를 새 프로젝트로 리팩토링하는 것이 아니다.

handoff §14 의 역할 분담상 이 공간의 담당은 다음이다.

```
공통 project spec 설계 · 문서 구조 관리 · architecture decision 정리
schema / scope / non-goal 정리 · 코드와 문서 consistency 관리
```

## 읽는 순서

1. `docs/handoff/finance_rule_graph_project_handoff.md` — 공통 맥락 (원본)
2. `docs/audit/00-comparison.md` — **여기서 시작.** 두 독립 감사의 차이와
   합쳐진 handoff 수정 목록 17항(§5), 사람이 결정할 4건(§6)
3. `docs/audit/claude-architecture-audit.md` — 기존 구조·스키마·흐름 실측
4. `docs/audit/codex-adversarial-review.md` — handoff 가정에 대한 반증
5. `docs/CONTEXT.md` — 기존 저장소의 진단 진실 원본 (발견 1~20)

## 하지 말 것

- **미정 사항을 임의로 확정하지 않는다.** `docs/spec/decisions/README.md` 의
  열린 결정은 사람이 정한다. 후보와 trade-off 를 정리하는 것이 이 공간의 일이다
- **구현하지 않는다.** handoff §17 — 현재 우선순위는 finance_verifier 완결이다
- **기존 저장소 코드를 수정하지 않는다.** `app/**`, `scripts/**` 는 감사 대상이며
  이 브랜치에서 손대지 않는다
- **`docs/archive/` 를 참조하지 않는다** (CONTEXT.md §7.7)

## 집필 원칙 — handoff §16

중요한 결정에는 네 항목을 함께 남긴다. `docs/spec/decisions/0000-template.md` 참고.

```
Decision · Evidence · Alternative · Why rejected
```

이 저장소의 발견 16(사전 등록은 오류를 막지 않고 드러낸다)이 근거다. 기준을 날짜와
함께 기록해두면 틀렸을 때 조용히 합리화하지 않고 무엇이 언제 왜 틀렸는지 지목할 수 있다.

## 현재 상태

| 문서 | 상태 |
|---|---|
| `docs/spec/project_spec.md` | 골격만. 12개 최소 항목 전부 **미정** |
| `docs/spec/schema.md` | 골격만. 노드/리터럴/논리식 구분표가 **최우선 블로킹 항목** |
| `docs/spec/evaluation.md` | 골격만. factorial matrix 와 지표 사전 등록 대기 |
| `docs/spec/decisions/` | 템플릿 + 열린 결정 목록 |
