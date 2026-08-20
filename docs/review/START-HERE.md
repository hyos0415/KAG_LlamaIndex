# GPT 프로젝트 작업 공간 — 문제 정의 비판 · 조사 · 서사

> 브랜치 `spec/gpt` · 기준 `audit/comparison` (39c2515)

## 이 워크트리의 성격

handoff §14 의 GPT 담당 범위를 위한 작업 공간이다.

```
문제 정의 비판 · KAG / Graph / 금융 AI 사례 조사 · schema 및 실험 설계 검토
failure interpretation · 결과 분석 · 발표 / 포트폴리오 서사 정리
"왜 이 기술이 필요한가?" 지속 검토
```

집필 대상이 아니라 **검토·조사 대상**이다. 공통 spec 집필은 `spec/cowork` 브랜치다.

---

## ⚠️ 읽는 순서를 먼저 결정할 것 — 독립성 문제

handoff §16 이 경고한 함정이 여기에 그대로 걸린다.

```
Claude 가 A 라는 가설을 docs 에 기록 → Codex 가 A 를 읽음 → GPT 가 A 를 읽음
→ 모두 A 라고 말함 →  이건 독립 검증이 아니다
```

**이미 이 저장소에는 감사 결론 3개가 커밋돼 있다.** GPT 에게 그것을 먼저 읽히면
GPT 의 비판은 독립 검증이 아니라 기존 결론의 재확인이 된다.

따라서 두 모드 중 하나를 **의식적으로 선택**하라.

### 모드 B — blind (권장, 먼저 할 것)

`docs/review/blind/README.md` 만 준다. handoff 원본과 원자료 위치만 있고
감사 결론은 없다. GPT 가 독립적으로 문제를 지목한 뒤, 아래 모드 A 결과와 비교한다.

이 비교가 실제 산출물이 된다 — 세 도구(Claude Code / Codex / GPT)가 서로를 보지 않고
같은 문제를 지목했다면 그건 근거가 되고, 갈라지면 그 지점이 검토 대상이다.

### 모드 A — 전체 (blind 이후)

읽는 순서:

1. `docs/handoff/finance_rule_graph_project_handoff.md` — 공통 맥락
2. `docs/audit/00-comparison.md` — 두 감사의 수렴/불일치, 합쳐진 수정 목록 17항
3. `docs/audit/codex-adversarial-review.md` — 반증 (GPT 과제와 가장 겹침)
4. `docs/audit/claude-architecture-audit.md` — 기존 구조 실측
5. `docs/CONTEXT.md` — 기존 저장소 진단 진실 원본 (발견 1~20)

---

## 하지 말 것

- **미정 사항을 임의로 확정하지 않는다** (CONTEXT §7.8). 결정은 사람 몫
- **구현 제안을 spec 확정으로 쓰지 않는다.** handoff §17 — 현재 우선순위는
  finance_verifier 완결
- **`docs/archive/` 참조 금지** (CONTEXT §7.7)

## 현재 상태

| 문서 | 상태 |
|---|---|
| `docs/review/agenda.md` | 검토 의제 — 조사 질문 목록. 답 없음 |
| `docs/review/research-log.md` | 사례 조사 기록 템플릿. 비어 있음 |
| `docs/review/narrative.md` | 포트폴리오 서사 골격. 미정 |
| `docs/review/blind/README.md` | blind 모드용 입력 (감사 결론 없음) |
