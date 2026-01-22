# Project Specification: AI News Arena (Bridging & Verification System)

## 1. Project Overview

- **Project Name:** AI News Arena
- **Goal:** A participatory platform where citizens propose social issues in article form, AI verifies them against news data, and citizens vote to identify significant issues.
- **Core Value:** Moving beyond "most viewed" news to "verified and socially significant" issues using AI and collective intelligence.

## 2. Problem Definitions

- **Agenda Setting Bias:** News agendas are currently set by a few editors, missing important but less "clickable" issues.
- **Verification Gap:** Citizens lack the tools to verify how their proposed issues relate to existing reporting.
- **Information Distortion:** Social media spreads emotionally charged but unverified information.

## 3. System Architecture (Revised & Optimized)

### 3.1 High-Level Data Flow

1. **Submission:** User submits a draft article/proposal.
2. **Ingestion:** Backend ingests text and triggers the **Async RLM Engine**.
3. **Verification (Hybrid):**
    - **GraphRAG:** Checks semantic context, entity relationships, and existing coverage.
    - **Python REPL (New):** Verifies statistical claims, logic, and data trends.
4. **Evaluation:** AI generates scores for Factuality, Novelty, and Impact.
5. **Ranking:** The **Bridging Algorithm** calculates scores based on diverse user support.

### 3.2 Directory Structure

Plaintext

`/
├── data/                   # News corpus & GraphRAG artifacts
├── core/                   # AI Logic
│   ├── graph_engine.py     # Async wrapper for GraphRAG (Local/Global)
│   ├── rlm_evaluator.py    # Recursive logic with Tool Selection (Search/Code)
│   └── code_sandbox.py     # Python REPL environment
├── server/                 # API Layer
│   ├── main.py             # FastAPI endpoints (Async)
│   └── bridging.py         # Matrix factorization/Consensus scoring
└── ui/                     # Presentation
    └── app.py              # Streamlit MVP [cite: 28]`

## 4. Core Technology Specifications

### Module A: Optimized GraphRAG Engine

- **Objective:** To analyze news complexity and social context using Knowledge Graphs.
- **Implementation Strategy:**
    - **Async Execution:** Must wrap GraphRAG calls in `asyncio` to prevent server blocking during retrieval.
    - **Dual-Mode Search:**
        - **Local Search:** For checking specific entities (People, Places) and their direct relationships.
        - **Global Search:** For understanding broader community themes and "Impact" assessment (using Map-Reduce summaries).
    - **Context Pruning:** Implement a token-limit logic to summarize retrieved sub-graphs before feeding them to the LLM to reduce latency and cost.

### Module B: Hybrid RLM (Recursive Language Model)

- **Objective:** To perform deep logical verification by decomposing claims.
- **Algorithm (Revised):**
    1. **Decomposition:** Break the user proposal into atomic claims.
    2. **Tool Selection (Classifier):** For each claim, determine if it needs:
        - `Tool: Knowledge_Search` (for historical facts/events).
        - `Tool: Code_Interpreter` (for statistics/logic/math).
    3. **Recursive Verification Loop:**
        - If confidence < Threshold (0.8), generate **Sub-queries**.
        - Recursively call the engine with sub-queries (up to `MAX_DEPTH=3`).
    4. **Trace Generation:** Output a JSON "Verification Trace" that explains the logic transparency to the user.

### Module C: Bridging Ranking Algorithm

- **Objective:** Prioritize issues with cross-group consensus rather than simple majority votes.
- **Logic:**
    - Score=log(TotalVotes)×(1+ConsensusFactor)
    - **ConsensusFactor:** Derived from the inverse variance of votes across different user clusters (e.g., ensuring an issue is supported by distinct demographic/interest groups).

## 5. Development Roadmap (MVP)

- **Scope:** A functioning demo from Article Submission → AI Verification → Voting → Ranking.
- **Exclusions:** User authentication, full security, and live media CMS integration.

### Phase 1: Foundation (Days 1-2)

- Set up FastAPI (Async) and Streamlit.
- Implement `NewsKnowledgeGraph` class using `graphrag`.
- Ingest sample news dataset.

### Phase 2: Intelligence (Days 3-5)

- Implement `RLMEvaluator` with `_verify_recursive` logic.
- Integrate `PythonREPL` for data/logic verification.
- Connect Claude API (primary) with fallback to OpenAI.

### Phase 3: Interaction (Days 6-7)

- Build the Bridging Score calculation logic.
- Finalize Streamlit UI for "Verification Trace" visualization.

## 6. Tech Stack

- **LLM:** Claude 3.5 Sonnet (Reasoning/Coding), OpenAI GPT-4o (Fallback).
- **RAG:** Microsoft GraphRAG (Knowledge Graph Indexing).
- **Backend:** Python 3.11+, FastAPI, Asyncio.
- **Frontend:** Streamlit.
- **Sandboxing:** E2B or LangChain PythonREPL (for safe code execution).