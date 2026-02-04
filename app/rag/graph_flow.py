import operator
from typing import Annotated, List, Union, TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from app.rag.langchain_solver import NewsLangChainSolver
from app.rag.decomposer import QueryDecomposer # 기존 로한 디컴포저 활용 가능

class AgentState(TypedDict):
    article_content: str
    facets: List[Dict[str, Any]]
    retrieved_docs: List[Any]
    rag_analysis: str
    graph_reasoning: str
    final_report: str
    need_graph: bool

class NewsAppGraph:
    def __init__(self):
        self.lc_solver = NewsLangChainSolver()
        self.decomposer = QueryDecomposer() # Need to wrap or adapt if needed
        
        # 그래프 빌드
        workflow = StateGraph(AgentState)
        
        # 노드 등록
        workflow.add_node("analyze_article", self.node_analyze)
        workflow.add_node("retrieve_rag", self.node_retrieve_rag)
        workflow.add_node("reason_graph", self.node_reason_graph)
        workflow.add_node("synthesize", self.node_synthesize)
        
        # 엣지 정의
        workflow.set_entry_point("analyze_article")
        workflow.add_edge("analyze_article", "retrieve_rag")
        
        # 조건부 엣지: 그래프 분석 필요 여부에 따라 분기
        workflow.add_conditional_edges(
            "retrieve_rag",
            self.should_use_graph,
            {
                "yes": "reason_graph",
                "no": "synthesize"
            }
        )
        workflow.add_edge("reason_graph", "synthesize")
        workflow.add_edge("synthesize", END)
        
        self.app = workflow.compile()

    async def node_analyze(self, state: AgentState):
        """기사 분석 및 검색 관점(Facets) 도출"""
        print("--- ARTICLE ANALYSIS ---")
        # 기존 디컴포저 로직 활용 (필요시 LC 규격으로 수정)
        facets = await self.decomposer.decompose_article(state["article_content"])
        return {"facets": [f.dict() for f in facets.facets], "need_graph": True} # 우선 True 고정

    async def node_retrieve_rag(self, state: AgentState):
        """LangChain 기반 하이브리드 RAG 수행"""
        print("--- RAG RETRIEVAL ---")
        result = await self.lc_solver.solve(state["article_content"])
        return {
            "rag_analysis": result["analysis"],
            "retrieved_docs": result["docs"]
        }

    async def node_reason_graph(self, state: AgentState):
        """LlamaIndex PGI 기반 심층 관계 분석"""
        print("--- GRAPH REASONING (PGI) ---")
        # RAG에서 검색된 문서를 바탕으로 심층 분석 수행
        result = await self.lc_solver.pgi_reasoning(
            docs=state["retrieved_docs"],
            query=f"다음 원문 기사와 관련된 인물/사건 관계를 지식 그래프로 분석해줘: {state['article_content'][:200]}"
        )
        return {"graph_reasoning": result["graph_analysis"]}

    async def node_synthesize(self, state: AgentState):
        """모든 정보를 통합하여 최종 리포트 작성"""
        print("--- FINAL SYNTHESIS ---")
        report = f"### [RAG 분석]\n{state['rag_analysis']}\n\n### [심층 관계 분석]\n{state.get('graph_reasoning', 'N/A')}"
        return {"final_report": report}

    def should_use_graph(self, state: AgentState):
        return "yes" if state.get("need_graph") else "no"

    async def run(self, article: str):
        config = {"configurable": {"thread_id": "1"}}
        inputs = {"article_content": article}
        
        # ainvoke를 사용하여 최종 상태를 직접 받아옴
        final_state = await self.app.ainvoke(inputs, config)
        return final_state
