from typing import List, Dict, Any, Optional
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.etl.storage import StorageManager

class NewsLangChainSolver:
    def __init__(self, model_name: str = "claude-sonnet-4-0"):
        self.llm = ChatAnthropic(model=model_name, temperature=0)
        self.storage_manager = StorageManager()
        self.retriever = self.storage_manager.get_hybrid_retriever()
        
    def get_rag_chain(self):
        """
        뉴스 분석 및 답변 생성을 위한 기본 RAG 체인
        """
        template = """당신은 매일경제 뉴스를 분석하는 전문 저널리스트 에이전트입니다.
제공된 [컨텍스트] 뉴스 기사들을 바탕으로 [기사 원문]에 대한 심층 분석을 수행하세요.

[기사 원문]: {article_content}

[컨텍스트 뉴스]:
{context}

주요 지침:
1. 관련 뉴스들 간의 연결 고리를 찾으세요.
2. 사실 관계(Fact)에 기반하여 분석하세요.
3. 산업에 미치는 영향이나 향후 전망을 포함하세요.

답변은 한국어로 작성하며, 전문적이고 통찰력 있는 논조를 유지하세요.
"""
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            prompt 
            | self.llm 
            | StrOutputParser()
        )
        return chain

    async def solve(self, article_content: str):
        """
        단일 기사에 대한 RAG 분석 수행
        """
        # 1. 검색
        docs = await self.retriever.ainvoke(article_content)
        context_text = "\n\n".join([d.page_content for d in docs])
        
        # 2. 분석 생성
        chain = self.get_rag_chain()
        result = await chain.ainvoke({
            "article_content": article_content,
            "context": context_text
        })
        
        return {
            "analysis": result,
            "sources": [d.metadata for d in docs],
            "docs": docs # 추후 PGI를 위해 원본 문서 객체 포함
        }

    async def pgi_reasoning(self, docs: List[Any], query: str):
        """
        검색된 문서들을 바탕으로 LlamaIndex PGI 심층 분석 수행
        """
        from llama_index.core import Document as LlamaDoc
        from llama_index.core.schema import NodeWithScore, TextNode
        from app.graph.jit_builder import JITGraphAnalyzer
        
        # LangChain Doc -> LlamaIndex Node 변환
        nodes = []
        for d in docs:
            node = TextNode(text=d.page_content, metadata=d.metadata)
            nodes.append(NodeWithScore(node=node, score=1.0))
            
        analyzer = JITGraphAnalyzer()
        response, index = analyzer.build_and_analyze(nodes, query)
        
        return {
            "graph_analysis": str(response),
            "triples": analyzer.get_graph_triples(index)
        }
