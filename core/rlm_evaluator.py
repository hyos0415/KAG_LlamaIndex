import asyncio
import json
import re
from typing import List, Dict, Any, Optional
from core.lib.rlm.utils.llm import OpenAIClient
from core.lib.rlm.rlm_repl import RLM_REPL

class RLMEvaluator:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5.1"):
        self.client = OpenAIClient(api_key=api_key, model=model)
        self.api_key = api_key
        self.model = model
        self.max_depth = 3

    async def evaluate(self, proposal: str) -> Dict[str, Any]:
        """
        Main entry point for evaluating a proposal.
        Decomposes the proposal into claims and verifies each.
        """
        # 1. Decomposition
        claims = await self.decompose_claims(proposal)
        
        verification_results = []
        for claim in claims:
            # 2. Tool Selection & 3. Verification
            result = await self.verify_claim(claim, depth=0)
            verification_results.append(result)
            
        # 4. Trace Generation
        trace = {
            "proposal": proposal,
            "claims": verification_results,
            "score": self.calculate_aggregate_score(verification_results)
        }
        return trace

    async def decompose_claims(self, text: str) -> List[str]:
        prompt = [
            {"role": "system", "content": "You are a helpful assistant that breaks down complex texts into atomic, verifiable claims."},
            {"role": "user", "content": f"Decompose the following text into atomic claims. Return a JSON list of strings (e.g. [\"claim1\", \"claim2\"]).\n\nText: {text}"}
        ]
        
        response = await asyncio.to_thread(self.client.completion, prompt)
        return self._parse_json_list(response, fallback=[text])

    async def verify_claim(self, claim: str, depth: int) -> Dict[str, Any]:
        if depth >= self.max_depth:
            return {"claim": claim, "verified": False, "reason": "Max depth reached", "confidence": 0.5}

        # Unified Verification: Use RLM for all claims
        # The RLM Agent is capable of both logical reasoning (code) and semantic analysis (sub-LLM queries).
        return await self.verify_with_rlm(claim)

    async def verify_with_rlm(self, claim: str) -> Dict[str, Any]:
        # Use RLM_REPL to verify
        # RLM_REPL handles its own context and recursion
        rlm = RLM_REPL(api_key=self.api_key, model=self.model, enable_logging=False)
        
        # General purpose query allowing for both code execution and semantic reasoning
        query = (
            f"Verify the following claim: '{claim}'.\n"
            "You can use Python code to calculate statistics or logic, OR use 'llm_query' "
            "to ask sub-questions if you need to reason about facts or semantics.\n"
            "Provide a final answer stating whether it is True, False, or Unverified, with a confidence score."
        )
        
        try:
            # RLM_REPL.completion takes (context, query). 
            # We pass an empty list for context initially, relying on the model's internal knowledge 
            # or data we might inject later (e.g. from GraphRAG).
            final_answer = await asyncio.to_thread(rlm.completion, context=[], query=query)
            return {
                "claim": claim,
                "verified": True,
                "output": final_answer,
                "confidence": 0.9 # Placeholder: extraction logic would go here
            }
        except Exception as e:
            return {
                "claim": claim,
                "verified": False,
                "error": str(e),
                "confidence": 0.0
            }
        
    def _parse_json_list(self, text: str, fallback: List[str]) -> List[str]:
        try:
            # Try to find JSON block
            match = re.search(r'[[.*]]', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return json.loads(text)
        except:
            return fallback
            
    def calculate_aggregate_score(self, results: List[Dict[str, Any]]) -> float:
        # Simple average of confidence
        if not results:
            return 0.0
        total = sum(r.get("confidence", 0) for r in results)
        return total / len(results)
