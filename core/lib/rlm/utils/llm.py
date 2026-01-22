"""
OpenAI Client wrapper specifically for GPT-5 models.
"""

import os
from typing import Optional, Union, List, Dict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.model = model
        self.client = OpenAI(api_key=self.api_key)

        # Implement cost tracking logic here.
    
    def completion(
        self,
        messages: Union[List[Dict[str, str]], str],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stop: Optional[List[str]] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> str:
        try:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            elif isinstance(messages, dict):
                messages = [messages]

            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                stop=stop,
                timeout=timeout,
                **kwargs
            )
            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"Error generating completion: {str(e)}")