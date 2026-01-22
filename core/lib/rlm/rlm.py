from abc import ABC, abstractmethod
from typing import List, Dict, Union

class RLM(ABC):
    @abstractmethod
    def completion(self, context: Union[List[str], str, Dict[str, str]], query: str) -> str:
        pass

    @abstractmethod
    def cost_summary(self) -> Dict[str, float]:
        pass

    @abstractmethod
    def reset(self):
        pass
