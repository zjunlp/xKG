from typing import Optional
from ..utils import get_llm_backend
import abc

class BaseTool(abc.ABC):
    def __init__(
        self,
        model: Optional[str] = None,
        memory: Optional[str] = None,
    ):
        self.model = model
        self.memory = memory
        self.backend = get_llm_backend()
        
    def set_model(self, model: str):
        self.model = model
    
    def set_memory(self, memory: str):
        self.memory = memory
        
    def run(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")
    
    