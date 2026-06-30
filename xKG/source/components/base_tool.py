from typing import Optional
from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar
from ..utils import get_llm_backend, get_process_path, sanitize_filename

_current_paper_var: ContextVar = ContextVar('_current_paper', default=None)


class BaseTool:

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

    @staticmethod
    @contextmanager
    def set_current_paper(paper):
        token = _current_paper_var.set(paper)
        try:
            yield
        finally:
            _current_paper_var.reset(token)

    @staticmethod
    def get_current_paper():
        return _current_paper_var.get()

    def get_output_path(self, component_name: str) -> Path:
        base_path = get_process_path()
        paper = _current_paper_var.get()
        if paper:
            paper_safe = sanitize_filename(paper.title)
            output_dir = base_path / f"{paper_safe}_{component_name}"
        else:
            output_dir = base_path / component_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def run(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")
