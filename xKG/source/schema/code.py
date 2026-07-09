"""
Code repository schema definitions
"""
from dataclasses import dataclass, field, fields
from typing import List, Optional, Dict, Any, Set

@dataclass
class File:
    name: str
    content: str = ""
    is_implementation: bool = False

@dataclass
class Code:
    # --- Exact Match ---
    name: str
    readme: str = ""
    file_tree: str = ""
    file_list: List[File] = field(default_factory=list)
    # --- LLM Extract ---
    overview: str = ""
    paper: Optional[str] = None    
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Code':
        known_field_names = {f.name for f in fields(cls)}
        
        init_data = {k: v for k, v in data.items() if k in known_field_names}

        if 'file_list' in init_data:
            init_data['file_list'] = [File(**f_data) for f_data in init_data['file_list']]
        
        return cls(**init_data)
    
@dataclass
class FileSnippet:
    file_name: str
    code_list: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileSnippet':
        known_field_names = {f.name for f in fields(cls)}
        init_data = {k: v for k, v in data.items() if k in known_field_names}
        return cls(**init_data)