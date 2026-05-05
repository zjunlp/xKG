"""
Paper数据类
"""

from dataclasses import dataclass, field, fields, asdict
from typing import List, Optional, Dict, Any, Set
import json

CONTRIBUTION_TYPE = ["Methodology", "Technique", "Finding", "Resource"]

@dataclass
class Section:
    id: str
    name: str = ""
    content: str = ""
    subsection: Optional[List['Section']] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Section':
        known_fields = {f.name for f in fields(cls)}
        
        init_data = {
            k: ([cls.from_dict(s) for s in v] if k == 'subsection' and v else v)
            for k, v in data.items() if k in known_fields
        }
        
        return cls(**init_data)

@dataclass
class Contribution:
    name: str
    type: str
    description: str
    components: Optional[List['Contribution']] = field(default_factory=list)
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Contribution':
        known_fields = {f.name for f in fields(cls)}
        init_data = {
            k: ([cls.from_dict(c) for c in v] if k == 'components' and v else v)
            for k, v in data.items() if k in known_fields
        }
        
        return cls(**init_data)

@dataclass
class Paper:
    # --- Exact Match ---
    title: str
    abstract: str = ""
    url: str = ""
    year: int = -1
    authors: List[str] = None
    sections: Optional[List[Section]] = None
    references: Optional[List[str]] = None
    equations: Optional[List[str]] = None
    # --- LLM Extract ---
    introduction: Optional[str] = None
    code_url: Optional[str] = None
    contributions: Optional[List[Contribution]] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Paper':
        known_fields = {f.name for f in fields(cls)}
        init_data = {}
        for k, v in data.items():
            if k not in known_fields:
                continue
            if k == 'sections' and v:
                init_data[k] = [Section.from_dict(s) for s in v]
            elif k == 'contributions' and v:
                init_data[k] = [Contribution.from_dict(c) for c in v]
            else:
                init_data[k] = v
        return cls(**init_data)