"""
代码-文本图的数据结构
"""
from .code import *
from .paper import *

@dataclass
class CodeBlock:
    implementation: str
    test: str
    documentation: str
    package: List[str] = field(default_factory=list)
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeBlock':
        known_field_names = {f.name for f in fields(cls)}
        init_data = {k: v for k, v in data.items() if k in known_field_names}
        return cls(**init_data)

@dataclass
class Technique:
    name: str
    description: str
    code: Optional[CodeBlock] = None
    weight: Optional[float] = 0.0
    components: Optional[List['Technique']] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Technique':
        init_data = data.copy()
        if 'code' in init_data and init_data.get('code'):
            init_data['code'] = CodeBlock.from_dict(init_data['code'])
        if 'components' in init_data and init_data.get('components'):
            init_data['components'] = [cls.from_dict(c) for c in init_data['components']]
            
        return cls(**{k: v for k, v in init_data.items() if k in {f.name for f in fields(cls)}})

@dataclass
class Node:
    paper_title: Optional[str] = None
    paper_abstract: Optional[str] = None
    paper_references: Optional[List[str]] = field(default_factory=list)
    techniques: Optional[List[Technique]] = field(default_factory=list)
    resources: Optional[List[str]] = field(default_factory=list)
    findings: Optional[List[str]] = field(default_factory=list)
    code_file: Optional[str] = None
    code_overview: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        init_data = data.copy()
        if 'techniques' in init_data and init_data.get('techniques'):
            init_data['techniques'] = [Technique.from_dict(t) for t in init_data['techniques']]

        return cls(**{k: v for k, v in init_data.items() if k in {f.name for f in fields(cls)}})