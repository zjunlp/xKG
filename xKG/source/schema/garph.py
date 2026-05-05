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
class VerifiableCodeBlock:
    """Code block data structure - corresponds to code structure in JSON"""
    implementation: str  # Main code implementation
    test: str = ""  # Test code
    documentation: str = ""  # Documentation
    package: List[str] = field(default_factory=list)  # Dependency package list
    language: str = "python"  # Programming language
    
    # Verification result attributes
    is_executable: bool = False
    is_syntactically_correct: bool = False
    error_messages: List[str] = field(default_factory=list)
    debug_attempts: int = 0
    execution_output: Optional[str] = None
    final_requirements: List[str] = field(default_factory=list)
    fixed_code: Optional[str] = None
    
    def __post_init__(self):
        if self.package is None:
            self.package = []
        if self.error_messages is None:
            self.error_messages = []
        if self.final_requirements is None:
            self.final_requirements = self.package.copy()
        if self.fixed_code is None:
            self.fixed_code = self.implementation
    
    @property
    def code(self) -> str:
        """Get main code"""
        return self.implementation
    
    @property
    def requirements(self) -> List[str]:
        """Get dependency package list"""
        return self.package
    
    @property
    def test_code(self) -> Optional[str]:
        """Get test code"""
        return self.test if self.test else None
    
    @classmethod
    def from_code(cls, code: str, requirements: List[str] = None, documentation: str = "", test_code: str = ""):
        """Create VerifiableCodeBlock instance from code string"""
        return cls(
            implementation=code,
            test=test_code,
            documentation=documentation,
            package=requirements or []
        )
        
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