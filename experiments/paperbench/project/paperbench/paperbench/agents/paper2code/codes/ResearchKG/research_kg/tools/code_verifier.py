"""
定义一个代码验证逻辑
- 可执行性：代码能无报错运行 让LLM进行self-debug 5~10次 claude
- 语法正确性：代码是有效的Python代码 

1. 找出所有requirements.txt 提供给LLM
2. 让LLM输出代码片段的时候 加上一个requirements.txt

实例：
```
code(imolementation + test)
```

```
documentation(操作手册)
```

```
rquirements.txt
matplotlib==3.7.3
matplotlib==3.3.4
nlp==0.4.0
seaborn==0.13.1
tabulate==0.9.0
torch==2.1.1
tqdm==4.65.0
transformer_lens==1.12.0
```
说这段代码片段里用到的 或者强依赖的包 从requirements.txt里找出来 放在这里
"""

import os
import re
import json
from pylatexenc.latexwalker import LatexEnvironmentNode, LatexMacroNode, LatexWalker
from sentence_transformers import SentenceTransformer
import bibtexparser
from typing import Optional, Tuple, List, Literal, Dict, Any
from collections import deque
import logging
from pathlib import Path
from dataclasses import asdict
import glob
from adalflow.core.types import Document
from adalflow.core.db import LocalDB
import faiss
from rank_bm25 import BM25Okapi
import numpy as np
from rapidfuzz import fuzz

from ..llm import extract_backtick_text, extract_backtick_texts, extract_object, REPOSITORY_ANALYZER_PROMPT, CODE_REWRITER_PROMPT
from .code_rag import CodeRAG
from ..schema import Code, File, Paper, Node, Technique, Contribution, FileSnippet, CodeBlock
from .base_tool import BaseTool
from ..utils import *


class CodeVerifier(BaseTool):
    def __init__(
        self,
        model: str,
        memory: Optional[str] = None,
    ):
        super().__init__(model, memory)
        