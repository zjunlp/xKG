""" 
构建code-text对
1. code-paper
2. 纯code
3. 纯paper
"""

import os
import json
from typing import Optional, Tuple, List, Dict
import logging
from dataclasses import asdict
import glob
import faiss
from rank_bm25 import BM25Okapi
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sentence_transformers import SentenceTransformer
import copy

from ..schema import Code, Paper, Node, Technique, Contribution, FileSnippet, CodeBlock
from .base_tool import BaseTool
from .code_verifier import CodeVerifier
from .code_rag import CodeRAG
from .paper_rag import PaperRAG
from ..utils import *
from ..llm import extract_backtick_text, extract_backtick_texts, extract_object, REPOSITORY_ANALYZER_PROMPT, CODE_REWRITER_PROMPT

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 2
REFERENCE_SIMILARITY_THRESHOLD = 95
RETRIEVE_TECHNIQUES= 10

class KnowledgeIndex:
    """
    知识索引库
    - 为 name 和 description 同时构建 BM25 和 FAISS 索引
    - 预先计算并存储所有向量，以支持高效的成对比较
    - 建立 Technique 对象到其唯一 ID 的快速反向映射
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        
        # 索引
        self.bm25_name: BM25Okapi = None
        self.faiss_name: faiss.Index = None
        self.bm25_desc: BM25Okapi = None
        self.faiss_desc: faiss.Index = None
        self.bm25_title: BM25Okapi = None
        
        # 数据和映射
        self.title_indexed_nodes: List[Node] = []
        self.flat_techniques: List[Technique] = []
        self.id_to_technique: Dict[int, Technique] = {}
        self.tech_to_id: Dict[Technique, int] = {}
        self.tech_id_to_node_id: Dict[int, int] = {}
        
        # 预计算的向量，用于高效成对比较
        self.name_vectors: np.ndarray = None
        self.desc_vectors: np.ndarray = None

    def _flatten_and_map_techniques(self, nodes: List[Node]):
        """扁平化所有 Node 中的 Technique，并建立正向和反向 ID 映射"""
        self.flat_techniques.clear()
        self.id_to_technique.clear()
        self.tech_to_id.clear()
        self.tech_id_to_node_id.clear()
        
        def _recursive_walk(techniques: List[Technique], node_id: int):
            for tech in techniques:
                tech_id = len(self.flat_techniques)
                self.flat_techniques.append(tech)
                self.id_to_technique[tech_id] = tech
                self.tech_to_id[id(tech)] = tech_id
                self.tech_id_to_node_id[tech_id] = node_id
                if tech.components:
                    _recursive_walk(tech.components, node_id)

        for node_idx, node in enumerate(nodes):
            if node.techniques:
                _recursive_walk(node.techniques, node_idx)

    def build_indexes(self, nodes: List[Node]):
        """为所有 Technique 构建索引并存储预计算的向量。"""
        logger.info("Building hybrid indexes and pre-calculating vectors...")
        
        self._flatten_and_map_techniques(nodes)
        if not self.flat_techniques:
            logger.warning("No techniques found to index.")
            return

        names_corpus = [tech.name for tech in self.flat_techniques]
        descs_corpus = [tech.description for tech in self.flat_techniques]

        # 1. 构建 BM25 索引 (同时为 name 和 description)
        logger.info("Building BM25 indexes for names and descriptions...")
        self.bm25_name = BM25Okapi([name.split() for name in names_corpus])
        self.bm25_desc = BM25Okapi([desc.split() for desc in descs_corpus])
        
        # 2. 编码并存储向量 (一次性完成)
        logger.info("Encoding names and descriptions for FAISS and pairwise comparison...")
        self.name_vectors = self.model.encode(names_corpus, show_progress_bar=True, convert_to_numpy=True).astype('float32')
        self.desc_vectors = self.model.encode(descs_corpus, show_progress_bar=True, convert_to_numpy=True).astype('float32')

        # 3. 构建 FAISS 索引
        if self.name_vectors.size > 0:
            dim = self.name_vectors.shape[1]
            # Name FAISS Index
            index_name = faiss.IndexFlatL2(dim)
            self.faiss_name = faiss.IndexIDMap(index_name)
            ids = np.array(range(len(self.flat_techniques))).astype('int64')
            self.faiss_name.add_with_ids(self.name_vectors, ids)

            # Description FAISS Index
            index_desc = faiss.IndexFlatL2(dim)
            self.faiss_desc = faiss.IndexIDMap(index_desc)
            self.faiss_desc.add_with_ids(self.desc_vectors, ids)

        logger.info(f"Hybrid indexes built successfully for {len(self.flat_techniques)} techniques.")
        
        # 4. 构建标题 BM25 索引
        logger.info("Building BM25 index for paper titles...")
        
        self.title_indexed_nodes.clear() # 清空旧数据
        nodes_with_titles = [node for node in nodes if node.paper_title]
        
        if nodes_with_titles:
            self.title_indexed_nodes = nodes_with_titles
            tokenized_titles = [node.paper_title.split() for node in self.title_indexed_nodes]
            self.bm25_title = BM25Okapi(tokenized_titles)
            logger.info(f"BM25 title index built for {len(self.title_indexed_nodes)} papers.")
        else:
            self.bm25_title = None
            logger.warning("No papers with titles found to build title index.")

# 给整个paper-code对进行编码
class GraphHandler(BaseTool):
    def __init__(
        self,
        model: Optional[str] = None,
        memory: Optional[str] = None,
        embedding_model: str = None
    ):
        super().__init__(model, memory)    
        if embedding_model is None:
            embedding_model = get_config().get('embedding_model', 'all-MiniLM-L6-v2')    
            
        self.paper_rag = PaperRAG()
        self.code_rag = CodeRAG()
        self.code_verifier = CodeVerifier(model=self.model) # 这里定义了CodeVerifier 如果有啥别的参数传入就同时给传入
        self.knowledge_base = None
        self.knowledge_index = KnowledgeIndex(model_name=embedding_model)
    
    @property
    def _prompt_rewrite_paper(self):
        prompt = """
# Task
Your task is to refine and enhance the description of a technical concept extracted from a research paper {paper}. The goal is to produce a clear, concise, and comprehensive description that accurately captures the essence of the technique.

# Input
1.  Technical Concept from the paper {paper}: 
{technique}

2. Relevant Excerpt of this Technique:
{excerpt}

# Output
Return a precise and comprehensive description, presented as a single, continuous paragraph written in a comprehensive, academic style. Avoid using bullet points, numbered lists, or other form of itemization.

1. Ensure the technique precisely matches the original description. DO NOT alter, expand, or reduce the scope of the technique. Ignore other related techniques and only FOCUS ON this technique.
2. Strictly adhering to the original description, augment its implementation details based on the provided excerpts. All formulas, parameter configurations, and implementation details must be extracted from the given excerpts, ensuring strict adherence to them. Avoid any summarization, inference, or omission.
3. If the excerpts offer no new information, leave the description unchanged. Your response MUST be based solely on the original description and provided excerpts. The inclusion of ANY external information or fabricated details is strictly forbidden!!!
4. Ensure that the provided description is precise, complete, and possesses sufficient detail to correspond to a specific implementation.

Now please think and reason carefully, and wrap your final answer between two ``` in the end.
        """
        return prompt
    
    @property
    def _prompt_check_snippets(self):
        prompt = """
# Task
Your task is to analyze a list of code files retrieved from a GitHub repository, and identify which files are directly relevant to the implementation of a specific technical concept defined in an academic paper {paper}.

# Input
1.  Technical Concept Definition from the paper {paper}: 
{technique}

2. Overview of the Code repository:
{overview}

3. Relevant Code Files:
{file_snippets}

# Output
Return a list of filenames formatted as ["xx", "xx", ...], sorted in **descending** order of relevance of the technical concept.
1. Exclude any file not DIRECTLY correspond to the concrete implementation and configurarion of this technique (e.g., tests, documentation, other technique implementation).
2. Confirm that a direct implementation exists within your provided file list. If no such implementation can be found, return None.
3. Return the nitrogen list even if there's only one file. 

Now please think and reason carefully, and wrap your final answer between two ``` in the end.
        """
        return prompt
    
    @property
    def _prompt_check_techniques(self):
        prompt = """
# Task
Your task is to analyze a list of technique implementations retrieved from the knowledge base, and identify which techniques are directly relevant to the implementation of a specific technical concept.

# Input
1. Technical Concept Definition: 
{technique}

2. Relevant Technique implementations:
{relevant_techniques}

# Output
Return a list of (technique_name, apply_guidance) tuples formatted as [("", ""), ("",""), ...], sorted in descending order of relevance to the technical concept. 
The guidance should be a short explanation of how the technique applies to the current scenario and what modifications are needed for adaptation. Use clear and definite wording, avoiding parentheses.

1. Exclude any techniques not relevant to the concrete implementation of this technique. 
2. Ensure the returned technique name exactly matches the original one.
3. For technologies with identical core definitions, keep the one whose application is most relevant.
4. If no such technique can be found, return None.
5. Return the nitrogen list even if there's only one relevant technique. 

Now please think and reason carefully, and wrap your final answer between two ``` in the end.
        """
        return prompt
    
    @property
    def _prompt_rewrite_code(self):
        prompt = """
# Task
Your task is to transform a collection of disparate source code snippets, which are the official implementation of a technique component from a research paper {paper}, into a single, self-contained, and executable code block. The final code block must be clean, well-documented, and easy for others to understand and run.

# Input
1. Abstract of the paper {paper}:
{abstract}

2. Technical Concept Definition from the paper {paper}:
{technique}

3. Relevant Code Files:
{file_snippets}

# Workflow
1. Analyze: Understand the technique's inputs, outputs and workflow from the paper. Focus ONLY on THIS technique, ignoring the mentioned context and related techniques.
2. Isolate & Extract: Based on the description of the technique, determine what is its PRECISE role and functionality, and extract ONLY the code you identified as belonging to {technique}. Other mentioned associated techniques **MUST BE IGNORED AND EXCLUDED**.
3. Refactor: Integrate the extracted code by removing hard-coded values, isolating the core algorithm, and standardizing it with proper documentation and type hints.
4. Assemble & Test: Build the final script and add an test block as a runnable example. Ensure accuracy and conciseness, avoiding unnecessary output.
5. Documentation: Write a brief and concise documentation of the code logic, configurable options, and usage in 5-10 sentences.

# Requirements
1. Dependency Management: Ensure all necessary imports and dependencies are included at the beginning of the code block.
2. Fidelity to the Original Technique: Strictly follow the description of the given technique to organize the code. ONLY focus on the implementation that DIRECTLY corresponds to THIS technique!!! (e.g., if the technique is a loss function definition, implement only the code for its calculation. Ignore all other parts of the algorithm's implementation, even if provided in the code snippets.)
3. Code Encapsulation and Documentation:
- Encapsulate the core logic of the technique into one or more functions/classes.
- Every function and class method must include a comprehensive docstring explaining its purpose, parameters, and return values.
- All function arguments and return values must have clear type hints.
- Preserve original parameters and comments from the source code.
4. Reproducibility and Testing:
- A main execution block, start with the comment `# TEST BLOCK`, is required at the end of the file, which serves as a practical usage example and a test case.
- The test case should use parameters from the code repository or paper. If missing, create and state your own defaults.
5. Fidelity to the Original Logic: 
- You must strictly adhere to the algorithmic logic present in the provided code snippets. Your role is to refactor and structure, not to re-implement or invent new logic.
- Minimal, necessary modifications are permitted (e.g., renaming variables for clarity, adapting function signatures for dependency injection), but the core computational steps must remain identical to the original author's implementation.
6. Documentation of Usage Scenarios: Provide a concise and fluent document of the code module’s core logic, configurable options, and usage. Limit the description to 5-10 clear and coherent sentences.

# Output
1. Implement the technique standalone without relying on external, undefined components. Return an executable code block and a corresponding documentation, each wrapped between two ```.
Example:
[... Reasoning Steps ...]
```python
[... Core Implementation of the technique ...]
[... Ignore other relevant techniques ...]
# TEST BLOCK
[... Example Usage ...]
```
The brief documentation of the code:
```
[...Brief Documentation ...]
```
2. Verify that the generated code does not exceed the scope of the technique's definition. If the technique requires integration with other modules to constitute a single code module, return None. If no direct implementation of the technique is found in the given code snippets, also return None.


Now, please proceed with the task, following the workflow and adhering to all requirements. Generate the final code block and documentation wrapped between two ``` separately at the end.
        """
        return prompt
    
    @property
    def _prompt_rewrite_restructure_code(self):
        prompt = """
# Task
Your task is to transform a collection of disparate source code snippets, which are the official implementation of a technique component from a research paper {paper}, into a single, self-contained, and executable code block. The final code block must be clean, well-documented, and easy for others to understand and run.

# Input
1. Abstract of the paper {paper}:
{abstract}

2. Technical Concept Definition from the paper {paper}:
{technique}

3. Sub-techniques and Associated Code:
{sub_techniques}

4. Relevant Code Files:
{file_snippets}

# Workflow
1. Analyze: Understand the technique's inputs, outputs and workflow from the paper.
2. Locate: Fully reuse the code of the provided sub-techniques. For any uncovered parts, locate the relevant implementation logic from the given code snippets.
3. Refactor: Integrate the extracted code by removing hard-coded values, isolating the core algorithm, and standardizing it with proper documentation and type hints.
4. Assemble & Test: Build the final script and add an test block as a runnable example. Ensure accuracy and conciseness, avoiding unnecessary output.
5. Documentation: Write a brief and concise documentation of the code logic, configurable options, and usage in 5-10 sentences.

# Requirements
1. Dependency Management: Ensure all necessary imports and dependencies are included at the beginning of the code block.
2. Fidelity to the Original Technique: Strictly follow the description of the given technique to organize the code. ONLY focus on the implementation that DIRECTLY corresponds to THIS technique!!! (e.g., if the technique is a loss function definition, implement only the code for its calculation. Ignore all other parts of the algorithm like model definition or training loop). Return None if no direct implementation is found.
3. Code Encapsulation and Documentation:
- Encapsulate the core logic of the technique into one or more functions/classes.
- Every function and class method must include a comprehensive docstring explaining its purpose, parameters, and return values.
- All function arguments and return values must have clear type hints.
- Preserve original parameters and comments from the source code.
4. Reproducibility and Testing:
- A main execution block, start with the comment `# TEST BLOCK`, is required at the end of the file, which serves as a practical usage example and a test case.
- The test case should use parameters from the code repository or paper. If missing, create and state your own defaults.
5. Fidelity to the Original Logic: 
- You must strictly adhere to the algorithmic logic present in the provided code snippets. Your role is to refactor and structure, not to re-implement or invent new logic.
- Minimal, necessary modifications are permitted (e.g., renaming variables for clarity, adapting function signatures for dependency injection), but the core computational steps must remain identical to the original author's implementation.
6. Documentation of Usage Scenarios: Provide a concise and fluent document of the code module’s core logic, configurable options, and usage. Limit the description to 5-10 clear and coherent sentences.

# Output
1. Implement the technique standalone without relying on external, undefined components. Return an executable code block and a corresponding documentation, each wrapped between two ```.
Example:
[... Reasoning Steps ...]
```python
[... Core Implementation of the technique ...]
[... Ignore other relevant techniques ...]
# TEST BLOCK
[... Example Usage ...]
```
The brief documentation of the code:
```
[...Brief Documentation ...]
```
2. Verify that the generated code does not exceed the scope of the technique's definition. If the technique requires integration with other modules to constitute a single code module, return None. If no direct implementation of the technique is found in the given code snippets, also return None.


Now, please proceed with the task, following the workflow and adhering to all requirements. Generate the final code block and documentation wrapped between two ``` separately at the end.
        """
        return prompt
    
    @property
    def _prompt_check_rewrite_code(self):
        prompt = """
# Task
Your task is to determine if the given code block strictly follows the provided technique description and relevant code files.

# Input
1. Technical Concept Definition from the paper {paper}:
{technique}

2. Relevant Code Files:
{file_snippets}

3. Implemented Code Block:
{code}

# Output
1. Return False if the implementation is unrelated to the technique.
2. Return False if the implementation contains core logic cannot be located in the given relevant code files.
3. Return False if the implementation contains logics not covered in the technique description (e.g., the technique defines a submodule, but the code implements the full algorithm).
3. Return True if the code implements exactly what is specified in the technique description without adding any unnecessary features beyond the technical concept, and strictly follows the implementation in the given code files.

Now please think and reason carefully, provide a detailed analysis process for the above criteria, and wrap your final answer between two ``` in the end.
        """
        return prompt
        
    @property
    def _prompt_decompose_task(self):
        prompt = """
# Task
Your task is to decompose a complex academic task into its automic fundamental techniques based on its description.

# Input
Academic Task Definition:
{description}

# Output
Return a list of (name, description) tuples in the format [("...", "..."), ("...", "...")], sorted by their importance to the task composition in descending order. Use clear and definite wording, avoiding parentheses.
1. Each tuple must represent a distinct, fundamental academic concept that is reusable and traceable in other literature.
2. Each tuple is explicitly mentioned or directly relevant to the target task.
3. Avoid overly broad or vague techniques; each should have a clear, specific code implementation. Avoid trivial techniques like Cosine Similarity that require no literature review.
4. If the task's implementation does not involve any specific academic concepts (e.g., purely engineering, configuration, or organizational task), simply return None.

Now please think and reason carefully, and wrap your final answer between two ``` in the end.
        """
        return prompt
    
    def _convert_to_technique(self, contrib: Contribution) -> Technique:
        return Technique(
            name=contrib.name,
            description=contrib.description,
            components=[self._convert_to_technique(c) for c in contrib.components]
        )

    def _get_leaves(self, tech: Technique) -> List[Technique]:
        if not tech.components:
            return [tech]
        all_leaves = []
        for component in tech.components:
            all_leaves.extend(self._get_leaves(component))
        return all_leaves

    def _get_all_nodes(self, tech: Technique) -> List[Technique]:
        nodes = [tech]
        if tech.components:
            for component in tech.components:
                nodes.extend(self._get_all_nodes(component))
        return nodes
    
    def _assign_weights(self, tech: Technique, base_weight: float) -> float:
        if not tech.components:
            tech.weight = base_weight
            return base_weight
        
        total_weight = sum(self._assign_weights(c, base_weight) for c in tech.components)
        tech.weight = total_weight
        return total_weight
    
    def _get_relevant_code(
        self, 
        query: str, 
        code: Code = None, # 用于构建详细的重排 prompt
        paper: Optional[Paper] = None, # 用于构建详细的重排 prompt
        technique: Optional[Technique] = None, # 同上
        llm_rerank: bool = True, 
        top_complete_file: int = None,
        top_files: int = None,
        model: str = None
    ) -> List[FileSnippet]:
        config_retriever = get_code_rag_config().get('retriever', {}).get('llm', {})
        if top_files is None:
            top_files = config_retriever.get('top_files', 5)
        if top_complete_file is None:
            top_complete_file = config_retriever.get('top_complete_file', 1) 
        if model is None:
            model = get_config().get('code_model', self.model)
            
        initial_snippets = self.code_rag(query)
        if not initial_snippets:
            logger.warning(f"Initial RAG search for query '{query[:50]}...' returned no snippets.")
            return []
        
        technique_des = json.dumps(
            {"name": technique.name, "description": technique.description}, 
            ensure_ascii=False, 
            indent=2
        )
        
        if llm_rerank:
            logger.info("Performing LLM re-ranking...")
            # 构建详细的重排 prompt
            rerank_prompt = self._prompt_check_snippets.format(
                paper=paper.title if paper else "N/A",
                technique=technique_des,
                overview=code.overview or "N/A",
                file_snippets=json.dumps([asdict(fs) for fs in initial_snippets], ensure_ascii=False, indent=2)
            )
            response = self.backend.query(
                system_message=REPOSITORY_ANALYZER_PROMPT,
                user_message=rerank_prompt,
                model=model,
            )
            filtered_file_names = extract_object(extract_backtick_text(response[0]))
            logger.debug(f"Technique: {technique.name}, {technique.description}")
            logger.debug(f"Filtered file names: {filtered_file_names}")
            if not filtered_file_names or not isinstance(filtered_file_names, list):
                logger.warning(f"LLM re-ranking did not identify any relevant code files for query '{query[:50]}...'.")
                return []

            # 根据重排结果重新组织文件列表
            snippet_map = {fs.file_name: fs for fs in initial_snippets}
            final_files = [snippet_map[name] for name in filtered_file_names if name in snippet_map]
            
            # 为排名靠前的文件加载完整内容
            if top_complete_file > 0:
                file_content_map = {f.name: f.content for f in code.file_list}
                max_size_limit_bytes = get_config().get('max_prompt_code_bytes', 20 * 1024)
                for fs_object in final_files[:top_complete_file]:
                    file_name = fs_object.file_name
                    full_content_str = file_content_map.get(file_name)
                    if not full_content_str:
                        logger.warning(f"Full content for '{file_name}' not found in code object. Keeping snippet.")
                        continue
                    if len(full_content_str.encode('utf-8')) > max_size_limit_bytes:
                        logger.info(f"Skipping full content for '{file_name}' due to size limit. Keeping snippet.")
                        continue
                    logger.info(f"Replacing snippet with full content for '{file_name}'.")
                    fs_object.code_list = [full_content_str]
        else:
            # 如果不进行重排，则直接使用 RAG 的原始结果
            logger.info("Skipping LLM re-ranking, using initial RAG results.")
            final_files = initial_snippets
        final_files = final_files[:top_files]
        if not final_files:
            return []        
        
        return final_files
        
    def _get_rewrited_code(
        self,
        paper: Paper,
        tech: Technique,
        model: str = None
    ) -> Optional[CodeBlock]:
        if model is None:
            model = get_config().get('code_model', self.model)
        
        technique_des = json.dumps(
            {"name": tech.name, "description": tech.description}, 
            ensure_ascii=False, 
            indent=2
        )
        
        sub_techs_with_code = [
            sub for sub in self._get_leaves(tech) 
            if sub.code and isinstance(sub.code, CodeBlock)
        ]

        if sub_techs_with_code:
            sub_techs_payload = [
                {
                    "name": sub.name,
                    "implementation": sub.code.implementation,
                    "documentation": sub.code.documentation
                } for sub in sub_techs_with_code
            ]
            sub_techs_json = json.dumps(sub_techs_payload, ensure_ascii=False, indent=2)

            prompt = self._prompt_rewrite_restructure_code.format(
                paper=paper.title,
                abstract=paper.abstract,
                technique=technique_des,
                sub_techniques=sub_techs_json,
                file_snippets=tech.code or "N/A"
            )
        else:
            if not tech.code:
                logger.warning(f"No original code snippets for leaf technique '{tech.name}'. Skipping.")
                return None

            prompt = self._prompt_rewrite_code.format(
                paper=paper.title,
                abstract=paper.abstract,
                technique=technique_des,
                file_snippets=tech.code
            )

        for _ in range(MAX_ATTEMPTS - 1):
            response = self.backend.query(
                system_message=CODE_REWRITER_PROMPT, user_message=prompt, model=model
            )
            blocks = extract_backtick_texts(response[0]) if response and response[0] else []
            if len(blocks) >= 2:
                break
        else:  
            logger.warning(f"Code rewriting for '{tech.name}' failed after {MAX_ATTEMPTS} attempts.")
            return None
        
        code, documentation = blocks[-2], blocks[-1]
        if not code or code.strip() in ["", "None", "null"]:
            logger.warning(f"Extracted code for '{tech.name}' is empty after rewriting.")
            return None
        
        # 分割实现与测试
        implementation, test = code.strip(), ""
        lines = code.splitlines()
        separator_idx = next((i for i, line in enumerate(lines) if "# TEST BLOCK" in line), next((i for i, line in enumerate(lines) if "TEST BLOCK" in line), len(lines)))
        implementation = "\n".join(lines[:separator_idx]).strip()
        test = "\n".join(lines[separator_idx + 1:]).strip()

        code = CodeBlock(
            implementation=implementation,
            test=test,
            documentation=documentation.strip()
        )
        
        # 二步检验
        if get_code_rag_config().get('llm_check_code', False):
            check_prompt = self._prompt_check_rewrite_code.format(
                paper=paper.title,
                technique=technique_des,
                file_snippets=tech.code,
                code=json.dumps(asdict(code), ensure_ascii=False, indent=2)
            )
            check_response = self.backend.query(
                system_message=REPOSITORY_ANALYZER_PROMPT,
                user_message=check_prompt,
                model=model,
            )
            response = extract_object(extract_backtick_text(check_response[0]))
            if response is not True:
                logger.warning(f"Rewritten code for '{tech.name}' did not pass fidelity check. Discarding.")
                return None
        
        return code
        
    def _rewrite_paper(self, paper: Paper, techniques: List[Technique]) -> None:
        """
        根据论文内容，重写给定 Technique 列表的描述。
        此函数会直接修改传入的 `techniques` 列表中的对象。
        """
        logger.info(f"Starting rewrite of paper contributions for '{paper.title}'...")

        ### MODIFIED ###
        # 不再从 paper.contributions 创建新的 Technique 对象
        # 直接使用传入的 `techniques` 参数
        if not techniques:
            logger.warning("No methodology/technique contributions provided to rewrite.")
            return

        # Step 2: 获取所有节点的扁平列表
        all_techniques = []
        for tech in techniques: # 使用传入的 techniques
            all_techniques.extend(self._get_all_nodes(tech))

        if not all_techniques:
            logger.warning(f"No nested techniques found in paper '{paper.title}'. Skipping rewrite.")
            return

        # 2. 准备 RAG 系统
        logger.info(f"Preparing paper RAG for '{paper.title}'...")
        paper_db_path = os.path.join(get_config().get('raw_data_path'), "paper", f"{sanitize_filename(paper.title)}.pkl")
        self.paper_rag.prepare_retriever(paper, paper_db_path)

        # 3. 遍历并直接修改传入的 technique 对象
        logger.info(f"Rewriting descriptions for {len(all_techniques)} contributions...")
        for c in all_techniques:
            # a. 构造查询
            query = f"{c.name}: {c.description}"
            
            # b. 使用RAG检索相关原文片段
            excerpts = self.paper_rag(query)
            if not excerpts:
                logger.warning(f"No relevant paper excerpts found for '{c.name}'. Skipping rewrite.")
                continue

            # c. 准备Prompt并调用LLM
            node_info = {"name": c.name, "description": c.description}
            rewrite_prompt = self._prompt_rewrite_paper.format(
                paper=paper.title,
                technique=json.dumps(node_info, ensure_ascii=False),
                excerpt=json.dumps(excerpts)
            )
            
            response = self.backend.query(
                user_message=rewrite_prompt,
                model=get_config().get('paper_model', self.model)
            )
            
            rewrited_description = extract_backtick_text(response[0])
            if rewrited_description and rewrited_description.strip() not in ["", "None", "null"]:
                # 核心：直接修改传入对象的属性
                c.description = rewrited_description.strip() 
                logger.debug(f"Successfully rewrote description for '{c.name}'.")
            else:
                logger.warning(f"LLM did not provide a valid new description for '{c.name}'.")


    def _map_code_paper(
        self, 
        paper: Paper, 
        code: Code, 
        techniques: List[Technique], ### MODIFIED: 接收 techniques 列表 ###
        database_path: str, 
        llm_rerank: bool
    ) -> None: ### MODIFIED: 返回值为 None，因为是就地修改 ###
        
        ### MODIFIED ###
        # Step 1: 直接使用传入的 techniques，不再重新创建
        if not techniques:
            logger.warning("No methodology/technique contributions provided to map.")
            return # 直接返回

        top_level_techniques = techniques

        # Step 2: 获取所有节点的扁平列表
        all_techniques = []
        for tech in top_level_techniques:
            all_techniques.extend(self._get_all_nodes(tech))
        
        # Step 3: 分配权重
        num_leaves = sum(1 for tech in all_techniques if not tech.components)
        if num_leaves == 0:
            return # 直接返回
        base_weight = 1.0 / num_leaves
        for tech in top_level_techniques:
            self._assign_weights(tech, base_weight)

        all_techniques.sort(key=lambda tech: tech.weight)
        
        # Step 4: 为所有节点执行 CodeRAG
        if not code:
            logger.info("No code repository provided. Skipping code mapping.")
            for tech in all_techniques:
                tech.code = None
            return # 直接返回

        database_path = os.path.join(get_config().get('raw_data_path'), "code", f"{code.name}.pkl") if database_path is None else database_path
        self.code_rag.prepare_retriever(code, database_path)
        
        logger.info("Performing RAG for all technique nodes...")
        for tech in all_techniques:
            query = f"{tech.name}: {tech.description}" # 这里会使用到可能已被重写过的 description
            relevant_files = self._get_relevant_code(query=query, code=code, paper=paper, technique=tech, llm_rerank=llm_rerank)
            tech.code = json.dumps([asdict(rf) for rf in relevant_files], ensure_ascii=False) if relevant_files else None

        # Step 5: 顺序为所有节点生成/整合代码
        logger.info("Generating/Integrating code for all techniques in bottom-up order...")
        if llm_rerank:
            for tech in all_techniques:
                tech.code = self._get_rewrited_code(paper=paper, tech=tech)

        logger.info("Code generation/integration completed for all techniques.")
        ### MODIFIED: 不再返回任何值 ###

    
    def exec_verify_node(self, node: Node) -> Node:
        # 这个函数可以自己单独调用
        # 外部给定一个node json的路径，譬如storage/kg/xx_graph.json
        # 然后外部load json, 调用node = Node.from_dict(data)来获取这个node
        # 然后就可以调用这个函数了
        
        # 处理后的paper_id
        paper_id = sanitize_filename(node.paper_title)
        
        # 处理后的techniques
        top_level_techniques = node.techniques
        if not top_level_techniques:
            logger.warning("No methodology/technique contributions found in the paper.")
            return []
        all_techniques = []
        for tech in top_level_techniques:
            all_techniques.extend(self._get_all_nodes(tech))
        all_techniques = [tech for tech in all_techniques if tech.code and isinstance(tech.code, CodeBlock)]
        
        # 插入code verifier逻辑
        self.code_verifier.get_verified_code(paper_id, all_techniques)
        
        # 返回更新后的techniques 注意返回的应该是顶层节点
        return node
    
    def generate_node(
        self, 
        paper: Paper = None, 
        code: Code = None, 
        llm_rerank: bool = True, 
        database_path: str = None,
        save_path: str = None,
        rewrite_paper: bool = None,
        verify_code: bool = None
    ) -> Node:
        if not paper:
            logger.error("Paper object must be provided to generate a graph node.")
            return None
        
        if rewrite_paper is None:
            rewrite_paper =  get_paper_rag_config().get('rag', False)
        if verify_code is None:
            verify_code = get_code_rag_config().get('exec_check_code', False)
            
        top_level_techniques = [
            self._convert_to_technique(c) for c in paper.contributions
            if c.type in ["Methodology", "Technique"]
        ]

        if rewrite_paper:
            logger.info("Rewriting paper contributions before node generation...")
            self._rewrite_paper(paper, top_level_techniques)
        
        self._map_code_paper(
            paper=paper, 
            code=code, 
            techniques=top_level_techniques, 
            database_path=database_path, 
            llm_rerank=llm_rerank
        )
                
        node = Node(
            paper_title=paper.title if paper else None,
            paper_abstract=paper.abstract if paper else None,
            paper_references=paper.references if paper else [],
            code_file=code.file_tree if code else None,
            code_overview=code.overview if code else None,
            techniques=top_level_techniques,
            resources=[f"{c.name}: {c.description}" for c in paper.contributions if c.type == "Resource"] if paper else [],
            findings=[f"{c.name}: {c.description}" for c in paper.contributions if c.type == "Finding"] if paper else []
        )
        
        if save_path:
            identifier = node.paper_title or f"node_{hash(node.code_overview or id(node))}"
            save_file_name = f"{sanitize_filename(identifier)}_graph_unverified.json"
            if verify_code:
                temp_dir_path = get_config().get("raw_data_path")
                os.makedirs(temp_dir_path, exist_ok=True)
                save_file = os.path.join(temp_dir_path, save_file_name)
            else:
                os.makedirs(save_path, exist_ok=True)
                save_file = os.path.join(save_path, save_file_name)
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(asdict(node), f, indent=2, ensure_ascii=False)
            logger.info(f"Graph Node data saved to {save_file}")   
        
        if verify_code:
            node = self.exec_verify_node(node)
            if save_path:
                identifier = node.paper_title or f"node_{hash(node.code_overview or id(node))}"
                os.makedirs(save_path, exist_ok=True)
                save_file = os.path.join(save_path, f"{sanitize_filename(identifier)}_graph_verified.json") 
                with open(save_file, "w", encoding="utf-8") as f:
                    json.dump(asdict(node), f, indent=2, ensure_ascii=False)
                logger.info(f"Verified graph Node data saved to {save_file}")   
        return node

    def load_kg(self, kg_dir: str) -> List[Node]:
        if not os.path.isdir(kg_dir):
            logger.warning(f"Knowledge graph directory not found: '{kg_dir}'")
            return []

        nodes: List[Node] = []
        json_file_paths = glob.glob(os.path.join(kg_dir, '*.json'))

        logger.info(f"Found {len(json_file_paths)} JSON files in '{kg_dir}'. Starting to load.")

        for file_path in json_file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                node_object = Node.from_dict(data)
                nodes.append(node_object)

            except json.JSONDecodeError as e:
                logger.warning(f"Skipping file due to invalid JSON format: '{file_path}'. Error: {e}")
            except TypeError as e:
                logger.warning(f"Skipping file due to mismatched structure (likely missing fields): '{file_path}'. Error: {e}")
            except Exception as e:
                logger.error(f"An unexpected error occurred while processing file '{file_path}'. Skipping. Error: {e}")

        logger.info(f"Successfully loaded {len(nodes)} out of {len(json_file_paths)} files to construct knowledge base.")
        self.knowledge_base = nodes
        return nodes
    
    def build_index(self):
        if not self.knowledge_base:
            logger.warning("Knowledge base is empty...")
            return
        logger.info(f"Starting to build knowledge index for {len(self.knowledge_base)} nodes...")
        self.knowledge_index.build_indexes(self.knowledge_base)
        
    def _calculate_technique_similarity(
        self, 
        tech1: Technique, 
        tech2: Technique,
        name_weight: float = 0.5,
        description_weight: float = 0.5,
        bm25_weight: float = 0.5, 
        embedding_weight: float = 0.5
    ) -> float:

        id2 = self.knowledge_index.tech_to_id.get(id(tech2))
        if id2 is None:
            return 0.0

        tokenized_query_name = tech1.name.split()
        name_bm25_score_raw = self.knowledge_index.bm25_name.get_scores(tokenized_query_name)[id2]
        name_bm25_norm = 1.0 / (1.0 + np.exp(-name_bm25_score_raw))
        
        # 1. name相似度计算
        query_name_vec = self.knowledge_index.model.encode([tech1.name])[0]
        candidate_name_vec = self.knowledge_index.name_vectors[id2]
        dot_product_name = np.dot(query_name_vec, candidate_name_vec)
        norm_name = np.linalg.norm(query_name_vec) * np.linalg.norm(candidate_name_vec)
        name_emb_score = (dot_product_name / norm_name + 1) / 2 if norm_name != 0 else 0.0
        total_name_score = (name_bm25_norm * bm25_weight) + (name_emb_score * embedding_weight)

        # 2. description相似度计算
        tokenized_query_desc = tech1.description.split()
        desc_bm25_score_raw = self.knowledge_index.bm25_desc.get_scores(tokenized_query_desc)[id2]
        desc_bm25_norm = 1.0 / (1.0 + np.exp(-desc_bm25_score_raw))
        query_desc_vec = self.knowledge_index.model.encode([tech1.description])[0]
        candidate_desc_vec = self.knowledge_index.desc_vectors[id2]
        dot_product_desc = np.dot(query_desc_vec, candidate_desc_vec)
        norm_desc = np.linalg.norm(query_desc_vec) * np.linalg.norm(candidate_desc_vec)
        desc_emb_score = (dot_product_desc / norm_desc + 1) / 2 if norm_desc != 0 else 0.0
        total_desc_score = (desc_bm25_norm * bm25_weight) + (desc_emb_score * embedding_weight)
        
        # 3. 最终加权融合
        return (total_name_score * name_weight) + (total_desc_score * description_weight)

    
    def _rerank_techniques_by_llm(self, description: str, candidates: List['Technique'], top_k: int = 5) -> List['Technique']:
        if not candidates:
            return []

        try:
            prompt = self._prompt_check_techniques.format(
                technique=description,
                relevant_techniques=json.dumps([
                    {
                        "name": t.name,
                        "description": t.description,
                        "documentation": t.code.documentation if t.code else ""
                    }
                    for t in candidates
                ])
            )
            
            response_text = self.backend.query(
                system_message=REPOSITORY_ANALYZER_PROMPT, user_message=prompt, model=self.model
            )[0]

            llm_ranked_items = extract_object(extract_backtick_text(response_text))

            if not llm_ranked_items or not isinstance(llm_ranked_items, list):
                logger.warning("No relevant techniques retrieved from LLM.")
                return []

            items_by_cleaned_name = {clean_string(t.name): t for t in candidates}
            reranked_results = []
            seen_names = set()

            for item in llm_ranked_items:
                if not (isinstance(item, (list, tuple)) and len(item) == 2):
                    continue

                name, new_description = item
                cleaned_name = clean_string(name)

                if cleaned_name in items_by_cleaned_name and cleaned_name not in seen_names:
                    technique_obj = items_by_cleaned_name[cleaned_name]

                    if technique_obj.code is None:
                        continue

                    separator = "\n\n" if technique_obj.code.documentation else ""
                    technique_obj.code.documentation += f"{separator}[Apply Guidance]:\n{new_description}"

                    reranked_results.append(technique_obj)
                    seen_names.add(cleaned_name)

        except Exception as e:
            logger.error(f"LLM reranking failed, falling back to original order. Error: {e}", exc_info=True)
            return candidates[:top_k]

        return reranked_results[:top_k] if reranked_results else candidates[:top_k]


    def retrieve_technique_by_technique(self, technique: Technique, top_k: int = 5, code_only: bool = True, return_code: bool = True, return_components: bool = False, llm_rerank: bool = True) -> List[Technique]:
        if not self.knowledge_index.flat_techniques:
            return []
        
        query_tech_id = self.knowledge_index.tech_to_id.get(id(technique))
        query_node_id = self.knowledge_index.tech_id_to_node_id.get(query_tech_id) if query_tech_id else None

        scores = []
        for i, candidate_tech in enumerate(self.knowledge_index.flat_techniques):
            candidate_node_id = self.knowledge_index.tech_id_to_node_id[i]

            if (candidate_tech is technique or
                (query_node_id is not None and candidate_node_id == query_node_id) or
                (code_only and not candidate_tech.code)):
                continue

            similarity = self._calculate_technique_similarity(technique, candidate_tech)
            scores.append((candidate_tech, similarity))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        technique_similarity_threshold = get_config().get('technique_similarity', 0)
        retrieve_num = max(RETRIEVE_TECHNIQUES, top_k * 2)
        results = [item for item in scores if item[1] >= technique_similarity_threshold][:retrieve_num]

        processed_results = []
        for tech, score in results:
            tech_copy = copy.deepcopy(tech)
            if not return_components:
                tech_copy.components = []
            if not return_code:
                all_nodes_in_copy = self._get_all_nodes(tech_copy)
                for node in all_nodes_in_copy:
                    node.code = None
            processed_results.append((tech_copy, score))
            
        if llm_rerank:
            techniqe_description = f"{technique.name}: {technique.description}"
            return self._rerank_techniques_by_llm(techniqe_description, [tr[0] for tr in processed_results], top_k=top_k)
        return processed_results[:top_k]

    def retrieve_technique_by_query(self, name: str , description: str = None, top_k: int = 5, code_only: bool = True, return_code: bool = True, return_components: bool = False, llm_rerank: bool = True) -> List[Tuple[Technique, float]]:
        if name:
            description = description or name
            technique = Technique(name=name, description=description)
            return self.retrieve_technique_by_technique(technique=technique, top_k=top_k, code_only=code_only, return_code=return_code, return_components=return_components, llm_rerank=llm_rerank)
        return []

    def retrieve_paper_by_paper(self, query_node: Node, top_k: int = 5, reference_bonus: float = 0.05, return_code: bool = True) -> List[Node]:
        if not self.knowledge_base:
            raise RuntimeError("Knowledge base is not loaded. Call load_kg() first.")
        if not self.knowledge_index.tech_to_id:
            raise RuntimeError("Index is not built. Call build_index() first.")

        scores = []
        for candidate_node in self.knowledge_base:
            if query_node.paper_title and candidate_node.paper_title and is_fuzzy_match(candidate_node.paper_title, query_node.paper_title):
                continue
            
            paper_score = 0.0
            
            if any(is_fuzzy_match(candidate_node.paper_title, ref) for ref in query_node.paper_references or []) or \
               any(is_fuzzy_match(query_node.paper_title, ref) for ref in candidate_node.paper_references or []):
                paper_score += reference_bonus
                
            technique_total_score = 0.0
            if query_node.techniques and candidate_node.techniques:
                for tech_q in query_node.techniques:
                    for tech_c in candidate_node.techniques:
                        similarity = self._calculate_technique_similarity(tech_q, tech_c)
                        technique_total_score += similarity * (tech_q.weight or 0.0) * (tech_c.weight or 0.0)
            
            paper_score += technique_total_score
            scores.append((candidate_node, paper_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        paper_similarity_threshold = get_config().get('paper_similarity', 0)
        results = [item for item in scores if item[1] >= paper_similarity_threshold][:top_k]
        
        if not return_code:
            processed_results = []
            for node, score in results:
                node_copy = copy.deepcopy(node)
                if node_copy.techniques:
                    all_techs = [t for top_tech in node_copy.techniques for t in self._get_all_nodes(top_tech)]
                    for tech in all_techs:
                        tech.code = None
                processed_results.append((node_copy, score))
            processed_results = [node[0] for node in processed_results]
            return processed_results
        
        results = [node[0] for node in results]
        return results
        
    def search_nodes_by_title(self, query_title: str, top_k: int = 5, return_code: bool = True) -> List[Tuple[Node, float]]:
        bm25_index = self.knowledge_index.bm25_title
        indexed_nodes = self.knowledge_index.title_indexed_nodes

        if not bm25_index:
            logger.warning("Title index is not available for searching.")
            return []

        tokenized_query = query_title.split()
        bm25_scores = bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if bm25_scores[idx] > 0:
                results.append((indexed_nodes[idx], bm25_scores[idx]))
        
        if not return_code:
            processed_results = []
            for node, score in results:
                node_copy = copy.deepcopy(node)
                if node_copy.techniques:
                    all_techs = [t for top_tech in node_copy.techniques for t in self._get_all_nodes(top_tech)]
                    for tech in all_techs:
                        tech.code = None
                processed_results.append((node_copy, score))
            return processed_results
            
        return results
    
    def find_node_by_paper_title(self, title: str, top_n_candidates: int = 5, fuzzy_match_threshold: int = 95, return_code: bool = True) -> Optional[Node]:
        for node, _ in self.search_nodes_by_title(title, top_k=top_n_candidates, return_code=return_code):
            if is_fuzzy_match(node.paper_title, title, threshold=fuzzy_match_threshold):
                return node
        return None

    def decompose_task_to_techniques(self, description: str, name: str = None) -> List[Technique]:
        if name:
            task_description = f"{name}: {description}"
        else:
            task_description = description
        prompt = self._prompt_decompose_task.format(description=task_description)
        response = self.backend.query(
            system_message=REPOSITORY_ANALYZER_PROMPT,
            user_message=prompt,
            model=get_config().get('paper_model', self.model)
        )
        response = extract_object(extract_backtick_text(response[0]))
        if not response:
            logger.warning("The task is irelevant to academic techniques.")
            return []
        if not isinstance(response, list):
            logger.warning("LLM failed to return a valid list of techniques for task decomposition. Return the original one.")
            return [Technique(name=name if name else description, description=description)]
        res = []
        for tech_name, tech_desc in response:
            tech = Technique(name=tech_name, description=tech_desc)
            res.append(tech)
        return res
            
        
