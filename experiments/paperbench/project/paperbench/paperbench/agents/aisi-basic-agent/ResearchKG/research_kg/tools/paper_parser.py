""" 
基本的解析paper的内容
"""
import os
import re
import json
from pylatexenc.latexwalker import LatexEnvironmentNode, LatexMacroNode, LatexWalker
import bibtexparser
from typing import Optional, Tuple, List, Literal, Dict
from collections import deque
import logging
from pathlib import Path
from dataclasses import asdict

from ..schema import Paper, Section, Contribution, CONTRIBUTION_TYPE
from .base_tool import BaseTool
from ..utils import *
from ..llm import extract_backtick_text, extract_object
from ..llm.prompt import *



logger = logging.getLogger(__name__)

class MatchExtractor(BaseTool):
    def __init__(
        self,
        model: str = None,
        memory: Optional[str] = None,
    ):
        super().__init__(model, memory)
    
    def run(
        self, 
        paper_path: str,
        save_path: str
    ) -> Paper:
        raise NotImplementedError("Not implemented yet.")
    
class LatexExtractor(MatchExtractor):
    def __init__(
        self,
        model: str,
        memory: Optional[str] = None,
    ):
        super().__init__(model, memory)
        
    @property
    def _prompt_get_references(self):
        prompt = """
# Task
You are provided with a .bbl file {bbl}. 
Please extract the titles of all the references in the .bbl file.

# Output
1. Output the extracted reference titles in the form of a string list.
2. If no reference is available, please return None.

Please wrap your final answer between two ``` in the end.
"""
        return prompt
    
    def _find_references(self, root_dir: Path) -> List[str]:
        """
        查找 .bib 文件，并用正则表达式提取所有标题。
        """
        bib_path = next(root_dir.rglob('*.bib'), None)
        
        if not bib_path:
            logger.info(f"No .bib file found in directory: {root_dir}. Try to find references through llm.")
            bib_path = next(root_dir.rglob('*.bbl'), None)
            if bib_path:
                with open(bib_path, 'r', encoding='utf-8', errors='ignore') as f:
                    bbl_content = f.read()
                reference_prompt = self._prompt_get_references.format(bbl=bbl_content)
                response = self.backend.query(
                    user_message=reference_prompt,
                    model=self.model
                )
                logger.debug(f"LLM response for references: {response[0]}")
                titles = extract_object(extract_backtick_text(response[0]))
                titles = [clean_string(title) for title in titles]
                return titles
            
        logger.info(f"Parsing references from: {bib_path}")
        
        try:
            with open(bib_path) as bibfile:
                bib_content = bibfile.read()
            bib_database = bibtexparser.loads(bib_content)
            titles = [entry.get('title', '').replace('\n', ' ').strip() for entry in bib_database.entries if 'title' in entry]
            titles = [clean_string(title) for title in titles if title]
            return titles
        except Exception as e:
            logger.error(f"Failed to read or parse .bib file {bib_path}: {e}")
            return []
            
    
    def _find_main_tex(self, root_dir: Path) -> Optional[Path]:
        """在指定目录中查找包含 '\\documentclass' 的主 .tex 文件。"""
        if not root_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {root_dir}")
            
        logger.debug(f"Searching for main .tex file in: {root_dir}")
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.tex'):
                    file_path = Path(dirpath) / filename
                    try:
                        with file_path.open('r', encoding='utf-8') as file:
                            for line in file:
                                if '\\documentclass' in line.split('%')[0]:
                                    logger.debug(f"Found main tex: {file_path}")
                                    return file_path
                    except Exception as e:
                        logger.warning(f"Failed to read {file_path}: {e}")
        return None

    def _load_tex_content(self, file_path: Path, already_read: set = None) -> List[str]:
        """递归地加载 .tex 文件及其所有 \\input 和 \\include 的内容。"""
        if already_read is None:
            already_read = set()
        
        if file_path in already_read or not file_path.exists():
            return []
        
        content = []
        already_read.add(file_path)
        
        try:
            with file_path.open('r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    match = re.match(r'\\(input|include)\{([^}]+)\}', line)
                    if match:
                        # 如果找到 \input 或 \include，递归加载
                        fname = match.group(2)
                        included_path = file_path.parent / (fname if fname.endswith('.tex') else f"{fname}.tex")
                        content.extend(self._load_tex_content(included_path, already_read))
                    else:
                        content.append(line)
        except Exception as e:
            logger.error(f"Error loading TeX content from {file_path}: {e}")
            
        return content
    
    def _get_valid_nodes(self, node_list: List) -> Optional[List]:
        """递归查找第一个包含 section 宏的节点列表。"""
        if any(isinstance(child, LatexMacroNode) and 'section' in child.macroname for child in node_list):
            return node_list 

        for node in node_list:
            if hasattr(node, 'nodelist') and node.nodelist is not None:
                result = self._get_valid_nodes(node.nodelist)
                if result:
                    return result
        return None
    
    def _construct_latex_node(self, content: List[str]) -> Tuple[str, List]:
        """从加载的 LaTeX 内容中构建节点树。"""
        start = 0
        end = len(content)
        for i, line in enumerate(content):
            if "\\begin{document}" in line:
                start = i
            if "\\end{document}" in line:
                end = i
                break # 找到 end{document} 就可以停止
        
        # 只处理 \begin{document} 和 \end{document} 之间的内容
        document_content = ''.join(content[start:end])
        
        lw = LatexWalker(document_content)
        content_nodes, _, _ = lw.get_latex_nodes()
        
        # 寻找包含 section 的有效节点部分
        valid_nodes = self._get_valid_nodes(content_nodes)
        
        return document_content, valid_nodes

    def _extract_labeled_environments(self, content_nodes: List, keyword: str) -> Dict[str, str]:
        """
        从节点中提取带标签的环境，如表格(table)和图形(figure)。
        """
        extracted_items = {}
        for node in content_nodes:
            if isinstance(node, LatexEnvironmentNode) and keyword in node.environmentname:
                label = None
                # 在环境的子节点中寻找标签
                for sub_node in node.nodelist:
                    if isinstance(sub_node, LatexMacroNode) and sub_node.macroname == 'label':
                        # 提取大括号内的标签名
                        label_match = re.search(r'\{([^}]+)\}', sub_node.latex_verbatim())
                        if label_match:
                            label = label_match.group(1)
                            break
                if label:
                    extracted_items[label] = node.latex_verbatim().strip()
        return extracted_items
    
    def _extract_equations(self, content_nodes: List) -> Dict[str, str]:
        """从节点中提取带标签或编号的方程。"""
        equations = {}
        valid_equations = set()
        eq_count = 1
        for node in content_nodes:
            if isinstance(node, LatexEnvironmentNode) and ('equation' in node.environmentname or 'align' in node.environmentname):
                content = node.latex_verbatim()
                if content not in valid_equations:
                    label = None
                    for sub_node in node.nodelist:
                        if isinstance(sub_node, LatexMacroNode) and sub_node.macroname == 'label':
                            label_match = re.search(r'\{([^}]+)\}', sub_node.latex_verbatim())
                            if label_match:
                                label = label_match.group(1)
                                break
                    
                    key = f"eq:{label}" if label else f'eq:{eq_count}'
                    equations[key] = content.strip()
                    
                    if not label:
                        eq_count += 1
                    valid_equations.add(content)
        return equations
    
    def _extract_section_content(self, content_nodes: List, start_idx: int, end_idx: int) -> Dict[str, str]:
        """提取单个章节内容的辅助函数。"""
        section = {"name": "", "content": ""}
        
        name_node_content = content_nodes[start_idx].latex_verbatim()
        name_match = re.search(r'\{([^}]+)\}', name_node_content)
        section["name"] = name_match.group(1) if name_match else ""
        
        if 'firstsection' in section["name"]:
            start_idx += 1
    
        for i in range(start_idx + 1, end_idx):
            cur_node = content_nodes[i]
            # if isinstance(cur_node, LatexEnvironmentNode) and ('table' in cur_node.environmentname or 'fig' in cur_node.environmentname):
            #     continue
            section["content"] += cur_node.latex_verbatim().strip()
        return section
    
    def _extract_sections(self, content_nodes: List) -> Optional[List[Section]]:
        """
        从节点中直接提取并构建一个包含两级嵌套的 Section 对象列表。
        """
        section_nodes = {
            idx: node.macroname
            for idx, node in enumerate(content_nodes)
            if isinstance(node, LatexMacroNode) and node.macroname in ('section', 'subsection')
        }
        section_positions = sorted(section_nodes.keys())
        logger.debug(f"Found section positions: {section_positions} with types: {[section_nodes[pos] for pos in section_positions]}")

        if not section_positions:
            return []

        root_sections: List[Section] = []        
        current_main_section: Optional[Section] = None 
        main_sec_num = 0
        sub_sec_num = 0

        try:
            for i, sec_pos in enumerate(section_positions):
                sec_type = section_nodes[sec_pos]
                
                sec_end = section_positions[i+1] if i + 1 < len(section_positions) else len(content_nodes)
                section_data = self._extract_section_content(content_nodes, sec_pos, sec_end)
                
                if sec_type == 'section':
                    # 开始一个新的主章节
                    main_sec_num += 1
                    sub_sec_num = 0 # 重置子章节计数器
                    
                    new_section = Section(
                        id=str(main_sec_num),
                        name=section_data.get("name", ""),
                        content=section_data.get("content", "")
                    )
                    logger.debug(f"Extracted section: {json.dumps(asdict(new_section))}")
                    root_sections.append(new_section)
                    current_main_section = new_section # 更新当前主章节的上下文
                
                elif sec_type == 'subsection':
                    # 检查是否存在一个可以依附的主章节
                    if current_main_section is None:
                        logger.warning(f"Found a subsection at index {sec_pos} before any main section. Skipping.")
                        continue
                    
                    # 创建一个子章节并添加到当前主章节
                    sub_sec_num += 1
                    
                    new_subsection = Section(
                        id=f"{current_main_section.id}.{sub_sec_num}",
                        name=section_data.get("name", ""),
                        content=section_data.get("content", "")
                    )
                    if current_main_section:
                        current_main_section.subsection.append(new_subsection)
                    else:
                        logger.warning(f"Current main section is None when trying to add subsection at index {sec_pos}. Skipping.")


        except Exception as e:
            logger.error(f"Error while extracting sections: {e}", exc_info=True)
            return None
            
        return root_sections

    def _inject_references(self, sections: List[Section], figures: Dict, tables: Dict):
        """
        递归地遍历 Section 对象列表，将图和表的LaTeX代码注入到第一次引用它们的位置。
        """
        all_references = {**figures, **tables}
        if not all_references:
            return

        for section in sections:
            if section.content:
                content = section.content
                processed_labels_in_section = set()
                new_content_parts = []
                last_end = 0

                for match in re.finditer(r'\\ref\{([^}]+)\}', content):
                    label = match.group(1)
                    start, end = match.span()
                    new_content_parts.append(content[last_end:end])

                    if label in all_references and label not in processed_labels_in_section:
                        injection_text = (
                            f"{all_references[label]}\n"
                        )
                        new_content_parts.append(injection_text)
                        processed_labels_in_section.add(label)
                    
                    last_end = end
                
                new_content_parts.append(content[last_end:])
                section.content = "".join(new_content_parts)

            if section.subsection:
                self._inject_references(section.subsection, figures, tables)

    def run(self, paper_path: str, save_path: str = None) -> Paper:
        """
        主入口函数：解析LaTeX项目，提取结构化信息，并保存为JSON。
        """
        # 路径提取与解析
        paper_dir = Path(paper_path)
        dir_name = paper_dir.name
        
        # 2. 加载元信息
        meta_info_path = paper_dir / f"{dir_name}.json"
        try:
            with meta_info_path.open('r', encoding='utf-8') as f:
                meta_data = json.load(f)
        except FileNotFoundError:
            # TODO: 进行arxiv元信息的重新获取
            raise FileNotFoundError(f"Meta info file not found: {meta_info_path}")
        paper = Paper.from_dict(meta_data)

        # 3. 加载并解析LaTeX内容
        main_tex_file = self._find_main_tex(paper_dir)
        if main_tex_file is None:
            logger.error(f"Failed to find main .tex file in {paper_dir}.")
            return None
        
        latex_content = self._load_tex_content(main_tex_file)
        _, content_nodes = self._construct_latex_node(latex_content)
        
        if not content_nodes:
            logger.error(f"Failed to extract content nodes from '{main_tex_file.name}'.")
            return None
        
        # 4. 从解析后的节点中提取各个部分
        references = self._find_references(paper_dir)
        sections = self._extract_sections(content_nodes)
        tables = self._extract_labeled_environments(content_nodes, keyword='tab')
        figures = self._extract_labeled_environments(content_nodes, keyword='fig')
        equations = self._extract_equations(content_nodes)        
        if sections:
            self._inject_references(sections, figures, tables)
        
        # 存储数据
        paper.sections = sections if sections else None
        paper.equations = [f"{key}: {value}" for key, value in equations.items()] if  equations else None
        paper.references = references if references else None
        paper_content = json.dumps(asdict(paper), ensure_ascii=False, indent=4)
        logger.info(f"Extracted paper content: \n\n{paper_content}\n\n...")
        
        if save_path:
            output_dir = Path(save_path)
            output_dir.mkdir(parents=True, exist_ok=True) # 确保输出目录存在
            output_path = output_dir / f"{dir_name}.json"
            with output_path.open("w", encoding='utf-8') as f:
                json.dump(asdict(paper), f, ensure_ascii=False, indent=4)
            logger.info(f"Successfully extracted and saved data to {save_path}")
        
        return paper
    
# TODO: implement Markdown parser
class MdExtractor(MatchExtractor):
    pass

# TODO: implement PDF parser
class PdfExtractor(MdExtractor):
    pass

class LLMExtractor(BaseTool ):
    def __init__(
        self,
        model: str,
        memory: Optional[str] = None,
    ):
        super().__init__(model, memory)
    
    @property
    def _prompt_get_github_url(self):
        prompt = """
# Task
You are provided with the article titled {title} and its corresponding abstract: {abstract}. Here are several GitHub links that may be related to the paper: {github_link}
Select the appropriate GitHub link that corresponds to the provided article and abstract.

# Output
1. If a matching link is found, please return the link.
2. If no relevant link is available, please return None.

Now please think and reason carefully, and wrap your final answer between two ``` in the end.
        """
        return prompt
    
    @property
    def _prompt_get_contributions(self):
        prompt = """
# Task
You are provided with the paper titled {title}. Here are the main sections of the paper: {sections}. Furthermore, key equations from the paper are provided to help you understand its specific algorithms: {equations}.
Your task is to analyze the provided research paper and identify its **Core Components**. For each Component, you must provide a clear, concise, and implementable definition.

# INSTRUCTIONS
1.  **Identify Core Components**: Read the paper to identify its primary components. A componnet is not limited to a single algorithm; it can be a novel methodology, reusable techniques, key insight/finding, open-source datasets/benchmarks, etc.**

2.  **Categorize Each Component**: Assign one of the following types to each component you identify:
    *   `Methodology`: A novel, end-to-end procedure proposed by the paper for solving a problem. This can be an entire algorithm or model architecture design that addresses a specific research challenge. It must correspond to a systematic and complete end-to-end code implementation. When composed of multiple atomic sub-techniques, represent using the "components" field. Ensure the methodology can be implemented standalone, instead of a generic theoretical definition or a high-level outline of a framework.
    *   `Technique`: A self-contained and algorithmically implementable component, applied within the paper's Methodology or Experiment Process. It is either a novel module from this work, or a traceable technique from prior research. When composed of multiple atomic sub-techniques, represent using the "components" field. Ensure each technique can be implemented standalone, requiring NO integration with other modules to constitute a single code module. Exclude theoretical points and experimental tricks not directly tied to code implementation. Move them to the "Finding" category.
    *   `Finding`: A significant empirical or theoretical insight which can refer to an intriguing experimental finding, a powerful theoretical proof, or a promising research direction.
    *   `Resource`: A PUBLICLY available dataset or benchmark originally constructed in this paper.

3.  **Define and Detail**: For each component, provide a detailed definition adhering to the following rules:
    *   **Fidelity**: All definitions must originate strictly from the provided paper. Do not invent details.
    *   **Atomicity & Modularity**: Each component, whether high-level or a component, should be defined as a distinct, self-contained unit. Explain its inputs, core logic, and outputs.
    *   **Reproducibility**: Retain as much original detail as possible. The definition should be comprehensive enough for an engineer or researcher to understand and implement it.
    *   **Structure**: If a `Methodology` or a `Technique` is composed of smaller `Technique`s, represent this hierarchical relationship using nested bullet points. This is crucial for understanding how the parts form the whole.

# OUTPUT FORMAT
Organize the extracted techniques into a list of dictionaries, with the final answer wrapped between two ``` markers. The keys for each dictionary are described below:
1. name: str, the name of the component, expressed as a concise and standardized academic term, intended to precisely capture its core identity while facilitating efficient indexing and retrieval from other literature.
2. type: str, One of `Methodology`, `Technique`, `Finding`, or `Resource`.
3. description: str, A detailed, self-contained explanation of the component, focusing on what it is, how it works, and its purpose. For implementable items, describe the whole process without missing any critical steps and implementation details. For insights, describe the core discovery. Maximize the retention of description and implementation details from the original text.
4. components: List[dict], Optional, If the component is a complex `Methodology` or `Techinque` composed of multiple smaller techniques, this field lists its key sub-techniques. Each sub-technique listed here must also be defined separately as a complete technique object following this same JSON schema (with `name`, `type` and `description` as dictionary keys), allowing for hierarchical and recursive decomposition. ATTENTION: Only `Methodology` and `Technique` can have `Technique` as its components!!! 

Now please think and reason carefully, and wrap your final answer between two ``` in the end.
        """
        return prompt
    
    def run(
        self, 
        paper: Paper,
        save_path: str = None
    ) -> Paper:
        # data preparation
        # TODO: When exceeding the context length, truncate each segment to the first 200 characters.
        # TODO: 支持传入上一步latex抽取的地址
        if paper.sections:
            sections_content = json.dumps([asdict(s) for s in paper.sections], ensure_ascii=False)
        
        # code url extraction
        # TODO: Github or Google在线搜索匹配的论文
        github_url_pattern = re.compile(r'https?://github\.com/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?/?')
        extracted_links = github_url_pattern.findall(sections_content)
        if extracted_links:
            code_prompt = self._prompt_get_github_url.format(
                title=paper.title,
                abstract=paper.abstract,
                github_link=", ".join(extracted_links) if extracted_links else "None"
            )
            response = self.backend.query(
                user_message=code_prompt,
                model=self.model
            )
            response = extract_object(extract_backtick_text(response[0]))
            paper.code_url = response if response in extracted_links else None
        logger.debug(f"Extracted code URL: {paper.code_url}")

        # introduction match
        for section in paper.sections:
            if 'introduction' in section.name.lower():
                paper.introduction = section.content
                break
        if not paper.introduction:
            logger.warning("No introduction section found. Defalut set as abstract.")
            paper.introduction = paper.abstract
        logger.debug(f"Extracted introduction: {paper.introduction}...")
        
        # contribution extract
        paper_model = get_config().get("paper_model", self.model)
        contribution_prompt = self._prompt_get_contributions.format(
            title=paper.title,
            sections=sections_content,
            equations=json.dumps(paper.equations) if paper.equations else "None"
        )
        response = self.backend.query(
                system_message=CONTRIBUTION_EXTRACTOR_PROMPT,
                user_message=contribution_prompt,
                model=paper_model
            )
        response = extract_object(extract_backtick_text(response[0]))
        if response and isinstance(response, list):
            contributions = []
            for item in response:
                try:
                    contribution = Contribution.from_dict(item)
                    if contribution.name and contribution.description and contribution.type in CONTRIBUTION_TYPE:
                        contributions.append(contribution)
                except Exception as e:
                    logger.warning(f"Failed to parse contribution item: {item} with error: {e}")
            paper.contributions = contributions if contributions else None
            logger.debug(f"Extracted contributions: {json.dumps([asdict(c) for c in contributions], ensure_ascii=False, indent=2)}")        
        
        # log the extracted content
        paper_content = json.dumps(asdict(paper), ensure_ascii=False, indent=4)
        logger.info(f"Extracted paper content: \n\n{paper_content}\n\n...") 
        
        if save_path:
            output_dir = Path(save_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / f"{sanitize_filename(paper.title)}_llm.json"
            with save_path.open("w", encoding='utf-8') as f:
                json.dump(asdict(paper), f, ensure_ascii=False, indent=4)
            logger.info(f"Successfully extracted and saved data to {save_path}")
            
        return paper


EXTRACTOR_REGISTRY = {
    "pdf": PdfExtractor,
    "latex": LatexExtractor,
    "md": MdExtractor,
}
PaperFormat = Literal["pdf", "latex", "md"]


class PaperParser(BaseTool):
    def __init__(
        self,
        model: str,
        memory: Optional[str] = None,
        match_extractor: Optional[MatchExtractor] = None,
        llm_extractor: Optional[LLMExtractor] = None,
    ):
        super().__init__(model, memory)
        self.match_extractor = match_extractor
        self.llm_extractor = llm_extractor
        
    def run(
        self, 
        paper_path: str,
        save_path: str = None,
        paper_format: PaperFormat = "latex"
    ) -> Paper:
        if paper_format not in EXTRACTOR_REGISTRY:
            raise ValueError(f"Unsupported paper format: {paper_format}")
        
        if not self.match_extractor:
            self.match_extractor = EXTRACTOR_REGISTRY[paper_format](
                model=self.model,
                memory=self.memory
            )
        
        if not self.llm_extractor:
            self.llm_extractor = LLMExtractor(
                model=self.model,
                memory=self.memory
            )
        
        # 1. 使用MatchExtractor进行结构化信息提取
        paper = self.match_extractor.run(paper_path = paper_path, save_path=save_path)
        if not paper:
            raise ValueError("MatchExtractor failed to extract paper content.")
        
        # 2. 使用LLMExtractor进行进一步的信息补全
        paper = self.llm_extractor.run(paper = paper, save_path=save_path)
        if not paper:
            raise ValueError("LLMExtractor failed to extract paper content.")
        
        return paper

