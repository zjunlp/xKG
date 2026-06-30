""" 
解析code库的内容
"""
import os
import re
import json
from typing import Optional, Tuple, List, Dict
import logging
from pathlib import Path
from dataclasses import asdict
import glob
from ..schema import Code, File
from .base_tool import BaseTool
from ..utils import *
from ..llm import extract_backtick_text, extract_object, REPOSITORY_ANALYZER_PROMPT

logger = logging.getLogger(__name__)

class CodeParser(BaseTool):
    def __init__(
        self,
        model: str,
        memory: Optional[str] = None,
    ):
        super().__init__(model, memory)
    
    @property
    def _prompt_get_code_overview(self):
        prompt = """
# Task
Analyze this GitHub repository {name} and create a structured overview of it.

# Input
1. The complete file tree of the project:
{file_tree}
2. The README file of the project:
{readme}

# Output
Create a detailed overview of the project, including:
- Overview (general information about the project)
- System Architecture (how the system is designed)
- Core Features (key functionality)
Organize the overview in a clear and structured markdown format.

Please wrap your final answer between two ``` in the end.
"""
        return prompt
    
    @property
    def _prompt_get_relevant_paper(self):
        prompt = """
# Task
Analyze this GitHub repository {name}, and determine whether this repository is directly associated with a specific academic paper.

# Input
The README file of the project:
{readme}

# Output
1. If you can find clear evidence that this repository is the official or direct code implementation of a specific academic paper, return the full title of the paper as a string.
2. If there is no sufficient evidence to identify a directly corresponding paper (e.g., only general descriptions, multiple papers, or no paper mentioned), return None.

Please wrap your final answer between two ``` in the end.
"""
        return prompt
    
    def get_file_tree(self, code_path: str) -> Tuple[str, str]:
        """
        Generates a file tree and extracts README content from a local repository.

        Args:
            code_path: The path to the local repository.

        Returns:
            A tuple containing (file_tree_string, readme_content_string).
            
        """
        if not code_path or not os.path.isdir(code_path):
            error_msg = f"Invalid or non-existent directory: '{code_path}'"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        file_paths = []
        readme_content = ""
        
        excluded_dirs = ['.git', '.idea', '.vscode', '__pycache__', 'node_modules', '.venv', 'venv']
        excluded_files = ['.DS_Store']

        try:
            for root, dirs, files in os.walk(code_path, topdown=True):
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in excluded_dirs]
                
                for file in files:
                    if file.startswith('.') or file in excluded_files:
                        continue
                    
                    rel_path = os.path.relpath(os.path.join(root, file), code_path)
                    file_paths.append(rel_path.replace('\\', '/'))

                    if file.lower() == 'readme.md' and not readme_content:
                        try:
                            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                                readme_content = f.read()
                        except Exception as e:
                            logger.warning(f"Could not read README '{os.path.join(root, file)}': {e}")
        
        except Exception as e:
            logger.error(f"Error traversing directory '{code_path}': {e}", exc_info=True)
            raise

        file_tree_str = '\n'.join(sorted(file_paths))
        return file_tree_str, readme_content

    def parse(
        self,
        code_path: str,
        code_extensions: List[str] = None,
        doc_extensions: List[str] = None,
        save_process: bool = None,
    ) -> Code:
        if code_extensions is None:
            code_extensions = get_code_rag_config().get("code_extensions")
        if doc_extensions is None:
            doc_extensions = get_code_rag_config().get("doc_extensions")

        should_save = save_process if save_process is not None else should_save_process()
        save_path = str(self.get_output_path("code_parser")) if should_save else None
        # get meta data
        name = os.path.basename(code_path)
        file_tree, readme = self.get_file_tree(code_path)
        
        # get overview
        overview_prompt = self._prompt_get_code_overview.format(name=name, file_tree=file_tree, readme=readme)
        response = self.backend.query(
            system_message=REPOSITORY_ANALYZER_PROMPT,
            user_message=overview_prompt,
            model=self.model,
        )
        overview = extract_backtick_text(response[0])
        
        # get paper
        paper_prompt = self._prompt_get_relevant_paper.format(name=name, readme=readme)
        response = self.backend.query(
            user_message=paper_prompt,
            model=self.model,
        )
        paper = extract_object(extract_backtick_text(response[0]))
        
        # get file list
        file_list = []
        for ext in code_extensions:
            files = glob.glob(f"{code_path}/**/*{ext}", recursive=True)
            for file in files:
                with open(file, "r", encoding="utf-8") as f:
                    content = f.read()
                relative_path = os.path.relpath(file, code_path)
                is_implementation = (
                        not relative_path.startswith("test_")
                        and not relative_path.startswith("app_")
                        and "test" not in relative_path.lower()
                    )
                file_object = File(
                    name=relative_path,
                    content=content,
                    is_implementation=is_implementation
                )
                file_list.append(file_object)
        for ext in doc_extensions:
            files = glob.glob(f"{code_path}/**/*{ext}", recursive=True)
            for file in files:
                with open(file, "r", encoding="utf-8") as f:
                    content = f.read()
                relative_path = os.path.relpath(file, code_path)
                file_object = File(
                    name=relative_path,
                    content=content,
                    is_implementation=False
                )
                file_list.append(file_object)
                
        # Save Result
        code = Code(
            name=name,
            readme=readme,
            file_tree=file_tree,
            file_list=file_list,
            overview=overview,
            paper=paper
        )
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            save_file = os.path.join(save_path, f"{name}_meta.json")
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(asdict(code), f, indent=4, ensure_ascii=False)
            logger.info(f"Code data saved to {save_file}")
        
        return code
                
