
import asyncio
import logging
import os
import sys
from inspect_ai.tool import Tool, ToolError, tool, ToolResult
from dataclasses import asdict
import json

# 使用主xKG包（通过pip install -e安装）
from xKG.source.schema.garph import Node, Technique
from xKG.source.interface.retrieve import (
    initialize_kg,
    find_similar_techniques,
    find_similar_papers
)

logger = logging.getLogger(__name__)

_kg_instance = None
_kg_init_lock = asyncio.Lock()

async def get_kg_instance():
    global _kg_instance
    # 如果实例已存在，直接返回
    if _kg_instance:
        return _kg_instance
    
    # 使用锁来防止多个并发任务同时初始化KG
    async with _kg_init_lock:
        # 再次检查，因为在等待锁的时候，可能另一个任务已经完成了初始化
        if _kg_instance:
            return _kg_instance
        
        logger.info("--- [KG] 正在初始化知识图谱，这可能需要一些时间...")
        try:
            # initialize_kg 是一个同步函数，我们用 to_thread 在后台线程中运行它
            # 以免阻塞 asyncio 事件循环
            graph = await asyncio.to_thread(initialize_kg)
            _kg_instance = graph
            logger.info("--- [KG] 知识图谱初始化完成。")
            return _kg_instance
        except Exception as e:
            logger.error(f"知识图谱初始化失败: {e}", exc_info=True)
            raise ToolError(f"无法初始化知识图谱: {e}")

# ==============================================================================
#  工具 1: 获取论文的关键技术 
# ==============================================================================
@tool
def get_overview() -> Tool:
    """
    定义一个工具，用于从指定的论文中提取关键技术。
    """
    async def execute() -> ToolResult:
        """
        Extracts and lists key academic techniques and contributions from the given academic paper.

        Returns:
            ToolResult: A formatted string containing key techniques and contributions
                        extracted from the paper.
        """
        file_path = "/home/guide.json"
        with open(file_path, 'r') as file:
            file = json.load(file)            
        res =  f"""
        Here are some key technologies and contributions extracted from the paper. 
        The following points are the paper's core contributions. Use them as a starting point, but your goal is to replicate EVERYTHING in the paper, not just these highlights. Implement algorithms and hyperparameters EXACTLY as the paper specifies.

        {json.dumps(file)}
        """
        return res
    
    return execute

# ==============================================================================
#  工具 2: 获取相似的技术
# ==============================================================================
@tool
def get_similar_techniques() -> Tool:
    """
    定义一个工具，根据给定的技术名称和描述，查找相似的技术。
    """
    async def execute(technique_name: str, technique_description: str) -> ToolResult:
        """
        Finds similar or related academic technique implementations based on the provided name and description.

        Args:
            technique_name (str): The name of the technique to search for.
            technique_description (str): A detailed description of the technique for more accurate matching.
        
        Returns:
            ToolResult: A formatted string with similar technique code implementations,
                        or a message if no techniques are found.
        
        Raises:
            ToolError: If `technique_name` or `technique_description` parameters are empty.
        """
        if not technique_name or not technique_description:
            raise ToolError("technique_name 和 technique_description 参数都不能为空。")

        logger.info(f"--- [Tool] 正在为技术 '{technique_name}' 查找相似项...")

        try:
            await get_kg_instance()
            
            # 假设 find_similar_techniques 返回一个技术列表
            similar_techs = await asyncio.to_thread(
                find_similar_techniques, name=technique_name, description=technique_description, code_only=False, top_k=1
            )

            if not similar_techs:
                return f"No similar techniques found for '{technique_name}'."

            # 格式化输出
            result = []
            for tech in similar_techs:
                tech = {
                    "name": tech.name,
                    "implementation": tech.code.implementation,
                    "test": tech.code.test,
                    "documenation": tech.code.documentation
                }
                result.append(tech)
            res =  f"""
            The following are relevant implementations concerning the technology {technique_name}, as retrieved from the knowledge base.
            Review the following content, which include code snippets to guide your implementation.

            WARNING: Do not copy this code directly. 
            Flexibly adapt and reuse the code based on the paper's settings, including the model architecture, input/output formats, loss functions, data preprocessing, evaluation metrics, parameter configurations, etc.
            
            In your final official code implementation, all simplifications, placeholders, or dummy implementations are forbidden.  Implement algorithms and hyperparameters EXACTLY as the paper specifies.
            
            {json.dumps(result)}

            """
            
            return res

        except Exception as e:
            logger.error(f"查找相似技术时发生错误: {e}", exc_info=True)
            return f"An internal error occurred while searching for similar techniques for '{technique_name}': {str(e)}"

    return execute

# ==============================================================================
#  工具 3: 获取论文关键技术的相关实现
# ==============================================================================
@tool
def get_full_techniques() -> Tool:
    """
    Defines a tool to extract key techniques from a specified paper 
    and retrieve relevant code implementations from the knowledge base.
    """
    async def execute() -> ToolResult:
        """
        Retrieves code snippets from the knowledge base that may be useful for implementing the paper.

        Returns:
            ToolResult: A formatted string containing relevant technique implementations from the knowledge base.
        """
        logger.info("--- [Tool] Starting to extract key techniques from the paper guide and find relevant implementations...")

        try:
            # 1. Read and parse the paper guide file
            file_path = "/home/guide.json"
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    guide_data = json.load(f)
            except FileNotFoundError:
                logger.error(f"Paper guide file not found: {file_path}")
                return ToolResult(f"Error: Paper guide file not found at {file_path}")
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON file: {file_path}", exc_info=True)
                return ToolResult(f"Error: Failed to parse JSON from {file_path}. The file might be corrupted.")

            # 2. Extract key techniques
            paper_node = Node.from_dict(guide_data)
            techniques = paper_node.techniques

            if not techniques:
                logger.warning("No techniques were found in the provided paper guide.")
                return ToolResult("No techniques found in the provided paper.")

            # 3. For each technique, find similar implementations
            await get_kg_instance()

            for technique in techniques:
                logger.info(f"Searching for similar implementations for technique: '{technique.name}'")
                similar_techs = await asyncio.to_thread(
                    find_similar_techniques, name=technique.name, description=technique.description, code_only=False, top_k=3
                )
                
                # Original logic preserved: If no similar techniques are found, just continue to the next one.
                if not similar_techs:
                    logger.warning(f"No relevant implementations found for '{technique.name}'.")
                    continue
                
                # Original logic preserved: Directly modify the 'code' attribute of the technique object.
                implementations = [
                    {
                        "name": tech.name, 
                        "implementation": tech.code.implementation, 
                        "test": tech.code.test, 
                        "documentation": tech.code.documentation
                    } for tech in similar_techs
                ]
                technique.code = f"[Relevant Implementations]: {json.dumps(implementations)}"

            # 4. Format the final output
            techniques_as_dict = [asdict(tech) for tech in techniques]
            res = f"""
            The following are relevant implementations concerning the key techniques in the paper, as retrieved from the knowledge base.
            Review the following content, which include code snippets to guide your implementation.

            WARNING: 
            1. Do not copy this code directly. Flexibly adapt and reuse the code based on the paper's settings, including the model architecture, input/output formats, loss functions, data preprocessing, evaluation metrics, parameter configurations, etc.
            2. Use them as a starting point, but your goal is to replicate EVERYTHING in the paper, not just these highlights. Implement algorithms and hyperparameters EXACTLY as the paper specifies.

            In your final official code implementation, all simplifications, placeholders, or dummy implementations are forbidden.
            {json.dumps(techniques_as_dict, indent=2)}
            """
            return res

        except Exception as e:
            logger.error(f"An unexpected error occurred while getting relevant technique implementations: {e}", exc_info=True)
            return f"An internal error occurred while getting relevant technique implementations: {str(e)}"
        
    return execute