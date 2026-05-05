"""
retrieve.py

这是 ResearchKG 项目的官方公共接口（Facade）。
所有外部调用都应该通过这个模块中的函数进行。
它封装了所有内部的复杂性，提供了简单、直接的函数来与知识图谱交互。

典型使用流程：
1. 调用 `initialize()` 来启动和加载知识图谱。
2. 调用 `find_*()` 系列函数进行查询。
"""
import logging
import json
from typing import Optional, List, Tuple

from ..components.graph_handler import GraphHandler
from ..schema.garph import Node, Technique
from ..schema.paper import Paper
from ..schema.code import Code
from ..utils.config import initialize_app, get_config

logger = logging.getLogger(__name__)


_handler_instance: Optional[GraphHandler] = None


def _get_handler() -> GraphHandler:
    """内部辅助函数，确保引擎已初始化并返回实例。"""
    if not _handler_instance:
        raise RuntimeError(
            "系统尚未初始化。在使用任何查询或构建功能之前，"
            "请务必先调用 `initialize()` 函数。"
        )
    return _handler_instance


# ==============================================================================
#  核心 API 函数
# ==============================================================================

def initialize_kg(profile: str = "basic-deepseek-v3"):
    """
    初始化知识图谱系统

    此函数会执行以下操作：
    1. 加载项目配置。
    2. 创建并持有一个全局的 GraphHandler 实例。
    3. 从配置的路径加载现有的知识图谱数据。
    4. 为加载的数据构建所有必要的索引（技术、标题等）。

    这个函数是幂等的，可以安全地多次调用，但只有第一次会执行实际的初始化。

    Args:
        profile (str): 使用的配置档案，例如 'basic-deepseek-v3'。
    """
    global _handler_instance
    if _handler_instance:
        logger.info("知识图谱系统已经初始化，无需重复操作。")
        return

    logger.info(f"正在初始化知识图谱系统 (Profile: {profile})...")
    initialize_app(profile_name=profile)
    config = get_config()

    # 创建并持有全局实例
    _handler_instance = GraphHandler(model=config.get('model'))
    
    # 加载和索引数据
    kg_path = config.get('kg_path')
    if kg_path:
        _handler_instance.load_kg(kg_path)
        _handler_instance.build_index()
    
    logger.info("Knowledge graph system initialization complete and ready.")


def find_node_by_paper_title(
    title: str, 
    return_code: bool = True
) -> Optional[Node]:
    """
    根据论文标题，精确查找知识图谱中的一个节点(Node)。

    Args:
        title (str): 要查找的论文标题。
        return_code (bool): 输出选项。False时，返回的Node对象及其子项的.code属性会被置为None。

    Returns:
        Optional[Node]: 如果找到匹配的Node对象，则返回该对象；否则返回None。
    """
    handler = _get_handler()
    # 直接调用并返回，将参数透传给 handler
    return handler.find_node_by_paper_title(title=title, return_code=return_code)


def find_similar_techniques(
    name: str,
    description: Optional[str] = None,
    top_k: int = 5,
    code_only: bool = True,
    return_code: bool = True,
    return_components: bool = False,
    llm_rerank: bool = True
) -> List[Technique]:
    """
    根据技术名称和描述，查找知识图谱中最相似的技术(Technique)
    适配于一个具体的技术点的情况 而非一个相对复杂点的或方案描述。

    Args:
        name (str): 技术的名称。
        description (Optional[str]): 技术的详细描述，如果为None，则使用name作为描述。
        top_k (int): 希望返回的最相似结果的数量。
        code_only (bool): 过滤选项。True=只返回有代码的, False=返回所有。
        return_code (bool): 输出选项。False时，返回的Technique对象的.code属性会被置为None。

    Returns:
        一个元组列表，每个元组包含一个 Technique 对象和它的相似度分数。
    """
    handler = _get_handler()
    # 直接调用并返回，将参数透传给 handler
    return handler.retrieve_technique_by_query(
        name=name, 
        description=description, 
        top_k=top_k, 
        code_only=code_only, 
        return_code=return_code,
        return_components=return_components,
        llm_rerank = llm_rerank
    )


def find_similar_papers(
    paper: str | Node, 
    top_k: int = 5,
    return_code: bool = True
) -> List[Node]:
    """
    根据一篇论文，查找知识图谱中最相关的其他论文。

    相关性是基于它们之间共享的技术和引用关系来计算的。

    Args:
        title (str): 作为查询基准的论文标题。
        top_k (int): 希望返回的最相关论文数量。
        return_code (bool): 输出选项。False时，返回的Node对象及其子项的.code属性会被置为None。

    Returns:
        一个元组列表，每个元组包含一个相关的 Node 对象和它的相关度分数。
    """
    handler = _get_handler()
    if isinstance(paper, str):
        # 首先根据标题找到查询的基准节点。必须获取带code的完整节点用于相似度计算。
        query_node = handler.find_node_by_paper_title(title=paper, return_code=True)
        if not query_node:
            logger.warning(f"无法找到基准论文: '{paper}'。无法进行相似论文查找。")
            return []
    elif isinstance(paper, Node):
        query_node = paper

    # 使用基准节点进行检索，并将用户的return_code偏好传递下去
    return handler.retrieve_paper_by_paper(
        query_node, 
        top_k=top_k, 
        return_code=return_code
    )


def decompose_and_find_techniques(
    description: str,
    name: Optional[str] = None,
    top_k: int = 5,
    code_only: bool = False,
    return_components: bool = False,
    return_code: bool = True,
    llm_rerank: bool = True
) -> List[Technique]:
    """
    将一个复杂的技术描述分解为多个子任务，并为每个子任务查找最相关的技术。
    适配于一个相对包含了多种技术的方案，不适合一个简短的概念。

    Args:
        description (str): 复杂技术的描述。
        top_k (int): 每个子任务希望返回的最相关技术数量。
        code_only (bool): 过滤选项。True=只返回有代码的, False=返回所有。
        return_code (bool): 输出选项。False时，返回的Technique对象的.code属性会被置为None。

    Returns:
        一个 Technique 对象列表。
    """
    handler = _get_handler()
    techniques = handler.decompose_task_to_techniques(description=description, name=name)
    if not techniques:
        logger.warning("The task decomposition did not yield any academic techniques.")
        return []
    total_techniques = []
    for tchnique in techniques:
        relevant_techniques = handler.retrieve_technique_by_query(
            name=tchnique.name,
            description=tchnique.description,
            top_k=1,
            code_only=code_only,
            return_code=return_code,
            return_components=return_components,
            llm_rerank=llm_rerank
        )
        if relevant_techniques:
            total_techniques.extend(relevant_techniques)
    # if llm_rerank:
    #     # 对所有找到的技术进行重新排序
    #     reranked_techniques = handler.rerank_techniques_by_llm(
    #         description=description,
    #         candidates=total_techniques,
    #         top_k=top_k,
    #     )
    #     return reranked_techniques
    return total_techniques[:top_k]
