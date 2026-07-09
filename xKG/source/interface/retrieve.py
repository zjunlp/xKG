"""
retrieve.py

This is the official public interface (Facade) of the xKG project.
All external calls should be made through the functions in this module.
It encapsulates all internal complexity and provides simple, direct functions
to interact with the knowledge graph.

Typical usage flow:
1. Call `initialize_kg()` to start and load the knowledge graph.
2. Call `find_*()` series functions to perform queries.
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
    """Internal helper function to ensure the engine is initialized and return the instance."""
    if not _handler_instance:
        raise RuntimeError(
            "System not yet initialized. Before using any query or build functionality, "
            "please make sure to call the `initialize_kg()` function first."
        )
    return _handler_instance


# ==============================================================================
# Core API functions
# ==============================================================================

def initialize_kg(profile: str = "basic-deepseek-v3"):
    """
    Initialize the knowledge graph system.

    This function performs the following operations:
    1. Load project configuration.
    2. Create and maintain a global GraphHandler instance.
    3. Load existing knowledge graph data from the configured path.
    4. Build all necessary indexes (techniques, titles, etc.) for the loaded data.

    This function is idempotent and can be safely called multiple times,
    but only the first call will perform actual initialization.

    Args:
        profile (str): The configuration profile to use, e.g. 'basic-deepseek-v3'.
    """
    global _handler_instance
    if _handler_instance:
        logger.info("Knowledge graph system is already initialized, no need to repeat.")
        return

    logger.info(f"Initializing knowledge graph system (Profile: {profile})...")
    initialize_app(profile_name=profile)
    config = get_config()

    # Create and maintain global instance
    _handler_instance = GraphHandler(model=config.get('model'))

    # Load and build index for data
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
    Find a node in the knowledge graph by exact paper title match.

    Args:
        title (str): The paper title to search for.
        return_code (bool): Output option. When False, the .code attribute of the returned Node object and its children will be set to None.

    Returns:
        Optional[Node]: The matching Node object if found; otherwise None.
    """
    handler = _get_handler()
    # Directly call and return, passing parameters through to the handler
    return handler.find_node_by_paper_title(title=title, return_code=return_code)


def find_similar_techniques(
    name: str,
    description: Optional[str] = None,
    top_k: int = 5,
    code_only: bool = True,
    return_code: bool = True,
    return_components: bool = False,
    llm_rerank: bool = True
) -> List[Tuple[Technique, float]]:
    """
    Find the most similar techniques in the knowledge graph by technique name and description.
    Suited for a specific technical concept rather than a complex multi-technique solution description.

    Args:
        name (str): The name of the technique.
        description (Optional[str]): A detailed description of the technique. If None, name is used as the description.
        top_k (int): The number of most similar results to return.
        code_only (bool): Filter option. True = only return techniques with code, False = return all.
        return_code (bool): Output option. When False, the .code attribute of the returned Technique objects will be set to None.
        llm_rerank (bool): Whether to use LLM for result reranking.

    Returns:
        A list of tuples, each containing a Technique object and its similarity score.
    """
    handler = _get_handler()
    # Directly call and return, passing parameters through to the handler
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
    Find the most related papers in the knowledge graph given a paper.

    Relevance is computed based on shared techniques and citation relationships between papers.

    Args:
        paper (str | Node): The paper title or Node object to use as the query baseline.
        top_k (int): The number of most related papers to return.
        return_code (bool): Output option. When False, the .code attribute of the returned Node objects and their children will be set to None.

    Returns:
        A list of tuples, each tuple containing a related Node object and its relevance score.
    """
    handler = _get_handler()
    if isinstance(paper, str):
        # First find the query baseline node by title. Must fetch complete node with code for similarity calculation.
        query_node = handler.find_node_by_paper_title(title=paper, return_code=True)
        if not query_node:
            logger.warning(f"Unable to find baseline paper: '{paper}'. Cannot perform similar paper search.")
            return []
    elif isinstance(paper, Node):
        query_node = paper

    # Use baseline node for retrieval and pass down user's return_code preference
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
    Decompose a complex technical description into multiple subtasks and find the most relevant techniques for each.
    Suited for solutions that encompass multiple techniques; not appropriate for a brief single concept.

    Args:
        description (str): The description of the complex technique/solution.
        top_k (int): The number of most relevant techniques to return for each subtask.
        code_only (bool): Filter option. True = only return techniques with code, False = return all.
        return_code (bool): Output option. When False, the .code attribute of the returned Technique objects will be set to None.
        llm_rerank (bool): Whether to use LLM for result reranking.

    Returns:
        A list of Technique objects (the top_k most similar techniques).
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
            # relevant_techniques is a list of tuples (Technique, score), extract just the technique
            total_techniques.extend([tech for tech, score in relevant_techniques])
    return total_techniques[:top_k]
