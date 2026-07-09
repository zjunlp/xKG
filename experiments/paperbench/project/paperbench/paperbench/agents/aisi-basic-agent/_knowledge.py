
import asyncio
import logging
import os
import sys
from inspect_ai.tool import Tool, ToolError, tool, ToolResult
from dataclasses import asdict
import json

# xKG lazy import: only actually called when KNOWLEDGE_ENABLED=true
_xkg_imported = False
Node = None
Technique = None
initialize_kg = None
find_similar_techniques = None
find_similar_papers = None


def _ensure_xkg_imported():
    """Lazy-import the xKG package, executed only on first use."""
    global _xkg_imported, Node, Technique, initialize_kg, find_similar_techniques, find_similar_papers
    if _xkg_imported:
        return
    try:
        from xKG.source.schema.garph import Node as _Node, Technique as _Technique
        from xKG.source.interface.retrieve import (
            initialize_kg as _initialize_kg,
            find_similar_techniques as _find_similar_techniques,
            find_similar_papers as _find_similar_papers
        )
        Node = _Node
        Technique = _Technique
        initialize_kg = _initialize_kg
        find_similar_techniques = _find_similar_techniques
        find_similar_papers = _find_similar_papers
        _xkg_imported = True
    except ImportError as e:
        raise ImportError(
            f"xKG package is not installed. Please ensure xKG is installed in the container via 'pip install -e /opt/xKG'.\n"
            f"Original error: {e}"
        )

logger = logging.getLogger(__name__)

_kg_instance = None
_kg_init_lock = asyncio.Lock()

async def get_kg_instance():
    global _kg_instance
    # If instance already exists, return it directly
    if _kg_instance:
        return _kg_instance

    # Ensure xKG is imported
    _ensure_xkg_imported()

    # Use a lock to prevent multiple concurrent tasks from initializing KG simultaneously
    async with _kg_init_lock:
        # Check again, because while waiting for the lock, another task may have completed initialization
        if _kg_instance:
            return _kg_instance

        logger.info("--- [KG] Initializing knowledge graph, this may take some time...")
        try:
            # initialize_kg is a synchronous function; we use to_thread to run it in a background thread
            # to avoid blocking the asyncio event loop
            graph = await asyncio.to_thread(initialize_kg)
            _kg_instance = graph
            logger.info("--- [KG] Knowledge graph initialization complete.")
            return _kg_instance
        except Exception as e:
            logger.error(f"Knowledge graph initialization failed: {e}", exc_info=True)
            raise ToolError(f"Unable to initialize knowledge graph: {e}")

# ==============================================================================
#  Tool 1: Get key techniques from the paper
# ==============================================================================
@tool
def get_overview() -> Tool:
    """
    Defines a tool to extract key techniques from a specified paper.
    """
    async def execute() -> ToolResult:
        """
        Extracts and lists key academic techniques and contributions from the given academic paper. This tool requires no arguments.

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
#  Tool 2: Get similar techniques
# ==============================================================================
@tool
def get_similar_techniques() -> Tool:
    """
    Defines a tool to find similar techniques based on a given technique name and description.
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
            raise ToolError("Both technique_name and technique_description parameters must not be empty.")

        logger.info(f"--- [Tool] Searching for similar items for technique '{technique_name}'...")

        try:
            await get_kg_instance()
            
            # Assuming find_similar_techniques returns a list of techniques
            similar_techs = await asyncio.to_thread(
                find_similar_techniques, name=technique_name, description=technique_description, code_only=True, top_k=1
            )

            if not similar_techs:
                return f"No similar techniques found for '{technique_name}'."

            # Format the output
            result = []
            for tech, score in similar_techs:
                tech_info = {
                    "name": tech.name,
                    "implementation": tech.code.implementation,
                    "test": tech.code.test,
                    "documenation": tech.code.documentation
                }
                result.append(tech_info)
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
            logger.error(f"Error occurred while searching for similar techniques: {e}", exc_info=True)
            return f"An internal error occurred while searching for similar techniques for '{technique_name}': {str(e)}"

    return execute

# ==============================================================================
#  Tool 3: Get relevant implementations of key paper techniques
# ==============================================================================
@tool
def get_full_techniques() -> Tool:
    """
    Defines a tool to extract key techniques from a specified paper 
    and retrieve relevant code implementations from the knowledge base.
    """
    async def execute() -> ToolResult:
        """
        Retrieves code snippets from the knowledge base that may be useful for implementing the paper. This tool requires no arguments.

        Returns:
            ToolResult: A formatted string containing relevant technique implementations from the knowledge base.
        """
        logger.info("--- [Tool] Starting to extract key techniques from the paper guide and find relevant implementations...")

        try:
            # Ensure xKG is imported (Node class is needed)
            _ensure_xkg_imported()

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
                    find_similar_techniques, name=technique.name, description=technique.description, code_only=True, top_k=3
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
                    } for tech, score in similar_techs
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