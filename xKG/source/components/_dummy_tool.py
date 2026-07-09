
from typing import Optional, Tuple, List, Literal, Dict
import logging

from ..schema import Code, File, Technique
from .base_tool import BaseTool
from ..utils import *
from ..llm import extract_backtick_text, extract_backtick_texts, extract_object, REPOSITORY_ANALYZER_PROMPT

# Register logger
logger = logging.getLogger(__name__)
# Can place some global variable definitions
GLOBAL_NUM = 1

class Component1():
    pass

class Component2():
    pass

class DummyTool(BaseTool):
    def __init__(
        self,
        model: str,
        memory: Optional[str] = None,
    ):
        super().__init__(model, memory)
        # Definitions of some dependent classes
        self.component1 = Component1()
        self.component2 = Component2()
    
    @property
    def _prompt_xxx(self):
        prompt = """
        User prompts used in this module are defined here
        {input}
        System prompt is placed in xKG/llm/prompt.py directory
        """
        return prompt

    def _xxx_xxx(self):
        """
        Internal functions only used within the class are placed here.
        Defined with _ prefix.
        """

    def xx_xx(self, input):
        """
        Function available for external use.
        """
        # A: Simple example: Calling LLM
        prompt = self._prompt_xxx.format(input=input)
        response = self.backend.query(
            user_message=prompt,  # Specify user prompt
            system_message=REPOSITORY_ANALYZER_PROMPT,  # Specify system prompt from xKG/llm/prompt.py, default used if not specified
            model=self.model,
        )
        # Complete response text in string format
        response_text = response[0]
        # Use logger consistently to display information
        logger.debug(f"LLM response: {response_text}")
        # Extract content between two ``` (get last one only)
        response_text = extract_backtick_text(response_text)
        # Extract content between two ``` (may have multiple, extract all)
        response_text_list = extract_backtick_texts(response_text)
        # Auto-extract object from string: json string extracted as dict/list/tuple, number string as number, boolean string as boolean, None and null unified to None, rest as string
        response_obj = extract_object(response_text)
        # API configuration can be directly set in xKG/.env file, initialize_app() will auto-load. Note: this function only needs to be executed once in main, refer to xKG/run.py

        # B: Simple example: Calling config
        global_config = get_config()
        # Get model field from config
        model = global_config.get("model")
        # Get code-related configuration
        code_config = get_code_rag_config()
        # Get code_extensions definition from code config
        code_extensions = code_config.get("code_extensions")
        # Config file path: xKG/config.yaml
        # Simple distinction: .env contains environment variables, config.yaml contains experiment parameters, set as needed
        # Debug iteration settings are suitable for config.yaml configuration

        # C: Simple example: Data passing. For frequent data passing, define dataclass in xKG/schema, see xKG/schema for implementation reference
        technique = Technique(
            name="technique name",
            description="technique description",
        )

        return technique
        
        
        
        
        