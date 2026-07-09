"""
Formatted parsing object definitions
"""
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

class Contribution(BaseModel):
    name: str = Field(description="Named entities appearing in the text.")
    type: str = Field(description="Named entities appearing in the text.")
    description: str = Field(description="Named entities appearing in the text.")
    components: Optional[List['Contribution']] = Field(description="Named entities appearing in the text.")