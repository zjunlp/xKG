""" 
LLM调用相关功能函数
"""
import backoff
import logging
import jsonschema
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin
from typing import Callable, Any
import ast
import json
import regex
import re

PromptType = str | dict | list
FunctionCallType = dict
OutputType = str | FunctionCallType

logger = logging.getLogger(__name__)

@backoff.on_predicate(
    wait_gen=backoff.expo,
    max_value=60,
    factor=1.5,
)
def backoff_create(
    create_fn: Callable, retry_exceptions: list[Exception], *args, **kwargs
):
    try:
        return create_fn(*args, **kwargs)
    except retry_exceptions as e:
        logger.info(f"Backoff exception: {e}")
        return False

def opt_messages_to_list(
    system_message: str | None, user_message: str | None
) -> list[dict[str, str]]:
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    else:
        messages.append({"role": "system", "content": "<none>"})
    if user_message:
        messages.append({"role": "user", "content": user_message})
    else:
        messages.append({"role": "user", "content": "<none>"})
    return messages

def compile_prompt_to_md(prompt: PromptType, _header_depth: int = 1) -> str:
    if isinstance(prompt, str):
        return prompt.strip() + "\n"
    elif isinstance(prompt, list):
        return "\n".join([f"- {s.strip()}" for s in prompt] + ["\n"])

    out = []
    header_prefix = "#" * _header_depth
    for k, v in prompt.items():
        if v:
            out.append(f"{header_prefix} {k}\n")
            out.append(compile_prompt_to_md(v, _header_depth=_header_depth + 1))
    return "\n".join(out)

def extract_object(text: Any) -> Any:
    if not isinstance(text, str):
        if isinstance(text, (bool, type(None), int, float)):
            return text
        text = str(text)
    else:
        text = text

    stripped_text = text.strip()
    if stripped_text == '':
        return None
    
    try:
        return json.loads(stripped_text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(stripped_text)
        except (ValueError, SyntaxError, MemoryError):
            pass

    pattern = regex.compile(r'''
        (
            \{
                (?: [^{}"'\[\]]+ | "(?:\\.|[^"\\])*" | '(?:\\.|[^'\\])*' | (?1) )*
            \}
            |
            \[
                (?: [^{}"'\[\]]+ | "(?:\\.|[^"\\])*" | '(?:\\.|[^'\\])*' | (?1) )*
            \]
            |
            "(?:\\.|[^"\\])*" | '(?:\\.|[^'\\])*'
            |
            -?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?
            |
            true | false | null | True | False | None
        )
    ''', regex.VERBOSE)

    matches = list(pattern.finditer(text))
    if not matches:
        return text

    last_match = matches[-1]
    json_string = last_match.group(0)

    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(json_string)
        except (ValueError, SyntaxError, MemoryError):
            print(f"Extracted string '{json_string}' is not a valid literal. Returning raw string.")
            return json_string  

def extract_backtick_texts(text: str) -> list[str]:
    pattern = r'```(?:[a-zA-Z]*)?\n([\s\S]+?)```'
    matches = re.findall(pattern, text)
    return [block.strip() for block in matches]

def extract_backtick_text(text: str) -> str:
    all_blocks = extract_backtick_texts(text)
    if all_blocks:
        return all_blocks[-1]
    return text

@dataclass
class FuncSpec(DataClassJsonMixin):
    name: str
    json_schema: dict 
    description: str

    def __post_init__(self):
        jsonschema.Draft7Validator.check_schema(self.json_schema)

    @property
    def as_openai_tool_dict(self):
        """Convert to OpenAI's function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema,
            },
        }

    @property
    def openai_tool_choice_dict(self):
        return {
            "type": "function",
            "function": {"name": self.name},
        }

    @property
    def as_anthropic_tool_dict(self):
        """Convert to Anthropic's tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.json_schema,  # Anthropic uses input_schema instead of parameters
        }

    @property
    def anthropic_tool_choice_dict(self):
        """Convert to Anthropic's tool choice format."""
        return {
            "type": "tool",  # Anthropic uses "tool" instead of "function"
            "name": self.name,
        }

