""" 
LLM后端调用接口
"""
import abc
import json
import logging
import time
from funcy import notnone, select_values
import openai
from .llm_utils import OutputType, opt_messages_to_list, backoff_create, FuncSpec
from .prompt import *

logger = logging.getLogger(__name__)

class LLMBackend(abc.ABC):
    """
    所有LLM后端实现的抽象基类（接口）
    """
    @abc.abstractmethod
    def query(
        self,
        user_message: str | None,
        model: str | None,
        system_message: str | None,
        func_spec: FuncSpec | None = None,
        **model_kwargs,
    ) -> tuple[OutputType, float, int, int, dict]:
        """
        执行一次查询并返回完整结果。
        """
        raise NotImplementedError

class OpenAIBackend(LLMBackend):
    """
    使用 openai-python 库与 OpenAI 兼容的 API 进行交互的后端实现。
    """
    
    TIMEOUT_EXCEPTIONS = (
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.InternalServerError,
    )

    def __init__(self, **client_kwargs):
        """
        初始化 OpenAI 后端。
        
        Args:
            **client_kwargs: 传递给 openai.OpenAI() 构造函数的参数，
                             例如 api_key, base_url, max_retries等。
        """
        if 'max_retries' not in client_kwargs:
            client_kwargs['max_retries'] = 0
            
        self._client = openai.OpenAI(**client_kwargs)

    def query(
        self,
        user_message: str | None,
        model: str | None,
        system_message: str | None = DEFAULT,
        func_spec: FuncSpec | None = None,
        **model_kwargs,
    ):
        """
        查询 OpenAI API，可选地使用函数调用。
        如果模型不支持函数调用，会优雅地降级为纯文本生成。
        """
        # FOR QWEN3 等模型的特殊处理
        # if system_message:
        #     system_message += "/no_think"
        logger.debug(
            f"OpenAI Request:"
            f"Sys Message: {system_message}, "
            f"User Message: {user_message}, "
        )
        filtered_kwargs: dict = select_values(notnone, model_kwargs)

        # 转换 system/user 消息为客户端需要的格式
        messages = opt_messages_to_list(system_message, user_message)

        # 如果请求了函数调用，则附加函数规范
        if func_spec is not None:
            filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
            filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

        completion = None
        t0 = time.time()

        # 尝试 API 调用
        try:
            completion = backoff_create(
                self._client.chat.completions.create,
                self.TIMEOUT_EXCEPTIONS,
                messages=messages,
                model=model,
                **filtered_kwargs,
            )
        except openai.BadRequestError as e:
            if "function calling" in str(e).lower() or "tools" in str(e).lower():
                logger.warning(
                    "Function calling was attempted but is not supported by this model. "
                    "Falling back to plain text generation."
                )
                filtered_kwargs.pop("tools", None)
                filtered_kwargs.pop("tool_choice", None)

                completion = backoff_create(
                    self._client.chat.completions.create,
                    self.TIMEOUT_EXCEPTIONS,
                    messages=messages,
                    model=model,
                    **filtered_kwargs,
                )
            else:
                raise

        req_time = time.time() - t0
        choice = completion.choices[0]

        output: OutputType
        if func_spec is None or "tools" not in filtered_kwargs:
            output = choice.message.content
        else:
            tool_calls = getattr(choice.message, "tool_calls", None)
            if not tool_calls:
                logger.warning(
                    "No function call was used despite function spec. Fallback to text.\n"
                    f"Message content: {choice.message.content}"
                )
                output = choice.message.content
            else:
                first_call = tool_calls[0]
                if first_call.function.name != func_spec.name:
                    logger.warning(
                        f"Function name mismatch: expected {func_spec.name}, "
                        f"got {first_call.function.name}. Fallback to text."
                    )
                    output = choice.message.content
                else:
                    try:
                        output = json.loads(first_call.function.arguments)
                    except json.JSONDecodeError as ex:
                        logger.error(
                            "Error decoding function arguments:\n"
                            f"{first_call.function.arguments}"
                        )
                        raise ex
        
        in_tokens = completion.usage.prompt_tokens if completion.usage else 0
        out_tokens = completion.usage.completion_tokens if completion.usage else 0

        info = {
            "system_fingerprint": completion.system_fingerprint,
            "model": completion.model,
            "created": completion.created,
        }
        logger.debug(
            f"OpenAI query completed in {req_time:.2f}s "
            f"Response: {str(output)}..."
        )
        return output, req_time, in_tokens, out_tokens, info


# TODO: support local models
class LocalBackend(LLMBackend):
    pass