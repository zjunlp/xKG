
from typing import Optional, Tuple, List, Literal, Dict
import logging

from ..schema import Code, File, Technique
from .base_tool import BaseTool
from ..utils import *
from ..llm import extract_backtick_text, extract_backtick_texts, extract_object, REPOSITORY_ANALYZER_PROMPT

# 注册logger
logger = logging.getLogger(__name__)
# 可以放一些全部变量定义
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
        # 一些依附的其它类的定义
        self.component1 = Component1()
        self.component2 = Component2()
    
    @property
    def _prompt_xxx(self):
        prompt = """
        本模块中一些用到的user prompt放在这里定义
        {input}
        sys prompt放在research_kg/llm/prompt.py目录下
        """
        return prompt
    
    def _xxx_xxx(self):
        """
        一些类内用到 外部不会用到的函数放在这里
        用_开头定义
        """
        
    def xx_xx(self, input):
        """
        可供外部运行的函数
        """
        # A: 简单的例子：调用LLM
        prompt = self._prompt_xxx.format(input=input)
        response = self.backend.query(
            user_message=prompt, # 指定user prompt
            system_message=REPOSITORY_ANALYZER_PROMPT, # 指定system prompt 放在research_kg/llm/prompt.py目录下 不指定的话就是默认prompt
            model=self.model,
        )
        # 回答的全部文本 字符串格式
        response_text = response[0]
        # 统一使用logger来显示信息
        logger.debug(f"LLM回答: {response_text}")
        # 从回答中 提取两个```之间的内容(只获取最后一个)
        response_text = extract_backtick_text(response_text)
        # 从回答中 提取两个```之间的内容(可能有多个 提取所有)
        response_text_list = extract_backtick_texts(response_text)
        # 从字符串中自动提取对象：json字符串提取为字典或列表或元组，数字字符串提取为数字，布尔字符串提取为布尔，None和null统一提取为None，其余字符串原样输出
        response_obj = extract_object(response_text)
        # 使用的api直接在ResearchKG/.env中配置即可 initialize_app()函数会自动加载 注：这个函数只要在main中执行一次即可 可参考research_kg/run.py中的实现
        
        # B: 简单的例子：调用config配置
        global_config = get_config()
        # 获取config中的model字段
        model = global_config.get("model")
        # 获取config中定义的code一类
        code_config = get_code_rag_config()
        # 获取code config中对于code_extensions的定义
        code_extensions = code_config.get("code_extensions")
        # 使用的config配置文件路径：ResearchKG/configs/config.yaml
        # 一些简单的区别：.env中配置了一些环境变量 config.yaml中则是实验的一些参数配置 按需设置
        # debug轮次的设置 就很适合在config.yaml中配置 配置好直接加载即可
        
        # C: 简单的例子：数据传递 如果需要频繁的数据传递 可以在research_kg/schema中定义dataclass 实现也参考research_kg/schema
        technique = Technique(
            name="technique name",
            description="technique description",
        )
        
        return technique
        
        
        
        
        