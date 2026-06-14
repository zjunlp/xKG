# research_kg/utils/config.py

import yaml
import sys
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from ..llm import LLMBackend, OpenAIBackend 

BACKEND_REGISTRY = {
    "openai": OpenAIBackend
}

FINAL_CONFIG: dict = None
LLM_INSTANCE: LLMBackend = None

# --- 辅助函数：深度合并字典 ---
def _deep_merge(source, destination):
    """递归地将 source 字典合并到 destination 字典中。"""
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            _deep_merge(value, node)
        else:
            destination[key] = value
    return destination

def _resolve_env_vars(config):
    """递归解析 ${VAR_NAME} 形式的环境变量。"""
    if isinstance(config, dict):
        return {k: _resolve_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_resolve_env_vars(i) for i in config]
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        var_name = config[2:-1]
        value = os.getenv(var_name)
        if value is None:
             raise ValueError(f"Environment variable '{var_name}' is not set but is required.")
        return value
    return config

def _setup_logging(log_level: str):
    """根据给定的级别字符串配置全局日志记录器。"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # 获取根日志记录器
    root_logger = logging.getLogger()
    
    # 如果已经配置过，则不再重复配置
    if root_logger.hasHandlers():
        return
        
    root_logger.setLevel(level)
    
    # 创建一个简单的控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # 添加处理器到根日志记录器
    root_logger.addHandler(console_handler)
    logging.info(f"Logging configured with level: {log_level.upper()}")

def initialize_app(profile_name: str = None):
    """
    加载、解析并返回最终的配置字典。
    """
    global FINAL_CONFIG, LLM_INSTANCE
    
    if FINAL_CONFIG is not None:
        logging.warning("Application is already initialized. Skipping re-initialization.")
        return
    
    # 1. 加载 .env 文件
    project_root = Path(__file__).parent.parent.parent
    load_dotenv(dotenv_path=project_root / ".env")

    # 2. 加载基础的 config.yaml 文件
    with open(project_root / "config.yaml", 'r') as f:
        config_data = yaml.safe_load(f)

    # 3. 解析命令行参数来决定 active_profile
    # 我们寻找 'profile=profile_name' 格式的参数
    active_profile_name = profile_name or config_data.get('active_profile', 'default')

    # 4. 获取选定的配置集
    if active_profile_name not in config_data:
        raise ValueError(f"Profile '{active_profile_name}' not found in config.yaml.")
    
    active_config = config_data[active_profile_name]
    
    # 5. 解析配置集中的环境变量
    final_config = _resolve_env_vars(active_config)
    
    # 6. 将profile名称也加入配置，方便使用
    final_config['profile_name'] = active_profile_name
    
    # 设置日志
    _setup_logging(final_config.get('log_level', 'INFO'))
    
    # 7. 设置llm backend
    api_type = final_config.get('api_type')
    if not api_type:
        raise ValueError("'api_type' not specified in the active profile.")

    backend_class = BACKEND_REGISTRY.get(api_type)
    if not backend_class:
        raise ValueError(f"Unsupported api_type: '{api_type}'. "
                         f"Available types: {list(BACKEND_REGISTRY.keys())}")

    logging.info(f"Creating LLM backend for api_type: '{api_type}'")
    llm_instance = backend_class()
    
    
    FINAL_CONFIG = final_config
    LLM_INSTANCE = llm_instance
    
    return


def get_config() -> dict:
    """获取全局共享的配置字典。如果尚未初始化，则会报错。"""
    if FINAL_CONFIG is None:
        raise RuntimeError("Configuration is not initialized. Please call initialize_app() first.")
    return FINAL_CONFIG

def get_llm_backend() -> LLMBackend:
    """获取全局共享的LLM后端实例。如果尚未初始化，则会报错。"""
    if LLM_INSTANCE is None:
        raise RuntimeError("LLM Backend is not initialized. Please call initialize_app() first.")
    return LLM_INSTANCE

def get_code_rag_config() -> dict:
    """获取嵌入器相关的配置子字典。如果尚未初始化，则会报错。"""
    config = get_config()
    embedder_config = config.get('code')
    if embedder_config is None:
        raise RuntimeError("'code' configuration not found in the global config.")
    return embedder_config