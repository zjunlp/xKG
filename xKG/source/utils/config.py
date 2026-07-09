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

# --- Helper function: Deep merge dictionaries ---
def _deep_merge(source, destination):
    """Recursively merge source dictionary into destination dictionary."""
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            _deep_merge(value, node)
        else:
            destination[key] = value
    return destination

def _resolve_env_vars(config):
    """Recursively resolve environment variables in ${VAR_NAME} format."""
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
    """Configure global logger based on the given level string."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Get root logger
    root_logger = logging.getLogger()

    # Skip reconfiguration if already configured
    if root_logger.hasHandlers():
        return

    root_logger.setLevel(level)

    # Create a simple console handler
    console_handler = logging.StreamHandler(sys.stdout)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)

    # Add handler to root logger
    root_logger.addHandler(console_handler)
    logging.info(f"Logging configured with level: {log_level.upper()}")
        
def initialize_app(profile_name: str = None, config_path: str = None, kg_path: str = None):
    """
    Load, parse, and set the global configuration dictionary.
    All relative paths will be converted to absolute paths relative to the xKG project root.

    Args:
        profile_name: Configuration profile name (default read from config.yaml)
        config_path: Externally specified config.yaml path (can also be set via XKG_CONFIG_PATH env var)
        kg_path: Externally specified KG data directory (can also be set via XKG_KG_PATH env var)
    """
    global FINAL_CONFIG

    package_root = Path(__file__).parent.parent.parent
    repo_root = package_root.parent
    load_dotenv(dotenv_path=repo_root / ".env")

    # 1. Determine config.yaml source
    config_file = Path(config_path or os.environ.get("XKG_CONFIG_PATH") or (package_root / "config.yaml"))

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    active_profile_name = profile_name or config_data.get('active_profile', 'default')

    if active_profile_name not in config_data:
        raise ValueError(f"Profile '{active_profile_name}' not found in config.yaml.")

    active_config = config_data[active_profile_name]
    final_config = _resolve_env_vars(active_config)
    final_config['profile_name'] = active_profile_name

    # 2. Convert all relative paths to absolute paths (relative to xKG project root)
    path_keys = ['raw_index_path', 'raw_papers_path', 'raw_code_path', 'kg_path', 'process_path']
    for key in path_keys:
        if key in final_config:
            path_value = final_config[key]
            # If already absolute path, keep unchanged; otherwise relative to package_root
            if not Path(path_value).is_absolute():
                final_config[key] = str((package_root / path_value).resolve())

    # 3. Allow environment variables to override kg_path and raw_index_path
    override_kg_path = kg_path or os.environ.get("XKG_KG_PATH")
    if override_kg_path:
        # If override_kg_path is relative, also convert to absolute path
        if not Path(override_kg_path).is_absolute():
            override_kg_path = str((package_root / override_kg_path).resolve())
        final_config['kg_path'] = override_kg_path
        final_config['raw_index_path'] = override_kg_path

    _setup_logging(final_config.get('log_level', 'INFO'))

    FINAL_CONFIG = final_config

    logging.info(f"[{os.getpid()}] Application configuration initialized for profile '{active_profile_name}'.")
    if override_kg_path:
        logging.info(f"[{os.getpid()}] KG path override: {override_kg_path}")


def get_config() -> dict:
    """Get the globally shared configuration dictionary. Raises error if not yet initialized."""
    if FINAL_CONFIG is None:
        raise RuntimeError("Configuration is not initialized. Please call initialize_app() first.")
    return FINAL_CONFIG

def get_llm_backend() -> LLMBackend:
    """Create and return a new LLM Backend instance on each call."""
    config = get_config()
    api_type = config.get('api_type')
    if not api_type:
        raise ValueError("'api_type' not specified in the active profile.")

    backend_class = BACKEND_REGISTRY.get(api_type)
    if not backend_class:
        raise ValueError(f"Unsupported api_type: '{api_type}'.")

    # Create and return a new instance on each call
    return backend_class()

def get_code_rag_config() -> dict:
    """Get the code RAG configuration sub-dictionary. Raises error if not yet initialized."""
    config = get_config()
    embedder_config = config.get('code')
    if embedder_config is None:
        raise RuntimeError("'code' configuration not found in the global config.")
    return embedder_config

def get_paper_rag_config() -> dict:
    """Get the paper RAG configuration sub-dictionary. Raises error if not yet initialized."""
    config = get_config()
    embedder_config = config.get('paper')
    if embedder_config is None:
        raise RuntimeError("'paper' configuration not found in the global config.")
    return embedder_config

def get_raw_papers_path() -> Path:
    """Get the raw papers storage path (absolute path relative to xKG project root)."""
    config = get_config()
    raw_papers_path = config.get('raw_papers_path', 'storage/raw/papers')
    # Convert to absolute path (relative to xKG project root)
    abs_path = Path(__file__).parent.parent.parent / raw_papers_path
    return abs_path.resolve()

def get_raw_code_path() -> Path:
    """Get the raw code storage path (absolute path relative to xKG project root)."""
    config = get_config()
    raw_code_path = config.get('raw_code_path', 'storage/raw/code')
    # Convert to absolute path (relative to xKG project root)
    abs_path = Path(__file__).parent.parent.parent / raw_code_path
    return abs_path.resolve()

def get_process_path() -> Path:
    """Get the processing stage output path (absolute path relative to xKG project root)."""
    config = get_config()
    process_path = config.get('process_path', 'storage/process')
    # Convert to absolute path (relative to xKG project root)
    abs_path = Path(__file__).parent.parent.parent / process_path
    return abs_path.resolve()

def get_kg_path() -> Path:
    """Get the knowledge graph storage path (absolute path relative to xKG project root)."""
    config = get_config()
    kg_path = config.get('kg_path', 'storage/kg')
    # Convert to absolute path (relative to xKG project root)
    abs_path = Path(__file__).parent.parent.parent / kg_path
    return abs_path.resolve()

def get_raw_index_path() -> Path:
    """Get the raw index storage path (absolute path relative to xKG project root)."""
    config = get_config()
    raw_index_path = config.get('raw_index_path', 'storage/raw/index')
    # Convert to absolute path (relative to xKG project root)
    abs_path = Path(__file__).parent.parent.parent / raw_index_path
    return abs_path.resolve()

def should_save_process() -> bool:
    """Get the configuration for whether to save process output."""
    config = get_config()
    return config.get('save_process', True)