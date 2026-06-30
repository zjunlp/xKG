"""单元测试 CodeVerifier 的各个组件"""

import logging
import json
from pathlib import Path
from source.components.code_verifier import DependencyExtractor, EnvironmentManager, CodeExecutor, CodeVerifier
from source.schema.garph import VerifiableCodeBlock
from source.utils import initialize_app, get_config

logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_dependency_extractor():
    """测试依赖提取"""
    logger.info("\n=== 测试 DependencyExtractor ===")
    
    extractor = DependencyExtractor()
    
    # 测试1：简单导入
    code = "import numpy\nimport os"
    deps = extractor.extract_from_imports(code)
    assert 'numpy' in deps and 'os' not in deps, f"Expected ['numpy'], got {deps}"
    logger.info(f"✓ 简单导入: {deps}")
    
    # 测试2：from 导入
    code = "from torch import nn\nfrom collections import defaultdict"
    deps = extractor.extract_from_imports(code)
    assert 'torch' in deps and 'collections' not in deps, f"Expected ['torch'], got {deps}"
    logger.info(f"✓ from 导入: {deps}")
    
    logger.info("✅ DependencyExtractor 测试通过")


def test_environment_manager():
    """测试环境管理器"""
    logger.info("\n=== 测试 EnvironmentManager ===")
    
    manager = EnvironmentManager(docker_image="xkg-coderunner:latest")
    logger.info(f"Docker 可用: {manager.client is not None}")
    
    # 测试本地执行
    success, stdout, stderr = manager._run_locally("print('hello')")
    assert success, f"本地执行失败: {stderr}"
    assert 'hello' in stdout, f"输出错误: {stdout}"
    logger.info("✓ 本地执行成功")
    
    logger.info("✅ EnvironmentManager 测试通过")


def test_verifiable_code_block():
    """测试代码块对象"""
    logger.info("\n=== 测试 VerifiableCodeBlock ===")
    
    block = VerifiableCodeBlock(
        implementation="x = 1\nprint(x)",
        test="assert x == 1",
        documentation="Test block",
        package=["numpy"],
        language="python"
    )
    
    assert block.implementation == "x = 1\nprint(x)"
    assert block.language == "python"
    logger.info("✓ VerifiableCodeBlock 创建成功")
    
    logger.info("✅ VerifiableCodeBlock 测试通过")


def test_load_unverified_json():
    """测试加载 unverified.json"""
    logger.info("\n=== 测试加载 unverified.json ===")
    
    json_path = Path("storage/process/graph_handler/How_Does_DPO_Reduce_Toxicity_A_Mechanistic_Neuron-Level_Analysis_unverified.json")
    
    if not json_path.exists():
        logger.warning(f"unverified.json 不存在: {json_path}")
        return
    
    with open(json_path) as f:
        data = json.load(f)
    
    logger.info(f"✓ 成功加载 JSON，包含 {len(data.get('techniques', []))} 个 techniques")
    
    # 检查是否有代码块
    def count_code_blocks(techs):
        count = 0
        for tech in techs:
            if tech.get('code'):
                count += 1
            if tech.get('components'):
                count += count_code_blocks(tech['components'])
        return count
    
    code_count = count_code_blocks(data.get('techniques', []))
    logger.info(f"✓ 找到 {code_count} 个代码块")
    
    logger.info("✅ JSON 加载测试通过")


if __name__ == "__main__":
    try:
        initialize_app()
        test_dependency_extractor()
        test_environment_manager()
        test_verifiable_code_block()
        test_load_unverified_json()
        logger.info("\n✅ 所有单元测试通过！")
    except Exception as e:
        logger.error(f"\n❌ 测试失败: {e}", exc_info=True)
        exit(1)
