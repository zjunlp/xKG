"""
简单的 CodeVerifier 测试脚本
测试依赖提取和 LLM 修复逻辑
"""

import logging
from xKG.source.components.code_verifier import DependencyExtractor
from xKG.source.schema.garph import VerifiableCodeBlock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dependency_extraction():
    """测试依赖提取"""
    logger.info("=== 测试依赖提取 ===")
    
    extractor = DependencyExtractor()
    
    # 测试代码 1：有外部依赖
    code1 = """
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import json
"""
    deps1 = extractor.extract_from_imports(code1)
    logger.info(f"代码 1 的依赖: {deps1}")
    assert 'numpy' in deps1
    assert 'pandas' in deps1
    assert 'sklearn' in deps1
    assert 'os' not in deps1  # stdlib
    assert 'json' not in deps1  # stdlib
    logger.info("✓ 依赖提取测试通过")
    
    # 测试代码 2：错误的语法
    code2 = """
import numpy
from torch import something
"""
    deps2 = extractor.extract_from_imports(code2)
    logger.info(f"代码 2 的依赖: {deps2}")
    assert 'numpy' in deps2
    assert 'torch' in deps2
    logger.info("✓ 正则表达式回退测试通过")


def test_verifiable_code_block():
    """测试 VerifiableCodeBlock 创建"""
    logger.info("=== 测试 VerifiableCodeBlock ===")
    
    block = VerifiableCodeBlock(
        implementation="print('hello')",
        test="assert 1 == 1",
        documentation="A simple test",
        package=["numpy"],
        language="python"
    )
    
    logger.info(f"创建了代码块: {block.documentation}")
    assert block.implementation == "print('hello')"
    assert block.language == "python"
    logger.info("✓ VerifiableCodeBlock 测试通过")


if __name__ == "__main__":
    try:
        test_dependency_extraction()
        test_verifiable_code_block()
        logger.info("\n✅ 所有测试通过!")
    except Exception as e:
        logger.error(f"\n❌ 测试失败: {e}", exc_info=True)
        exit(1)
