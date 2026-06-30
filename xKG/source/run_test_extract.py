import os
import json
import logging
from dataclasses import asdict

from .utils import initialize_app, get_config
from .components import PaperParser, CodeParser, GraphHandler
from .components.base_tool import BaseTool

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STORAGE_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage")
PAPER_PATH = os.path.join(STORAGE_ROOT, "raw/paper/2411.06424v3")
CODE_PATH = os.path.join(STORAGE_ROOT, "raw/code/Yushi-Y_dpo-toxic-neurons")
KG_SAVE_PATH = os.path.join(STORAGE_ROOT, "kg_raw")


def main():
    initialize_app()
    config = get_config()

    logger.info(f"Profile: {config['profile_name']}")
    logger.info(f"Model: {config['model']}")

    # Verify paths exist
    assert os.path.exists(PAPER_PATH), f"Paper path not found: {PAPER_PATH}"
    assert os.path.exists(CODE_PATH), f"Code path not found: {CODE_PATH}"

    # Phase 1: Paper Parsing
    logger.info("=" * 60)
    logger.info("Phase 1: Paper Parsing")
    logger.info("=" * 60)

    paper_parser = PaperParser(model=config['model'])
    paper = paper_parser.parse(paper_path=PAPER_PATH, paper_format="latex")

    assert paper is not None, "PaperParser returned None"
    logger.info(f"Paper title: {paper.title}")
    logger.info(f"Sections: {len(paper.sections) if paper.sections else 0}")
    logger.info(f"Contributions: {len(paper.contributions) if paper.contributions else 0}")
    logger.info(f"Code URL: {paper.code_url}")

    # Phase 2: Code Parsing
    logger.info("=" * 60)
    logger.info("Phase 2: Code Parsing")
    logger.info("=" * 60)

    code_parser = CodeParser(model=config['model'])
    code = code_parser.parse(code_path=CODE_PATH)

    assert code is not None, "CodeParser returned None"
    logger.info(f"Code name: {code.name}")
    logger.info(f"File count: {len(code.file_list)}")
    logger.info(f"Overview length: {len(code.overview) if code.overview else 0} chars")

    # Phase 3: Generate Node
    logger.info("=" * 60)
    logger.info("Phase 3: Generate Node")
    logger.info("=" * 60)

    graph_handler = GraphHandler(model=config['model'])
    node = graph_handler.generate_node(
        paper=paper,
        code=code,
        llm_rerank=True,
        save_process=True,
    )

    assert node is not None, "GraphHandler.generate_node returned None"
    logger.info(f"Node paper_title: {node.paper_title}")
    logger.info(f"Techniques: {len(node.techniques) if node.techniques else 0}")

    # Print summary
    logger.info("=" * 60)
    logger.info("ALL PHASES COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
