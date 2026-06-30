#!/usr/bin/env python
"""
Test script for CorpusCollector.collect_corpus()

Loads a Paper from JSON and tests the corpus collection pipeline.
"""

import json
import logging
import sys
from pathlib import Path

from .components.corpus_collector import CorpusCollector
from .schema.paper import Paper
from .utils import initialize_app

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    initialize_app()

    json_path = "/Users/luoyujie/Documents/Code/xKG/tmp/A_Mechanistic_Understanding_of_Alignment_Algorithms_A_Case_Study_on_DPO_and_Toxicity_llm.json"
    logger.info(f"Loading paper from {json_path}")
    with open(json_path, 'r') as f:
        paper_data = json.load(f)
    paper = Paper.from_dict(paper_data)
    logger.info(f"✓ Loaded paper: '{paper.title}'")
    logger.info(f"  Abstract length: {len(paper.abstract)} chars")
    logger.info(f"  References count: {len(paper.references) if paper.references else 0}")

    # Initialize CorpusCollector
    logger.info("\nInitializing CorpusCollector...")
    collector = CorpusCollector(
        model="DeepSeek-V3.2" # Use a lightweight model for testing
    )
    logger.info("✓ CorpusCollector initialized")

    # Test collect_corpus
    logger.info("\nStarting corpus collection...")
    try:
        results = collector.collect_corpus(paper)

        logger.info(f"\n✓ Corpus collection completed!")
        logger.info(f"  Found {len(results)} related papers with code")

        for i, (title, paper_path, code_path) in enumerate(results, 1):
            logger.info(f"\n  [{i}] {title}")
            logger.info(f"      Paper: {paper_path}")
            logger.info(f"      Code:  {code_path}")

        return 0
    except Exception as e:
        logger.error(f"✗ Error during corpus collection: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
