#!/usr/bin/env python3
"""
Unified KG retrieval script.

Usage:
    python -m xKG.scripts.retrieve_kg --mode node --title "Paper A"
    python -m xKG.scripts.retrieve_kg --mode techniques --query "attention"
    python -m xKG.scripts.retrieve_kg --mode papers --title "Paper A"
    python -m xKG.scripts.retrieve_kg --mode search --query "complex description"
"""

import sys
import argparse
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def retrieve_node(title: str) -> int:
    """Retrieve single node by paper title."""
    try:
        from xKG.source.interface.retrieve import initialize_kg, find_node_by_paper_title

        logger.info(f"Retrieving: '{title}'")
        initialize_kg()

        node = find_node_by_paper_title(title=title, return_code=True)
        if not node:
            logger.error(f"✗ Not found")
            return 1

        logger.info(f"✓ Retrieved")
        logger.info(f"  Techniques: {len(node.techniques)}, Resources: {len(node.resources)}, Findings: {len(node.findings)}")
        logger.info(f"  Abstract: {node.paper_abstract[:100]}...")

        for i, tech in enumerate(node.techniques[:3], 1):
            logger.info(f"  [{i}] {tech.name} (weight={tech.weight:.2f})")

        return 0

    except Exception as e:
        logger.error(f"✗ {e}")
        return 1


def retrieve_techniques(query: str, top_k: int = None, code_only: bool = None, llm_rerank: bool = None) -> int:
    """Find techniques by simple query (embedding-based search with optional LLM reranking)."""
    try:
        from xKG.source.interface.retrieve import initialize_kg, find_similar_techniques

        logger.info(f"Searching techniques: '{query}'")
        initialize_kg()

        start = time.time()

        # Get results with LLM reranking (default enabled for better relevance)
        results = find_similar_techniques(
            name=query,
            description=query,
            top_k=top_k or 5,
            code_only=code_only if code_only is not None else True,
            return_code=True,
            llm_rerank=llm_rerank if llm_rerank is not None else True  # Default enabled
        )

        elapsed = time.time() - start

        logger.info(f"✓ Found {len(results)} in {elapsed:.2f}s\n")

        for i, result in enumerate(results, 1):
            tech = result[0] if isinstance(result, tuple) else result
            code_indicator = '✓' if tech.code else '✗'
            logger.info(f"  [{i}] {tech.name} (code={code_indicator})")
            if tech.description:
                logger.info(f"      Description: {tech.description[:80]}...")

        return 0

    except Exception as e:
        logger.error(f"✗ {e}")
        return 1


def retrieve_papers(title: str, top_k: int = None) -> int:
    """Find similar papers."""
    try:
        from xKG.source.interface.retrieve import initialize_kg, find_similar_papers

        logger.info(f"Searching similar papers: '{title}'")
        initialize_kg()

        start = time.time()
        kwargs = {'paper': title}
        if top_k is not None:
            kwargs['top_k'] = top_k

        papers = find_similar_papers(**kwargs)
        elapsed = time.time() - start

        logger.info(f"✓ Found {len(papers)} in {elapsed:.2f}s\n")

        for i, node in enumerate(papers, 1):
            logger.info(f"  [{i}] {node.paper_title}")
            logger.info(f"      Techniques: {len(node.techniques)}")

        return 0

    except Exception as e:
        logger.error(f"✗ {e}")
        return 1


def retrieve_search(query: str, top_k: int = None, code_only: bool = None, llm_rerank: bool = None) -> int:
    """Decompose and search complex query."""
    try:
        from xKG.source.interface.retrieve import initialize_kg, decompose_and_find_techniques

        logger.info(f"Searching (complex): '{query}'")
        initialize_kg()

        start = time.time()
        kwargs = {'description': query, 'name': None}
        if top_k is not None:
            kwargs['top_k'] = top_k
        if code_only is not None:
            kwargs['code_only'] = code_only
        if llm_rerank is not None:
            kwargs['llm_rerank'] = llm_rerank

        techniques = decompose_and_find_techniques(**kwargs)
        elapsed = time.time() - start

        logger.info(f"✓ Found {len(techniques)} in {elapsed:.2f}s\n")

        for i, tech in enumerate(techniques, 1):
            logger.info(f"  [{i}] {tech.name} (weight={tech.weight:.2f}, code={'✓' if tech.code else '✗'})")
            if tech.description:
                logger.info(f"      {tech.description[:80]}...")

        return 0

    except Exception as e:
        logger.error(f"✗ {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--mode', required=True, choices=['node', 'techniques', 'papers', 'search'])
    parser.add_argument('--title', default=None, help='Paper title (for node/papers modes)')
    parser.add_argument('--query', default=None, help='Query (for techniques/search modes)')
    parser.add_argument('--limit', type=int, default=None, help='Max results')
    parser.add_argument('--code-only', action='store_true', default=None, help='Only with code')
    parser.add_argument('--llm-rerank', action='store_true', default=None, help='LLM re-ranking')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("xKG Retrieval")
    logger.info("=" * 80)

    try:
        if args.mode == 'node':
            if not args.title:
                logger.error("--title required")
                return 1
            return retrieve_node(args.title)

        elif args.mode == 'techniques':
            if not args.query:
                logger.error("--query required")
                return 1
            code_only = args.code_only if args.code_only is not None else None
            llm_rerank = args.llm_rerank if args.llm_rerank is not None else None
            return retrieve_techniques(args.query, args.limit, code_only, llm_rerank)

        elif args.mode == 'papers':
            if not args.title:
                logger.error("--title required")
                return 1
            return retrieve_papers(args.title, args.limit)

        elif args.mode == 'search':
            if not args.query:
                logger.error("--query required")
                return 1
            code_only = args.code_only if args.code_only is not None else None
            llm_rerank = args.llm_rerank if args.llm_rerank is not None else None
            return retrieve_search(args.query, args.limit, code_only, llm_rerank)

    except KeyboardInterrupt:
        logger.info("\nInterrupted")
        return 130
    except Exception as e:
        logger.error(f"✗ {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
