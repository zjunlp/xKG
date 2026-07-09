#!/usr/bin/env python3
"""
Collect related papers and code for a target paper/node.

Usage:
    python -m xKG.scripts.collect_corpus --title "Paper Title"
    python -m xKG.scripts.collect_corpus --title "Title 1" "Title 2" --output-dir ./corpus
    python -m xKG.scripts.collect_corpus --node path/to/node.json --output-dir ./corpus
"""

import json
import os
import sys
import argparse
import logging
from typing import Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _generate_single_node(
    paper_path: str,
    code_path: Optional[str] = None,
) -> bool:
    """
    Generate a KG node from a single paper_path + code_path pair.

    Returns True on success, False on failure.
    """
    from xKG.source.utils.config import get_config, get_kg_path
    from xKG.source.utils.utils import sanitize_filename
    from xKG.source.components.paper_parser import PaperParser
    from xKG.source.components.code_parser import CodeParser
    from xKG.source.components.graph_handler import GraphHandler
    from dataclasses import asdict

    config = get_config()
    model = config.get('model', 'o3-mini')
    paper_model = config.get('paper_model', model)
    code_model = config.get('code_model', model)
    kg_path = str(get_kg_path())

    try:
        if not paper_path or not os.path.exists(paper_path):
            logger.warning(f"  Paper not found: {paper_path}, skipping")
            return False

        paper = PaperParser(model=paper_model).parse(paper_path=paper_path, paper_format="latex")
        if not paper:
            logger.warning(f"  Failed to parse paper: {paper_path}")
            return False

        code = None
        if code_path and os.path.exists(code_path):
            try:
                code = CodeParser(model=code_model).parse(code_path=code_path)
                if code:
                    logger.info(f"  ✓ Code parsed: {len(code.file_list)} files")
            except Exception as e:
                logger.warning(f"  Failed to parse code: {e}")

        node = GraphHandler(model=model).generate_node(
            paper=paper,
            code=code,
            llm_rerank=config.get('code', {}).get('llm_rerank', True),
            rag_paper=config.get('paper', {}).get('rag', True),
            verify_code=False,
            keep_raw_index=True,
        )
        if not node:
            logger.warning(f"  Node generation failed: {paper_path}")
            return False

        os.makedirs(kg_path, exist_ok=True)
        node_path = os.path.join(kg_path, f"{sanitize_filename(paper.title)}_graph.json")
        with open(node_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(node), f, indent=2, ensure_ascii=False)

        logger.info(f"  ✓ Node saved to {node_path}")
        return True

    except Exception as e:
        logger.warning(f"  Error generating node for {paper_path}: {e}")
        return False


def _generate_corpus_nodes(
    corpus_list: list,
    parallel: int = 2,
) -> int:
    """
    Generate KG nodes for collected corpus papers in parallel.

    Args:
        corpus_list: List of (title, paper_path, code_path) tuples
        parallel: Number of parallel workers (default: 2)

    Returns:
        Number of successfully generated nodes
    """
    try:
        from xKG.source.utils.config import initialize_app
        initialize_app()
    except:
        pass

    total = len(corpus_list)

    def _process_one(item):
        idx, (title, paper_path, code_path) = item
        logger.info(f"[{idx}/{total}] Generating node for: {title[:60]}...")
        return _generate_single_node(paper_path, code_path)

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        results = list(executor.map(_process_one, enumerate(corpus_list, 1)))
    success_count = sum(1 for r in results if r)

    logger.info(f"Node generation complete: {success_count}/{total} succeeded")
    return success_count


def collect_corpus_for_input(
    target_input: Union[str, dict],
    output_dir: str,
    max_papers: int = 10,
    generate_node: bool = True,
    parallel_parse: int = 2,
) -> Tuple[bool, int]:
    """
    Collect corpus for a single target (paper title or node JSON file).

    Pipeline:
      1. If target_input is a title string:
         - Download paper from arXiv
         - Parse with PaperParser
      2. If target_input is a node JSON file path:
         - Load node
         - Convert to Paper object
      3. Collect corpus using CorpusCollector
      4. Save results to output_dir
      5. (Optional) Generate and save KG nodes for each corpus paper

    Args:
        target_input: Paper title (string) or node JSON file path
        output_dir: Directory to save results
        max_papers: Maximum papers to collect (default: 10)
        generate_node: Whether to generate KG nodes for collected papers (default: True)
        parallel_parse: Parallel workers for node generation (default: 2)

    Returns:
        (success: bool, num_results: int)
    """
    try:
        from xKG.source.utils.config import initialize_app, get_config
        from xKG.source.components.corpus_collector import CorpusCollector

        try:
            initialize_app()
        except:
            pass

        config = get_config()
        model = config.get('model', 'o3-mini')

        # Determine input type and get target info
        target_title = None
        if isinstance(target_input, str):
            if os.path.isfile(target_input):
                # Node JSON file
                try:
                    with open(target_input, 'r', encoding='utf-8') as f:
                        node_data = json.load(f)
                    target_title = node_data.get('paper_title')
                    logger.info(f"[{target_title}] Loaded from {target_input}")

                    # Use collect_corpus_from_node
                    collector = CorpusCollector(model=model)
                    results = collector.collect_corpus_from_node(node_data)
                except Exception as e:
                    logger.error(f"Failed to process node JSON: {e}")
                    return False, 0
            else:
                # Paper title string
                target_title = target_input

                # Use collect_corpus_from_title
                collector = CorpusCollector(model=model)
                logger.info(f"[{target_title}] Collecting corpus (max_papers={max_papers})...")
                results = collector.collect_corpus_from_title(target_title)
        else:
            logger.error(f"Invalid target_input type: {type(target_input)}")
            return False, 0

        if not target_title:
            logger.error(f"No target title found")
            return False, 0

        # Trim to max_papers
        results = results[:max_papers]
        num_results = len(results)
        corpus_data = [
            {'title': title, 'paper_path': paper_path, 'code_path': code_path}
            for title, paper_path, code_path in results
        ]

        logger.info(f"[{target_title}] ✓ Collected {num_results} papers: {corpus_data}")

        # Step 5: Generate KG nodes for each corpus paper
        if generate_node and num_results > 0:
            logger.info(f"[{target_title}] Generating KG nodes for {num_results} corpus papers...")
            _generate_corpus_nodes(results, parallel=parallel_parse)

        return True, num_results

    except Exception as e:
        logger.error(f"[{target_input}] ✗ {e}")
        import traceback
        traceback.print_exc()
        return False, 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Input: either --title (one or more) or --node (single file)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--title', nargs='+', help='Paper title(s)')
    input_group.add_argument('--node', help='Node JSON file path')

    # Output
    parser.add_argument('--output-dir', default=None, help='Output directory (default: config kg_path)')
    parser.add_argument('--max-papers', type=int, default=10, help='Max papers per target (default: 10)')

    # Processing
    parser.add_argument('--generate-node', nargs='?', const=True, default=None,
                        type=lambda x: x.lower() == 'true', help='Generate KG nodes for collected papers (default: True)')
    parser.add_argument('--parallel-collection', type=int, default=2, help='Parallel workers for paper collection (default: 2)')
    parser.add_argument('--parallel-parse', type=int, default=2, help='Parallel workers for node generation (default: 2)')

    args = parser.parse_args()

    # Initialize config to get defaults
    try:
        from xKG.source.utils.config import initialize_app, get_config
        initialize_app()
        config = get_config()
    except:
        config = {}

    # Resolve generate_node
    generate_node = args.generate_node if args.generate_node is not None else True

    # Collect targets
    targets = []
    if args.title:
        targets = args.title
    elif args.node:
        targets = [args.node]

    # Resolve output_dir
    if args.output_dir is None:
        output_dir = config.get('kg_path', './storage/kg')
    else:
        output_dir = args.output_dir
    output_dir = os.path.abspath(output_dir)

    # Process targets
    def run_target(target):
        return target, *collect_corpus_for_input(
            target_input=target,
            output_dir=output_dir,
            max_papers=args.max_papers,
            generate_node=generate_node,
            parallel_parse=args.parallel_parse,
        )

    max_workers = min(args.parallel_collection, len(targets))
    failed = []
    total_papers = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for target, success, num_papers in executor.map(run_target, targets):
            if not success:
                failed.append(target)
            total_papers += num_papers

    # Summary
    if len(targets) > 1:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Results: {len(targets) - len(failed)}/{len(targets)} succeeded")
        logger.info(f"Total papers collected: {total_papers}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"{'=' * 80}")

    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
