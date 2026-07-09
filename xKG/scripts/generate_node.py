#!/usr/bin/env python3
"""
Generate knowledge graph nodes from paper titles.

Usage:
    python -m xKG.scripts.generate_node --title "Paper Title"
    python -m xKG.scripts.generate_node --title "Title 1" "Title 2" --output-dir ./out --model gpt-4o
"""

import json
import os
import sys
import argparse
import logging
import tempfile
import shutil
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_node_from_title(title: str, output_dir: str = None, output_name: str = None, **kwargs) -> int:
    """Generate a node from a paper title. Returns 0 on success, 1 on failure."""
    from xKG.source.utils.config import initialize_app, get_config, get_raw_papers_path, get_raw_code_path
    from xKG.source.utils.utils import sanitize_filename
    from xKG.source.components.paper_parser import PaperParser
    from xKG.source.components.code_parser import CodeParser
    from xKG.source.components.graph_handler import GraphHandler
    from xKG.source.components.web_fetcher import WebFetcher

    try:
        initialize_app(profile_name=kwargs.get('profile'))
    except:
        pass

    config = get_config()
    model = kwargs.get('model') or config.get('model', 'o3-mini')
    paper_model = kwargs.get('paper_model') or config.get('paper_model', model)
    code_model = kwargs.get('code_model') or config.get('code_model', model)
    output_dir = output_dir or config.get('kg_path', './output')

    logger.info(f"[{title}] Starting (model={model}, paper_model={paper_model}, code_model={code_model})")

    try:
        # Step 1: Fetch paper and code from web
        fetch_code = kwargs.get('fetch_code') if kwargs.get('fetch_code') is not None else True
        logger.info(f"[{title}] Fetching paper{' and code' if fetch_code else ''}...")
        web_fetcher = WebFetcher(model=model)
        paper_path, code_path = web_fetcher.fetch_paper_code_pair(
            title=title,
            paper_save_path=str(get_raw_papers_path()),
            code_save_path=str(get_raw_code_path()),
            fetch_code=fetch_code,
        )
        if not paper_path:
            logger.error(f"[{title}] Failed to fetch paper")
            return 1
        logger.info(f"[{title}] ✓ Paper: {paper_path}")
        if fetch_code:
            if code_path:
                logger.info(f"[{title}] ✓ Code: {code_path}")
            else:
                logger.info(f"[{title}] ⚠ No code repository found")

        # Step 2: Parse paper
        logger.info(f"[{title}] Parsing paper...")
        paper_parser = PaperParser(model=paper_model)
        paper = paper_parser.parse(paper_path=paper_path, paper_format="latex")
        if not paper:
            logger.error(f"[{title}] Paper parsing failed")
            return 1
        contributions_count = len(paper.contributions) if paper.contributions else 0
        logger.info(f"[{title}] ✓ {len(paper.sections)} sections, {contributions_count} contributions")

        # Step 3: Parse code if available
        code = None
        if code_path:
            logger.info(f"[{title}] Parsing code...")
            code_parser = CodeParser(model=model)
            code = code_parser.parse(code_path=code_path)
            if code:
                logger.info(f"[{title}] ✓ Code parsed: {len(code.file_list)} files")

        # Step 4: Generate node from paper + code
        logger.info(f"[{title}] Generating node...")
        llm_rerank = kwargs.get('llm_rerank') if kwargs.get('llm_rerank') is not None else config.get('code', {}).get('llm_rerank', True)
        rag_paper = kwargs.get('rag_paper') if kwargs.get('rag_paper') is not None else config.get('paper', {}).get('rag', True)
        verify_code = kwargs.get('verify_code') if kwargs.get('verify_code') is not None else config.get('code', {}).get('exec_check_code', False)

        node = GraphHandler(model=model).generate_node(
            paper=paper,
            code=code,
            llm_rerank=llm_rerank,
            rag_paper=rag_paper,
            verify_code=verify_code,
            keep_raw_index=True,
        )
        if not node:
            logger.error(f"[{title}] Node generation failed")
            return 1

        # Step 5: Save
        os.makedirs(output_dir, exist_ok=True)
        fname = output_name or f"{sanitize_filename(paper.title)}_graph.json"
        out_path = os.path.join(output_dir, fname)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(node), f, indent=2, ensure_ascii=False)
        logger.info(f"[{title}] ✓ Saved to {out_path}")
        return 0

    except Exception as e:
        logger.error(f"[{title}] ✗ {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--title', nargs='+', required=True, help='Paper title(s)')
    parser.add_argument('--profile', default=None, help='Config profile name (default: active_profile in config.yaml)')
    parser.add_argument('--output-dir', default=None, help='Output directory (default: config kg_path)')
    parser.add_argument('--output-name', default=None, help='Output filename (default: {title}_graph_unverified.json)')
    parser.add_argument('--model', default=None, help='General model (default: config)')
    parser.add_argument('--paper-model', default=None, help='Paper model (default: config)')
    parser.add_argument('--code-model', default=None, help='Code model (default: config)')
    parser.add_argument('--fetch-code', nargs='?', const=True, default=None, type=lambda x: x.lower() == 'true', help='Fetch and process code (default: True)')
    parser.add_argument('--llm-rerank', nargs='?', const=True, default=None, type=lambda x: x.lower() == 'true', help='Use LLM for code re-ranking')
    parser.add_argument('--rag-paper', nargs='?', const=True, default=None, type=lambda x: x.lower() == 'true', help='Enable paper RAG rewrite')
    parser.add_argument('--no-rag-paper', action='store_false', dest='rag_paper', help='Disable paper RAG rewrite')
    parser.add_argument('--verify-code', nargs='?', const=True, default=None, type=lambda x: x.lower() == 'true', help='Verify code loop execution')
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir) if args.output_dir else None

    def run_title(title):
        return title, generate_node_from_title(
            title, output_dir, output_name=args.output_name,
            profile=args.profile,
            model=args.model, paper_model=args.paper_model, code_model=args.code_model,
            fetch_code=args.fetch_code, llm_rerank=args.llm_rerank, rag_paper=args.rag_paper,
            verify_code=args.verify_code
        )

    max_workers = min(5, len(args.title))
    failed = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for title, ret in executor.map(run_title, args.title):
            if ret != 0:
                failed.append(title)

    if len(args.title) > 1:
        logger.info(f"Done: {len(args.title) - len(failed)}/{len(args.title)} succeeded")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
