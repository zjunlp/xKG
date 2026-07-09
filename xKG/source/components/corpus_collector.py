import os
import logging
from typing import List, Optional, Tuple
from pathlib import Path

from ..schema import Paper
from .base_tool import BaseTool
from .web_fetcher import WebFetcher, OpenAlexFetcher, ArxivFetcher
from ..utils import get_raw_papers_path, get_raw_code_path, should_save_process

from ..llm import extract_backtick_text, extract_object

logger = logging.getLogger(__name__)

class CorpusCollector(BaseTool):
    def __init__(
        self,
        model: str,
        memory: Optional[str] = None,
        paper_save_path: Optional[str] = None,
        code_save_path: Optional[str] = None,
    ):
        super().__init__(model, memory)
        self._fetcher = WebFetcher(model, memory)
        self._openalex = OpenAlexFetcher()
        self._arxiv = ArxivFetcher(model=model)
        self.paper_save_path = paper_save_path or str(get_raw_papers_path())
        self.code_save_path = code_save_path or str(get_raw_code_path())
        logger.info(f"CorpusCollector initialized: papers={self.paper_save_path}, code={self.code_save_path}")

    @property
    def _prompt_rank_references(self):
        return """
# Task
You are given a research paper and its reference list.
Carefully analyze the METHODOLOGY and CORE TECHNIQUES used in the paper.
Select the 10 references most relevant to the paper's core methodological approach and technical implementation.

Paper Title: {title}

Paper Full Text:
{full_text}

Reference List:
{references}

# Ranking Criteria:
1. Direct method application: Papers that directly apply or improve the specific methods/algorithms applied
2. Technical implementation details: Related work on implementation techniques
3. Methodological foundation: Papers that provide the core theoretical/methodological basis for the techniques used
4. Relevant applications: Papers that use similar techniques in different contexts

# Output
1. Return a JSON list of up to 10 reference titles, ordered by relevance using the above criteria.
2. Prioritize papers that provide or share methods and techniques directly used in the paper.
3. Exclude any review/survey papers with no code implementations.
4. If fewer than 10 references exist, return all relevant ones.
5. Fix broken/fragmented candidate titles: if a relevant title has single letters separated by spaces that clearly form a word, merge them back into the correct word.

Wrap your final answer between two ``` in the end.
"""

    @property
    def _prompt_generate_queries(self):
        return """
# Task
You are given a research paper. Generate 5 concise search queries to find papers
that are most related to the core implementation techniques in this paper.
Each query should be related to specific domain, method, and technique from the paper.

Paper Title: {title}

Paper Full Text:
{full_text}

Contributions and Techniques: {contributions}

# Output
1. Return a JSON list of exactly 5 search query strings.
2. Queries must be strongly relevant to the paper. Avoid overly broad domain terms (CV, NLP) or overly specific implementation details (loss function implementations).
3. Ensure queries are diverse and cover different aspects of the paper's domain, method and techniques.
4. Ensure queries include specific technical approaches as scope qualifiers (e.g., 'LLM xxx') to reduce cross-domain matches.

Wrap your final answer between two ``` in the end.
"""

    @property
    def _prompt_rerank_papers(self):
        return """
# Task
You are given a target research paper and a list of candidate related papers.
Re-rank the candidates by relevance to the target paper's core problem domain and core implementation techniques.

Target Paper Title: {title}
Target Paper Abstract: {abstract}
Target Paper Contributions and Techniques: {contributions}

Candidate Papers:
{candidates}

# Output
1. Prioritize papers that provide or share methods and techniques directly used in the paper.
2. Exclude the target paper itself if it appears in the list.
3. Exclude any review/survey papers with no code implementations.
4. Only include papers directly relevant to the target paper's PRIMARY DOMAIN. 
5. Return a JSON list of paper titles ordered by relevance.

Wrap your final answer between two ``` in the end.
"""

    def _rank_references(self, paper: Paper, full_text: str = "") -> List[str]:
        """Use LLM to rank paper.references by relevance using full paper text, return top 10 titles."""
        import re
        from pathlib import Path

        if not paper.references:
            return []

        # If full_text not provided, try to load from paper sections or .tex files
        if not full_text:
            # First try to use paper.sections if available
            if paper.sections:
                full_text = ""
                for section in paper.sections:
                    full_text += f"\n{section.name}:\n{section.content}\n"
            else:
                # Fallback to abstract
                full_text = f"Abstract:\n{paper.abstract}"

        response = self.backend.query(
            user_message=self._prompt_rank_references.format(
                title=paper.title,
                full_text=full_text,
                references=paper.references,
            ),
            model=self.model,
        )
        result = extract_object(extract_backtick_text(response[0]))
        if isinstance(result, list):
            return [str(t) for t in result[:10]]
        logger.warning("_rank_references: LLM did not return a list")
        return []

    def _generate_queries(self, paper: Paper, full_text: str = "") -> List[str]:
        """Use LLM to generate search queries from the paper with full text and contributions."""
        import re

        # If full_text not provided, construct from paper sections
        if not full_text:
            if paper.sections:
                full_text = ""
                for section in paper.sections:
                    full_text += f"\n{section.name}:\n{section.content}\n"
            else:
                full_text = f"Abstract:\n{paper.abstract}"

        # Extract contributions description
        contributions_str = ""
        if paper.contributions:
            contributions_str = "; ".join(
                getattr(c, "description", str(c)) for c in paper.contributions
            )

        response = self.backend.query(
            user_message=self._prompt_generate_queries.format(
                title=paper.title,
                full_text=full_text,
                contributions=contributions_str,
            ),
            model=self.model,
        )
        result = extract_object(extract_backtick_text(response[0]))
        if isinstance(result, list):
            return [str(q) for q in result[:5]]
        logger.warning("_generate_queries: LLM did not return a list")
        return []

    def _collect_technique_titles(self, paper: Paper) -> List[str]:
        """
        Generate 5 queries, call paper_search_fuzzy for each query with LLM verification,
        then keep top 2 results per query. Returns deduplicated titles.
        """
        queries = self._generate_queries(paper)
        titles: List[str] = []
        seen: set = set()

        for query in queries:
            # Call paper_search_fuzzy which handles search + LLM verification, returns titles only
            result_titles = self._fetcher.paper_search_fuzzy(
                query=query,
                max_results=10,
                paper=paper,  # Pass paper for LLM verification
            )

            if not result_titles:
                continue

            count = 0
            for title in result_titles:
                if not title:
                    continue
                key = title.lower().strip()
                if key not in seen and key != paper.title.lower().strip():
                    seen.add(key)
                    titles.append(title)
                    count += 1
                if count >= 2:
                    break

        return titles

    def _download_all(
        self, titles: List[str]
    ) -> List[Tuple[str, str, str]]:
        """
        Download paper + code for each title.
        Returns list of (title, paper_path, code_path) where both paths are valid.
        """
        results = []
        total = len(titles)
        for idx, title in enumerate(titles, 1):
            logger.info(f"[{idx}/{total}] Downloading: '{title[:60]}...'")
            try:
                paper_path, code_path = self._fetcher.paper_search_exact(
                    title,
                    paper_save_path=self.paper_save_path,
                    code_save_path=self.code_save_path,
                )
                if paper_path and code_path:
                    logger.info(f"  ✓ Success: paper + code")
                    results.append((title, paper_path, code_path))
                elif paper_path:
                    logger.info(f"  ⚠ Skipping: paper found but no code")
                elif code_path:
                    logger.info(f"  ⚠ Skipping: code found but no paper")
                else:
                    logger.info(f"  ✗ Skipping: paper not found on arXiv")
            except Exception as e:
                logger.warning(f"  ✗ Download failed: {e}")
        logger.info(f"Downloaded: {len(results)}/{total} papers with code")
        return results

    def _rerank_and_trim(
        self, paper: Paper, candidates: List[Tuple[str, str, str]]
    ) -> List[Tuple[str, str, str]]:
        """LLM rerank, exclude the target paper, return top 10."""
        contributions_str = ""
        if paper.contributions:
            contributions_str = "; ".join(
                getattr(c, "description", str(c)) for c in paper.contributions
            )

        titles = [t for t, _, _ in candidates]
        response = self.backend.query(
            user_message=self._prompt_rerank_papers.format(
                title=paper.title,
                abstract=paper.abstract,
                contributions=contributions_str,
                candidates=titles,
            ),
            model=self.model,
        )
        result = extract_object(extract_backtick_text(response[0]))
        if not isinstance(result, list):
            logger.warning("_rerank_and_trim: LLM did not return a list, keeping original order")
            return candidates[:10]

        ranked_titles = [str(t).lower().strip() for t in result]
        title_to_entry = {t.lower().strip(): (t, pp, cp) for t, pp, cp in candidates}

        reranked = []
        for rt in ranked_titles:
            if rt in title_to_entry:
                reranked.append(title_to_entry[rt])
            if len(reranked) >= 10:
                break

        return reranked

    @staticmethod
    def _flatten_techniques(techniques: list) -> list:
        """Recursively flatten nested techniques/components into a flat list."""
        flat = []
        for tech in techniques:
            if not isinstance(tech, dict):
                continue
            flat.append(tech)
            if tech.get('components'):
                flat.extend(CorpusCollector._flatten_techniques(tech['components']))
        return flat

    def _node_to_paper(self, node: dict) -> Optional[Paper]:
        """Convert a Node dict to a Paper object for corpus collection."""
        try:
            all_techniques = self._flatten_techniques(node.get('techniques', []))

            technique_descriptions = []
            contributions = []
            for tech in all_techniques:
                name = tech.get('name', '')
                desc = tech.get('description', '')
                if name and desc:
                    technique_descriptions.append(f"Technique: {name}\n{desc}")
                    contributions.append({'name': name, 'type': 'Technique', 'description': desc})

            paper_data = {
                'title': node.get('paper_title', 'Unknown'),
                'abstract': node.get('paper_abstract', ''),
                'url': None,
                'year': 2024,
                'authors': [],
                'sections': [],
                'references': node.get('paper_references', []),
                'equations': [],
                'introduction': '\n\n'.join(technique_descriptions) if technique_descriptions else node.get('paper_abstract', ''),
                'code_url': None,
                'contributions': contributions,
            }

            paper = Paper.from_dict(paper_data)
            logger.info(f"Converted node to Paper: {paper.title}")
            logger.info(f"  References: {len(paper.references or [])}, Techniques (flat): {len(all_techniques)}")
            return paper

        except Exception as e:
            logger.error(f"Failed to convert node to Paper: {e}", exc_info=True)
            return None

    def collect_corpus_from_title(
        self,
        title: str,
        save_process: bool = None,
    ) -> List[Tuple[str, str, str]]:
        """
        Collect corpus for a paper given its title.

        Args:
            title: Paper title to search on arXiv
            save_process: Whether to save results (default: config setting)

        Returns:
            List of (title, paper_path, code_path) for related papers
        """
        from .paper_parser import PaperParser
        from ..utils import get_raw_papers_path, get_raw_code_path

        logger.info(f"Collecting corpus for title: '{title}'")

        # Step 1: Download paper
        logger.info(f"  Downloading paper from arXiv...")
        paper_path, _ = self._fetcher.fetch_paper_code_pair(
            title=title,
            paper_save_path=str(get_raw_papers_path()),
            code_save_path=str(get_raw_code_path()),
            fetch_code=False,  # Don't fetch code for target paper
        )
        if not paper_path:
            logger.error(f"Failed to fetch paper: {title}")
            return []

        # Step 2: Parse paper
        logger.info(f"  Parsing paper...")
        try:
            paper_parser = PaperParser(model=self.model)
            paper = paper_parser.parse(paper_path=paper_path, paper_format="latex", save_process=False)
            if not paper:
                logger.error(f"Failed to parse paper")
                return []
            logger.info(f"  ✓ Paper parsed: {len(paper.sections)} sections")
        except Exception as e:
            logger.error(f"Paper parsing failed: {e}")
            return []

        # Step 3: Collect corpus
        logger.info(f"  Collecting related papers...")
        return self.collect_corpus_from_paper(paper, save_process=save_process)

    def collect_corpus_from_paper(
        self,
        target_paper: Paper,
        save_process: bool = None,
    ) -> List[Tuple[str, str, str]]:
        """
        Collect related papers with code for a given Paper object.

        Args:
            paper: Paper object to collect corpus for
            save_process: Whether to save final results. If None, uses config setting.

        Returns:
            List of (title, paper_path, code_path), up to 10 entries.
        """
        import json

        logger.info(f"Collecting corpus for: '{target_paper.title}'")

        should_save = save_process if save_process is not None else should_save_process()

        reference_titles = self._rank_references(target_paper)
        logger.info(f"Ranked references: {len(reference_titles)}")

        technique_titles = self._collect_technique_titles(target_paper)
        logger.info(f"Technique search titles: {len(technique_titles)}")

        seen: set = set()
        all_titles: List[str] = []
        for t in reference_titles + technique_titles:
            key = t.lower().strip()
            if key not in seen:
                seen.add(key)
                all_titles.append(t)

        logger.info(f"Total unique candidates to download: {len(all_titles)}")

        downloaded = self._download_all(all_titles)
        logger.info(f"Successfully downloaded (paper+code): {len(downloaded)}")

        if not downloaded:
            return []

        final = self._rerank_and_trim(target_paper, downloaded)
        logger.info(f"Final collect size: {len(final)}")

        if should_save:
            try:
                output_dir = self.get_output_path("corpus_collector")
                save_file = output_dir / "paper_code_pair.json"
                corpus_data = [
                    {"title": title, "paper_path": paper_path, "code_path": code_path}
                    for title, paper_path, code_path in final
                ]
                with open(save_file, 'w', encoding='utf-8') as f:
                    json.dump(corpus_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Corpus saved to {save_file}")
            except Exception as e:
                logger.error(f"Failed to save corpus: {e}", exc_info=True)

        return final

    def collect_corpus_from_node(
        self,
        target_node: dict,
        save_process: bool = None,
    ) -> List[Tuple[str, str, str]]:
        """
        Collect related papers with code for a given Node object.

        This method converts a node (knowledge graph node with paper metadata and techniques)
        into a Paper object and then collects related corpus.

        Args:
            node: Node dict with keys like:
                  - paper_title: str
                  - paper_abstract: str
                  - paper_references: List[str]
                  - techniques: List[dict] with 'name', 'description', 'components'
                  - code_overview: Optional[str]
            save_process: Whether to save final results. If None, uses config setting.

        Returns:
            List of (title, paper_path, code_path), up to 10 entries.
        """
        import json

        # Convert node to Paper object
        paper_title = target_node.get('paper_title', 'Unknown')
        logger.info(f"Collecting corpus for node: '{paper_title}'")

        # Construct Paper object from node
        paper = self._node_to_paper(target_node)
        if not paper:
            logger.error(f"Failed to convert node to Paper")
            return []

        # Use existing collect_corpus logic with converted Paper
        return self.collect_corpus_from_paper(paper, save_process=save_process)