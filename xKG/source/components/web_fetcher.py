"""
Information collection
"""
import re
import requests
import feedparser
import os
import tarfile
import time
from rapidfuzz import fuzz
from typing import List, Optional, Tuple
import logging
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..schema import Code, Paper, Node, Technique, Contribution, FileSnippet, CodeBlock
from urllib.parse import urlparse, urlunparse, quote
from .base_tool import BaseTool
from ..llm import extract_backtick_text, extract_object
from ..utils import *

logger = logging.getLogger(__name__)
MAX_ARXIV_RESULTS = 50

# ──────────────────────────────────────────────────────────────────────────────
# GithubFetcher  —  responsible for cloning repositories
# ──────────────────────────────────────────────────────────────────────────────

class GithubFetcher(BaseTool):
    def __init__(
        self,
        model: Optional[str] = None,
        memory: Optional[str] = None,
    ):
        super().__init__(model, memory)

    def extract_repo_name(self, repo_url_or_path: str) -> str:
        url_parts = repo_url_or_path.rstrip('/').split('/')
        if len(url_parts) >= 5:
            owner = url_parts[-2]
            repo = url_parts[-1].replace(".git", "")
            return f"{owner}_{repo}"
        return url_parts[-1].replace(".git", "")

    @property
    def _prompt_verify_repo(self):
        return """
# Task
You are given a research paper and a GitHub repository README excerpt. Determine whether this repository is the official implementation or a direct reproduction of the paper.

Paper title: {title}
Paper abstract: {abstract}

README excerpt:
{readme_snippet}

# Output
Answer "yes" if this repository implements the paper described above, or "no" if it does not.

Please wrap your final answer between two ``` in the end.
        """

    def _get_readme_snippet(self, full_name: str, max_chars: int = 3000) -> Optional[str]:
        """Fetch the first `max_chars` characters of a repo's README. Returns None if not found."""
        for branch in ['main', 'master']:
            for filename in ['README.md', 'readme.md', 'README.rst', 'README']:
                url = f"https://raw.githubusercontent.com/{full_name}/{branch}/{filename}"
                try:
                    resp = requests.get(url, timeout=10)
                    if resp.status_code == 200:
                        return resp.text[:max_chars]
                except Exception:
                    continue
        return None

    def verify_repo(
        self,
        full_name: str,
        title: str,
        abstract: str,
        max_chars: int = 3000,
        fuzzy_threshold: int = 75,
    ) -> bool:
        """
        Two-stage verification that a GitHub repo implements a given paper.

        Stage 1 — fuzzy pre-filter: check if the paper title appears in the
                   README (rapidfuzz partial_ratio >= fuzzy_threshold).
        Stage 2 — LLM confirm: ask the model to confirm the match using the
                   full README snippet and paper abstract.

        Args:
            full_name:       GitHub "owner/repo" string.
            title:           Paper title.
            abstract:        Paper abstract.
            max_chars:       How many README characters to read.
            fuzzy_threshold: Minimum partial_ratio score for the pre-filter.

        Returns:
            True only if both stages pass.
        """
        snippet = self._get_readme_snippet(full_name, max_chars)
        if snippet is None:
            return False

        # Stage 1: fast fuzzy pre-filter
        fuzzy_score = fuzz.partial_ratio(title.lower(), snippet.lower())
        if fuzzy_score < fuzzy_threshold:
            logger.debug(f"Repo '{full_name}' fuzzy score {fuzzy_score} below threshold {fuzzy_threshold}")
            return False

        # Stage 2: LLM confirmation
        response = self.backend.query(
            user_message=self._prompt_verify_repo.format(
                title=title,
                abstract=abstract,
                readme_snippet=snippet,
            ),
            model=self.model,
        )
        answer = str(extract_object(extract_backtick_text(response[0]))).strip().lower()
        logger.debug(f"LLM repo verification for '{full_name}': {answer} (fuzzy score: {fuzzy_score})")
        return answer == "yes"

    def fetch(
        self,
        repo_url: str,
        save_path: str,
        access_token: str = None,
    ) -> str:
        """
        Clone a Git repository to `save_path`.

        Returns:
            str: Local path of the cloned repository.
        """
        try:
            code_save_path = os.path.join(save_path, self.extract_repo_name(repo_url))
            logger.info(f"Preparing to clone repository to {code_save_path}")
            subprocess.run(
                ["git", "--version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            if os.path.exists(code_save_path) and os.listdir(code_save_path):
                logger.warning(f"Repository already exists at {code_save_path}. Using existing.")
                return code_save_path

            os.makedirs(code_save_path, exist_ok=True)

            clone_url = repo_url
            access_token = os.getenv('GITHUB_TOKEN') if access_token is None else access_token
            if access_token:
                parsed = urlparse(repo_url)
                clone_url = urlunparse(
                    (parsed.scheme, f"{access_token}@{parsed.netloc}", parsed.path, '', '', '')
                )
                logger.info("Using access token for authentication")

            logger.info(f"Cloning repository from {repo_url} to {code_save_path}")

            proxy_url = os.getenv('GITHUB_PROXY') or os.getenv('http_proxy') or os.getenv('https_proxy')

            clone_cmd = ["git"]
            if proxy_url:
                logger.info(f"Using proxy for clone: {proxy_url}")
                clone_cmd += ["-c", f"http.https://github.com.proxy={proxy_url}"]
            clone_cmd += ["clone", "--depth=1", "--single-branch", clone_url, code_save_path]

            result = subprocess.run(
                clone_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            logger.info("Repository cloned successfully")
            logger.info(f"Git stdout: {result.stdout.decode('utf-8')}")

            return code_save_path

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if isinstance(e.stderr, str) else e.stderr.decode('utf-8', errors='ignore')
            if access_token and access_token in error_msg:
                error_msg = error_msg.replace(access_token, "***TOKEN***")
            raise ValueError(f"Error during cloning: {error_msg}")
        except Exception as e:
            raise ValueError(f"An unexpected error occurred: {str(e)}")


# ──────────────────────────────────────────────────────────────────────────────
# ArxivFetcher  —  responsible for exact title search on arXiv and downloading LaTeX source
# ──────────────────────────────────────────────────────────────────────────────

class ArxivFetcher(BaseTool):
    RATE_LIMIT_SECONDS = 3

    def __init__(
        self,
        model: str,
        memory: Optional[str] = None,
    ):
        super().__init__(model, memory)
        self._last_request_time = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.RATE_LIMIT_SECONDS:
            time.sleep(self.RATE_LIMIT_SECONDS - elapsed)
        self._last_request_time = time.time()

    @property
    def _prompt_match_title(self):
        return """
# Task
You are provided with a list of articles retrieved based on the paper title "{title}": {article_list}. Please check the provided article list to determine if it contains an article that is essentially the same as the target paper, disregarding minor differences in spelling, punctuation, line breaks, or trivial words.

# Output
1. If such an article exists, return its index in the article list (starting from 0).
2. If no matching article is found, please return -1.

Please wrap your final answer between two ``` in the end.
        """

    @property
    def _prompt_get_github_url(self):
        return """
# Task
You are provided with the article titled {title} and its corresponding abstract: {abstract}. Here are several GitHub links that may be related to the paper: {github_link}
Select the appropriate GitHub link that corresponds to the provided article and abstract.

# Output
1. If a matching link is found, please return the link.
2. If all links point to cited third-party repositories rather than the paper's official code, return None.

Now please think and reason carefully, and wrap your final answer between two ``` in the end.
        """

    def get_arxiv_response(self, query, max_results=30):
        cleaned_query = clean_string(query)
        base_url = 'http://export.arxiv.org/api/query?'

        logger.info(f"Step 1: Precise TITLE search for '{cleaned_query}'...")
        try:
            resp = requests.get(base_url, params={'search_query': f'ti:"{cleaned_query}"', 'max_results': max_results})
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)
            if feed.entries:
                logger.info("Success with precise TITLE search!")
                return feed
        except Exception as e:
            logger.warning(f"Step 1 failed: {e}")

        logger.warning("Falling back to Step 2: Keyword TITLE search...")
        keyword_query = " AND ".join(f"ti:{word}" for word in cleaned_query.split())
        try:
            resp = requests.get(base_url, params={'search_query': keyword_query, 'max_results': 50})
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)
            if feed.entries:
                logger.info(f"Success with keyword TITLE search! Found {len(feed.entries)} results.")
                return feed
        except Exception as e:
            logger.warning(f"Step 2 failed: {e}")

        logger.warning("Falling back to Step 3: Global Keyword search...")
        try:
            resp = requests.get(base_url, params={'search_query': cleaned_query, 'max_results': 50})
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)
            if feed.entries:
                logger.info(f"Success with global keyword search! Found {len(feed.entries)} results.")
                return feed
        except Exception as e:
            logger.error(f"All 3 search steps failed: {e}")
            return None

        logger.error(f"All search attempts for '{query}' returned 0 results.")
        return None

    def download_arxiv_latex(self, arxiv_id: str, save_path: str):
        """Download and extract the LaTeX source for `arxiv_id` into `save_path/<arxiv_id>/`."""
        download_path = os.path.join(save_path, f'{arxiv_id}_source.tar.gz')
        unzip_path = os.path.join(save_path, arxiv_id)

        if os.path.exists(unzip_path):
            logger.info(f"LaTeX source already exists: {unzip_path}. Skipping.")
            return

        url = f'https://arxiv.org/e-print/{arxiv_id}'
        try:
            response = requests.get(url)
            if response.status_code != 200:
                logger.error(f"Failed to download. Status: {response.status_code} for {url}")
                return
            os.makedirs(os.path.dirname(download_path), exist_ok=True)
            with open(download_path, 'wb') as f:
                f.write(response.content)
            logger.debug(f"Downloaded: {os.path.basename(download_path)}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error while downloading {url}: {e}")
            return

        try:
            with tarfile.open(download_path, 'r:gz') as tar:
                tar.extractall(path=unzip_path)
                logger.debug(f"Extracted to: {unzip_path}")
            os.remove(download_path)
        except tarfile.TarError as e:
            logger.error(f"Failed to extract tar file {download_path}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during extraction of {download_path}: {e}")

    def _read_paper_text(self, paper_dir: str) -> str:
        """Concatenate all .tex files found under `paper_dir`."""
        tex_contents = []
        for root, _, files in os.walk(paper_dir):
            for fname in sorted(files):
                if fname.endswith('.tex'):
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                            tex_contents.append(f.read())
                    except Exception as e:
                        logger.warning(f"Failed to read {fpath}: {e}")
        return "\n".join(tex_contents)

    def extract_repo_link(self, text_content: str, title: str, abstract: str) -> Optional[str]:
        """
        Scan `text_content` for GitHub URLs and use LLM to pick the one that is
        the paper's official implementation.

        Returns:
            Validated GitHub repo URL, or None if not found.
        """
        pattern = re.compile(
            r'(.{0,50})(https?://github\.com/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?/?)(.{0,50})',
            re.DOTALL,
        )
        matches = pattern.findall(text_content)
        if not matches:
            logger.info(f"No GitHub links found in content for: '{title}'")
            return None

        snippets = [f"...{b.strip()} {url} {a.strip()}..." for b, url, a in matches]
        response = self.backend.query(
            user_message=self._prompt_get_github_url.format(
                title=title,
                abstract=abstract,
                github_link="\n".join(snippets),
            ),
            model=self.model,
        )
        logger.debug(f"LLM GitHub extraction response: {response[0]}")

        result = extract_object(extract_backtick_text(response[0]))
        check = re.compile(r'https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?$')
        if result and check.match(str(result).strip()):
            return str(result).strip()
        return None

    def fetch(self, query: str, save_path: str = "./") -> Optional[str]:
        """
        Exact-title search on arXiv, download LaTeX source, save metadata JSON.

        Returns:
            Local directory path (paper_path) consumed by PaperParser, or None.
        """
        entries = None
        for attempt in range(2):
            try:
                feed = self.get_arxiv_response(query)
                if feed and feed.entries:
                    entries = feed.entries
                    break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
            time.sleep(2)

        if not entries:
            logger.error(f"No paper matched on arXiv for query: '{query}'")
            return None

        title_list = [entry.title for entry in entries]
        response = self.backend.query(
            user_message=self._prompt_match_title.format(title=query, article_list=title_list),
            model=self.model,
        )
        idx = extract_object(extract_backtick_text(response[0]))

        if not isinstance(idx, int) or idx < 0 or idx >= len(entries):
            logger.error(f"LLM did not return a valid index for query: '{query}'")
            return None

        entry = entries[idx]
        arxiv_id = entry.link.split('/')[-1]
        logger.debug(f"Query: '{query}' → matched: '{entry.title}' (id={arxiv_id})")

        arxiv_json = {
            "title": entry.get("title", "").replace("\n", "").strip(),
            "abstract": entry.get("summary", "").strip(),
            "url": entry.get("link", ""),
            "year": int(entry.get("published", "")[:4]) if entry.get("published") else -1,
            "authors": [a.get("name", "") for a in entry.get("authors", [])],
        }

        self.download_arxiv_latex(arxiv_id, save_path)

        paper_path = os.path.join(save_path, arxiv_id)
        json_path = os.path.join(paper_path, f'{arxiv_id}.json')
        os.makedirs(paper_path, exist_ok=True)
        with open(json_path, "w", encoding='utf-8') as f:
            json.dump(arxiv_json, f, ensure_ascii=False, indent=4)

        return paper_path

# ──────────────────────────────────────────────────────────────────────────────
# OpenAlexFetcher  —  responsible for fuzzy search of related papers
# ──────────────────────────────────────────────────────────────────────────────
class OpenAlexFetcher:
    """Handles OpenAlex paper search. Search only — download via ArxivFetcher."""

    _BASE_URL = "https://api.openalex.org/works"

    @staticmethod
    def _reconstruct_abstract(inverted_index: dict) -> str:
        if not inverted_index:
            return ""
        pairs = []
        for word, positions in inverted_index.items():
            for pos in positions:
                pairs.append((pos, word))
        pairs.sort()
        return " ".join(w for _, w in pairs)

    def search(self, query: str, max_results: int = 10) -> List[dict]:
        """
        Search OpenAlex for papers matching `query`.

        Returns:
            List of dicts with keys: title, abstract, url, year, authors, arxiv_id.
        """
        try:
            resp = requests.get(
                self._BASE_URL,
                params={"search": query, "per-page": min(max_results * 2, 200)},
                timeout=30,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
        except Exception as e:
            logger.warning(f"OpenAlex search failed for '{query}': {e}")
            return []

        papers: List[dict] = []
        seen: set = set()
        for item in results:
            title = item.get("display_name", "").strip()
            if not title or title.lower() in seen:
                continue

            # Filter out survey/review papers
            title_lower = title.lower()
            if any(keyword in title_lower for keyword in ["survey", "review", "overview"]):
                logger.debug(f"Filtered out survey/review paper: {title}")
                continue

            seen.add(title.lower())

            arxiv_id = None
            for loc in item.get("locations", []):
                lp = loc.get("landing_page_url") or ""
                if "arxiv.org" in lp:
                    arxiv_id = lp.rstrip("/").split("/")[-1]
                    break
            if not arxiv_id:
                best_oa = item.get("best_oa_location") or {}
                lp = best_oa.get("landing_page_url") or ""
                if "arxiv.org" in lp:
                    arxiv_id = lp.rstrip("/").split("/")[-1]

            abstract_inv = item.get("abstract_inverted_index")
            papers.append({
                "title": title,
                "abstract": self._reconstruct_abstract(abstract_inv) if abstract_inv else "",
                "url": item.get("doi") or item.get("id", ""),
                "year": item.get("publication_year") or -1,
                "authors": [
                    a.get("author", {}).get("display_name", "")
                    for a in item.get("authorships", [])
                ],
                "arxiv_id": arxiv_id,
            })
            if len(papers) >= max_results:
                break

        return papers

# ──────────────────────────────────────────────────────────────────────────────
# WebFetcher  —  unified dispatcher: responsible for paper and code downloading
# ──────────────────────────────────────────────────────────────────────────────

class WebFetcher(BaseTool):
    """
    High-level dispatcher that coordinates ArxivFetcher and GithubFetcher.

    Public API:
        paper_search_exact(title, ...)  → (paper_path, code_path) Uses arXiv exact title search to download paper by default
        paper_search_fuzzy(query, ...)  → List[(paper_path, code_path)] Uses OpenAlex fuzzy search for papers, then downloads via arXiv
        code_search_exact(url, ...) → code_path Directly clone a GitHub repository
        code_search_paper(title, ...) → code_path Search for related code repos by paper title, select official implementation
        code_search_fuzzy(query, ...) → List[code_path] Search for related code repos by keywords, return multiple results for downstream filtering

    Note: paper: Paper = None is optional. If None, no LLM verification is performed and the raw search results are returned. If a paper object is provided, its title and abstract are used for LLM verification to filter and return the most relevant results.


    """

    def __init__(
        self,
        model: str = None,
        memory: str = None,
    ):
        super().__init__(model, memory)
        self._arxiv = ArxivFetcher(model, memory)
        self._github = GithubFetcher(model, memory)
        self._openalex = OpenAlexFetcher()

    @property
    def _prompt_verify_paper(self):
        return """
    # Task
    You are given a research paper and the query keyword for a related technique, figure out papers strongly related to that technique query.

    # Input
    Paper Title: {title}
    Paper Abstract: {abstract}
    Query Keyword: {query}
    Retrieved Paper List: {paper_list}

    # Output
    Return a list of filtered papers that are strongly related to the query keyword.

    1. Exclude papers without resources or code implementations.
    2. Exclude papers that follow a different technical approach from the target query (e.g., CV vs. NLP, LLM vs. CNN/RNN).
    3. Sort by technical relevance, and retain only papers strongly related to the target paper and share technical techniques. Strictly ensure quality.
    4. If no relevant paper can be found, return None. Return the paper list even if there's only one relevant paper.

    Now please think and reason step by step, and wrap your final answer between two ``` in the end.
        """

    @property
    def _prompt_select_official_repo(self):
        return """
# Task
You are given a research paper and a list of candidate GitHub repositories. Select the repository that is the official implementation or direct reproduction of the paper. If no such repository exists, return None.

# Input
Paper Title: {title}
Paper Abstract: {abstract}

Candidate Repositories:
{candidates}

# Output
1. Select the repository that is DIRECTLY the paper's official implementation or authoritative reproduction.
2. The repository README should explicitly mention this paper or clearly indicate it implements the paper's methods.
3. If multiple repositories could be official implementations, select the one with higher authority (e.g., official org, more stars, most recent).
4. If no repository is a direct implementation of this paper, return None.
5. Return either a single repository name/URL or None.

Wrap your final answer between two ``` in the end.
"""
    
    # ── Private helpers ──────────────────────────────────────────────────────

    def _load_paper_meta(self, paper_path: str) -> Tuple[str, str]:
        """Return (title, abstract) from the metadata JSON inside paper_path."""
        dir_name = os.path.basename(paper_path)
        json_path = os.path.join(paper_path, f"{dir_name}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                return meta.get("title", ""), meta.get("abstract", "")
            except Exception as e:
                logger.warning(f"Could not read metadata from {json_path}: {e}")
        return "", ""

    def _find_github_url(
        self,
        paper_path: str,
        title: str,
        abstract: str,
    ) -> Optional[str]:
        """
        Two-stage GitHub URL discovery:
          1. Extract from the paper's LaTeX source (via ArxivFetcher).
          2. Fall back to GitHub API search, verified by README content.
        """
        # Stage 1: LaTeX source
        text_content = self._arxiv._read_paper_text(paper_path)
        github_url = self._arxiv.extract_repo_link(text_content, title, abstract)
        if github_url:
            logger.info(f"GitHub URL found in LaTeX: {github_url}")
            return github_url

        # Stage 2: GitHub API search
        logger.info(f"No GitHub URL in LaTeX. Searching GitHub for: '{title}'")
        return self._search_github_for_paper(title, abstract)

    def _search_github_for_paper(self, title: str, abstract: str) -> Optional[str]:
        """
        Search the GitHub API for a repo implementing the paper.
        Collects README snippets from all candidates and uses LLM to select
        the one that is the official implementation or direct reproduction.

        Returns:
            GitHub repo URL of the official/direct implementation, or None.
        """
        headers = {'Accept': 'application/vnd.github+json'}
        token = os.getenv('GITHUB_TOKEN')
        if token:
            headers['Authorization'] = f'Bearer {token}'

        # Normalize title: remove punctuation, keep only alphanumeric words
        cleaned_title = clean_string(title)
        search_query = cleaned_title + " in:readme"
        logger.info(f"Searching GitHub for paper in README: '{search_query}'")

        try:
            resp = requests.get(
                "https://api.github.com/search/repositories",
                params={'q': search_query, 'sort': 'stars', 'order': 'desc', 'per_page': 50},
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
            items = resp.json().get('items', [])
            logger.debug(f"GitHub search found {len(items)} candidates")
        except Exception as e:
            logger.warning(f"GitHub API search failed for '{title}': {e}")
            return None

        if not items:
            return None

        # Parallel fetch README snippets from all candidates
        def fetch_readme(item):
            snippet = self._github._get_readme_snippet(item['full_name'], max_chars=2000)
            if snippet:
                return {
                    'name': item['full_name'],
                    'url': item['html_url'],
                    'readme': snippet,
                    'stars': item.get('stargazers_count', 0),
                }
            return None

        candidates_with_readmes = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_readme, item): item for item in items}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    candidates_with_readmes.append(result)

        if not candidates_with_readmes:
            logger.info(f"No README found for any candidates for: '{title}'")
            return None

        # Format candidates for LLM
        candidates_str = "\n".join([
            f"[{i+1}] {c['name']} (⭐ {c['stars']})\nREADME:\n{c['readme'][:1000]}...\n"
            for i, c in enumerate(candidates_with_readmes)
        ])

        # Use LLM to select the official implementation
        response = self.backend.query(
            user_message=self._prompt_select_official_repo.format(
                title=title,
                abstract=abstract,
                candidates=candidates_str,
            ),
            model=self.model,
        )

        result = extract_object(extract_backtick_text(response[0]))
        if result is None or result == "None":
            logger.info(f"LLM found no official implementation for: '{title}'")
            return None

        # Match the LLM result to one of our candidates
        result_str = str(result).lower().strip()
        for candidate in candidates_with_readmes:
            if candidate['name'].lower() in result_str or candidate['url'].lower() in result_str:
                logger.info(f"GitHub search matched repo: {candidate['url']}")
                return candidate['url']

        logger.warning(f"LLM selected a repo but couldn't match it in candidates for: '{title}'")
        return None

    def _download_by_arxiv_id(
        self,
        arxiv_id: str,
        title: str,
        abstract: str,
        save_path: str,
    ) -> Optional[str]:
        """
        Directly download a paper by its arXiv ID (bypassing LLM title matching).
        Writes a minimal metadata JSON so downstream tools can read it.

        Returns:
            Local paper_path, or None if download failed.
        """
        self._arxiv.download_arxiv_latex(arxiv_id, save_path)
        paper_path = os.path.join(save_path, arxiv_id)
        if not os.path.exists(paper_path):
            return None
        json_path = os.path.join(paper_path, f"{arxiv_id}.json")
        if not os.path.exists(json_path):
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"title": title, "abstract": abstract}, f, ensure_ascii=False, indent=4)
        return paper_path

    def _clone_code(self, github_url: str, code_save_path: str) -> Optional[str]:
        """Clone `github_url` and return code_path, or None on failure."""
        try:
            return self._github.fetch(github_url, save_path=code_save_path)
        except Exception as e:
            logger.error(f"Failed to clone '{github_url}': {e}")
            return None

    def fetch_paper_code_pair(
        self,
        title: str,
        paper_save_path: str = "./",
        code_save_path: str = "./",
        fetch_code: bool = True,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Fetch paper LaTeX and code for a given title.

        Pipeline:
            1. ArxivFetcher.fetch(title)      → paper_path (LaTeX)
            2. (Optional) Extract GitHub URL from paper  → github_url
            3. (Optional) Clone repository               → code_path

        Args:
            title: Paper title to search on arXiv
            paper_save_path: Directory to save paper LaTeX
            code_save_path: Directory to save cloned code
            fetch_code: Whether to fetch and clone code repository (default: True)

        Returns:
            (paper_path, code_path) where either may be None on failure
        """
        paper_path = self._arxiv.fetch(title, save_path=paper_save_path)
        if not paper_path:
            logger.error(f"fetch_paper_code_pair: could not download paper for '{title}'")
            return None, None

        # Only fetch code if requested
        code_path = None
        if fetch_code:
            title_meta, abstract_meta = self._load_paper_meta(paper_path)
            title_meta = title_meta or title
            github_url = self._find_github_url(paper_path, title_meta, abstract_meta)
            code_path = self._clone_code(github_url, code_save_path) if github_url else None

        return paper_path, code_path

    def paper_search_exact(
        self,
        title: str,
        paper_save_path: str = "./",
        code_save_path: str = "./",
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Download a paper by exact title and find its code repository.
        """
        return self.fetch_paper_code_pair(title, paper_save_path, code_save_path)

    def paper_search_fuzzy(
        self,
        query: str,
        max_results: int = 10,
        arxiv_first: bool = True,
        paper: Paper = None,
    ) -> List[str]:
        """
        Find papers by keyword via arXiv/OpenAlex search, optionally filter with LLM verification.

        Returns:
            List of paper titles (verified/reranked if paper provided, else all search results)
        """
        # Search via arXiv (new default) or OpenAlex
        arxiv_feed = self._arxiv.get_arxiv_response(query)
        if arxiv_first and arxiv_feed and arxiv_feed.entries:
            # Extract titles from arXiv results
            papers = [{"title": entry.title.strip()} for entry in arxiv_feed.entries]
        else:
            # Fallback to OpenAlex if arXiv has no results
            papers = self._openalex.search(query)
            if not papers:
                logger.warning(f"paper_search_fuzzy: No results from arXiv or OpenAlex for '{query}'")
                return []

        # LLM verification/reranking if paper provided
        if paper:
            title_list = [p["title"] for p in papers]
            response = self.backend.query(
                user_message=self._prompt_verify_paper.format(
                    title=paper.title,
                    abstract=paper.abstract or "",
                    query=query,
                    paper_list=title_list,
                ),
                model=self.model,
            )
            result = extract_object(extract_backtick_text(response[0]))
            if result is None:
                return []
            if isinstance(result, list):
                keep = {str(r).lower() for r in result}
                papers = [p for p in papers if p["title"].lower() in keep]
                papers = papers[:max_results]  # Limit to max_results after filtering

        # Return only titles 
        return [p["title"] for p in papers]

    def paper_search_fuzzy_with_download(
        self,
        query: str,
        max_results: int = 10,
        paper_save_path: str = "./",
        code_save_path: str = "./",
        paper: Paper = None,
    ) -> List[Tuple[Optional[str], Optional[str]]]:
        """
        Find papers by keyword, verify with LLM if paper provided, then download.

        Returns:
            List of (paper_path, code_path) pairs.
        """
        # Get verified titles
        titles = self.paper_search_fuzzy(
            query=query,
            max_results=max_results,
            paper_save_path=paper_save_path,
            code_save_path=code_save_path,
            paper=paper,
        )

        results: List[Tuple[Optional[str], Optional[str]]] = []
        for title in titles:
            if not title:
                continue

            paper_path = None
            # Try to fetch from arXiv
            paper_path = self._arxiv.fetch(title, save_path=paper_save_path)

            if not paper_path:
                logger.warning(f"paper_search_fuzzy_with_download: could not download '{title}'")
                results.append((None, None))
                continue

            title_meta, abstract_meta = self._load_paper_meta(paper_path)
            github_url = self._find_github_url(paper_path, title_meta or title, abstract_meta or "")
            code_path = self._clone_code(github_url, code_save_path) if github_url else None
            results.append((paper_path, code_path))

        return results

    def code_search_exact(self, url: str, code_save_path: str = "./") -> Optional[str]:
        """Clone a GitHub repository directly by URL."""
        return self._clone_code(url, code_save_path)

    def code_search_paper(
        self,
        title: str,
        paper_save_path: str = "./",
        code_save_path: str = "./",
        abstract: str = "",
    ) -> Optional[str]:
        """
        Find and clone the official code repository for a paper by its title.

        Downloads the paper from arXiv to extract the GitHub URL from LaTeX,
        falls back to GitHub API search if not found.
        """
        paper_path = self._arxiv.fetch(title, save_path=paper_save_path)
        if paper_path:
            title_meta, abstract_meta = self._load_paper_meta(paper_path)
            github_url = self._find_github_url(paper_path, title_meta or title, abstract_meta or abstract)
        else:
            logger.warning(f"code_search_paper: could not download '{title}', falling back to GitHub search")
            github_url = self._search_github_for_paper(title, abstract)

        if not github_url:
            logger.error(f"code_search_paper: no GitHub URL found for '{title}'")
            return None
        return self._clone_code(github_url, code_save_path)

    def code_search_fuzzy(
        self,
        query: str,
        max_results: int = 5,
        code_save_path: str = "./",
    ) -> List[Optional[str]]:
        """
        Search GitHub for repositories matching `query` (sorted by stars).
        No LLM verification — returns multiple candidates for downstream filtering.
        """
        headers = {"Accept": "application/vnd.github+json"}
        token = os.getenv("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            resp = requests.get(
                "https://api.github.com/search/repositories",
                params={"q": query, "sort": "stars", "order": "desc", "per_page": max_results},
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
            items = resp.json().get("items", [])
        except Exception as e:
            logger.warning(f"code_search_fuzzy: GitHub search failed for '{query}': {e}")
            return []

        results = []
        for item in items:
            result = self._clone_code(item["html_url"], code_save_path)
            results.append(result)
            time.sleep(0.5)
        return results