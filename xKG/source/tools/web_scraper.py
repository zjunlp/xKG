""" 
收集信息
"""
import re
import requests
import feedparser
import os
import tarfile
import time
from rapidfuzz import fuzz
from typing import Optional
import logging
import json
import subprocess
from urllib.parse import urlparse, urlunparse, quote
from .base_tool import BaseTool
from ..llm import extract_backtick_text, extract_object
from ..utils import *

logger = logging.getLogger(__name__)    
MAX_ARXIV_RESULTS = 50

class WebScraper(BaseTool):
    def __init__(
        self, model: str = None, 
        memory: str | None = None
    ):
        super().__init__(model, memory)
        
    def run(
        self, 
        query, 
        save_path="./"
    ) -> str:
        """
        根据查询 从网络中爬取相关知识 并保存在对应路径中
        """
        pass

class GithubScraper(WebScraper):
    def __init__(
        self,
        model: Optional[str] = None,
        memory: Optional[str] = None,
    ):
        super().__init__(model, memory)
    
    def repo_filter(self, repo_url: str) -> bool:
        pass
    
    def extract_repo_name(self, repo_url_or_path: str) -> str:
        # Extract owner and repo name to create unique identifier
        url_parts = repo_url_or_path.rstrip('/').split('/')

        if len(url_parts) >= 5:
            # GitHub URL format: https://github.com/owner/repo
            owner = url_parts[-2]
            repo = url_parts[-1].replace(".git", "")
            repo_name = f"{owner}_{repo}"
        else:
            repo_name = url_parts[-1].replace(".git", "")
        return repo_name
    
    def run(self, 
            repo_url: str, 
            save_path: str, 
            access_token: str = None
        ) -> str:
        """
        Downloads a Git repository to a specified local path.

        Args:
            repo_url (str): The URL of the Git repository to clone.
            save_path (str): The local directory where the repository will be cloned.
            access_token (str, optional): Access token for private repositories.

        Returns:
            str: The output message from the `git` command.
        """
        try:
            code_save_path = os.path.join(save_path, self.extract_repo_name(repo_url))
            # Check if Git is installed
            logger.info(f"Preparing to clone repository to {code_save_path}")
            subprocess.run(
                ["git", "--version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Check if repository already exists
            if os.path.exists(code_save_path) and os.listdir(code_save_path):
                # Directory exists and is not empty
                logger.warning(f"Repository already exists at {code_save_path}. Using existing repository.")
                return code_save_path

            # Ensure the local path exists
            os.makedirs(code_save_path, exist_ok=True)

            # Prepare the clone URL with access token if provided
            clone_url = repo_url
            access_token = os.getenv('GITHUB_TOKEN') if access_token is None else access_token
            if access_token:
                parsed = urlparse(repo_url)
                # Determine the repository type and format the URL accordingly
                # Format: https://{token}@{domain}/owner/repo.git
                # Works for both github.com and enterprise GitHub domains
                clone_url = urlunparse((parsed.scheme, f"{access_token}@{parsed.netloc}", parsed.path, '', '', ''))
                logger.info("Using access token for authentication")

            # Clone the repository
            logger.info(f"Cloning repository from {repo_url} to {code_save_path}")
            
            # Proxy Setting
            proxy_url = os.getenv('GITHUB_PROXY') or os.getenv('http_proxy') or os.getenv('https_proxy')
            if proxy_url:
                logger.info(f"Setting up a dedicated proxy for github.com: {proxy_url}")
                try:
                    # 使用 http.https://github.com.proxy 这个关键参数
                    result = subprocess.run(
                        ["git", "config", "--global", "http.https://github.com.proxy", proxy_url],
                        check=True,        
                        capture_output=True,  
                        text=True           
                    )
                    logger.info("Successfully set the git proxy for github.com.")
                except subprocess.CalledProcessError as e:
                    logger.error("Failed to set git proxy for github.com.")
                    logger.error(f"Stderr: {e.stderr}")
            
            # We use repo_url in the log to avoid exposing the token in logs
            result = subprocess.run(
                ["git", "clone", "--depth=1", "--single-branch", clone_url, code_save_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            logger.info("Repository cloned successfully")
            logger.info(f"Git stdout: {result.stdout.decode('utf-8')}")
            # unset http_proxy
            if proxy_url:
                unset_result = subprocess.run(
                    ["git", "config", "--global", "--unset", "http.https://github.com.proxy"],
                    capture_output=True,  
                    text=True           
                )
                if unset_result.returncode not in [0, 5]:
                    logger.warning(f"Could not unset git proxy cleanly. Stderr: {unset_result.stderr.strip()}")
                else:
                    logger.info("Unset the git proxy for github.com (or it was not set).")

                
            return code_save_path
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if isinstance(e.stderr, str) else e.stderr.decode('utf-8', errors='ignore')
            # Sanitize error message to remove any tokens
            if access_token and access_token in error_msg:
                error_msg = error_msg.replace(access_token, "***TOKEN***")
            raise ValueError(f"Error during cloning: {error_msg}")
        except Exception as e:
            raise ValueError(f"An unexpected error occurred: {str(e)}")

class ArxivScraper(WebScraper):
    def __init__(
        self,
        model: str,
        memory: Optional[str] = None,
    ):
        super().__init__(model, memory)
    
    @property
    def _prompt_match_title(self):
        prompt = """
# Task
You are provided with a list of articles retrieved based on the paper title "{title}": {article_list}. Please check the provided article list to determine if it contains an article that is essentially the same as the target paper, disregarding minor differences in spelling, punctuation, line breaks, or trivial words.

# Output
1. If such an article exists, return its index in the article list (starting from 0).
2. If no matching article is found, please return -1.

Please wrap your final answer between two ``` in the end.
        """
        return prompt
    
    def get_arxiv_response(self, query, start=0, max_results=30, sort_by='relevance', sort_order='descending'):
        cleaned_query = clean_string(query)
        base_url = 'http://export.arxiv.org/api/query?'
        
        logger.info(f"Step 1: Attempting CORRECT Precise TITLE search for '{cleaned_query}'...")
        search_query = f'ti:"{cleaned_query}"'
        params = {'search_query': search_query, 'max_results': max_results}
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            feed = feedparser.parse(response.text)
            if feed.entries:
                logger.info("Success with Correct Precise TITLE search!")
                return feed
        except Exception as e:
            logger.warning(f"Step 1 failed: {e}")

        logger.warning("Step 1 failed. Falling back to Step 2: Correct Keyword TITLE search...")
        keyword_query = " AND ".join(f"ti:{word}" for word in cleaned_query.split())
        params = {'search_query': keyword_query, 'max_results': 50}
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            feed = feedparser.parse(response.text)
            if feed.entries:
                logger.info(f"Success with Correct Keyword TITLE search! Found {len(feed.entries)} results.")
                return feed
        except Exception as e:
            logger.warning(f"Step 2 failed: {e}")

        logger.warning("Step 2 failed. Falling back to Step 3: Global Keyword search...")
        params = {'search_query': cleaned_query, 'max_results': 50}
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            feed = feedparser.parse(response.text)
            if feed.entries:
                logger.info(f"Success with Global Keyword search! Found {len(feed.entries)} results.")
                return feed
        except Exception as e:
            logger.error(f"FATAL: All 3 search steps failed: {e}")
            return None
            
        logger.error(f"All search attempts for '{query}' returned 0 results.")
        return None
    
    def download_arxiv_latex(self, arxiv_id, save_path):
        # 路径定义
        download_path = os.path.join(save_path, f'{arxiv_id}_source.tar.gz')
        unzip_path = os.path.join(save_path, f'{arxiv_id}')
        
        logger.debug(f"Download path: {download_path}, Unzip path: {unzip_path}")
        
        # 修复：改进文件存在性检查，检查最终产物（解压目录）
        if os.path.exists(unzip_path):
            logger.info(f"Unzipped directory already exists: {unzip_path}. Skipping process.")
            return

        url = f'https://arxiv.org/e-print/{arxiv_id}'
        
        try:
            response = requests.get(url)
            # 检查状态码并提供更明确的错误信息
            if response.status_code != 200:
                logger.error(f"Failed to download. Status code: {response.status_code} for URL: {url}")
                return
            
            # 确保保存目录存在
            os.makedirs(os.path.dirname(download_path), exist_ok=True)
            with open(download_path, 'wb') as f:
                f.write(response.content)
            logger.debug(f"Paper source has been downloaded: {os.path.basename(download_path)}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"A network error occurred while downloading {url}: {e}")
            return

        try:
            with tarfile.open(download_path, 'r:gz') as tar:
                tar.extractall(path=unzip_path)
                logger.debug(f"Successfully unzipped file to: {unzip_path}")
            os.remove(download_path)
        except tarfile.TarError as e:
            logger.error(f"Failed to extract tar file {download_path}: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during extraction of {download_path}: {e}")    
    
    def run(
        self, 
        query, 
        save_path="./"
    ) -> str:
        retry = 2
        entries = None
        for attempt in range(retry):
            try:
                feed = self.get_arxiv_response(query)
                if feed and feed.entries:
                    entries = feed.entries
                    break
                else:
                    logger.warning(f"Attempt {attempt + 1} failed: No entries returned. Retrying...")
                    time.sleep(2)
            except Exception as e:
                time.sleep(2)
                logger.warning(f"Attempt {attempt + 1} failed with an exception: {e}. Retrying...")
            
        if not entries:
            logger.error(f"No paper matched in arxiv after all retries for query: {query}")
            return
        
        # TODO: use similarity match to do this
        title_list = [entry.title for entry in entries]
        prompt = self._prompt_match_title.format(title=query, article_list=title_list)
        response = self.backend.query(
            user_message=prompt,
            model = self.model
        )
        logger.debug(f"LLM response for title matching: {response}")
        idx = extract_object(extract_backtick_text(response[0]))

        if not isinstance(idx, int) or idx < 0 or idx >= len(entries):
            logger.error(f"LLM did not return a valid index. No paper matched in arxiv for query: {query}")
            return

        entry = entries[idx]
        arxiv_id = entry.link.split('/')[-1]
        
        logger.debug(f"Original query: '{query}'")
        logger.debug(f"Matched title by LLM: '{entry.title}'")
        
        arxiv_json = {
                "title": entry.get("title", "").replace("\n", "").strip(),
                "abstract": entry.get("summary", "").strip(),
                "url": entry.get("link", ""),
                "year": int(entry.get("published", "")[:4]) if entry.get("published") else -1,
                "authors": [author.get("name", "") for author in entry.get("authors", [])]
        }
        logger.debug(f"Extracted metadata: {json.dumps(arxiv_json, indent=2)}")
        
        # download source latex
        self.download_arxiv_latex(arxiv_id, save_path)
        
        # download meta info
        paper_path = os.path.join(save_path, f'{arxiv_id}')
        json_path = os.path.join(paper_path, f'{arxiv_id}.json')
        
        content = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding='utf-8') as f:
                    content = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Could not decode existing JSON file at {json_path}. It will be overwritten.")
                content = {}
        
        content = arxiv_json
        
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=4)
            
        return paper_path
        
