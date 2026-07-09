import logging 
logging.getLogger("adaflow").setLevel(logging.ERROR)
from adalflow.core.component import DataComponent
from adalflow.core.model_client import ModelClient
from copy import deepcopy
from typing import List, TypeVar, Sequence, Union, Dict, Any, Optional
from tqdm import tqdm
from adalflow.core.types import (
    Document,
    RetrieverOutput,
)
from adalflow.core.embedder import (
    BatchEmbedder,
    BatchEmbedderOutputType,
    BatchEmbedderInputType,
    Embedder,
)
from dataclasses import asdict
import adalflow as adal
from adalflow.components.model_client import TransformersClient, OpenAIClient
from adalflow.core.db import LocalDB
from adalflow.components.data_process import TextSplitter, ToEmbeddings
import os

import tiktoken
import json
from adalflow.components.retriever.faiss_retriever import FAISSRetriever
from .embedder import BaseEmbedder
from ..utils import get_code_rag_config
from ..schema import Code, FileSnippet


logger = logging.getLogger(__name__)

MAX_EMBEDDING_TOKENS = 8192

class CodeRAG:
    def __init__(self):
        """
        Initialize CodeRAG
        """
        self.config = get_code_rag_config()
        self.embedder = BaseEmbedder(self.config.get("embedder", {}))
        self.retriever = None
        logger.info("CodeRAG initialized.")

    def _repo_to_documents(self, code: Code, implementation_only: bool = True) -> List[Document]:
        logger.debug(f"Converting Code object to Document objects: {json.dumps(asdict(code), ensure_ascii=False)}")
        documents = []
        max_tokens = MAX_EMBEDDING_TOKENS
        
        try:
            model_name = self.embedder.config["model_kwargs"]["model"]
            encoding = tiktoken.encoding_for_model(model_name)
            special_tokens = sorted(encoding.special_tokens_set, key=len, reverse=True)
        except Exception as e:
            logger.error(f"Could not initialize tiktoken encoding: {e}. Special tokens will not be sanitized.")
            special_tokens = []

        for file in tqdm(code.file_list, desc="Processing files into documents"):
            if implementation_only and not file.is_implementation:
                continue
            
            content = file.content

            processed_content = content
            for token in special_tokens:
                processed_content = processed_content.replace(token, "") 

            token_count = self.embedder.count_tokens(processed_content)

            if token_count > max_tokens:
                logger.warning(f"Skipping large file '{file.name}': Token count ({token_count}) exceeds limit of {max_tokens}.")
                continue
                
            doc = Document(
                text=processed_content,
                meta_data={
                    "file_path": file.name,
                    "is_implementation": file.is_implementation,
                    "type": file.name.split('.')[-1] if '.' in file.name else 'unknown',
                    "title": file.name,
                    "token_count": token_count,
                },
            )
            
            documents.append(doc)
        
        logger.info(f"Converted {len(documents)} files into initial Document objects.")
        return documents
    
    def prepare_retriever(self, code: Code = None, db_path: str = None):
        """
        Prepare the retriever. If a local database exists, load it; otherwise, create a new one.
        """
        # embed_and_save_to_db processes the code base
        if not code:
            logger.debug(f"Code object is empty or None:{json.dumps(asdict(code), ensure_ascii=False)}")
            transformed_docs = self.embedder.embed_and_save_to_db(db_path=db_path)
        else:
            documents = self._repo_to_documents(code)
            transformed_docs = self.embedder.embed_and_save_to_db(documents, db_path)

        if not transformed_docs:
            raise ValueError("No documents available to create a retriever.")
        
        # Get retriever parameters from config
        retriever_config = self.config.get("retriever", {}).get("faiss", {})
        
        logger.info("Creating FAISS retriever...")
        try:
            self.retriever = FAISSRetriever(
                **retriever_config,
                embedder=self.embedder.embedder, 
                documents=transformed_docs,
                document_map_func=lambda doc: doc.vector
            )
            logger.info("FAISS retriever created successfully.")
        except Exception as e:
            logger.error(f"Failed to create FAISS retriever: {e}")
            raise

    def __call__(self, query: str) -> List[FileSnippet]:
        """
        Retrieve relevant code snippets based on the query and organize them into a list of CodeSnippet objects grouped by file.
        """
        if not self.retriever:
            raise RuntimeError("Retriever is not prepared. Call prepare_retriever() first.")
        
        logger.info(f"Retrieving documents for query: '{query}'")
        
        retriever_outputs = self.retriever(query)
        
        if not retriever_outputs:
            logger.warning("Retriever did not return any output.")
            return []
            
        first_output = retriever_outputs[0]
        if hasattr(first_output, 'doc_scores') and first_output.doc_scores is not None:
            logger.debug("--- Retrieved Snippet Scores ---")
            for doc_index, score in zip(first_output.doc_indices, first_output.doc_scores):
                doc = self.embedder.transformed_docs[doc_index]
                file_path = doc.meta_data.get('file_path', 'unknown')
                logger.debug(f"  - File: {file_path}, Score: {score:.4f}")
            logger.debug("---------------------------------")
        else:
            logger.debug("Retriever output does not contain similarity scores (doc_scores).")

        retrieved_docs = [
            self.embedder.transformed_docs[doc_index]
            for doc_index in first_output.doc_indices
        ]
        
        if not retrieved_docs:
            logger.warning("No documents retrieved for the given query.")
            return []

        logger.info(f"Retrieved {len(retrieved_docs)} documents, grouping by file.")

        docs_by_file = {}
        for doc in retrieved_docs:
            file_path = doc.meta_data.get('file_path', 'unknown')
            if file_path not in docs_by_file:
                docs_by_file[file_path] = []
            docs_by_file[file_path].append(doc.text)

        output_snippets: List[FileSnippet] = []
        for file_path, code_fragments in docs_by_file.items():
            snippet = FileSnippet(
                file_name=file_path,
                code_list=code_fragments
            )
            output_snippets.append(snippet)
            
        return output_snippets