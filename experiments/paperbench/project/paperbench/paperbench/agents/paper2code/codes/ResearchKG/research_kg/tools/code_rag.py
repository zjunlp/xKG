from adalflow.core.component import DataComponent
from adalflow.core.model_client import ModelClient
from copy import deepcopy
from typing import List, TypeVar, Sequence, Union, Dict, Any, Optional
from tqdm import tqdm
from ..utils import get_code_rag_config
from ..schema import Code, FileSnippet
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
import logging 
import tiktoken
import json
from adalflow.components.retriever.faiss_retriever import FAISSRetriever
logger = logging.getLogger(__name__)
MAX_EMBEDDING_TOKENS = 8192
CLIENT_CLASS_MAP = {
    "OpenAIClient": OpenAIClient,
    "TransformersClient": TransformersClient,
}
class BaseEmbedder:
    def __init__(self) -> None:
        logger.info("Initializing BaseEmbedder...")
        self.config = get_code_rag_config().get("embedder", {})
        
        model_kwargs = self.config.get("model_kwargs", {})
        if not model_kwargs or "model" not in model_kwargs:
            raise ValueError("`model_kwargs` with a `model` key is required in the configuration.")
        
        model_client_name = self.config["model_client"]        
        if model_client_name not in CLIENT_CLASS_MAP:
            raise ValueError(f"Unknown model_client '{model_client_name}'. "
                             f"Available options are: {list(CLIENT_CLASS_MAP.keys())}")
        
        model_client_class = CLIENT_CLASS_MAP[model_client_name]
        if model_client_name == "OpenAIClient":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            model_client = model_client_class(api_key=api_key, base_url=base_url)
        else:
            model_client = model_client_class()
            
        self.embedder = adal.Embedder(
            model_client=model_client,
            model_kwargs=model_kwargs,
        )
        
        self.db: Optional[LocalDB] = None
        self.transformed_docs: Optional[List[Document]] = None
        logger.info("BaseEmbedder initialized successfully.")

    def _validate_and_filter_embeddings(self, documents: List[Document]) -> List[Document]:
        if not documents:
            logger.warning("No documents provided for embedding validation.")
            return []

        valid_documents = []
        embedding_sizes = {}

        for i, doc in enumerate(documents):
            if not hasattr(doc, 'vector') or doc.vector is None:
                logger.warning(f"Document {i} has no embedding vector, skipping.")
                continue
            try:
                if isinstance(doc.vector, list):
                    embedding_size = len(doc.vector)
                elif hasattr(doc.vector, 'shape'):
                    embedding_size = doc.vector.shape[0] if len(doc.vector.shape) == 1 else doc.vector.shape[-1]
                else:
                    logger.warning(f"Document {i} has an invalid embedding vector type: {type(doc.vector)}, skipping.")
                    continue

                if embedding_size > 0:
                    embedding_sizes[embedding_size] = embedding_sizes.get(embedding_size, 0) + 1

            except Exception as e:
                logger.warning(f"Error checking embedding size for document {i}: {e}, skipping.")
                continue

        if not embedding_sizes:
            logger.error("No valid embeddings found in any documents.")
            return []
    
        target_size = max(embedding_sizes, key=embedding_sizes.get)
        total_valid = embedding_sizes[target_size]
        logger.info(f"Target embedding size set to: {target_size} (found in {total_valid} documents).")

        for size, count in embedding_sizes.items():
            if size != target_size:
                logger.warning(f"Found {count} documents with incorrect embedding size {size}, which will be filtered out.")

        for doc in documents:
            if not hasattr(doc, 'vector') or doc.vector is None:
                continue

            try:
                if isinstance(doc.vector, list):
                    embedding_size = len(doc.vector)
                elif hasattr(doc.vector, 'shape'):
                    embedding_size = doc.vector.shape[0] if len(doc.vector.shape) == 1 else doc.vector.shape[-1]
                else:
                    continue
                
                if embedding_size == target_size:
                    valid_documents.append(doc)
            except Exception:
                continue
        
        filtered_count = len(documents) - len(valid_documents)
        if filtered_count > 0:
             logger.warning(f"Filtered out {filtered_count} documents due to embedding issues.")
        logger.info(f"Embedding validation complete: {len(valid_documents)} documents remaining.")

        return valid_documents

    def count_tokens(self, text: str) -> int:
        """
        使用 tiktoken 计算文本的 token 数量。如果失败，则回退到基于字符的估算
        """
        try:
            model_name = self.config["model_kwargs"]["model"]
            encoding = tiktoken.encoding_for_model(model_name)
            processed_text = text
            special_tokens = sorted(
                encoding.special_tokens_set, key=len, reverse=True
            )
            for token in special_tokens:
                processed_text = processed_text.replace(token, "")
            return len(encoding.encode(processed_text, disallowed_special=()))
        except Exception as e:
            logger.warning(f"Tiktoken token counting failed for model '{self.config['model_kwargs']['model']}': {e}. Falling back to character-based estimation.")
            return len(text) // 4
        
    def _prepare_data_pipeline(self) -> adal.Sequential:
        """
        构建数据处理流水线，包括文本切分和嵌入
        """
        splitter_config = self.config.get("text_splitter", {})
        batch_size = self.config.get("batch_size", 32)
        
        splitter = TextSplitter(**splitter_config)
        embedder_transformer = ToEmbeddings(
            embedder=self.embedder, batch_size=batch_size
        )

        # 将切分器和嵌入器串联起来
        data_transformer = adal.Sequential(
            splitter, embedder_transformer
        ) 
        return data_transformer
    
    def embed_and_save_to_db(self, documents: List[Document] = None, db_path: str = None):
        if db_path and os.path.exists(db_path):
            logger.info(f"Loading existing database from '{db_path}'...")
            try:
                # 1. 加载数据库状态
                self.db = LocalDB.load_state(filepath=db_path)
                
                # 2. 尝试获取已转换的文档
                documents = self.db.get_transformed_data(key="split_and_embed")
                
                # 3. 如果成功获取到非空文档列表，则任务完成，直接返回
                if documents:
                    logger.info(f"Loaded {len(documents)} documents from existing database.")
                    self.transformed_docs = documents
                    return self.transformed_docs
                
                logger.warning("Existing database is empty. Will create a new one.")

            except Exception as e:
                logger.error(f"Error loading existing database: {e}. Will create a new one.")

        # 创建新数据库
        if not documents:
            raise ValueError("No documents were generated from the provided code. Cannot create an empty database.")
        
        logger.info(f"Creating new database at '{db_path}'...")
        
        self.db = LocalDB()
        data_pipeline = self._prepare_data_pipeline()
        self.db.register_transformer(transformer=data_pipeline, key="split_and_embed")
        
        self.db.load(documents)
        
        logger.info("Transforming data (splitting and embedding). This may take a while...")
        self.db.transform(key="split_and_embed")

        self.transformed_docs = self.db.get_transformed_data(key="split_and_embed")
        
        if not self.transformed_docs:
            raise RuntimeError("Pipeline execution resulted in no valid documents after processing. Cannot proceed.")
        
        logger.info(f"Saving new database with {len(self.transformed_docs)} documents to '{db_path}'...")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db.save_state(filepath=db_path)
        
        return self.transformed_docs

    
class CodeRAG:
    def __init__(self):
        """
        初始化 CodeRAG
        """
        self.config = get_code_rag_config()
        self.embedder = BaseEmbedder()
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
        准备检索器。如果本地有数据库，则加载；否则，创建新的。
        """
        # embed_and_save_to_db 处理code base
        if not code:
            logger.debug(f"Code object is empty or None:{json.dumps(asdict(code), ensure_ascii=False)}")
            transformed_docs = self.embedder.embed_and_save_to_db(db_path=db_path)
        else:
            documents = self._repo_to_documents(code)
            transformed_docs = self.embedder.embed_and_save_to_db(documents, db_path)

        if not transformed_docs:
            raise ValueError("No documents available to create a retriever.")
        
        # 从配置中获取 retriever 参数
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
        根据查询检索相关的代码片段，并将其组织成按文件分类的 CodeSnippet 对象列表
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