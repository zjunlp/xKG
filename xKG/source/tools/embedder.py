import logging 
logging.getLogger("adaflow").setLevel(logging.ERROR)
from typing import List, Optional
from adalflow.core.types import (
    Document,
)
import adalflow as adal
from adalflow.components.model_client import TransformersClient, OpenAIClient
from adalflow.core.db import LocalDB
from adalflow.components.data_process import TextSplitter, ToEmbeddings
import os

import tiktoken
from ..utils import get_code_rag_config

logger = logging.getLogger(__name__)

MAX_EMBEDDING_TOKENS = 8192

CLIENT_CLASS_MAP = {
    "OpenAIClient": OpenAIClient,
    "TransformersClient": TransformersClient,
}

class BaseEmbedder:
    def __init__(self, config) -> None:
        logger.info("Initializing BaseEmbedder...")
        self.config = config
        
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