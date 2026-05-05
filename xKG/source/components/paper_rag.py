import logging

import tiktoken
logging.getLogger("adaflow").setLevel(logging.ERROR)
from typing import List
from dataclasses import asdict
import json
from adalflow.core.types import Document
from adalflow.components.retriever.faiss_retriever import FAISSRetriever
from .embedder import BaseEmbedder
from ..schema import Paper, Section 
from ..utils import get_paper_rag_config

logger = logging.getLogger(__name__)

class PaperRAG:
    """
    一个用于处理学术论文（Paper对象）的、配置驱动的RAG实现。
    """
    def __init__(self):
        """
        初始化 PaperRAG。
        - 加载 'paper' 专属配置。
        - 将 'embedder' 子配置传递给 BaseEmbedder。
        """
        self.config = get_paper_rag_config()
        
        # 将 embedder 配置注入 BaseEmbedder
        embedder_config = self.config.get("embedder", {})
        if not embedder_config:
            raise ValueError("Embedder configuration is missing in 'paper' config.")
        self.embedder = BaseEmbedder(embedder_config)
        
        self.retriever = None
        logger.info("PaperRAG initialized.")

    def _paper_to_documents(self, paper: Paper) -> List[Document]:
        """
        将 Paper 对象的 sections 转换为 Document 列表，并清理特殊 tokens。
        """
        if not paper.sections:
            logger.warning(f"Paper '{paper.title}' has no sections. No documents will be created.")
            return []

        try:
            model_name = self.embedder.config["model_kwargs"]["model"]
            encoding = tiktoken.encoding_for_model(model_name)
            special_tokens = sorted(encoding.special_tokens_set, key=len, reverse=True)
        except (AttributeError, KeyError, Exception) as e:
            logger.error(f"Could not initialize tiktoken encoding: {e}. Special tokens will not be sanitized.")
            special_tokens = []
        
        documents = []
        for section in paper.sections:
            if not section:
                continue
            
            if isinstance(section, Section):
                raw_doc_text = json.dumps(asdict(section), ensure_ascii=False)
            elif isinstance(section, dict):
                raw_doc_text = json.dumps(section, ensure_ascii=False)
            elif isinstance(section, str):
                raw_doc_text = section
            else:
                logger.warning(f"Unexpected section type in paper '{paper.title}': {type(section)}. Skipping this section.")
                continue
            
            processed_doc_text = raw_doc_text
            for token in special_tokens:
                processed_doc_text = processed_doc_text.replace(token, "")

            meta_data = {"paper_title": paper.title}
            if paper.url:
                meta_data["paper_url"] = paper.url
            if paper.year and paper.year > 0:
                meta_data["paper_year"] = paper.year
            
            doc = Document(
                text=processed_doc_text,
                meta_data=meta_data,
            )
            documents.append(doc)
        
        logger.info(f"Converted {len(documents)} sections from paper '{paper.title}' into Document objects.")
        return documents
    
    def prepare_retriever(self, paper: Paper, db_path: str = None):
        """
        准备检索器。如果本地有数据库，则加载；否则，基于论文内容创建新的。
        """
        documents = self._paper_to_documents(paper)
        if not documents:
            raise ValueError(f"No valid documents could be generated from the paper '{paper.title}'.")

        transformed_docs = self.embedder.embed_and_save_to_db(documents, db_path)

        if not transformed_docs:
            raise ValueError("No documents available to create a retriever after embedding process.")
        
        retriever_config = self.config.get("retriever", {}).get("faiss", {})
        if not retriever_config:
            logger.warning("FAISS retriever configuration not found. Using default settings.")

        logger.info("Creating FAISS retriever from configuration...")
        try:
            self.retriever = FAISSRetriever(
                **retriever_config, # 使用配置动态设置 top_k 等参数
                embedder=self.embedder.embedder, 
                documents=transformed_docs,
                document_map_func=lambda doc: doc.vector
            )
            logger.info(f"FAISS retriever created successfully with config: {retriever_config}")
        except Exception as e:
            logger.error(f"Failed to create FAISS retriever: {e}")
            raise

    def __call__(self, query: str) -> List[str]:
        """
        根据查询检索相关的论文片段，并返回文本列表。
        """
        if not self.retriever:
            raise RuntimeError("Retriever is not prepared. Call prepare_retriever() first.")
        
        logger.info(f"Retrieving document snippets for query: '{query}'")
        retriever_outputs = self.retriever(query)
        
        if not retriever_outputs:
            logger.warning("Retriever did not return any output.")
            return []
            
        first_output = retriever_outputs[0]
        
        retrieved_docs = [
            self.embedder.transformed_docs[doc_index]
            for doc_index in first_output.doc_indices
        ]
        
        if not retrieved_docs:
            logger.warning("No documents retrieved for the given query.")
            return []
        
        return [doc.text for doc in retrieved_docs]

