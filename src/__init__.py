"""
RAG-cy 项目模块入口

此项目实现了基于检索增强生成(RAG)的企业知识库系统，
支持多种向量数据库(Faiss, Weaviate)和检索策略。
"""

# 主要模块
from .pipeline import Pipeline, RunConfig
from .text_splitter import TextSplitter
from .ingestion import VectorDBIngestor, BM25Ingestor
from .weaviate_ingestion import WeaviateDBIngestor
from .retrieval import VectorRetriever, BM25Retriever, HybridRetriever
from .weaviate_retrieval import WeaviateRetriever, WeaviateHybridRetriever
from .questions_processing import QuestionsProcessor
from .pdf_parsing import PDFParser
from . import pdf_mineru
from .parsed_reports_merging import PageTextPreparation
from .tables_serialization import TableSerializer

__version__ = "1.0.0"
__author__ = "RAG-cy Team"

__all__ = [
    # Pipeline
    'Pipeline', 'RunConfig',
    
    # Text Processing
    'TextSplitter',
    
    # Ingestion
    'VectorDBIngestor', 'BM25Ingestor', 'WeaviateDBIngestor',
    
    # Retrieval
    'VectorRetriever', 'BM25Retriever', 'HybridRetriever',
    'WeaviateRetriever', 'WeaviateHybridRetriever',
    
    # Question Answering
    'QuestionsProcessor',
    
    # PDF Processing
    'PDFParser', 'pdf_mineru', 'PageTextPreparation',
    
    # Others
    'TableSerializer'
]