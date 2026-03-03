import json
import logging
from typing import List, Tuple, Dict, Union
from pathlib import Path
import numpy as np
from src.reranking import LLMReranker
import hashlib
import pandas as pd
import time

# 尝试导入Weaviate客户端
try:
    import weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    print("Weaviate客户端未安装。请运行: pip install weaviate-client")


_log = logging.getLogger(__name__)


class WeaviateRetriever:
    def __init__(self, url: str = "http://localhost:8080", api_key: str = None, class_name: str = "DocumentChunk"):
        """
        初始化Weaviate检索器
        :param url: Weaviate实例的URL
        :param api_key: 认证密钥（如需要）
        :param class_name: Weaviate类名
        """
        if not WEAVIATE_AVAILABLE:
            raise ImportError("Weaviate客户端未安装。请运行: pip install weaviate-client")
        
        # 初始化Weaviate客户端
        auth_config = None
        if api_key:
            auth_config = weaviate.auth.AuthApiKey(api_key)
        
        self.client = weaviate.Client(
            url=url,
            auth_client_secret=auth_config
        )
        
        # 测试连接
        try:
            self.client.is_ready()
            print(f"Weaviate检索器连接成功: {url}")
        except Exception as e:
            print(f"Weaviate检索器连接失败: {e}")
            raise
        
        self.class_name = class_name

    def retrieve_by_company_name(
        self, 
        company_name: str, 
        query: str, 
        llm_reranking_sample_size: int = None, 
        top_n: int = 3, 
        return_parent_pages: bool = False
    ) -> List[Dict]:
        """
        按公司名检索相关文本块，返回向量距离最近的top_n个块
        """
        # 获取查询的嵌入向量
        query_embedding = self._get_query_embedding(query)
        
        # 构建Weaviate查询
        near_vector = {
            "vector": query_embedding,
            "distance": 0.2  # 允许的最大距离，越小越相关
        }
        
        # 创建GraphQL查询
        result = (
            self.client.query
            .get(self.class_name, ["text", "page", "sha1", "company_name", "file_name", "chunk_id"])
            .with_near_vector(near_vector)
            .with_where({
                "path": ["company_name"],
                "operator": "Equal",
                "valueString": company_name
            })
            .with_limit(top_n)
            .do()
        )
        
        # 处理结果
        objects = result.get("data", {}).get("Get", {}).get(self.class_name, [])
        
        retrieval_results = []
        seen_pages = set()
        
        for obj in objects:
            properties = obj.get("properties", {})
            distance = obj.get("_additional", {}).get("distance", 0.5)
            
            result_item = {
                "distance": round(1.0 - float(distance), 4),  # Weaviate返回的是距离，转换为相似度
                "page": properties.get("page", 0),
                "text": properties.get("text", ""),
                "sha1": properties.get("sha1", ""),
                "company_name": properties.get("company_name", ""),
                "file_name": properties.get("file_name", "")
            }
            
            if return_parent_pages:
                if properties.get("page") not in seen_pages:
                    seen_pages.add(properties.get("page"))
                    retrieval_results.append(result_item)
            else:
                retrieval_results.append(result_item)
        
        return retrieval_results

    def retrieve_by_sha1(
        self, 
        sha1: str, 
        query: str, 
        top_n: int = 3
    ) -> List[Dict]:
        """
        按SHA1检索相关文本块
        """
        # 获取查询的嵌入向量
        query_embedding = self._get_query_embedding(query)
        
        # 构建GraphQL查询
        result = (
            self.client.query
            .get(self.class_name, ["text", "page", "company_name", "file_name", "chunk_id"])
            .with_near_vector({
                "vector": query_embedding,
                "distance": 0.2
            })
            .with_where({
                "path": ["sha1"],
                "operator": "Equal",
                "valueString": sha1
            })
            .with_limit(top_n)
            .do()
        )
        
        # 处理结果
        objects = result.get("data", {}).get("Get", {}).get(self.class_name, [])
        
        retrieval_results = []
        for obj in objects:
            properties = obj.get("properties", {})
            distance = obj.get("_additional", {}).get("distance", 0.5)
            
            result_item = {
                "distance": round(1.0 - float(distance), 4),
                "page": properties.get("page", 0),
                "text": properties.get("text", ""),
                "sha1": sha1,
                "company_name": properties.get("company_name", ""),
                "file_name": properties.get("file_name", "")
            }
            
            retrieval_results.append(result_item)
        
        return retrieval_results

    def retrieve_all_by_company(self, company_name: str) -> List[Dict]:
        """
        检索公司相关的所有文档
        """
        result = (
            self.client.query
            .get(self.class_name, ["text", "page", "sha1", "company_name", "file_name", "chunk_id"])
            .with_where({
                "path": ["company_name"],
                "operator": "Equal",
                "valueString": company_name
            })
            .do()
        )
        
        objects = result.get("data", {}).get("Get", {}).get(self.class_name, [])
        
        all_docs = []
        for obj in objects:
            properties = obj.get("properties", {})
            result_item = {
                "distance": 0.5,  # 默认距离
                "page": properties.get("page", 0),
                "text": properties.get("text", ""),
                "sha1": properties.get("sha1", ""),
                "company_name": properties.get("company_name", ""),
                "file_name": properties.get("file_name", "")
            }
            all_docs.append(result_item)
        
        return all_docs

    def hybrid_search(
        self, 
        query: str, 
        company_name: str = None,
        alpha: float = 0.7,  # 0=纯关键词搜索，1=纯向量搜索
        top_n: int = 3
    ) -> List[Dict]:
        """
        执行混合搜索（向量+关键词）
        :param query: 查询字符串
        :param company_name: 公司名称筛选条件
        :param alpha: 混合权重，0表示完全基于关键词，1表示完全基于向量
        :param top_n: 返回结果数量
        """
        # Weaviate的hybrid search支持关键词和向量的结合
        hybrid_query = (
            self.client.query
            .get(self.class_name, ["text", "page", "sha1", "company_name", "file_name"])
            .with_hybrid(
                query=query,
                alpha=alpha
            )
        )
        
        # 如果指定了公司名，添加过滤条件
        if company_name:
            hybrid_query = hybrid_query.with_where({
                "path": ["company_name"],
                "operator": "Equal",
                "valueString": company_name
            })
        
        hybrid_query = hybrid_query.with_limit(top_n).do()
        
        objects = hybrid_query.get("data", {}).get("Get", {}).get(self.class_name, [])
        
        retrieval_results = []
        for obj in objects:
            properties = obj.get("properties", {})
            score = obj.get("_additional", {}).get("score", 0.5)
            
            result_item = {
                "distance": round(float(score), 4),
                "page": properties.get("page", 0),
                "text": properties.get("text", ""),
                "sha1": properties.get("sha1", ""),
                "company_name": properties.get("company_name", ""),
                "file_name": properties.get("file_name", "")
            }
            
            retrieval_results.append(result_item)
        
        return retrieval_results

    def keyword_search(
        self, 
        query: str, 
        company_name: str = None, 
        top_n: int = 3
    ) -> List[Dict]:
        """
        基于关键词的全文搜索
        """
        # 使用hybrid search但将alpha设为0，仅进行关键词搜索
        hybrid_query = (
            self.client.query
            .get(self.class_name, ["text", "page", "sha1", "company_name", "file_name"])
            .with_hybrid(
                query=query,
                alpha=0  # 仅关键词搜索
            )
        )
        
        # 如果指定了公司名，添加过滤条件
        if company_name:
            hybrid_query = hybrid_query.with_where({
                "path": ["company_name"],
                "operator": "Equal",
                "valueString": company_name
            })
        
        hybrid_query = hybrid_query.with_limit(top_n).do()
        
        objects = hybrid_query.get("data", {}).get("Get", {}).get(self.class_name, [])
        
        retrieval_results = []
        for obj in objects:
            properties = obj.get("properties", {})
            score = obj.get("_additional", {}).get("score", 0.5)
            
            result_item = {
                "distance": round(float(score), 4),
                "page": properties.get("page", 0),
                "text": properties.get("text", ""),
                "sha1": properties.get("sha1", ""),
                "company_name": properties.get("company_name", ""),
                "file_name": properties.get("file_name", "")
            }
            
            retrieval_results.append(result_item)
        
        return retrieval_results

    def _get_query_embedding(self, text: str):
        """
        获取查询文本的嵌入向量
        注意：这里仍使用与原项目相同的嵌入方式以保持一致性
        """
        # 这里复用了原项目中的嵌入获取方式
        import os
        import dashscope
        from dashscope import TextEmbedding
        
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
        
        try:
            rsp = TextEmbedding.call(
                model="text-embedding-v1",
                input=[text]
            )
            
            if 'output' in rsp and 'embeddings' in rsp['output']:
                emb = rsp['output']['embeddings'][0]
                if emb['embedding'] is None or len(emb['embedding']) == 0:
                    raise RuntimeError(f"DashScope返回的embedding为空")
                return emb['embedding']
            elif 'output' in rsp and 'embedding' in rsp['output']:
                if rsp['output']['embedding'] is None or len(rsp['output']['embedding']) == 0:
                    raise RuntimeError("DashScope返回的embedding为空")
                return rsp['output']['embedding']
            else:
                raise RuntimeError(f"DashScope embedding API返回格式异常: {rsp}")
        except Exception as e:
            print(f"获取查询嵌入向量失败: {e}")
            raise

    def close(self):
        """
        关闭Weaviate客户端连接
        """
        # Weaviate客户端通常不需要显式关闭
        print("Weaviate检索器连接已断开")


class WeaviateHybridRetriever:
    """
    结合Weaviate向量检索和LLM重排的混合检索器
    """
    def __init__(self, url: str = "http://localhost:8080", api_key: str = None, class_name: str = "DocumentChunk"):
        self.weaviate_retriever = WeaviateRetriever(url, api_key, class_name)
        self.reranker = LLMReranker()
        
    def retrieve_by_company_name(
        self, 
        company_name: str, 
        query: str, 
        llm_reranking_sample_size: int = 28,
        documents_batch_size: int = 10,
        top_n: int = 6,
        llm_weight: float = 0.7,
        return_parent_pages: bool = False
    ) -> List[Dict]:
        """
        使用Weaviate检索并用LLM重排结果
        """
        t0 = time.time()
        
        # 首先用Weaviate获取初步结果
        print("[计时] [WeaviateHybridRetriever] 开始Weaviate检索 ...")
        vector_results = self.weaviate_retriever.retrieve_by_company_name(
            company_name=company_name,
            query=query,
            top_n=llm_reranking_sample_size,
            return_parent_pages=return_parent_pages
        )
        
        t1 = time.time()
        print(f"[计时] [WeaviateHybridRetriever] Weaviate检索耗时: {t1-t0:.2f} 秒")
        
        # 如果结果不足，尝试使用混合搜索
        if len(vector_results) < llm_reranking_sample_size:
            print("[WeaviateHybridRetriever] Weaviate检索结果不足，尝试混合搜索 ...")
            additional_results = self.weaviate_retriever.hybrid_search(
                query=query,
                company_name=company_name,
                top_n=llm_reranking_sample_size - len(vector_results)
            )
            vector_results.extend(additional_results)
        
        # 使用LLM对结果进行重排
        print("[计时] [WeaviateHybridRetriever] 开始LLM重排 ...")
        reranked_results = self.reranker.rerank_documents(
            query=query,
            documents=vector_results,
            documents_batch_size=documents_batch_size,
            llm_weight=llm_weight
        )
        
        t2 = time.time()
        print(f"[计时] [WeaviateHybridRetriever] LLM重排耗时: {t2-t1:.2f} 秒")
        print(f"[计时] [WeaviateHybridRetriever] 总耗时: {t2-t0:.2f} 秒")
        
        return reranked_results[:top_n]

    def close(self):
        """
        关闭内部组件连接
        """
        self.weaviate_retriever.close()


# 使用示例
def example_usage():
    """
    WeaviateRetriever使用示例
    """
    try:
        # 初始化Weaviate检索器
        retriever = WeaviateRetriever(url="http://localhost:8080")
        
        # 示例检索
        # results = retriever.retrieve_by_company_name(
        #     company_name="中芯国际",
        #     query="公司的营业收入是多少？",
        #     top_n=3
        # )
        #
        # for result in results:
        #     print(f"页面: {result['page']}, 相似度: {result['distance']}")
        #     print(f"文本: {result['text'][:200]}...")
        #     print("-" * 50)
        
    except Exception as e:
        print(f"Weaviate检索器初始化失败: {e}")
        print("\n要使用Weaviate，请确保:")
        print("1. 安装客户端: pip install weaviate-client")
        print("2. 启动Weaviate服务")
        print("3. 确保服务在指定URL上运行")


if __name__ == "__main__":
    example_usage()