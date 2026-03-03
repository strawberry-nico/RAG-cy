import os
import json
from typing import List, Union
from pathlib import Path
from tqdm import tqdm
import hashlib

from dotenv import load_dotenv
import dashscope
from dashscope import TextEmbedding
import uuid

# 尝试导入Weaviate客户端，如果未安装则提供指导
try:
    import weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    print("Weaviate客户端未安装。请运行: pip install weaviate-client")


class WeaviateDBIngestor:
    def __init__(self, url: str = "http://localhost:8080", api_key: str = None):
        """
        初始化Weaviate数据库连接
        :param url: Weaviate实例的URL，默认为本地实例
        :param api_key: 认证密钥（如需要）
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
            print(f"Weaviate数据库连接成功: {url}")
        except Exception as e:
            print(f"Weaviate数据库连接失败: {e}")
            raise

    def _create_schema(self, class_name: str):
        """
        创建Weaviate类模式，定义数据结构
        """
        schema = {
            "class": class_name,
            "description": "Document chunks from reports",
            "properties": [
                {
                    "name": "text",
                    "dataType": ["text"],
                    "description": "The content of the document chunk"
                },
                {
                    "name": "page",
                    "dataType": ["int"],
                    "description": "Page number in original document"
                },
                {
                    "name": "sha1",
                    "dataType": ["string"],
                    "description": "SHA1 hash of the original document"
                },
                {
                    "name": "company_name",
                    "dataType": ["string"],
                    "description": "Company name associated with the document"
                },
                {
                    "name": "file_name",
                    "dataType": ["string"],
                    "description": "Original file name"
                }
            ]
        }
        
        # 检查类是否存在
        if self.client.schema.exists(class_name):
            print(f"Weaviate类 {class_name} 已存在")
            return
        
        # 创建新类
        self.client.schema.create_class(schema)
        print(f"Weaviate类 {class_name} 创建成功")

    def _get_embeddings(self, text: Union[str, List[str]], model: str = "text-embedding-v1") -> List[float]:
        """
        获取文本或文本块的嵌入向量（使用阿里云DashScope）
        """
        if isinstance(text, str) and not text.strip():
            raise ValueError("Input text cannot be an empty string.")
        
        # 保证 input 为一维字符串列表或单个字符串
        if isinstance(text, list):
            text_chunks = text
        else:
            text_chunks = [text]

        # 类型检查，确保每一项都是字符串
        if not all(isinstance(x, str) for x in text_chunks):
            raise ValueError(f"所有待嵌入文本必须为字符串类型！实际类型: {[type(x) for x in text_chunks]}")

        # 过滤空字符串
        text_chunks = [x for x in text_chunks if x.strip()]
        if not text_chunks:
            raise ValueError("所有待嵌入文本均为空字符串！")
        
        embeddings = []
        MAX_BATCH_SIZE = 25
        LOG_FILE = 'embedding_error.log'
        
        for i in range(0, len(text_chunks), MAX_BATCH_SIZE):
            batch = text_chunks[i:i+MAX_BATCH_SIZE]
            resp = TextEmbedding.call(
                model=TextEmbedding.Models.text_embedding_v1,
                input=batch
            )
            
            # 兼容单条和多条输入
            if 'output' in resp and 'embeddings' in resp['output']:
                for emb in resp['output']['embeddings']:
                    if emb['embedding'] is None or len(emb['embedding']) == 0:
                        error_text = batch[emb.get('text_index', 0)] if emb.get('text_index') is not None else None
                        with open(LOG_FILE, 'a', encoding='utf-8') as f:
                            f.write(f"DashScope返回的embedding为空，text_index={emb.get('text_index', None)}，文本内容如下：\n{error_text}\n{'-'*60}\n")
                        raise RuntimeError(f"DashScope返回的embedding为空，text_index={emb.get('text_index', None)}，文本内容已写入 {LOG_FILE}")
                    embeddings.append(emb['embedding'])
            elif 'output' in resp and 'embedding' in resp['output']:
                if resp['output']['embedding'] is None or len(resp['output']['embedding']) == 0:
                    with open(LOG_FILE, 'a', encoding='utf-8') as f:
                        f.write(f"DashScope返回的embedding为空，文本内容如下：\n{batch[0] if batch else None}\n{'-'*60}\n")
                    raise RuntimeError(f"DashScope返回的embedding为空，文本内容已写入 {LOG_FILE}")
                embeddings.append(resp.output.embedding)
            else:
                raise RuntimeError(f"DashScope embedding API返回格式异常: {resp}")
                
        return embeddings

    def process_reports(self, all_reports_dir: Path, class_name: str = "DocumentChunk"):
        """
        批量处理所有报告，将向量数据上传到Weaviate数据库
        :param all_reports_dir: 存放JSON报告的目录
        :param class_name: Weaviate类名
        """
        # 创建类模式
        self._create_schema(class_name)
        
        all_report_paths = list(all_reports_dir.glob("*.json"))
        processed_count = 0
        
        for report_path in tqdm(all_report_paths, desc="Processing reports for Weaviate"):
            # 加载报告
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            # 获取文档元信息
            metainfo = report_data.get("metainfo", {})
            sha1 = metainfo.get("sha1", "")
            company_name = metainfo.get("company_name", "")
            file_name = metainfo.get("file_name", "")
            
            # 处理文档中的每个chunk
            chunks = report_data.get("content", {}).get("chunks", [])
            
            for idx, chunk in enumerate(chunks):
                # 准备数据对象
                data_object = {
                    "text": chunk.get("text", ""),
                    "page": chunk.get("page", 0),
                    "sha1": sha1,
                    "company_name": company_name,
                    "file_name": file_name,
                    "chunk_id": idx,
                    "length_tokens": chunk.get("length_tokens", 0)
                }
                
                # 获取嵌入向量
                try:
                    embedding = self._get_embeddings(data_object["text"])[0]
                except Exception as e:
                    print(f"获取嵌入向量失败，跳过该chunk: {e}")
                    continue
                
                # 生成唯一标识符
                object_uuid = str(uuid.uuid5(
                    uuid.NAMESPACE_DNS, 
                    f"{sha1}_{idx}_{chunk.get('text', '')[:50]}"
                ))
                
                # 添加到Weaviate
                try:
                    self.client.data_object.create(
                        data_object=data_object,
                        class_name=class_name,
                        vector=embedding,
                        uuid=object_uuid
                    )
                except Exception as e:
                    print(f"添加对象到Weaviate失败: {e}")
                    continue
            
            processed_count += 1
        
        print(f"成功处理 {processed_count} 个报告，数据已上传到Weaviate数据库")
    
    def update_document(self, sha1: str, new_chunks: List[dict], class_name: str = "DocumentChunk"):
        """
        更新特定文档的向量数据
        :param sha1: 文档的SHA1标识符
        :param new_chunks: 新的文本块列表
        :param class_name: Weaviate类名
        """
        # 首先删除现有文档的所有chunks
        where_filter = {
            "path": ["sha1"],
            "operator": "Equal",
            "valueString": sha1
        }
        
        try:
            # 删除现有对象
            self.client.batch.delete_objects(
                class_name=class_name,
                where=where_filter
            )
            print(f"已删除文档 {sha1} 的所有向量数据")
        except Exception as e:
            print(f"删除现有对象时出错: {e}")
        
        # 添加新数据
        for idx, chunk in enumerate(new_chunks):
            data_object = {
                "text": chunk.get("text", ""),
                "page": chunk.get("page", 0),
                "sha1": sha1,
                "company_name": chunk.get("company_name", ""),
                "file_name": chunk.get("file_name", ""),
                "chunk_id": idx,
                "length_tokens": chunk.get("length_tokens", 0)
            }
            
            # 获取嵌入向量
            try:
                embedding = self._get_embeddings(data_object["text"])[0]
            except Exception as e:
                print(f"获取嵌入向量失败，跳过该chunk: {e}")
                continue
            
            # 生成唯一标识符
            object_uuid = str(uuid.uuid5(
                uuid.NAMESPACE_DNS, 
                f"{sha1}_{idx}_{chunk.get('text', '')[:50]}"
            ))
            
            # 添加到Weaviate
            try:
                self.client.data_object.create(
                    data_object=data_object,
                    class_name=class_name,
                    vector=embedding,
                    uuid=object_uuid
                )
            except Exception as e:
                print(f"添加对象到Weaviate失败: {e}")
                continue
        
        print(f"文档 {sha1} 已更新")
    
    def delete_document(self, sha1: str, class_name: str = "DocumentChunk"):
        """
        删除特定文档的所有向量数据
        :param sha1: 文档的SHA1标识符
        :param class_name: Weaviate类名
        """
        where_filter = {
            "path": ["sha1"],
            "operator": "Equal",
            "valueString": sha1
        }
        
        try:
            result = self.client.batch.delete_objects(
                class_name=class_name,
                where=where_filter
            )
            print(f"已删除文档 {sha1} 的所有向量数据，删除数量: {result.get('results', {}).get('matches', 0)}")
        except Exception as e:
            print(f"删除对象时出错: {e}")
    
    def close(self):
        """
        关闭Weaviate客户端连接
        """
        # Weaviate客户端通常不需要显式关闭
        print("Weaviate客户端连接已断开")


# 使用示例函数
def example_usage():
    """
    WeaviateDBIngestor使用示例
    """
    # 注意：这仅作演示，实际使用时需要先启动Weaviate服务
    # Weaviate可以通过Docker轻松运行: 
    # docker run -d --name weaviate -p 8080:8080 -e QUERY_DEFAULTS_LIMIT=25 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true -e DEFAULT_VECTORIZER_MODULE=text2vec-transformers -e TRANSFORMERS_INFERENCE_API=http://t2v-transformers:8080 weaviate/weaviate:1.16.3
    
    try:
        # 初始化Weaviate数据库
        ingestor = WeaviateDBIngestor(url="http://localhost:8080")
        
        # 处理报告
        # ingestor.process_reports(Path("./path/to/reports"), "DocumentChunk")
        
    except Exception as e:
        print(f"Weaviate初始化失败: {e}")
        print("\n要使用Weaviate，请按以下步骤操作:")
        print("1. 安装客户端: pip install weaviate-client")
        print("2. 启动Weaviate服务（如使用Docker）:")
        print("   docker run -d --name weaviate -p 8080:8080 \\")
        print("   -e QUERY_DEFAULTS_LIMIT=25 \\")
        print("   -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \\")
        print("   -e DEFAULT_VECTORIZER_MODULE=text2vec-transformers \\")
        print("   -e TRANSFORMERS_INFERENCE_API=http://t2v-transformers:8080 \\")
        print("   weaviate/weaviate:1.16.3")


if __name__ == "__main__":
    example_usage()