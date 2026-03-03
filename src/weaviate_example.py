"""
Weaviate 集成使用说明

此文件演示了如何在 RAG-cy 项目中使用 Weaviate 向量数据库。
Weaviate 提供了更好的 CRUD 操作支持，解决了原 FAISS 方案在数据更新方面的局限性。
"""

from pathlib import Path
from src.pipeline import Pipeline, RunConfig


def setup_weaviate_example():
    """
    配置使用 Weaviate 数据库的管道示例
    """
    # 设置数据集根目录
    root_path = Path("data/stock_data")
    
    # 创建支持 Weaviate 的运行配置
    weaviate_config = RunConfig(
        use_weaviate_db=True,  # 启用 Weaviate
        weaviate_url="http://localhost:8080",  # Weaviate 服务地址
        weaviate_api_key=None,  # 如果需要认证，则填写 API 密钥
        weaviate_class_name="DocumentChunk",  # Weaviate 类名
        
        # 其他配置保持不变
        parent_document_retrieval=True,
        llm_reranking=True,
        parallel_requests=4,
        submission_file=True,
        pipeline_details="Custom pdf parsing + Weaviate vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = qwen-turbo",
        answering_model="qwen-turbo-latest",
        config_suffix="_weaviate"
    )
    
    # 初始化使用 Weaviate 的管道
    pipeline = Pipeline(root_path, run_config=weaviate_config)
    
    return pipeline


def run_full_pipeline_with_weaviate():
    """
    运行完整流程（使用 Weaviate）
    """
    pipeline = setup_weaviate_example()
    
    print('1. 将PDF转化为结构化数据')
    # pipeline.export_reports_to_markdown('【财报】中芯国际：中芯国际2024年年度报告.pdf') 
    
    print('2. 将规整后报告分块，便于后续向量化，输出到 databases/chunked_reports')
    pipeline.chunk_reports() 
    
    print('3. 从分块报告创建 Weaviate 向量数据库')
    weaviate_ingestor = pipeline.create_weaviate_db()  # 现在创建的是 Weaviate 数据库
    
    if weaviate_ingestor:
        print('4. 处理问题并生成答案（使用 Weaviate 检索）')
        pipeline.process_questions() 
        
        print('清理资源')
        weaviate_ingestor.close()
        
    print('完成')


def update_document_example():
    """
    演示如何更新文档（这是 Weaviate 相对于 FAISS 的优势）
    """
    from src.weaviate_ingestion import WeaviateDBIngestor
    from src.weaviate_retrieval import WeaviateRetriever
    
    # 初始化 Weaviate 客户端
    ingestor = WeaviateDBIngestor(
        url="http://localhost:8080",
        api_key=None
    )
    
    # 更新特定文档的示例
    # ingestor.update_document(
    #     sha1="some-document-sha1", 
    #     new_chunks=[{
    #         "text": "updated text content...",
    #         "page": 1,
    #         "company_name": "公司名称",
    #         "file_name": "文件名",
    #         "length_tokens": 100
    #     }]
    # )
    
    # 或者删除文档
    # ingestor.delete_document(sha1="some-document-sha1")
    
    return ingestor


if __name__ == "__main__":
    print("Weaviate 集成示例")
    print("="*50)
    
    # 展示如何启动 Weaviate 服务
    print("要使用 Weaviate，请先启动 Weaviate 服务：")
    print("方法1：使用 Docker（推荐）")
    print("""docker run -d \\
  --name weaviate \\
  -p 8080:8080 \\
  -e QUERY_DEFAULTS_LIMIT=25 \\
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \\
  -e DEFAULT_VECTORIZER_MODULE='none' \\
  -e CLUSTER_HOSTNAME=hostname \\
  weaviate/weaviate:latest""")
    
    print("\n方法2：使用 pip 安装客户端")
    print("pip install weaviate-client")
    
    print("\n完成设置后，可以运行完整流程：")
    print("# run_full_pipeline_with_weaviate()")