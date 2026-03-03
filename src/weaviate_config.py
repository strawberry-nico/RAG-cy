# Weaviate 配置示例

# Weaviate 服务配置
WEAVIATE_CONFIG = {
    "url": "http://localhost:8080",  # Weaviate 服务地址
    "api_key": None,  # 如需要认证则填写
    "class_name": "DocumentChunk"  # Weaviate 类名
}

# 启动 Weaviate 的 Docker 命令示例
DOCKER_START_COMMAND = """
docker run -d \\
  --name weaviate \\
  -p 8080:8080 \\
  -e QUERY_DEFAULTS_LIMIT=25 \\
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \\
  -e DEFAULT_VECTORIZER_MODULE=text25-transformers \\
  -e TRANSFORMERS_INFERENCE_API=http://t2v-transformers:8080 \\
  -e CLUSTER_HOSTNAME=hostname \\
  weaviate/weaviate:latest
"""

# 或者使用免向量化模块的版本（使用外部嵌入）
DOCKER_START_COMMAND_WITHOUT_VECTORIZER = """
docker run -d \\
  --name weaviate \\
  -p 8080:8080 \\
  -e QUERY_DEFAULTS_LIMIT=25 \\
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \\
  -e DEFAULT_VECTORIZER_MODULE='none' \\
  -e CLUSTER_HOSTNAME=hostname \\
  weaviate/weaviate:latest
"""