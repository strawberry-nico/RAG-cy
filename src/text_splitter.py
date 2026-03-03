import json
import tiktoken
from pathlib import Path
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import os

# 文本分块工具类，支持按页分块、表格插入、token统计等
class TextSplitter():
    def _get_serialized_tables_by_page(self, tables: List[Dict]) -> Dict[int, List[Dict]]:
        """按页分组已序列化表格，便于后续插入到对应页面分块中"""
        tables_by_page = {}
        for table in tables:
            if 'serialized' not in table:
                continue
                
            page = table['page']
            if page not in tables_by_page:
                tables_by_page[page] = []
            
            table_text = "\n".join(
                block["information_block"] 
                for block in table["serialized"]["information_blocks"]
            )
            
            tables_by_page[page].append({
                "page": page,
                "text": table_text,
                "table_id": table["table_id"],
                "length_tokens": self.count_tokens(table_text)
            })
            
        return tables_by_page

    def _split_report(self, file_content: Dict[str, any], serialized_tables_report_path: Optional[Path] = None) -> Dict[str, any]:
        """将报告按页分块，保留markdown表格内容，可选插入序列化表格块。"""
        chunks = []
        chunk_id = 0
        
        tables_by_page = {}
        if serialized_tables_report_path is not None:
            # 加载序列化表格，按页分组
            with open(serialized_tables_report_path, 'r', encoding='utf-8') as f:
                parsed_report = json.load(f)
            tables_by_page = self._get_serialized_tables_by_page(parsed_report.get('tables', []))
        
        for page in file_content['content']['pages']:
            # 普通文本分块
            page_chunks = self._split_page(page)
            for chunk in page_chunks:
                chunk['id'] = chunk_id
                chunk['type'] = 'content'
                chunk_id += 1
                chunks.append(chunk)
            
            # 插入序列化表格分块
            if tables_by_page and page['page'] in tables_by_page:
                for table in tables_by_page[page['page']]:
                    table['id'] = chunk_id
                    table['type'] = 'serialized_table'
                    chunk_id += 1
                    chunks.append(table)
        
        file_content['content']['chunks'] = chunks
        return file_content

    def count_tokens(self, string: str, encoding_name="o200k_base"):
        # 统计字符串的token数，支持自定义编码
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(string)
        token_count = len(tokens)
        return token_count

    def _split_page(self, page: Dict[str, any], chunk_size: int = 300, chunk_overlap: int = 50) -> List[Dict[str, any]]:
        """将单页文本分块，保留原始markdown表格。"""
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(page['text'])
        chunks_with_meta = []
        for chunk in chunks:
            chunks_with_meta.append({
                "page": page['page'],
                "length_tokens": self.count_tokens(chunk),
                "text": chunk
            })
        return chunks_with_meta

    #对 json 文件分块，输出还是 json
    def split_all_reports(self, all_report_dir: Path, output_dir: Path, serialized_tables_dir: Optional[Path] = None):
        """
        批量处理目录下所有报告（json文件），对每个报告进行文本分块，并输出到目标目录。
        如果提供了序列化表格目录，会尝试将表格内容插入到对应页面的分块中。
        主要用于后续向量化和检索的预处理。
        参数：
            all_report_dir: 存放待处理报告json的目录
            output_dir: 分块后输出的目标目录
            serialized_tables_dir: （可选）存放序列化表格的目录
        """
        # 获取所有报告文件路径
        all_report_paths = list(all_report_dir.glob("*.json"))
        
        # 遍历每个报告文件
        for report_path in all_report_paths:
            serialized_tables_path = None
            # 如果提供了表格序列化目录，查找对应表格文件
            if serialized_tables_dir is not None:
                serialized_tables_path = serialized_tables_dir / report_path.name
                if not serialized_tables_path.exists():
                    print(f"警告：未找到 {report_path.name} 的序列化表格报告")
                
            # 读取报告内容
            with open(report_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)
                
            # 分块处理，插入表格分块（如有）
            updated_report = self._split_report(report_data, serialized_tables_path)
            # 确保输出目录存在
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 写入分块后的报告到目标目录
            with open(output_dir / report_path.name, 'w', encoding='utf-8') as file:
                json.dump(updated_report, file, indent=2, ensure_ascii=False)
                
        # 输出处理文件数统计
        print(f"已分块处理 {len(all_report_paths)} 个文件")

    def split_markdown_file(self, md_path: Path, chunk_size: int = 30, chunk_overlap: int = 5):
        """
        按行分割 markdown 文件，每个分块记录起止行号和内容。
        :param md_path: markdown 文件路径
        :param chunk_size: 每个分块的最大行数
        :param chunk_overlap: 分块重叠行数
        :return: 分块列表
        """
        with open(md_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        chunks = []
        i = 0
        total_lines = len(lines)
        while i < total_lines:
            start = i
            end = min(i + chunk_size, total_lines)
            chunk_text = ''.join(lines[start:end])
            chunks.append({
                'lines': [start + 1, end],  # 行号从1开始
                'text': chunk_text
            })
            i += chunk_size - chunk_overlap
        return chunks

    def split_markdown_reports(self, all_md_dir: Path, output_dir: Path, chunk_size: int = 30, chunk_overlap: int = 5, subset_csv: Path = None):
        """
        批量处理目录下所有 markdown 文件，分块并输出为 json 文件到目标目录。
        :param all_md_dir: 存放 .md 文件的目录
        :param output_dir: 输出 .json 文件的目录
        :param chunk_size: 每个分块的最大行数
        :param chunk_overlap: 分块重叠行数
        :param subset_csv: subset.csv 路径，用于建立 file_name 到 company_name 的映射
        """
        # 建立 file_name（去扩展名）到 company_name 的映射
        file2company = {}
        file2sha1 = {}
        if subset_csv is not None and os.path.exists(subset_csv):
            # 优先尝试 utf-8，失败则尝试 gbk
            try:
                df = pd.read_csv(subset_csv, encoding='utf-8')
            except UnicodeDecodeError:
                print('警告：subset.csv 不是 utf-8 编码，自动尝试 gbk 编码...')
                df = pd.read_csv(subset_csv, encoding='gbk')
            # 自动识别主键列
            if 'file_name' in df.columns:
                for _, row in df.iterrows():
                    file_no_ext = os.path.splitext(str(row['file_name']))[0]
                    file2company[file_no_ext] = row['company_name']
                    if 'sha1' in row:
                        file2sha1[file_no_ext] = row['sha1']
            elif 'sha1' in df.columns:
                for _, row in df.iterrows():
                    file_no_ext = str(row['sha1'])
                    file2company[file_no_ext] = row['company_name']
                    file2sha1[file_no_ext] = row['sha1']
            else:
                raise ValueError('subset.csv 缺少 file_name 或 sha1 列，无法建立文件名到公司名的映射')
        
        all_md_paths = list(all_md_dir.glob("*.md"))
        output_dir.mkdir(parents=True, exist_ok=True)
        for md_path in all_md_paths:
            chunks = self.split_markdown_file(md_path, chunk_size, chunk_overlap)
            output_json_path = output_dir / (md_path.stem + ".json")
            # 查找 company_name 和 sha1
            file_no_ext = md_path.stem
            company_name = file2company.get(file_no_ext, "")
            sha1 = file2sha1.get(file_no_ext, "")
            # metainfo 只保留 sha1、company_name、file_name 字段
            metainfo = {"sha1": sha1, "company_name": company_name, "file_name": md_path.name}
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump({"metainfo": metainfo, "content": {"chunks": chunks}}, f, ensure_ascii=False, indent=2)
            print(f"已处理: {md_path.name} -> {output_json_path.name}")
        print(f\"共分割 {len(all_md_paths)} 个 markdown 文件\")\n\n    def split_json_reports(self, all_json_dir: Path, output_dir: Path, chunk_size: int = 30, chunk_overlap: int = 5, subset_csv: Path = None):\n        \"\"\"\n        批量处理目录下的JSON格式文档，保留页码和结构信息，分块并输出为标准格式\n        :param all_json_dir: 存放JSON格式文档的目录\n        :param output_dir: 输出chunked_reports的目录\n        :param chunk_size: 每个块的最大行数\n        :param chunk_overlap: 块重叠行数\n        :param subset_csv: subset.csv路径，用于建立file_name到company_name的映射\n        \"\"\"\n        # 建立file_name（去扩展名）到company_name的映射\n        file2company = {}\n        file2sha1 = {}\n        if subset_csv is not None and os.path.exists(subset_csv):\n            try:\n                df = pd.read_csv(subset_csv, encoding='utf-8')\n            except UnicodeDecodeError:\n                print('警告：subset.csv 不是 utf-8 编码，自动尝试 gbk 编码...')\n                df = pd.read_csv(subset_csv, encoding='gbk')\n            \n            # 自动识别主键列\n            if 'file_name' in df.columns:\n                for _, row in df.iterrows():\n                    file_no_ext = os.path.splitext(str(row['file_name']))[0]\n                    file2company[file_no_ext] = row['company_name']\n                    if 'sha1' in row:\n                        file2sha1[file_no_ext] = row['sha1']\n            elif 'sha1' in df.columns:\n                for _, row in df.iterrows():\n                    file_no_ext = str(row['sha1'])\n                    file2company[file_no_ext] = row['company_name']\n                    file2sha1[file_no_ext] = row['sha1']\n            else:\n                raise ValueError('subset.csv 缺少 file_name 或 sha1 列，无法建立文件名到公司名的映射')\n\n        all_json_paths = list(all_json_dir.glob(\"*.json\"))\n        output_dir.mkdir(parents=True, exist_ok=True)\n\n        for json_path in all_json_paths:\n            with open(json_path, 'r', encoding='utf-8') as f:\n                data = json.load(f)\n            \n            output_json_path = output_dir / (json_path.stem + \".json\")\n            \n            # 检索company_name和sha1\n            file_no_ext = json_path.stem\n            company_name = file2company.get(file_no_ext, \"\")\n            sha1 = file2sha1.get(file_no_ext, \"\")\n            \n            # 从JSON结构中提取chunks，保留页码信息\n            chunks = []\n            chunk_id = 0\n            \n            # 根据MinerU返回的JSON格式提取内容\n            # 假设JSON格式包含pages数组，每个page有page_number和content\n            if 'pages' in data:\n                for page_data in data['pages']:\n                    page_num = page_data.get('page_number', page_data.get('page_idx', page_data.get('page', 0)))\n                    \n                    # 获取页面内容\n                    page_content = \"\"\n                    if 'content' in page_data:\n                        if isinstance(page_data['content'], list):\n                            # 如果content是blocks列表\n                            for block in page_data['content']:\n                                if isinstance(block, dict) and 'text' in block:\n                                    page_content += block['text'] + \"\\n\"\n                        elif isinstance(page_data['content'], str):\n                            page_content = page_data['content']\n                    elif 'text' in page_data:\n                        # 直接是text字段\n                        page_content = page_data['text']\n                    elif 'blocks' in page_data:\n                        # 如果是blocks结构\n                        for block in page_data['blocks']:\n                            if isinstance(block, dict) and 'text' in block:\n                                page_content += block['text'] + \"\\n\"\n                    \n                    # 创建chunk，保留页码信息\n                    if page_content.strip():\n                        chunks.append({\n                            'id': chunk_id,\n                            'page': page_num,\n                            'text': page_content,\n                            'length_tokens': self.count_tokens(page_content)\n                        })\n                        chunk_id += 1\n            \n            # 构建输出结构\n            output_data = {\n                \"metainfo\": {\n                    \"sha1\": sha1,\n                    \"company_name\": company_name,\n                    \"file_name\": json_path.name\n                },\n                \"content\": {\n                    \"chunks\": chunks\n                }\n            }\n            \n            with open(output_json_path, 'w', encoding='utf-8') as f:\n                json.dump(output_data, f, ensure_ascii=False, indent=2)\n            \n            print(f\"已处理: {json_path.name} -> {output_json_path.name}\")\n        \n        print(f\"共分块处理 {len(all_json_paths)} 个JSON文件\")
