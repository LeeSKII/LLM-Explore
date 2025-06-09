# 使用 mammoth 库将 docx 文件转换为html，解决表格合并的问题，然后将文本内容添加到 Document 对象中，并将 Document 对象添加到 DocumentKnowledgeBase 对象中。
# 解决agno中docx文件进knowledge base丢失表格的问题
# markitdown 库兼容ing问题太多，且无法处理表格合并项问题

from agno.document.base import Document
from agno.knowledge.document import DocumentKnowledgeBase
from agno.vectordb.lancedb import LanceDb
from agno.vectordb.search import SearchType
from agno.embedder.openai import OpenAIEmbedder
from agno.document.chunking.fixed import FixedSizeChunking
from pathlib import Path
import mammoth

import os
from dotenv import load_dotenv

load_dotenv()
api_key=os.getenv("QWEN_API_KEY")
base_url=os.getenv("QWEN_API_BASE_URL")
embedding_model_id = 'text-embedding-v4'

vector_db = LanceDb(
    table_name="contact_table",
    uri="C:\\Lee\\work\\contract\\db\\tmp\\contact_vectors.lancedb",
    search_type=SearchType.hybrid,
    embedder=OpenAIEmbedder(id=embedding_model_id,api_key=api_key,base_url=base_url, dimensions=2048),
)

documents = []

style_map = """
p =>
b =>
i =>
"""

for file in Path("C:\Lee\work\contract\精简\锅炉合同").glob("*.docx"):
    print(file)
    result = mammoth.convert_to_html(file,style_map=style_map,include_default_style_map=False)
    doc_content = result.value
    doc = Document(content=doc_content)
    documents.append(doc)


knowledge_base = DocumentKnowledgeBase(
    documents=documents,
    vector_db=vector_db,
    chunking_strategy=FixedSizeChunking(chunk_size=8192),
)

knowledge_base.load(recreate=False)

