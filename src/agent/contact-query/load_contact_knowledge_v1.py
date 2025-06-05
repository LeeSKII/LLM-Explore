# 此版本直接读取docx文件会造成表格数据丢失

from agno.knowledge.docx import DocxKnowledgeBase
from agno.document.base import Document
from agno.vectordb.lancedb import LanceDb
from agno.vectordb.search import SearchType
from agno.embedder.openai import OpenAIEmbedder
from pathlib import Path

import os
from dotenv import load_dotenv
load_dotenv()
api_key=os.getenv("QWEN_API_KEY")
base_url=os.getenv("QWEN_API_BASE_URL")
embedding_model_id = 'text-embedding-v3'

vector_db = LanceDb(
    table_name="contact_table",
    uri="E:\\PythonProject\\LLM-Explore\\src\\agent\\contact-query\\tmp\\contact_vectors.lancedb",
    search_type=SearchType.hybrid,
    embedder=OpenAIEmbedder(id=embedding_model_id,api_key=api_key,base_url=base_url, dimensions=1024),
)

knowledge_base = DocxKnowledgeBase(
    path=Path(r'E:\Temp\docx_files'),
    # Table name: ai.website_documents
    vector_db=vector_db,
)

knowledge_base.load(recreate=False)

