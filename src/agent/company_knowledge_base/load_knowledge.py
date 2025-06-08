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
    uri="D:\\projects\\LLM-Explore\\src\\agent\\contact-query\\tmp\\contact_vectors.lancedb",
    search_type=SearchType.hybrid,
    embedder=OpenAIEmbedder(id=embedding_model_id,api_key=api_key,base_url=base_url, dimensions=2048),
)

documents = []

for file in Path("D:\work\others\docx_files").glob("*.docx"):
    print(file)
    result = mammoth.convert_to_html(file)
    doc_content = result.value
    doc = Document(content=doc_content)
    documents.append(doc)


knowledge_base = DocumentKnowledgeBase(
    documents=documents,
    vector_db=vector_db,
    chunking_strategy=FixedSizeChunking(chunk_size=8192),
)

knowledge_base.load(recreate=False)

