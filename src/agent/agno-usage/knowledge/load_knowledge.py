from agno.knowledge.website import WebsiteKnowledgeBase
from agno.vectordb.lancedb import LanceDb
from agno.vectordb.search import SearchType
from agno.embedder.openai import OpenAIEmbedder

import os
from dotenv import load_dotenv
load_dotenv()
api_key=os.getenv("QWEN_API_KEY")
base_url=os.getenv("QWEN_API_BASE_URL")
embedding_model_id = 'text-embedding-v3'

vector_db = LanceDb(
    table_name="website_vectors",
    uri="tmp/website_vectors.lancedb",
    search_type=SearchType.hybrid,
    embedder=OpenAIEmbedder(id=embedding_model_id,api_key=api_key,base_url=base_url, dimensions=1024),
)

knowledge_base = WebsiteKnowledgeBase(
    urls=["https://docs.agno.com/introduction"],
    # Number of links to follow from the seed URLs
    max_links=10,
    # Table name: ai.website_documents
    vector_db=vector_db,
)

knowledge_base.load(recreate=False)

