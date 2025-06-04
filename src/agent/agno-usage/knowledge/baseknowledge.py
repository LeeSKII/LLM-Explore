from typing import Iterator
from agno.agent import Agent, RunResponse
from agno.models.openai.like import OpenAILike
from agno.utils.pprint import pprint_run_response
from agno.vectordb.lancedb import LanceDb
from agno.vectordb.search import SearchType
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge import AgentKnowledge
from pathlib import Path

import os
from dotenv import load_dotenv
load_dotenv()
api_key=os.getenv("QWEN_API_KEY")
base_url=os.getenv("QWEN_API_BASE_URL")
local_base_url = 'http://192.168.0.166:8000/v1'
local_model_name = 'Qwen3-235B'
model_name = 'qwen-plus-latest'
embedding_model_id = 'text-embedding-v3'

local_settings = {
  'api_key' : '123',
  'base_url' : local_base_url,
  'id' : local_model_name
}

qwen_settings = {
  'api_key' : api_key,
  'base_url' : base_url,
  'id' : model_name
}

settings = qwen_settings

# 通过load_knowledge函数加载database到指定的url vector database的table_name中

vector_db = LanceDb(
    table_name="website_vectors",
    uri="tmp/website_vectors.lancedb",
    search_type=SearchType.hybrid,
    embedder=OpenAIEmbedder(id=embedding_model_id,api_key=api_key,base_url=base_url, dimensions=1024),
)

knowledge = AgentKnowledge(vector_db=vector_db)

agent = Agent(model=OpenAILike(**settings),knowledge=knowledge,show_tool_calls=True,telemetry=False)

agent.print_response(message='what is 5 level agentic system?',stream=True,stream_intermediate_steps=True)