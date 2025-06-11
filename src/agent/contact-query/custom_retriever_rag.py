import re
from agno.agent import Agent
from agno.models.openai import OpenAILike
from agno.vectordb.lancedb import LanceDb
from agno.vectordb.search import SearchType
from agno.embedder.openai import OpenAIEmbedder
from agno.tools.reasoning import ReasoningTools
from agno.knowledge import AgentKnowledge
from agno.playground import Playground, serve_playground_app
from typing import Optional
import dashscope
from http import HTTPStatus

#------------------ settings ------------------
import os
from dotenv import load_dotenv
load_dotenv()
api_key=os.getenv("QWEN_API_KEY")
base_url=os.getenv("QWEN_API_BASE_URL")
local_base_url = 'http://192.168.0.166:8000/v1'
local_model_name = 'Qwen3-235B'
model_name = 'qwen-plus-latest'
embedding_model_id = 'text-embedding-v4'
dashscope_api_key = os.getenv("QWEN_API_KEY")

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

deepseek_settings = {
  'api_key' : os.getenv("DEEPSEEK_API_KEY"),
  'base_url' : os.getenv("DEEPSEEK_API_BASE_URL"),
  'id' : 'deepseek-chat'
}

settings = qwen_settings
#------------------ settings ------------------

vector_db = LanceDb(
    table_name="contact_table",
    uri="C:\\Lee\\work\\contract\\db\\tmp\\contact_vectors.lancedb",
    search_type=SearchType.hybrid,
    embedder=OpenAIEmbedder(id=embedding_model_id,api_key=api_key,base_url=base_url, dimensions=2048),
)

# Define the custom retriever
# This is the function that the agent will use to retrieve documents
def retriever(
    query: str, agent: Optional[Agent] = None, num_documents: int = 5, **kwargs
) -> Optional[list[dict]]:
    """
    Custom retriever function to search the vector database for relevant documents.
    """
    try:
        pass
    except Exception as e:
        print(f"Error during vector database search: {str(e)}")
        return None
      
def text_rerank(query,documents,api_key,threshold=0.5):
    resp = dashscope.TextReRank.call(
        model="gte-rerank-v2",
        query=query,
        documents=documents,
        top_n=10,
        return_documents=False,
        api_key=api_key,
    )
    if resp.status_code == HTTPStatus.OK:
        # print(resp)
        results = resp.output.results
        results = [result for result in results if result.relevance_score > threshold]
        return results
    else:
        # print(resp)
        return None
    
def retriever_with_rerank():
    results = vector_db.search(query,limit=20)
    content_list = [result.content for result in results]
    document_chunk_size = 3000 # max number of characters in each document chunk for reranker
    documents = [content[:document_chunk_size] for content in content_list]
    
    reranker_results = text_rerank(query,documents,api_key=dashscope_api_key)
    
    content_list_rerank = []
    if reranker_results is None:
        raise Exception("Reranker failed")
    if len(reranker_results) == 0:
        pass
    else:
        for result in reranker_results:
            content_list_rerank.append(content_list[result.index])
    
    return content_list_rerank
 
if __name__ == '__main__':
    # query = '包钢新体系烧结机余热项目 余热锅炉'
    query = '包钢炼铁厂烧结合同项目设备'
    
    results = vector_db.search(query,limit=20)
    content_list = [result.content for result in results]
    document_chunk_size = 3000 # max number of characters in each document chunk for reranker
    documents = [content[:document_chunk_size] for content in content_list]
    reranker_results = text_rerank(query,documents,api_key=dashscope_api_key)
    
    content_list_rerank = []
    if reranker_results is None:
        raise Exception("Reranker failed")
    if len(reranker_results) == 0:
        pass
    else:
        for result in reranker_results:
            content_list_rerank.append(content_list[result.index])
   
    for content in content_list_rerank:
        print(content[:100])
        print('-'*30)
    # for content in content_list:
    #     print(content[:100])
    #     print('-'*30)
