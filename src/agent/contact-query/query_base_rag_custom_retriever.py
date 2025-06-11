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
from agno.utils.log import log_debug

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

settings = deepseek_settings
#------------------ settings ------------------

vector_db = LanceDb(
    table_name="contact_table",
    uri="C:\\Lee\\work\\contract\\db\\tmp\\contact_vectors.lancedb",
    search_type=SearchType.hybrid,
    embedder=OpenAIEmbedder(id=embedding_model_id,api_key=api_key,base_url=base_url, dimensions=2048),
)

def text_rerank(query,documents,api_key,threshold=0.4):
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
    
def retriever_with_rerank(query,num_documents=60):
    results = vector_db.search(query,limit=num_documents)
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
            content_list_rerank.append({"content":content_list[result.index],"relevance_score":result.relevance_score})
    
    return content_list_rerank

def retriever(
    query: str, agent: Optional[Agent] = None, num_documents: int = 20, **kwargs
) -> Optional[list[dict]]:
    """
    Custom retriever function to search the vector database for relevant documents.
    """
    try:
        # log_debug(f"\n\n\n\n\nretriever: {query}\n\n\n\n\n")
        result = retriever_with_rerank(query,40)
        return result
    except Exception as e:
        print(f"Error during vector database search: {str(e)}")
        return None

# knowledge_base = AgentKnowledge(vector_db=vector_db)

# add_references=True 就是传统rag的做法，会将找到的相关文档都放在用户的context中，而现代的做法会通过search_knowledge_base进行工具查询然后得到相关结果，建议先不使用add_references=True，如果性能有问题再进行相关测试进行评估

agent = Agent(
    model=OpenAILike(**settings),
    name='Contact_Query_Agent',
    instructions=['查询合同详情的时候请列出所有相关合同的数据，严禁遗漏任何条目','查询设备详情的时候请列出所有相关合同及设备的数据，严禁遗漏任何条目，无关数据不要列出','如果查询的是某个项目合同中的设备信息，先使用search_knowledge_base工具，参数传递项目名称，找到对应的项目合同信息，再全文搜索设备信息','禁止虚构和假设任何数据','如果需要进行合同比对的时候，请按需**分别**查出所有项目后再进行比对','必须使用简体中文回复'],
    search_knowledge=True,
    retriever=retriever,
    add_history_to_messages=True,
    num_history_responses=20,
    # tools=[ReasoningTools(add_instructions=True)],
    markdown=True,
    # add_references=True,
    # stream=True,
    stream_intermediate_steps=True,
    telemetry=False,
    debug_mode=True,
)

# content_list_rerank = retriever_with_rerank('包钢炼铁厂烧结项目余热锅炉价格')
# for content in content_list_rerank:
#     print(content[:100])
#     print('-'*30)

agent.print_response(message='包钢炼铁厂烧结项目和宝山钢铁余热项目的余热锅炉价格相差多少')

# response = agent.run(message='包头钢铁烧结机余热项目和宝钢德盛烧结余热项目的余热锅炉价格相差多少')
# print(response)

# app = Playground(agents=[agent]).get_app()

# if __name__ == "__main__":
#     serve_playground_app("query_base_rag:app", reload=True)