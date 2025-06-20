import os
from agno.agent import Agent,AgentKnowledge,RunResponse,RunEvent,RunResponseEvent
from agno.models.openai.like import OpenAILike
from dotenv import load_dotenv
from agno.vectordb.lancedb import LanceDb
from agno.vectordb.search import SearchType
from agno.embedder.openai import OpenAIEmbedder
import chainlit as cl
from agno.knowledge import AgentKnowledge
import asyncio
from agno.tools.reasoning import ReasoningTools
from pathlib import Path
from agno.tools import tool
import json
from typing import Optional
import dashscope
from http import HTTPStatus
from agno.utils.log import log_debug
import lancedb
from openai import OpenAI
import pandas as pd

load_dotenv()

api_key=os.getenv("QWEN_API_KEY")
base_url=os.getenv("QWEN_API_BASE_URL")
local_base_url = 'http://192.168.0.166:8000/v1'
local_model_name = 'Qwen3-235B'
model_name = 'qwen-plus-latest'
embedding_model_id = 'text-embedding-v4'
dashscope_api_key = os.getenv("QWEN_API_KEY")

temperature = 0.1
local_settings = {
  'api_key' : '123',
  'base_url' : local_base_url,
  'id' : local_model_name,
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

settings = local_settings

# Available commands in the UI
COMMANDS = [
    {
        "id": "Reasoning",
        "icon": "sparkle",
        "description": "Reasoning deep",
        "button": True,
        "persistent": True
    },
]

instructions = ['查询合同详情的时候请列出所有相关合同的数据，严禁遗漏任何条目',
                '查询设备详情的时候请列出所有相关合同及设备的数据，严禁遗漏任何条目，无关数据不要列出',
                '如果查询的是某个项目合同中的设备信息，先使用search_knowledge_base工具，参数传递项目名称，找到对应的项目合同信息，再全文搜索设备信息',
                '需要在相关匹配用户的查询需求和提供的背景知识，例如项目名称，名称，供应商名称等，严禁使用非用户指定的查询内容作为回答，例如：用户指定查询A项目，但是返回了B项目的信息，这将严重违背用户的查询意愿。',
                '如果提供的背景知识没有用户需要查询的信息，请告知用户没有在知识库搜索到相关数据',
                '查询合同详情的时候请列出所有数据，严禁遗漏任何条目',
                '禁止虚构和假设任何数据',
                '如果需要进行合同比对的时候，请按需**分别**查出所有项目后再进行比对',
                '合同查询结果请按年份从新到旧排列',
                '合同查询结果请按合同金额从高到低排列',
                '必须使用简体中文回复',
                ]

db = lancedb.connect("C:/Lee/work/db/contract_full_lancedb") 
table = db.open_table("contract_table")

def get_embedding(text,model='text-embedding-v4',dimensions=2048):
    client = OpenAI(
        api_key=api_key, 
        base_url=base_url
    )

    completion = client.embeddings.create(
        model=model,
        input=text,
        dimensions=dimensions, # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
        encoding_format="float"
    )
    
    return completion.data[0].embedding

def text_rerank(query,documents,api_key,threshold=0.1):
    resp = dashscope.TextReRank.call(
        model="gte-rerank-v2",
        query=query,
        documents=documents,
        return_documents=False,
        api_key=api_key,
    )
    if resp.status_code == HTTPStatus.OK:
        # print(resp)
        results = resp.output.results
        scores = [result.relevance_score for result in results]
        log_debug(f"\n\n查询到相关系数: {scores}\n\n")
        results = [result for result in results if result.relevance_score > threshold]
        return results
    else:
        # print(resp)
        return None

def retriever_with_rerank(query,num_documents=30):
    # results = vector_db.search(query,limit=num_documents)
    # Vector search with filters (pre-filtering is the default)
    
    embedding = get_embedding(query)
    search_vector_results = table.search(embedding,vector_column_name="vector").nprobes(256).limit(num_documents).to_pandas()  
    search_like_results = table.search().where(f"doc LIKE '%{query}%'").limit(num_documents).to_pandas()
    
    search_merged = pd.concat([search_like_results, search_vector_results], ignore_index=True)
    search_results = search_merged.drop_duplicates(subset='meta_str', keep='first')
    
    log_debug(f"\n\n查询到文档数: {len(search_results)}\n\n")
    content_list = search_results['doc'].tolist()

    document_chunk_size = 5000 # max number of characters in each document chunk for reranker
    documents = [content[:document_chunk_size] for content in content_list]
    # log_debug(f"\n\n\n\n\n查询到文档: {documents}\n\n\n\n\n")
    reranker_results = text_rerank(query,documents,api_key=dashscope_api_key, threshold=0.15)
    
    content_list_rerank = []
    if reranker_results is None:
        raise Exception("Reranker failed")
    if len(reranker_results) == 0:
        pass
    else:
        for result in reranker_results:
            content_list_rerank.append({"content":content_list[result.index],"relevance_score":result.relevance_score})
    
    return content_list_rerank

@tool(name="search_knowledge_base")
def search_knowledge_base(query: str) -> Optional[list[dict]]:
    '''
    根据语义相关度检索知识库，获取相关合同信息.
    
    Args:
        query (str): 搜索知识库的查询关键字.
        
    Returns:
        Optional[list[dict]]: 包含文档内容和相关度得分的字典列表.
    '''
    try:
        # log_debug(f"\n\n\n\n\nretriever: {query}\n\n\n\n\n")
        result = retriever_with_rerank(query,30)
        return result
    except Exception as e:
        print(f"Error during vector database search: {str(e)}")
        return None

@tool(name="search_knowledge_base_with_year")
def search_knowledge_base_with_year(query: str,year:int) -> Optional[list[dict]]:
    '''
    优先根据年份过滤然后根据语义相关度检索知识库.
    
    Args:
        query (str): 搜索知识库的查询关键字.
        
        year (int): 合同年份.
        
    Returns:
        Optional[list[dict]]: 包含文档内容和相关度得分的字典列表.
    '''
    try:
        num_documents = 30
        embedding = get_embedding(query)
        search_vector_results = table.search(embedding,vector_column_name="vector").where(f"year>={year}").nprobes(256).limit(num_documents).to_pandas()  
        search_like_results = table.search().where(f"doc LIKE '%{query}%' AND year>={year}").limit(num_documents).to_pandas()
        
        search_merged = pd.concat([search_like_results, search_vector_results], ignore_index=True)
        search_results = search_merged.drop_duplicates(subset='meta_str', keep='first')
        
        log_debug(f"\n\n查询到文档数: {len(search_results)}\n\n")
        content_list = search_results['doc'].tolist()

        document_chunk_size = 5000 # max number of characters in each document chunk for reranker
        documents = [content[:document_chunk_size] for content in content_list]
        # log_debug(f"\n\n\n\n\n查询到文档: {documents}\n\n\n\n\n")
        reranker_results = text_rerank(query,documents,api_key=dashscope_api_key, threshold=0.1)
        
        content_list_rerank = []
        if reranker_results is None:
            raise Exception("Reranker failed")
        if len(reranker_results) == 0:
            pass
        else:
            for result in reranker_results:
                content_list_rerank.append({"content":content_list[result.index],"relevance_score":result.relevance_score})
        
        return content_list_rerank
    except Exception as e:
        print(f"Error during vector database search: {str(e)}")
        return None



@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="项目合同信息查询",
            message="包头钢铁新体系项目采购设备",
            icon="/public/idea.svg",
            ),
        cl.Starter(
            label="多项目余热锅炉价格对比",
            message="湖南华菱涟钢项目和揭阳大南海石化工业区危险废物焚烧以及宝山钢铁四烧结余热锅炉价格对比。",
            icon="/public/learn.svg",
            ),
        cl.Starter(
            label="联合查询合同信息",
            message="山东永锋余热发电、河北东海特钢项目余热锅炉价格对比。",
            icon="/public/terminal.svg",
            ),
        cl.Starter(
            label="单项目查询合同信息",
            message="泉州闽光余热发电项目合同数据。",
            icon="/public/write.svg",
            ),
        cl.Starter(
            label="其它设备的合同信息",
            message="增压风机的合同有哪些。",
            icon="/public/write.svg",
            ),
        cl.Starter(
            label="根据年份查询合同信息",
            message="2024年后余热锅炉的合同有哪些。",
            icon="/public/write.svg",
            )
        ]

@cl.on_chat_start
async def init_agent():
    # await cl.context.emitter.set_commands(COMMANDS)
    
    # knowledge_base = AgentKnowledge(vector_db=vector_db,num_documents=5)
    agent = Agent(
      model=OpenAILike(**settings,temperature=temperature),
      name='Contact_Query_Agent',
      instructions=instructions,
      tools=[search_knowledge_base,search_knowledge_base_with_year],
      add_history_to_messages=True,
      num_history_responses=20,
      markdown=True,
      add_datetime_to_instructions=True,
      stream=True,
      stream_intermediate_steps=True,
      telemetry=False,
      debug_mode=True,
    )

    cl.user_session.set("agent", agent)

@cl.on_message
async def on_message(msg: cl.Message):
    # Process message with or without explicit search command
    agent:Agent = cl.user_session.get("agent")
    
    # agent:Agent = cl.user_session.get("agent")
    
    message = cl.Message(content="")
    user_query = msg.content 
    run_start_step = None

    for response in await cl.make_async(agent.run)(user_query, stream=True):
        # if response.event != RunEvent.run_response:
        #     print(response.event,"----",response)
        if response.event != 'RunResponseContent' and run_start_step:
            await run_start_step.remove()
        if response.event == 'RunResponseContent':
            await message.stream_token(response.content)
        elif response.event == 'ToolCallStarted':
            tool = response.tool
            async with cl.Step(name=tool.tool_name,id=tool.tool_call_id) as tool_call_step:
                tool_args_str = json.dumps(tool.tool_args, indent=2, ensure_ascii=False)
                tool_call_step.input = f"Tool Args: {tool_args_str}"
                                
        elif response.event == 'ReasoningStarted':
            # name = response.event+f":{response.content.title}" # 使用动态name会有繁忙图标问题
            async with cl.Step(name='reasoning',default_open=False) as reasoning_step:
                # reasoning_step.output = response.reasoning_content
                reasoning_step.input = json.dumps({"title":response.content.title,"action":response.content.action}, indent=2, ensure_ascii=False)
                reasoning_step.output = response.reasoning_content
                # await reasoning_step.stream_token(response.reasoning_content)
                # await reasoning_step.update()
        elif response.event == 'RunStarted':
            async with cl.Step(name="合同查询 Agent 开始执行...") as run_start_step:
                pass
        else:
            pass
            
    await message.send()

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)