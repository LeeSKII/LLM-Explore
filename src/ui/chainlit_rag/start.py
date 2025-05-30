import os
from agno.agent import Agent,AgentKnowledge
from agno.models.openai.like import OpenAILike
from dotenv import load_dotenv
from agno.vectordb.lancedb import LanceDb
from agno.vectordb.search import SearchType
from agno.embedder.openai import OpenAIEmbedder
import chainlit as cl
from agno.knowledge.docx import DocxKnowledgeBase
from agno.knowledge import AgentKnowledge
import asyncio
from pathlib import Path

load_dotenv()

api_key=os.getenv("QWEN_API_KEY")
base_url=os.getenv("QWEN_API_BASE_URL")
local_base_url = 'http://192.168.0.166:8000/v1'
local_model_name = 'Qwen3-235B'
model_name = 'qwen3-235b-a22b'
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

settings = local_settings

vector_db = LanceDb(
    table_name="contact_file",
    uri=r"E:\PythonProject\LLM-Explore\src\agent\agno-usage\knowledge\tmp\lancedb",
    search_type=SearchType.hybrid,
    embedder=OpenAIEmbedder(id=embedding_model_id,api_key=api_key,base_url=base_url, dimensions=1024),
)

# knowledge_base = DocxKnowledgeBase(path=Path(r"D:\【个人工作】\【2025】\AI解析合同\安阳钢铁集团有限责任公司综利公司烧结机头灰资源化处置项目（运营）"),vector_db=vector_db)
# 使用已有的vectordb不需要使用load方法
knowledge_base = AgentKnowledge(vector_db=vector_db)

@cl.on_chat_start
def init_agent():
    # 需要关闭telemetry，否则会有额外的post请求到agno官网
    agent = Agent(model=OpenAILike(**settings),markdown=True,telemetry=False,knowledge=knowledge_base)
    # 在使用知识库之前，需要加载将用于检索的嵌入，异步加载，可以显著提高加载大型知识库时的性能
    # asyncio.run(knowledge_base.aload(recreate=False))
    # knowledge_base.load(recreate=False)
    cl.user_session.set("agent", agent)

@cl.on_message
async def on_message(msg: cl.Message):
    message = cl.Message(content="")
    user_query = msg.content
    agent:Agent = cl.user_session.get("agent")

    # # Streaming the final answer 可以生效
    for chunk in await cl.make_async(agent.run)(user_query, stream=True):
        await message.stream_token(chunk.content)
    
    await message.send()

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)