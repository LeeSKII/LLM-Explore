import os
from agno.agent import Agent,AgentKnowledge,RunResponse,RunEvent
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
import json

from traitlets import default

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



@cl.on_chat_start
def init_agent():
    vector_db = LanceDb(
      table_name="contact_table",
      uri="D:\\projects\\LLM-Explore\\src\\agent\\contact-query\\tmp\\contact_vectors.lancedb",
      search_type=SearchType.hybrid,
      embedder=OpenAIEmbedder(id=embedding_model_id,api_key=api_key,base_url=base_url, dimensions=2048),
    )
    knowledge_base = AgentKnowledge(vector_db=vector_db)
    agent = Agent(
      model=OpenAILike(**settings),
      name='Contact_Query_Agent',
      instructions=['查询合同详情的时候请列出所有数据，严禁遗漏任何条目','禁止虚构和假设任何数据','如果需要进行合同比对的时候，请按需**分别**查出所有项目后再进行比对','必须使用简体中文回复'],
      knowledge=knowledge_base,
      add_history_to_messages=True,
      num_history_responses=20,
      tools=[ReasoningTools(add_instructions=True)],
      markdown=True,
      # add_references=True,
      stream=True,
      stream_intermediate_steps=True,
      telemetry=False,
      debug_mode=False,
    )
    # asyncio.run(knowledge_base.aload(recreate=False))
    # knowledge_base.load(recreate=False)
    cl.user_session.set("agent", agent)

@cl.on_message
async def on_message(msg: cl.Message):
    message = cl.Message(content="")
    user_query = msg.content
    agent:Agent = cl.user_session.get("agent")

    # # Streaming the final answer 可以生效
    for response in await cl.make_async(agent.run)(user_query, stream=True):
        if response.event != RunEvent.run_response:
            print(response.event,"----",response)
        if response.event == RunEvent.run_response:
            await message.stream_token(response.content)
        elif response.event == RunEvent.tool_call_started:
            for tool in response.tools:
              async with cl.Step(name=tool.tool_name) as tool_call_step:
                  tool_args_str = json.dumps(tool.tool_args, indent=2, ensure_ascii=False)
                  tool_call_step.input = f"Tool Args: {tool_args_str}"
                  # await tool_call_step.stream_token(tool_args_str)                      
        elif response.event == RunEvent.reasoning_step:
            async with cl.Step(name=response.event+f":{response.content.title}",default_open=False) as reasoning_step:
                await reasoning_step.stream_token(response.reasoning_content)
        elif response.event == RunEvent.run_started:
            async with cl.Step(name="Agent 开始执行...") as run_start:
                pass
        else:
            await run_start.remove()
            
    await message.send()

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)