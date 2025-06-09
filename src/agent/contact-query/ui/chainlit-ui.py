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

deepseek_settings = {
    'api_key' : os.getenv("DEEPSEEK_API_KEY"),
    'base_url' : os.getenv("DEEPSEEK_API_BASE_URL"),
    'id' : 'deepseek-chat'
}

settings = deepseek_settings

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

@cl.on_chat_start
async def init_agent():
    await cl.context.emitter.set_commands(COMMANDS)
    vector_db = LanceDb(
      table_name="contact_table",
      uri="C:\\Lee\\work\\contract\\db\\tmp\\contact_vectors.lancedb",
      search_type=SearchType.hybrid,
      embedder=OpenAIEmbedder(id=embedding_model_id,api_key=api_key,base_url=base_url, dimensions=2048),
    )
    knowledge_base = AgentKnowledge(vector_db=vector_db,num_documents=10)
    agent = Agent(
      model=OpenAILike(**settings),
      name='Contact_Query_Agent',
      instructions=['查询合同详情的时候请列出所有数据，严禁遗漏任何条目','禁止虚构和假设任何数据','如果需要进行合同比对的时候，请按需**分别**查出所有项目后再进行比对','必须使用简体中文回复'],
      knowledge=knowledge_base,
      add_history_to_messages=True,
      num_history_responses=20,
      markdown=True,
      add_datetime_to_instructions=True,
      # add_references=True,
      stream=True,
      stream_intermediate_steps=True,
      telemetry=False,
      debug_mode=False,
    )
    
    agent_reasoning = Agent(
      model=OpenAILike(**settings),
      name='Contact_Query_Agent',
      instructions=['查询合同详情的时候请列出所有包含价格的设备数据，严禁遗漏任何条目','禁止虚构和假设任何数据','如果需要进行合同比对的时候，请按需**分别**查出所有项目后再进行比对','必须使用简体中文回复'],
      knowledge=knowledge_base,
      add_history_to_messages=True,
      num_history_responses=20,
      tools=[ReasoningTools(add_instructions=True)],
      markdown=True,
      add_datetime_to_instructions=True,
      # add_references=True,
      stream=True,
      stream_intermediate_steps=True,
      telemetry=False,
      debug_mode=False,
    )

    cl.user_session.set("agent", agent)
    cl.user_session.set("agent_reasoning", agent_reasoning)

@cl.on_message
async def on_message(msg: cl.Message):
    # Process message with or without explicit search command
    if msg.command == "Reasoning":
        agent:Agent = cl.user_session.get("agent_reasoning")
    else:
        agent:Agent = cl.user_session.get("agent")
    
    # agent:Agent = cl.user_session.get("agent")
    
    message = cl.Message(content="")
    user_query = msg.content 
    run_start_step = None

    for response in await cl.make_async(agent.run)(user_query, stream=True):
        # if response.event != RunEvent.run_response:
        #     print(response.event,"----",response)
        if response.event != RunEvent.run_response and run_start_step:
            await run_start_step.remove()
        if response.event == RunEvent.run_response:
            await message.stream_token(response.content)
        elif response.event == RunEvent.tool_call_started:
            for tool in response.tools:
              # 通过id屏蔽重复项的出现
              async with cl.Step(name=tool.tool_name,id=tool.tool_call_id) as tool_call_step:
                  tool_args_str = json.dumps(tool.tool_args, indent=2, ensure_ascii=False)
                  tool_call_step.input = f"Tool Args: {tool_args_str}"
                      
        elif response.event == RunEvent.reasoning_step:
            # name = response.event+f":{response.content.title}" # 使用动态name会有繁忙图标问题
            async with cl.Step(name='reasoning',default_open=False) as reasoning_step:
                # reasoning_step.output = response.reasoning_content
                reasoning_step.input = json.dumps({"title":response.content.title,"action":response.content.action}, indent=2, ensure_ascii=False)
                reasoning_step.output = response.reasoning_content
                # await reasoning_step.stream_token(response.reasoning_content)
                # await reasoning_step.update()
        elif response.event == RunEvent.run_started:
            async with cl.Step(name="合同查询 Agent 开始执行...") as run_start_step:
                pass
        else:
            pass
            
    await message.send()

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)