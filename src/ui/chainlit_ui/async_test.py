import time
import chainlit as cl
from agno.agent import Agent
from agno.models.openai.like import OpenAILike
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

temperature = 0.01
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

def sync_func():
    time.sleep(35)
    return "Hello!"

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
                "no_think"
                ]

def process_func(query):
    result = sync_func()
    return result

def search_knowledge_base(query):
    result = process_func(query)
    return result

@cl.on_message
async def main(msg: cl.Message):
    agent = Agent(
        model=OpenAILike(**settings,temperature=temperature),
        name='Contact_Query_Agent',
        instructions=instructions,
        tools=[search_knowledge_base],
        add_history_to_messages=True,
        num_history_responses=20,
        markdown=True,
        stream=True,
        stream_intermediate_steps=True,
        telemetry=False,
        debug_mode=True,
      )
    
    message = cl.Message(content="")
    user_query = msg.content 
    print(user_query)
    async for chunk in await agent.arun(user_query):  # 遍历异步生成器
        print("收到数据块:", chunk)
        if chunk.event == 'RunResponseContent':
            await message.stream_token(chunk.content)
        # await message.stream_token(chunk)  # 逐步发送到前端
    
    print("流式传输完成")
    await message.send()

    # for response in await cl.make_async(agent.run)(user_query, stream=True):
    #     if response.event == 'RunResponseContent':
    #         await message.stream_token(response.content)          
    # await message.send()
    
    # answer = await cl.make_async(sync_func)()
    # await cl.Message(
    #     content=answer,
    # ).send()