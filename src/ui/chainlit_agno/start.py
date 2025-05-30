import os
import time
from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from openai import AsyncOpenAI
from dotenv import load_dotenv
from regex import F
load_dotenv()
import chainlit as cl

api_key=os.getenv("QWEN_API_KEY")
base_url=os.getenv("QWEN_API_BASE_URL")
local_base_url = 'http://192.168.0.166:8000/v1'
local_model_name = 'Qwen3-235B'
model_name = 'qwen3-235b-a22b'

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

client = AsyncOpenAI(
    api_key='123', base_url="http://192.168.0.166:8000/v1",
)

@cl.on_chat_start
def init_agent():
    # 需要关闭telemetry，否则会有额外的post请求到agno官网,造成延迟
    agent = Agent(model=OpenAILike(**settings),markdown=True,telemetry=False)
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