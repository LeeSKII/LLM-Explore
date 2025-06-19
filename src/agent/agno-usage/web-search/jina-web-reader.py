from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from agno.tools.jina import JinaReaderTools

import os
from dotenv import load_dotenv
load_dotenv()
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

agent = Agent(model=OpenAILike(**settings),tools=[JinaReaderTools(max_content_length=30000)], show_tool_calls=True,debug_mode=True,markdown=True,telemetry=False)

agent.print_response("研读并分析这个网站的内容 https://news.mysteel.com/a/25061807/9BDBECC1EFB7B600.html")