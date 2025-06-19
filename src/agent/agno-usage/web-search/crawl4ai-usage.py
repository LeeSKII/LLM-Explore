from typing import Iterator
from agno.agent import Agent, RunResponse
from agno.models.openai.like import OpenAILike
from agno.utils.pprint import pprint_run_response
from agno.tools.crawl4ai import Crawl4aiTools

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

agent = Agent(model=OpenAILike(**settings),tools=[Crawl4aiTools(max_length=None)], show_tool_calls=True,telemetry=False)

agent.print_response("研读并分析 https://news.mysteel.com/a/25061707/BE68EFBE12136264.html")