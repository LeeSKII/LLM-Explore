import os
from dotenv import load_dotenv
from agno.agent import Agent, RunResponse
from agno.models.openai.like import OpenAILike

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

agent = Agent(model=OpenAILike(**local_settings), markdown=True)
agent.print_response("如何身无分文在1年内挣到100W.",stream=True)