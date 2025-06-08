from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from dotenv import load_dotenv
import os

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
    'id' : 'deepseek-reasoner'
}

settings = deepseek_settings

agent = Agent(model=OpenAILike(**settings))
agent.print_response(
    "今天是礼拜三，再过10天是星期几，一步步思考",
    stream=True,
)