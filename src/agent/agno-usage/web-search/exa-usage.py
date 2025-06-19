import os
from dotenv import load_dotenv
from agno.agent import Agent, RunResponse
from agno.tools.reasoning import ReasoningTools
from agno.models.openai.like import OpenAILike
from agno.tools.exa import ExaTools

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

agent = Agent(name='web-search-agent',
              model=OpenAILike(**qwen_settings),
              tools=[ReasoningTools(add_instructions=True,add_few_shot=True),ExaTools(num_results=10,text_length_limit=3000)],
              description="You are a brilliant analytical. You can find the latest news, weather reports, and stock prices of any company or industry. Just ask anything related to the web search domain." ,
              success_criteria="don't miss any query, hard find the resources you need, and provide accurate information.",
              goal='Use Simplify Chinese to response.',
              add_datetime_to_instructions=True,
              markdown=True,
              debug_mode=True,telemetry=False)
agent.print_response("钢铁冶金行业各种矿石的价格，例如，铁、钼等技术经济指标评价分析",stream=True)