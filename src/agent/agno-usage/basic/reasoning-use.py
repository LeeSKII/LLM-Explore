import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from agno.tools.yfinance import YFinanceTools
from agno.tools.reasoning import ReasoningTools
from agno.tools.tavily import TavilyTools

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

agent = Agent(model=OpenAILike(**local_settings),tools=[ReasoningTools(add_instructions=True),TavilyTools()],instructions=[
         "Use tables to display data.",
        "Include sources in your response.",
        "Only include the report in your response. No other text.",
    ], markdown=True,show_tool_calls=True)
agent.print_response("写一篇关于英伟达的报告，需要引用最近5年的数据，尤其是关注2025年和2024年.",stream=True,show_full_reasoning=True,stream_intermediate_steps=True,)