import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from agno.tools.yfinance import YFinanceTools

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

agent = Agent(model=OpenAILike(**local_settings),tools=[YFinanceTools(stock_price=True)],instructions=[
        "Use tables to display data.",
        "Only include the table in your response. No other text.",
    ], markdown=True)
agent.print_response("今天中国五矿的股票价格.",stream=True)