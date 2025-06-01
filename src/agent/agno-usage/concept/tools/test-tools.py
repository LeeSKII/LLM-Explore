from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from agno.tools.reasoning import ReasoningTools
from agno.tools import tool
from agno.tools.yfinance import YFinanceTools
import logging

# logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s:%(filename)s:%(lineno)d - %(message)s',datefmt='%Y-%m-%d %H:%M:%S',level=logging.DEBUG) 

#------------------ settings ------------------
import os
from dotenv import load_dotenv
load_dotenv()
api_key=os.getenv("QWEN_API_KEY")
base_url=os.getenv("QWEN_API_BASE_URL")
local_base_url = 'http://192.168.0.166:8000/v1'
local_model_name = 'Qwen3-235B'
model_name = 'qwen-plus-latest'

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

settings = qwen_settings
#------------------ settings ------------------

agent = Agent(
    model=OpenAILike(**settings),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    show_tool_calls=True,
    description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
    instructions=["Format your response using markdown and use tables to display data where possible."],
    debug_mode=True,
    telemetry=False
)
agent.print_response("Share the NVDA stock price and analyst recommendations", markdown=True)