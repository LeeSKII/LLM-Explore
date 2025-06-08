from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from agno.tools.reasoning import ReasoningTools
from agno.tools import tool
import random
import logging
from agno.playground import Playground, serve_playground_app

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

# 大多数情况下编写定义工具，通过tool decorator装饰器装饰
# @tool(show_result=True, stop_after_tool_call=True)
@tool()
def get_weather(city:str)->str:
    '''Get the weather of a city.'''
    weather_condition = ['Sunny', 'Cloudy', 'Rainy', 'Snowy']
    return random.choice(weather_condition)
    

# reasoning tools 提供了think 和 analyze 两个tool，显式设置add_instructions=True，会在system_prompt中添加框架自带的这两个tools使用的instruction指令
# 相当于教模型如何去reasoning
# 如果再设置add_few_shot=True，则会在system_prompt中添加框架自带的few_shot_example到system_prompt中
agent = Agent(model=OpenAILike(**settings),telemetry=False,tools=[get_weather,ReasoningTools(add_instructions=True,add_few_shot=True)],markdown=True,debug_mode=True)

# if __name__ == '__main__':
#     # logging.info('Starting agent...')
#     agent.print_response(message='今天长沙适合什么活动',stream=True,show_full_reasoning=True,stream_intermediate_steps=True)

app = Playground(agents=[agent]).get_app()
if __name__ == "__main__":
    serve_playground_app("1_tools_instructions:app", reload=True)