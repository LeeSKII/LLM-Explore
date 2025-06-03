from agno.agent import Agent
from agno.app.fastapi.app import FastAPIApp
from agno.app.fastapi.serve import serve_fastapi_app
from agno.models.openai import OpenAILike

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

settings = local_settings
#------------------ settings ------------------

agent = Agent(model=OpenAILike(**settings),add_datetime_to_instructions=True,telemetry=False)

# Async router by default (use_async=True)
app = FastAPIApp(agent=agent).get_app()


# 使用post请求端口的v1/run方法，参数通过form表单提交,必填参数为message
if __name__ == "__main__":
    # 使用当前的脚本名称.
    serve_fastapi_app("fastapi-usage:app", port=8001, reload=True)