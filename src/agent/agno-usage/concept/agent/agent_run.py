from typing import Iterator
from agno.agent import Agent, RunResponse
from agno.models.openai.like import OpenAILike
from agno.utils.pprint import pprint_run_response

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

agent = Agent(model=OpenAILike(**settings))


# Run agent and return the response as a variable
response: RunResponse = agent.run("Tell me a 5 second short story about a robot")
# Run agent and return the response as a stream
response_stream: Iterator[RunResponse] = agent.run("Tell me a 5 second short story about a lion", stream=True)

# Print the response in markdown format
pprint_run_response(response, markdown=True)
# Print the response stream in markdown format
pprint_run_response(response_stream, markdown=True)