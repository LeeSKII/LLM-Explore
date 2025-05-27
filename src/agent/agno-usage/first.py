import os
from dotenv import load_dotenv
from agno.agent import Agent, RunResponse
from agno.models.openai.like import OpenAILike

load_dotenv()
api_key=os.getenv("QWEN_API_KEY")
base_url=os.getenv("QWEN_API_BASE_URL")
model_name = 'qwen3-235b-a22b'


agent = Agent(model=OpenAILike(id=model_name,base_url=base_url,api_key=api_key), markdown=True)
agent.print_response("如何身无分文在1年内挣到100W.",stream=True)