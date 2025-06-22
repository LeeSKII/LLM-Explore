from langgraph.graph import StateGraph,START,END
import os
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_API_BASE_URL")

llm = ChatOpenAI(api_key=api_key, base_url=base_url,model="deepseek-chat",max_retries=3,temperature=0.01)

prompt_template = PromptTemplate.from_template("hello,{name}")

prompt = prompt_template.format(name="world")

print(prompt)