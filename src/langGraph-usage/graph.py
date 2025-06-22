from stat import filemode
from langgraph.graph import StateGraph,START,END
import os
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from dotenv import load_dotenv

import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s:%(filename)s:%(lineno)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                      logging.FileHandler("langgraph.log",mode='a', encoding='utf-8'),  # 关键：指定 UTF-8
                      logging.StreamHandler()  # 可选：同时输出到控制台
                    ]
) 

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_API_BASE_URL")

llm = ChatOpenAI(api_key=api_key, base_url=base_url,model="deepseek-chat",max_retries=3,temperature=0.01)

class State(TypedDict):
    topic:str
    outline:str
    paper:str

# Nodes

def outline_node(state:State):
    logging.info(f"Generating outline for {state['topic']}")
    prompt = PromptTemplate.from_template("根据{topic}，生成一篇文章的大纲。").format(topic=state['topic'])
    msg = llm.invoke(prompt)
    logging.info(f"Generated outline: {msg.content}")
    return {'outline':msg.content}

def write_node(state:State):
    logging.info(f"Writing paper for {state['topic']}")
    prompt = PromptTemplate.from_template("请用{outline}来写一篇{topic}的文章。").format(outline=state['outline'],topic=state['topic'])
    msg = llm.invoke(prompt)
    logging.info(f"Generated paper: {msg.content}")
    return {'paper':msg.content}

# Build Graph

workflow = StateGraph(State)

workflow.add_node("outline_node",outline_node)
workflow.add_node("write_node",write_node)

workflow.add_edge(START,"outline_node")
workflow.add_edge("outline_node","write_node")
workflow.add_edge("write_node",END)

graph = workflow.compile()

# Run Graph

init_state = {'topic':"如何使用LLM根据每周搜集到的100篇行业报告文本汇总成每周行业动态?"}

state = graph.invoke(init_state)

print(state)