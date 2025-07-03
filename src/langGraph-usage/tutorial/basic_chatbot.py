from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("QWEN_API_KEY")
base_url = os.getenv("QWEN_API_BASE_URL")

class State(TypedDict):
    messages:Annotated[list,add_messages]
    
graph_builder = StateGraph(State)

llm = ChatOpenAI(model="qwen-plus-latest",api_key=api_key,base_url=base_url,temperature=0.01)

def chat_bot_node(state:State)->State:
    message = llm.invoke(state["messages"])
    return {"messages": [message]}

graph_builder.add_node("chat_bot_node",chat_bot_node)
graph_builder.add_edge(START,"chat_bot_node")
graph_builder.add_edge("chat_bot_node",END)

graph = graph_builder.compile()

result_node = graph.invoke({"messages": ["hi"]})

print(result_node)
