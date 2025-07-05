from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage,HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
import json

load_dotenv()

api_key = os.getenv("QWEN_API_KEY")
base_url = os.getenv("QWEN_API_BASE_URL")

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@tool
def sum(a: int, b: int) -> int:
    """Sum two numbers."""
    return a + b

tools_list = [multiply,sum]

# 使用prebuilt的tool node，state中必须有键名为messages，并且使用add_messages reducer函数处理该消息
class State(TypedDict):
    messages:Annotated[list,add_messages]
    
graph_builder = StateGraph(State)

llm = ChatOpenAI(model="qwen-plus-latest",api_key=api_key,base_url=base_url,temperature=0.01)

llm_with_tools = llm.bind_tools(tools_list)


def chat_bot_node(state:State)->State:
    message = llm_with_tools.invoke(state["messages"])
    return {"messages": [message]}

# tool_node的输入是messages，其中最后一个信息是AIMessage，并且包含了tool_calls，输出则是append了ToolMessage的messages
tool_node = ToolNode(tools=tools_list)

def route_tools(state:State):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    if hasattr(ai_message,'tool_calls') and len(ai_message.tool_calls) > 0:
        return "call_tool"
    else:
        return END

graph_builder.add_node("chat_bot_node",chat_bot_node)
graph_builder.add_node("call_tool",tool_node)
graph_builder.add_edge(START,"chat_bot_node")
graph_builder.add_edge("call_tool","chat_bot_node")
graph_builder.add_conditional_edges("chat_bot_node",route_tools,{"call_tool":"call_tool",END:END})

graph = graph_builder.compile()

result_node = graph.invoke({"messages": [HumanMessage(content="3 * 34的结果再加上45是多少")]})

for message in result_node['messages']:
    print(message)
