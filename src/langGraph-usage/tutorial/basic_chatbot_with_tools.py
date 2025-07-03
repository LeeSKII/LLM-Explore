from turtle import st
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage,HumanMessage
import json

load_dotenv()

api_key = os.getenv("QWEN_API_KEY")
base_url = os.getenv("QWEN_API_BASE_URL")

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

class State(TypedDict):
    messages:Annotated[list,add_messages]
    
graph_builder = StateGraph(State)

llm = ChatOpenAI(model="qwen-plus-latest",api_key=api_key,base_url=base_url,temperature=0.01)

llm_with_tools = llm.bind_tools([multiply])

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

def chat_bot_node(state:State)->State:
    message = llm_with_tools.invoke(state["messages"])
    return {"messages": [message]}

tool_node = BasicToolNode(tools=[multiply])

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

result_node = graph.invoke({"messages": [HumanMessage(content="3 * 34是多少")]})

print(result_node['messages'][-1])
