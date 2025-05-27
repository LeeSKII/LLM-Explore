import json
import os
import time
import datetime
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()
import chainlit as cl
from mcp import ClientSession

currentDateTime = datetime.datetime.now().strftime("%Y-%m-%d")

system_prompt = f'''You are a helpful assistant.The current date is {currentDateTime}.
'''
regular_tools = [{
    "name": "show_linear_ticket",
    "description": "Displays a Linear ticket in the UI with its details. Use this tool after retrieving ticket information to show a visual representation of the ticket. The tool will create a card showing the ticket title, status, assignee, deadline, and tags. This provides a cleaner presentation than text-only responses.",
    "input_schema": {
        "type": "object",
        "properties": {"title": {"type": "string"}, "status": {"type": "string"}, "assignee": {"type": "string"}, "deadline": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}},
        "required": ["title", "status", "assignee", "deadline", "tags"]
    }
}]

async def show_linear_ticket(title, status, assignee, deadline, tags):
    props = {
        "title": title,
        "status": status,
        "assignee": assignee,
        "deadline": deadline,
        "tags": tags
    }
    print("props", props)
    ticket_element = cl.CustomElement(name="LinearTicket", props=props)
    await cl.Message(content="", elements=[ticket_element], author="show_linear_ticket").send()
    return "the ticket was displayed to the user: " + str(props)

def flatten(xss):
    return [x for xs in xss for x in xs]

client = AsyncOpenAI(
    api_key=os.getenv("QWEN_API_KEY"), base_url=os.getenv("QWEN_API_BASE_URL"),
)

@cl.on_mcp_connect
async def on_mcp(connection, session: ClientSession):
    # List available tools
    result = await session.list_tools()
    
    # Process tool metadata
    tools = [{
        "name": t.name,
        "description": t.description,
        "input_schema": t.inputSchema,
    } for t in result.tools]
    
    # Store tools for later use
    mcp_tools = cl.user_session.get("mcp_tools", {})
    mcp_tools[connection.name] = tools
    cl.user_session.set("mcp_tools", mcp_tools)
    
@cl.step(type="tool") 
async def call_tool(tool_use):
    tool_name = tool_use.name
    tool_input = tool_use.input
    
    current_step = cl.context.current_step
    current_step.name = tool_name
    
    # Identify which mcp is used
    mcp_tools = cl.user_session.get("mcp_tools", {})
    mcp_name = None

    for connection_name, tools in mcp_tools.items():
        if any(tool.get("name") == tool_name for tool in tools):
            mcp_name = connection_name
            break
    
    if not mcp_name:
        current_step.output = json.dumps({"error": f"Tool {tool_name} not found in any MCP connection"})
        return current_step.output
    
    mcp_session, _ = cl.context.session.mcp_sessions.get(mcp_name)
    
    if not mcp_session:
        current_step.output = json.dumps({"error": f"MCP {mcp_name} not found in any MCP connection"})
        return current_step.output
    
    try:
        current_step.output = await mcp_session.call_tool(tool_name, tool_input)
    except Exception as e:
        current_step.output = json.dumps({"error": str(e)})
    
    return current_step.output

@cl.on_chat_start
async def start_chat():
    cl.user_session.set("chat_messages", [])
    cl.user_session.set("regular_tools", regular_tools)

@cl.on_message
async def on_message(msg: cl.Message):
    mcp_tools = cl.user_session.get("mcp_tools", {})
    regular_tools = cl.user_session.get("regular_tools", [])
    # Flatten the tools from all MCP connections
    tools = flatten([tools for _, tools in mcp_tools.items()]) + regular_tools
    print([tool.get("name") for tool in tools])
    start = time.time()
    stream = await client.chat.completions.create(
        model="qwen3-235b-a22b",
        messages=[
            {"role": "system", "content": system_prompt},
            *cl.chat_context.to_openai(),
        ],
        stream=True,
    )

    # Flag to track if we've exited the thinking step
    thinking_completed = False

    # Streaming the thinking
    async with cl.Step(name="Thinking") as thinking_step:
        async for chunk in stream:
            delta = chunk.choices[0].delta
            reasoning_content = getattr(delta, "reasoning_content", None)
            # 注意由于VLLM本地部署的模型第一个chunk出现了reasoning_content丢失的bug，所以这里需要判断reasoning_content是否为空，同时使用content判断是否为空指示是否还在思考中
            content = getattr(delta, "content", None)
            if not reasoning_content and not content:
                continue 
            # end of fix for VLLM bug
            if reasoning_content is not None and not thinking_completed:
                await thinking_step.stream_token(reasoning_content)
            elif not thinking_completed:
                # Exit the thinking step
                thought_for = round(time.time() - start)
                thinking_step.name = f"Thought for {thought_for}s"
                await thinking_step.update()
                thinking_completed = True
                break

    final_answer = cl.Message(content="")

    # Streaming the final answer
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            await final_answer.stream_token(delta.content)

    await final_answer.send()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)