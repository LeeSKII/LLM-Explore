import chainlit as cl
import os
from litellm import acompletion

api_key = os.getenv("GEMINI_API_KEY")
settings = {
    'api_key':api_key,
    "model": "gemini/gemini-2.0-flash",
    "temperature": 0.7,
    'max_retries':3
}

@cl.on_chat_start
def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )

@cl.on_message
async def on_message(message:cl.Message):
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")
    await msg.send() # 先发送一个空消息占位

    full_response_content = "" # 用于累积完整的响应文本

    try:
        stream = await acompletion(
            messages=message_history,
            stream=True,
            **settings
        )

        async for part in stream:
            if token := part.choices[0].delta.content or "":
                full_response_content += token
                await msg.stream_token(token)
    except Exception as e:
        error_message = f"调用 LiteLLM 时出错: {str(e)}"
        await cl.Message(content=error_message).send()
        # 如果出错，从历史记录中移除最后一条用户消息，避免污染后续对话
        if message_history and message_history[-1]["role"] == "user":
            message_history.pop()
        cl.user_session.set("message_history", message_history)
        return

    # 将助手的完整响应添加到历史记录
    if full_response_content:
        message_history.append({"role": "assistant", "content": full_response_content})
    
    cl.user_session.set("message_history", message_history)