import chainlit as cl
import os
from litellm import acompletion
from chainlit.input_widget import Select, Switch, Slider

api_key = os.getenv("GEMINI_API_KEY")
settings = {
    "model": "gemini/gemini-2.0-flash",
    "temperature": 0.7,
    'max_retries':3
}

@cl.on_chat_start
async def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )
    cl.user_session.set("settings", settings)
    await cl.ChatSettings(
        [
          Select(
                id="model",
                label="AI Model",
                values=["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-pro", "gemini-1.5-flash","gemini-1.5-flash-8b"],
                initial_index=0,
            ),
            # Switch(id="Streaming", label="OpenAI - Stream Tokens", initial=True),
            Slider(
                id="temperature",
                label="Temperature",
                initial=1,
                min=0,
                max=2,
                step=0.1,
            ),    
        ]
      ).send()
    
@cl.on_settings_update
async def setup_agent(settings):
    settings['model'] = f'gemini/{settings["model"]}'
    cl.user_session.set("settings", settings)
    print(settings)

@cl.on_message
async def on_message(message:cl.Message):
    message_history = cl.user_session.get("message_history")
    settings = cl.user_session.get("settings")

    message_history.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")

    try:
        stream = await acompletion(
            api_key=api_key,
            messages=message_history,
            stream=True,
            **settings
        )

        async for part in stream:
            if token := part.choices[0].delta.content or "":
                await msg.stream_token(token)
        message_history.append({"role": "assistant", "content": msg.content})
        await msg.update()
    except Exception as e:
        await msg.stream_token(f"Error: {str(e)}")
        await msg.update()
        raise