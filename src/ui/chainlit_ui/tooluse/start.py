import os
import time
import datetime
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()
import chainlit as cl

currentDateTime = datetime.datetime.now().strftime("%Y-%m-%d")

system_prompt = f'''The assistant is K.The current date is {currentDateTime}.
'''

client = AsyncOpenAI(
    api_key=os.getenv("QWEN_API_KEY"), base_url=os.getenv("QWEN_API_BASE_URL"),
)


@cl.on_message
async def on_message(msg: cl.Message):
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