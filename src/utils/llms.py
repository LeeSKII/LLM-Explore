import os
from dotenv import load_dotenv
from typing import List
from openai import OpenAI

load_dotenv()
api_key = os.getenv("API_KEY")
base_url = os.getenv("API_BASE_URL")


def chat_with_llm(message:List,api_key:str,base_url:str,model_name:str='google/gemini-2.5-pro-exp-03-25:free'):
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=message
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f'Error: {e}'