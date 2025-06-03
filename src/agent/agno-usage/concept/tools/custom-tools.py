from concurrent.futures import ThreadPoolExecutor
from agno.agent import Agent
from agno.tools import tool
from agno.models.openai import OpenAILike
from agno.tools.hackernews import HackerNewsTools
import json
import httpx
from typing import Any,Callable,Dict

#------------------ settings ------------------
import os
from dotenv import load_dotenv
load_dotenv()
api_key=os.getenv("QWEN_API_KEY")
base_url=os.getenv("QWEN_API_BASE_URL")
local_base_url = 'http://192.168.0.166:8000/v1'
local_model_name = 'Qwen3-235B'
model_name = 'qwen-plus-latest'

local_settings = {
  'api_key' : '123',
  'base_url' : local_base_url,
  'id' : local_model_name
}

qwen_settings = {
  'api_key' : api_key,
  'base_url' : base_url,
  'id' : model_name
}

settings = local_settings
#------------------ settings ------------------

# 使用了hooks就必须显式的调用function，然后返回结果，否则不会执行tool
def logger_hook(function_name:str,function_call:Callable,arguments:Dict[str,Any]):
    """Hooks function that wraps the tool execution"""
    print(f"Calling {function_name} with arguments {arguments}")
    result = function_call(**arguments)
    print(f"Result of {function_name}: {result}")
    return result

@tool(
    name='fetch_hacker_news_stories',     # Custom tool name (otherwise the function name is used)
    description='Get the top stories from Hacker News.',    # Custom tool description (otherwise the function docstring is used)
    show_result=False,    # Show the result of the tool in the console
    tool_hooks=[logger_hook],
    cache_results=True,    # Cache the result of the tool for faster access
    cache_dir='/tmp/agents_cache',
    cache_ttl=3600    # Cache the result for 1 hour
)
def get_top_hacker_news_stories(num_stories=10)->str:
    """
    Get the top stories from Hacker News.
    
    Args:
        num_stories (int): Number of stories to return.
        
    Returns:
        str: JSON string of the top stories.
    """
    
    # Fetch the top stories ID
    
    response = httpx.get('https://hacker-news.firebaseio.com/v0/topstories.json')
    story_ids = response.json()
    
    # Fetch story details
    
    stories = []
    
    for story_id in story_ids[:num_stories]:
        story_response = httpx.get(f'https://hacker-news.firebaseio.com/v0/item/{story_id}.json')
        story = story_response.json()
        if 'text' in story:
            story.pop('text')
        stories.append(story)
    
    return json.dumps(stories)

@tool(
    name='fetch_hacker_news_stories',
    cache_results=True,    # Cache the result of the tool for faster access
    cache_dir='/tmp/agents_cache',
    cache_ttl=3600    # Cache the result for 1 hour
)
def get_top_hacker_news_stories_concurrent(num_stories: int = 5) -> str:
    """
    Get the top stories from Hacker News.
    
    Args:
        num_stories (int): Number of stories to return.
        
    Returns:
        str: JSON string of the top stories.
    """
    try:
        response = httpx.get(
            'https://hacker-news.firebaseio.com/v0/topstories.json',
            timeout=10
        )
        response.raise_for_status()
        story_ids = response.json()
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        return json.dumps({"error": f"Top stories fetch failed: {str(e)}"})
    
    if not isinstance(story_ids, list):
        return json.dumps({"error": "Invalid API response format"})
    
    num_stories = min(num_stories, len(story_ids))
    
    # 并发获取故事详情
    def fetch_story(story_id: int) -> Dict[str, Any]:
        try:
            response = httpx.get(
                f'https://hacker-news.firebaseio.com/v0/item/{story_id}.json',
                timeout=10
            )
            response.raise_for_status()
            story = response.json()
            story.pop('text', None)  # 安全移除text字段
            return story
        except (httpx.RequestError, json.JSONDecodeError) as e:
            return None
    
    with ThreadPoolExecutor() as executor:
        stories = list(executor.map(fetch_story, story_ids[:num_stories]))
    
    # 过滤失败结果
    valid_stories = [s for s in stories if s is not None]
    
    return json.dumps({
        "count": len(valid_stories),
        "stories": valid_stories
    })

agent = Agent(model=OpenAILike(**settings),markdown=True,tools=[get_top_hacker_news_stories_concurrent],telemetry=False)
agent.print_response(message = 'Show me the top stories from Hacker News')
        
# if __name__=='__main__':
#     print(get_top_hacker_news_stories(5))
        
    
    

