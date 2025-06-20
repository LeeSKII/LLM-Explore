import asyncio
from crawl4ai import AsyncWebCrawler,CrawlerRunConfig
from pydantic import BaseModel,Field
import os
from openai import OpenAI
from dotenv import load_dotenv
import time
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

settings = qwen_settings


class Web_link(BaseModel):
    title:str = Field(description="新闻|信息标题")
    link: str = Field(
        description="网页超链接"
    )
    date: str = Field(None,description="链接发布日期")

class Link_Collection(BaseModel):
    collections:list[Web_link] = Field(None,description="新闻集合")

client: OpenAI = OpenAI(
        api_key="EMPTY",
        base_url=local_base_url,
)

current_date = time.strftime("%Y-%m-%d", time.localtime())

def guided_link_json_completion(client: OpenAI,prompt:str, model="Qwen3-235B"):
    json_schema = Link_Collection.model_json_schema()
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": f"今天是:{current_date},提取网页中七天内发布的新闻标题和链接,不要遗漏任何新闻，如果没有7天内的新闻,请返回空collections对象,请按格式输出。",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        extra_body={"guided_json": json_schema,"chat_template_kwargs": {"enable_thinking": False},},
    )
    return completion.choices[0].message.reasoning_content



async def main():
    config = CrawlerRunConfig(
        scan_full_page=True,       # 滚动整个页面触发懒加载
        scroll_delay=0.5,           # 每次滚动后暂停0.5秒
        # wait_for="body > div.w.clear.list-page > div.right-b > div" ,
        # css_selector="body > div.w.clear.list-page > div.right-b > div"  
    )
    async with AsyncWebCrawler() as crawler:
        url = "https://www.chinaisa.org.cn/gxportal/xfgl/portal/list.html?columnId=c42511ce3f868a515b49668dd250290c80d4dc8930c7e455d0e6e14b8033eae2"
        url= "https://www.chinaisa.org.cn/gxportal/xfgl/portal/list.html?columnId=3238889ba0fa3aabcf28f40e537d440916a361c9170a4054f9fc43517cb58c1e"
        url = "https://www.chinaisa.org.cn/gxportal/xfgl/portal/list.html?columnId=ae2a3c0fd4936acf75f4aab6fadd08bc6371aa65bdd50419e74b70d6f043c473"
        url = "https://www.custeel.com/s1001/more.jsp?group=1001&cat=1001026"
        url="https://www.custeel.com/caijing/more.jsp?topic=425"
        result = await crawler.arun(url, config=config)
        print(result.markdown)  # Print first 300 chars
        prompt = result.markdown
        response = guided_link_json_completion(client,prompt)
        print(response)
        

if __name__ == "__main__":
    asyncio.run(main())
