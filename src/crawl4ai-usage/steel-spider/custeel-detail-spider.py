import asyncio
from crawl4ai import AsyncWebCrawler,CrawlerRunConfig
import pandas as pd
import os
from pydantic import BaseModel,Field

class Web_link(BaseModel):
    title:str = Field(description="链接标题")
    link: str = Field(
        description="网页超链接"
    )
    date: str = Field(None,description="链接发布日期")

async def main():
    df = pd.read_csv(r'D:\projects\LLM-Explore\src\crawl4ai-usage\steel-spider\link_collection.csv')
    # 转换为Pydantic对象列表
    list_web_link = [Web_link(**row) for row in df.to_dict(orient='records')]
    for web_link in list_web_link:
        result = await crawler(web_link.link)
        print(result["content"][:100])
        result_df = pd.DataFrame([{'title': web_link.title,'url': web_link.link, 'content': result['content']}])
        file_path = 'content.csv'
        # 判断文件是否已存在
        file_exists = os.path.isfile(file_path)
        result_df.to_csv('content.csv', index=False,mode='a',header=not file_exists, encoding='utf-8')

async def crawler(url):
    config = CrawlerRunConfig(
        wait_for="#main_c",
        css_selector="#main_c"
    )
    async with AsyncWebCrawler() as crawler:
        try:
            result = await crawler.arun(url,config=config)
            content = result.markdown if hasattr(result, 'markdown') else ''
            return {'url': url, 'content': content}
        except Exception as e:
            print(f"Error: url={url}, error={e}")
            return {'url': url, 'content': f'Error: {e}'}

if __name__ == "__main__":
    asyncio.run(main())
