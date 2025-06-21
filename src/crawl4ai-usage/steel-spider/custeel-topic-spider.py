import asyncio
from crawl4ai import AsyncWebCrawler,CrawlerRunConfig
import pandas as pd

async def main():
    config = CrawlerRunConfig(
        css_selector=".box-2right"
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun("https://www.custeel.com/caijing/news-jj.shtml",config=config)
        print(result.markdown)

if __name__ == "__main__":
    asyncio.run(main())
