import asyncio
from crawl4ai import AsyncWebCrawler,CrawlerRunConfig

async def main():
    config = CrawlerRunConfig(
        process_iframes=True, 
        # wait_for="js:() => document.querySelector('#viewer') !== null",
        # css_selector="#viewer"
        css_selector="#article_content"
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun("https://www.chinaisa.org.cn/gxportal/xfgl/portal/content.html?articleId=6a34293c4c03ffffd003eed1b89d775a354bf84b4731a712a34c60c61f0a8bd9&columnId=ae2a3c0fd4936acf75f4aab6fadd08bc6371aa65bdd50419e74b70d6f043c473",config=config)
        print(result.markdown)  # Print first 300 chars

if __name__ == "__main__":
    asyncio.run(main())
