import asyncio
from crawl4ai import AsyncWebCrawler,CrawlerRunConfig

async def main():
    config = CrawlerRunConfig(
        process_iframes=True, 
        # wait_for="js:() => document.querySelector('#viewer') !== null",
        # css_selector="#viewer"
        # wait_for="#body .col-lg-8",
        css_selector=".col-lg-8"
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun("https://www.custeel.com/reform/view.mv?articleID=7995625&group=1001&cat=1001026",config=config)
        print(result.markdown)  # Print first 300 chars

if __name__ == "__main__":
    asyncio.run(main())
