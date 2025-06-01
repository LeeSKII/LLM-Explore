from textwrap import dedent
from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from agno.tools.reasoning import ReasoningTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools import tool
import finnhub
import logging

# logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s:%(filename)s:%(lineno)d - %(message)s',datefmt='%Y-%m-%d %H:%M:%S',level=logging.DEBUG) 

#------------------ settings ------------------
import os
from dotenv import load_dotenv
from sqlalchemy import desc
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

deepseek_settings = {
    'api_key' : os.getenv("DEEPSEEK_API_KEY"),
    'base_url' : os.getenv("DEEPSEEK_API_BASE_URL"),
    'id' : 'deepseek-chat'
}

settings = deepseek_settings
#------------------ settings ------------------
FINNHUB_API_KEY = 'd0tqkcpr01qlvahdqd00d0tqkcpr01qlvahdqd0g'

#------------------ settings ------------------

finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

@tool()
def get_stock_symbol(query:str):
    """
    Search for best-matching symbols based on your query. You can input anything from symbol, security's name to ISIN and Cusip.

    Args:
        query: Query text can be symbol, name, isin, or cusip.

    Returns:
        object: Stock symbol information.
    """
    return finnhub_client.symbol_lookup(query)

@tool()
def get_company_profile(symbol:str):
    '''
    Get general information of a company. You can query by symbol, ISIN or CUSIP.
    
    Args:
        symbol: Symbol of the company: AAPL e.g.

    Returns:
        object: Company profile information.
    '''
    return finnhub_client.company_profile2(symbol=symbol)

@tool()
def get_market_news():
    '''
    Get latest market news.
    
    Returns:
        object: Market news information.
    '''
    return finnhub_client.general_news('general', min_id=0)

@tool()
def get_company_news(symbol:str, _from:str, to:str):
    '''
    List latest company news by symbol. 
    
    Args:
        symbol: Symbol of the company: AAPL e.g.
        _from: From date in the format yyyy-MM-dd.
        to: To date in the format yyyy-MM-dd.
        
    Returns:
        object: Company news information.
    '''
    return finnhub_client.company_news(symbol=symbol, _from=_from, to=to)

@tool()
def get_company_peers(symbol:str):
    '''
    Get company peers. Return a list of peers operating in the same country and sector/industry.
    
    Args:
        symbol: Symbol of the company: AAPL e.g.
        
    Returns:
        object: Company peers information.
    '''
    return finnhub_client.company_peers(symbol=symbol)

@tool()
def get_company_financials(symbol:str):
    '''
    Get company basic financials such as margin, P/E ratio, 52-week high/low etc.
    
    Args:
        symbol: Symbol of the company: AAPL e.g.
        
    Returns:
        object: Company financials information.
    '''
    res = finnhub_client.company_basic_financials(symbol=symbol, metric='all')
    return res['metric']

@tool()
def get_stock_insider_transactions(symbol:str, _from:str, to:str):
    '''
    Get company insider transactions data.
    
    Args:
        symbol: Symbol of the company: AAPL e.g.
        _from: From date in the format yyyy-MM-dd.
        to: To date in the format yyyy-MM-dd.
        
    Returns:
        object: Stock insider transactions information.
    '''
    return finnhub_client.stock_insider_transactions(symbol=symbol, _from=_from, to=to)

@tool()
def get_stock_insider_sentiment(symbol:str, _from:str, to:str):
    '''
    Get insider sentiment data for US companies. The MSPR ranges from -100 for the most negative to 100 for the most positive which can signal price changes in the coming 30-90 days.
    
    Args:
        symbol: Symbol of the company: AAPL e.g.
        _from: From date in the format yyyy-MM-dd.
        to: To date in the format yyyy-MM-dd.
        
    Returns:
        object: Stock insider sentiment information.
    '''
    return finnhub_client.stock_insider_sentiment(symbol=symbol, _from=_from, to=to)

@tool()
def get_financials_reported(symbol:str, _from:str, to:str,freq:str='annual'):
    '''
    Get company financials as reported.
    
    Args:
        symbol: Symbol of the company: AAPL e.g.
        freq: Frequency. Can be either `annual` or `quarterly`. Default to `annual`.
        _from: From date in the format yyyy-MM-dd.
        to: To date in the format yyyy-MM-dd.
        
    Returns:
        object: Financials reported information.
    '''
    return finnhub_client.financials_reported(symbol=symbol, freq=freq, _from=_from, to=to)

@tool()
def get_recommendation_trends(symbol:str):
    '''
    Get latest analyst recommendation trends for a company.
    
    Args:
        symbol: Symbol of the company: AAPL e.g.
        
    Returns:
        object: Recommendation trends information.
    '''
    return finnhub_client.recommendation_trends(symbol=symbol)

@tool()
def get_company_earnings_surprises(symbol:str):
    '''
    Get company last 4 quarters earnings surprise.
    
    Args:
        symbol: Symbol of the company: AAPL e.g.
        
    Returns:
        object: Company earnings information.
    '''
    return finnhub_client.company_earnings(symbol=symbol, limit=4)

@tool()
def get_quote(symbol:str):
    '''
    Get real-time quote data for US stocks. 
    
    Args:
        symbol: Symbol of the company: AAPL e.g.
        
    Returns:
        object: Company quote information.
    '''
    return finnhub_client.quote(symbol=symbol)

# Fundamental

# print(finnhub_client.symbol_lookup('apple'))

# print(finnhub_client.general_news('general', min_id=0))

# Company Profile 2
# print(finnhub_client.company_profile2(symbol='AAPL'))

# Company News
# Need to use _from instead of from to avoid conflict
# print(finnhub_client.company_news('AAPL', _from="2025-06-01", to="2025-06-10"))

# Company Peers
# print(finnhub_client.company_peers('NVDA'))

# res = finnhub_client.company_basic_financials('AAPL', 'all')

# print(res['metric'])

# print(finnhub_client.stock_insider_transactions('AAPL', '2025-05-01', '2025-05-31'))

# print(finnhub_client.stock_insider_sentiment('AAPL', '2025-05-01', '2025-05-31'))

# print(finnhub_client.financials_reported(symbol='AAPL', freq='annual',from_='2024-01-01', to='2025-5-31'))

# print(finnhub_client.recommendation_trends('AAPL'))

# print(finnhub_client.company_earnings('TSLA', limit=4))

# print(finnhub_client.quote('AAPL'))

tools = [get_stock_symbol, get_company_profile,get_market_news, get_company_news, get_company_peers, get_company_financials, get_stock_insider_transactions, get_stock_insider_sentiment, get_financials_reported, get_recommendation_trends, get_company_earnings_surprises, get_quote]

agent = Agent(
    model=OpenAILike(**settings),
    # tools=[ReasoningTools(add_instructions=True,add_few_shot=True),get_stock_symbol],
    tools=tools+[ReasoningTools(add_instructions=True,add_few_shot=True)]+[DuckDuckGoTools()],
    show_tool_calls=True,
    description=dedent("""\
        You are Professor X-1000, a distinguished AI research scientist with expertise
        in analyzing and synthesizing complex information. Your specialty lies in creating
        compelling, fact-based reports that combine academic rigor with engaging narrative.

        Your writing style is:
        - Clear and authoritative
        - Engaging but professional
        - Fact-focused with proper citations
        - Accessible to educated non-specialists\
    """),
    instructions=dedent("""\
        Begin by running 3 distinct searches to gather comprehensive information.
        Analyze and cross-reference sources for accuracy and relevance.
        Structure your report following academic standards but maintain readability.
        Include only verifiable facts with proper citations.
        Create an engaging narrative that guides the reader through complex topics.
        End with actionable takeaways and future implications.
        必须使用简体中文回复.\
    """),
    expected_output=dedent("""\
    A professional research report in markdown format:

    # {Compelling Title That Captures the Topic's Essence}

    ## Executive Summary
    {Brief overview of key findings and significance}

    ## Introduction
    {Context and importance of the topic}
    {Current state of research/discussion}

    ## Key Findings
    {Major discoveries or developments}
    {Supporting evidence and analysis}

    ## Implications
    {Impact on field/society}
    {Future directions}

    ## Key Takeaways
    - {Bullet point 1}
    - {Bullet point 2}
    - {Bullet point 3}

    ## References
    - [Source 1](link) - Key finding/quote
    - [Source 2](link) - Key finding/quote
    - [Source 3](link) - Key finding/quote

    ---
    Report generated by Professor X-1000
    Advanced Research Systems Division
    Date: {current_date}\
    """),
    debug_mode=True,
    telemetry=False
)
agent.print_response("Anthropic公司前景如何,有发行股票吗", markdown=True)