import json
import time
import logging
import os
import re
import configparser
import enum
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("deep_researcher.log")
    ]
)
logger = logging.getLogger("deep_researcher")

# ======================== 配置管理 ========================

@dataclass
class DeepResearcherConfig:
    """配置数据类，存储系统配置信息"""
    # LLM配置
    llm_provider: str = "openai"  # 支持: openai, ollama, local
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2000
    llm_server_address: str = "127.0.0.1:11434"
    
    # 搜索配置
    search_engine: str = "searx"  # 支持: searx, google, duckduckgo
    search_result_count: int = 10
    search_concurrent_limit: int = 3
    
    # 浏览器配置
    browser_headless: bool = True
    browser_stealth_mode: bool = True
    browser_timeout: int = 30
    browser_max_pages: int = 20
    
    # 研究配置
    max_research_depth: int = 3
    max_research_time: int = 3600  # 秒
    include_citations: bool = True
    
    # 系统配置
    work_dir: str = "./workspace"
    cache_dir: str = "./cache"
    output_format: str = "markdown"  # 支持: markdown, html, pdf

    @classmethod
    def from_file(cls, config_file: str) -> 'DeepResearcherConfig':
        """从配置文件加载配置"""
        config = configparser.ConfigParser()
        config.read(config_file)
        
        # 创建默认配置
        dr_config = cls()
        
        # 读取LLM配置
        if 'LLM' in config:
            llm_section = config['LLM']
            dr_config.llm_provider = llm_section.get('provider', dr_config.llm_provider)
            dr_config.llm_model = llm_section.get('model', dr_config.llm_model)
            dr_config.llm_temperature = llm_section.getfloat('temperature', dr_config.llm_temperature)
            dr_config.llm_max_tokens = llm_section.getint('max_tokens', dr_config.llm_max_tokens)
        
        if 'LOCAL_LLM' in config:
            local_llm_section = config['LOCAL_LLM']
            dr_config.llm_server_address = local_llm_section.get('server_address', dr_config.llm_server_address)
        
        # 读取搜索配置
        if 'SEARCH' in config:
            search_section = config['SEARCH']
            dr_config.search_engine = search_section.get('engine', dr_config.search_engine)
            dr_config.search_result_count = search_section.getint('result_count', dr_config.search_result_count)
            dr_config.search_concurrent_limit = search_section.getint('concurrent_limit', dr_config.search_concurrent_limit)
        
        # 读取浏览器配置
        if 'BROWSER' in config:
            browser_section = config['BROWSER']
            dr_config.browser_headless = browser_section.getboolean('headless', dr_config.browser_headless)
            dr_config.browser_stealth_mode = browser_section.getboolean('stealth_mode', dr_config.browser_stealth_mode)
            dr_config.browser_timeout = browser_section.getint('timeout', dr_config.browser_timeout)
            dr_config.browser_max_pages = browser_section.getint('max_pages', dr_config.browser_max_pages)
        
        # 读取研究配置
        if 'RESEARCH' in config:
            research_section = config['RESEARCH']
            dr_config.max_research_depth = research_section.getint('max_depth', dr_config.max_research_depth)
            dr_config.max_research_time = research_section.getint('max_time', dr_config.max_research_time)
            dr_config.include_citations = research_section.getboolean('include_citations', dr_config.include_citations)
        
        # 读取系统配置
        if 'SYSTEM' in config:
            system_section = config['SYSTEM']
            dr_config.work_dir = system_section.get('work_dir', dr_config.work_dir)
            dr_config.cache_dir = system_section.get('cache_dir', dr_config.cache_dir)
            dr_config.output_format = system_section.get('output_format', dr_config.output_format)
        
        return dr_config

# ======================== LLM提供商接口 ========================

class LLMProvider:
    """LLM提供商接口，提供与大型语言模型交互的统一接口"""
    
    def __init__(self, config: DeepResearcherConfig):
        self.config = config
        self.provider_type = config.llm_provider
        self.model = config.llm_model
        self.temperature = config.llm_temperature
        self.max_tokens = config.llm_max_tokens
        self.server_address = config.llm_server_address
        
        # 初始化相应的LLM客户端
        if self.provider_type == "openai":
            self._init_openai()
        elif self.provider_type == "ollama":
            self._init_ollama()
        elif self.provider_type == "local":
            self._init_local()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider_type}")
    
    def _init_openai(self):
        """初始化OpenAI客户端"""
        # 实际实现中应导入openai库并配置API密钥
        # 这里简化处理
        logger.info(f"Initialized OpenAI client with model {self.model}")
        self.client = "openai_client"  # 实际实现中应是实际的客户端对象
    
    def _init_ollama(self):
        """初始化Ollama客户端"""
        # 实际实现中应建立与ollama服务的连接
        # 这里简化处理
        logger.info(f"Initialized Ollama client with model {self.model} at {self.server_address}")
        self.client = "ollama_client"  # 实际实现中应是实际的客户端对象
    
    def _init_local(self):
        """初始化本地LLM客户端"""
        # 实际实现中应建立与本地LLM服务的连接
        # 这里简化处理
        logger.info(f"Initialized Local LLM client with model {self.model} at {self.server_address}")
        self.client = "local_llm_client"  # 实际实现中应是实际的客户端对象
    
    def generate(self, prompt: str, **kwargs) -> str:
        """生成LLM回应"""
        # 在实际实现中，这里应调用相应LLM客户端的API
        # 这里简化处理，仅记录日志并返回模拟响应
        
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        logger.debug(f"LLM Request: {prompt[:100]}... [temperature={temperature}, max_tokens={max_tokens}]")
        
        # 模拟LLM响应
        # 在实际实现中，这里应是对应LLM的实际调用
        response = f"LLM response to: {prompt[:50]}..."
        
        logger.debug(f"LLM Response: {response[:100]}...")
        return response
    
    def generate_json(self, prompt: str, **kwargs) -> Dict:
        """生成JSON格式的LLM回应"""
        try:
            response = self.generate(prompt, **kwargs)
            # 尝试解析JSON
            json_response = json.loads(response)
            return json_response
        except json.JSONDecodeError:
            # JSON解析失败时的处理
            logger.warning("Failed to parse JSON from LLM response. Attempting to fix...")
            # 尝试修复JSON
            fixed_prompt = f"""
            Your previous response could not be parsed as valid JSON. 
            Please provide a valid JSON format for the following request:
            
            {prompt}
            
            Return only valid JSON, nothing else.
            """
            try:
                fixed_response = self.generate(fixed_prompt, **kwargs)
                return json.loads(fixed_response)
            except json.JSONDecodeError:
                logger.error("Still failed to parse JSON from LLM response.")
                # 返回空字典作为后备
                return {}

# ======================== 工具接口 ========================

class SearchTool:
    """搜索工具，用于执行Web搜索"""
    
    def __init__(self, config: DeepResearcherConfig):
        self.config = config
        self.engine = config.search_engine
        self.result_count = config.search_result_count
        self.concurrent_limit = config.search_concurrent_limit
    
    def search(self, query: str) -> List[Dict[str, str]]:
        """执行搜索并返回结果"""
        # 在实际实现中，这里应调用具体搜索引擎的API
        # 这里简化处理，返回模拟搜索结果
        
        logger.info(f"Searching for: {query}")
        
        # 模拟搜索结果
        results = [
            {
                "title": f"Result {i} for {query}",
                "url": f"https://example.com/result{i}",
                "snippet": f"This is a snippet for result {i} related to {query}.",
                "date": "2025-05-24"
            }
            for i in range(1, self.result_count + 1)
        ]
        
        logger.info(f"Found {len(results)} results for query: {query}")
        return results

class Browser:
    """浏览器工具，用于网页导航和内容提取"""
    
    def __init__(self, config: DeepResearcherConfig):
        self.config = config
        self.headless = config.browser_headless
        self.stealth_mode = config.browser_stealth_mode
        self.timeout = config.browser_timeout
        self.max_pages = config.browser_max_pages
        self.driver = None
        self._initialize_browser()
    
    def _initialize_browser(self):
        """初始化浏览器"""
        # 在实际实现中，这里应初始化Selenium WebDriver或类似工具
        # 这里简化处理，仅记录初始化日志
        logger.info(f"Initializing browser: headless={self.headless}, stealth_mode={self.stealth_mode}")
        self.driver = "selenium_webdriver"  # 实际实现中应是实际的WebDriver对象
    
    def navigate(self, url: str) -> bool:
        """导航到指定URL"""
        # 在实际实现中，这里应使用WebDriver导航到目标URL
        # 这里简化处理，仅记录导航日志
        logger.info(f"Navigating to: {url}")
        return True
    
    def get_page_content(self) -> str:
        """获取当前页面内容"""
        # 在实际实现中，这里应提取页面HTML或文本内容
        # 这里简化处理，返回模拟页面内容
        return "<html><body><h1>Example Page</h1><p>This is example content.</p></body></html>"
    
    def get_links(self) -> List[Dict[str, str]]:
        """获取当前页面上的链接"""
        # 在实际实现中，这里应提取页面上的链接
        # 这里简化处理，返回模拟链接
        return [
            {"text": "Link 1", "url": "https://example.com/link1"},
            {"text": "Link 2", "url": "https://example.com/link2"},
            {"text": "Link 3", "url": "https://example.com/link3"}
        ]
    
    def get_current_url(self) -> str:
        """获取当前页面URL"""
        # 在实际实现中，这里应返回当前页面的URL
        # 这里简化处理，返回模拟URL
        return "https://example.com/current"
    
    def get_form_inputs(self) -> List[Dict[str, str]]:
        """获取当前页面上的表单输入字段"""
        # 在实际实现中，这里应提取页面上的表单字段
        # 这里简化处理，返回模拟表单字段
        return [
            {"name": "username", "type": "text", "required": True},
            {"name": "password", "type": "password", "required": True},
            {"name": "remember", "type": "checkbox", "required": False}
        ]
    
    def fill_form_inputs(self, field_values: Dict[str, str]) -> bool:
        """填写表单字段"""
        # 在实际实现中，这里应填写实际表单字段
        # 这里简化处理，仅记录填写日志
        logger.info(f"Filling form with values: {field_values}")
        return True
    
    def submit_form(self) -> bool:
        """提交表单"""
        # 在实际实现中，这里应提交表单
        # 这里简化处理，仅记录提交日志
        logger.info("Submitting form")
        return True
    
    def close(self):
        """关闭浏览器"""
        # 在实际实现中，这里应关闭WebDriver
        # 这里简化处理，仅记录关闭日志
        logger.info("Closing browser")

class Parser:
    """内容解析器基类"""
    
    @abstractmethod
    def parse(self, content: str) -> str:
        """解析内容"""
        pass

class HTMLParser(Parser):
    """HTML解析器"""
    
    def parse(self, content: str) -> str:
        """解析HTML内容"""
        # 在实际实现中，这里应使用BeautifulSoup或类似工具提取文本内容
        # 这里简化处理，使用正则表达式简单提取
        text = re.sub(r'<[^>]+>', ' ', content)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

class PDFParser(Parser):
    """PDF解析器"""
    
    def parse(self, content: str) -> str:
        """解析PDF内容"""
        # 在实际实现中，这里应使用PyPDF2或类似工具解析PDF
        # 这里简化处理，返回模拟解析结果
        return "This is extracted content from a PDF document."

class TableParser(Parser):
    """表格解析器"""
    
    def parse(self, content: str) -> str:
        """解析表格内容"""
        # 在实际实现中，这里应解析HTML表格或CSV内容
        # 这里简化处理，返回模拟解析结果
        return "This is extracted content from a table structure."

class CodeParser(Parser):
    """代码解析器"""
    
    def parse(self, content: str) -> str:
        """解析代码内容"""
        # 在实际实现中，这里应处理代码块的格式化
        # 这里简化处理，返回模拟解析结果
        return "This is extracted and formatted code content."

# ======================== 代理系统 ========================

class BaseAgent(ABC):
    """基础代理类，所有特定代理的父类"""
    
    def __init__(self, llm_provider: LLMProvider, config: DeepResearcherConfig):
        self.llm_provider = llm_provider
        self.config = config
        self.name = self.__class__.__name__
        self.logger = logging.getLogger(f"deep_researcher.{self.name}")
    
    @abstractmethod
    def process(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """处理任务并返回结果"""
        pass

class ActionType(enum.Enum):
    """代理操作类型枚举"""
    NAVIGATE = "navigate"
    SEARCH = "search"
    GO_BACK = "go_back"
    FILL_FORM = "fill_form"
    EXIT = "exit"
    EXTRACT = "extract"
    ANALYZE = "analyze"
    INTEGRATE = "integrate"

class SearchAgent(BaseAgent):
    """搜索代理，负责生成搜索查询、执行搜索和评估结果"""
    
    def __init__(self, llm_provider: LLMProvider, config: DeepResearcherConfig):
        super().__init__(llm_provider, config)
        self.search_tool = SearchTool(config)
        self.search_history = []
        self.knowledge_gaps = []
    
    def process(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """处理搜索任务"""
        research_topic = task.get("research_topic")
        existing_knowledge = context.get("existing_knowledge", "")
        knowledge_gaps = context.get("knowledge_gaps", [])
        search_depth = task.get("search_depth", 1)
        
        self.logger.info(f"Processing search task for topic: {research_topic}")
        
        # 生成搜索查询
        queries = self.generate_search_queries(research_topic, existing_knowledge, knowledge_gaps)
        search_results = []
        
        # 执行搜索
        for query in queries:
            results = self.execute_search(query)
            evaluated_results = self.evaluate_search_results(results, research_topic)
            
            # 对结果排序（按评分降序）
            sorted_results = sorted(
                evaluated_results, 
                key=lambda x: x.get("relevance_score", 0) + x.get("value_score", 0),
                reverse=True
            )
            
            # 添加到搜索结果
            search_results.extend(sorted_results[:3])  # 每个查询取前3个最相关结果
        
        # 识别知识缺口
        if existing_knowledge:
            new_gaps = self.identify_knowledge_gaps(existing_knowledge, research_topic)
            knowledge_gaps = list(set(knowledge_gaps + new_gaps))
        
        return {
            "action": ActionType.NAVIGATE.value,
            "search_results": search_results,
            "knowledge_gaps": knowledge_gaps,
            "search_queries": queries,
            "search_depth": search_depth,
            "status": "completed"
        }
    
    def generate_search_queries(self, research_topic: str, existing_knowledge: str = None, knowledge_gaps: List[str] = None) -> List[str]:
        """为研究主题生成有效的搜索查询"""
        query_gen_prompt = f"""
        为研究主题"{research_topic}"生成5个有效的搜索查询。
        
        已有知识：
        {existing_knowledge or "尚无收集到的信息"}
        
        已识别的知识缺口：
        {', '.join(knowledge_gaps) if knowledge_gaps else "尚未识别知识缺口"}
        
        生成的查询应当：
        1. 针对性强，使用精确的关键词
        2. 涵盖不同角度和子主题
        3. 包含必要的限定词以提高结果质量
        4. 避免与已执行过的搜索重复
        
        已执行过的搜索：
        {[q["query"] for q in self.search_history]}
        
        返回格式：每行一个搜索查询，不要编号
        """
        
        queries_text = self.llm_provider.generate(query_gen_prompt)
        queries = [q.strip() for q in queries_text.split('\n') if q.strip()]
        
        # 记录查询历史
        for query in queries:
            self.search_history.append({
                "query": query,
                "timestamp": time.time(),
                "research_topic": research_topic
            })
        
        return queries[:5]  # 最多返回5个查询
    
    def execute_search(self, query: str) -> List[Dict[str, str]]:
        """执行搜索并返回结果"""
        search_results = self.search_tool.search(query)
        return search_results
    
    def evaluate_search_results(self, results: List[Dict[str, str]], research_topic: str) -> List[Dict[str, Any]]:
        """评估搜索结果的相关性和信息价值"""
        if not results:
            return []
            
        eval_prompt = f"""
        评估以下搜索结果对研究主题"{research_topic}"的价值。
        
        为每个结果评分：
        - 相关性（0-10）：内容与研究主题的相关程度
        - 信息价值（0-10）：提供的信息可能的深度和价值
        - 可信度（0-10）：来源的可靠性和权威性
        - 新颖性（0-10）：提供新信息的可能性
        
        搜索结果：
        {json.dumps(results[:10], indent=2)}
        
        返回JSON格式的评估结果，包含每个结果的URL和四项评分。格式如下：
        [
            {{
                "url": "结果URL",
                "title": "结果标题",
                "relevance_score": 相关性评分,
                "value_score": 信息价值评分,
                "credibility_score": 可信度评分,
                "novelty_score": 新颖性评分,
                "overall_score": 综合评分(四项平均)
            }},
            ...
        ]
        """
        
        try:
            evaluation = self.llm_provider.generate_json(eval_prompt)
            
            # 确保评估结果格式正确
            if isinstance(evaluation, list):
                return evaluation
            else:
                return self._fallback_evaluation(results, research_topic)
                
        except Exception as e:
            self.logger.error(f"Error evaluating search results: {e}")
            return self._fallback_evaluation(results, research_topic)
    
    def _fallback_evaluation(self, results: List[Dict[str, str]], research_topic: str) -> List[Dict[str, Any]]:
        """搜索结果评估的后备实现"""
        evaluated_results = []
        
        for result in results:
            # 简单评估（固定分数）
            evaluation = {
                "url": result.get("url", ""),
                "title": result.get("title", ""),
                "relevance_score": 7,  # 默认相关性
                "value_score": 7,     # 默认信息价值
                "credibility_score": 7,  # 默认可信度
                "novelty_score": 7,    # 默认新颖性
                "overall_score": 7     # 默认综合评分
            }
            evaluated_results.append(evaluation)
        
        return evaluated_results
    
    def identify_knowledge_gaps(self, collected_info: str, research_topic: str) -> List[str]:
        """识别研究过程中的知识缺口"""
        gaps_prompt = f"""
        分析已收集的信息，识别关于研究主题"{research_topic}"的知识缺口。
        
        已收集信息概述：
        {collected_info}
        
        识别以下类型的知识缺口：
        1. 缺少的关键事实或数据
        2. 未探索的重要子主题
        3. 缺乏的对立观点或批评
        4. 时间线或历史发展中的空白
        5. 未来趋势或预测的缺失
        
        返回格式：列出5个最关键的知识缺口，每行一个
        """
        
        gaps_text = self.llm_provider.generate(gaps_prompt)
        gaps = [g.strip() for g in gaps_text.split('\n') if g.strip()]
        
        # 更新知识缺口列表
        self.knowledge_gaps = list(set(self.knowledge_gaps + gaps))
        
        return gaps
    
    def adjust_search_strategy(self, feedback: str, research_topic: str) -> List[str]:
        """根据反馈调整搜索策略"""
        strategy_prompt = f"""
        基于以下反馈，为研究主题"{research_topic}"调整搜索策略：
        
        当前搜索历史：
        {json.dumps(self.search_history, indent=2)}
        
        已识别的知识缺口：
        {self.knowledge_gaps}
        
        反馈信息：
        {feedback}
        
        提出搜索策略调整：
        1. 应该使用哪些新的关键词或短语？
        2. 应该避免哪些已证明无效的搜索方向？
        3. 应该增加哪些限定词以提高结果精确度？
        4. 是否需要更专业/更通用的搜索表述？
        
        返回调整后的3-5个具体的新搜索查询，每行一个。
        """
        
        adjusted_queries_text = self.llm_provider.generate(strategy_prompt)
        adjusted_queries = [q.strip() for q in adjusted_queries_text.split('\n') if q.strip()]
        
        # 更新搜索历史
        for query in adjusted_queries:
            self.search_history.append({
                "query": query,
                "timestamp": time.time(),
                "research_topic": research_topic,
                "is_adjusted": True
            })
        
        return adjusted_queries

class BrowserAgent(BaseAgent):
    """浏览器代理，负责网页导航、内容提取和表单交互"""
    
    def __init__(self, llm_provider: LLMProvider, config: DeepResearcherConfig):
        super().__init__(llm_provider, config)
        self.browser = Browser(config)
        
        self.current_page = None
        self.navigable_links = []
        self.notes = []
        self.research_focus = None
        self.visited_urls = set()
        self.content_quality_scores = {}
    
    def process(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """处理浏览任务"""
        action = task.get("action")
        research_topic = task.get("research_topic", "")
        search_results = task.get("search_results", [])
        
        self.research_focus = research_topic
        
        if action == ActionType.NAVIGATE.value:
            if "url" in task:
                # 导航到指定URL
                url = task["url"]
                return self.navigate_to_url(url)
            elif search_results:
                # 选择搜索结果中最相关的URL
                best_result = search_results[0]
                url = best_result.get("url", "")
                return self.navigate_to_url(url)
                
        elif action == ActionType.GO_BACK.value:
            # 返回操作
            return self._go_back()
            
        elif action == ActionType.FILL_FORM.value:
            # 填写表单
            form_data = task.get("form_data", {})
            return self.fill_form(form_data)
            
        return {
            "status": "error",
            "message": f"Unsupported action: {action}",
            "action": ActionType.EXIT.value
        }
    
    def navigate_to_url(self, url: str) -> Dict[str, Any]:
        """导航到指定URL并提取内容"""
        if url in self.visited_urls:
            return {
                "status": "already_visited",
                "url": url,
                "action": ActionType.GO_BACK.value
            }
            
        try:
            navigation_result = self.browser.navigate(url)
            
            if not navigation_result:
                return {
                    "status": "navigation_failed",
                    "url": url,
                    "action": ActionType.GO_BACK.value,
                    "message": "Failed to navigate to URL"
                }
            
            self.current_page = url
            self.visited_urls.add(url)
            
            # 提取页面内容
            content = self.browser.get_page_content()
            links = self.browser.get_links()
            self.navigable_links = links
            
            # 评估内容与研究重点的相关性
            relevance_score = self._evaluate_content_relevance(content)
            self.content_quality_scores[url] = relevance_score
            
            # 提取关键信息
            extracted_info = self._extract_key_information(content)
            
            if extracted_info:
                self.notes.append({
                    "url": url,
                    "info": extracted_info,
                    "relevance": relevance_score,
                    "timestamp": time.time()
                })
            
            # 决定下一步动作
            research_state = {
                "visited_urls": list(self.visited_urls),
                "notes_count": len(self.notes),
                "current_relevance": relevance_score
            }
            
            next_action = self.decide_next_action(content, research_state)
                
            return {
                "status": "success",
                "action": next_action.get("action", ActionType.EXIT.value),
                "url": url,
                "content_summary": content[:500] + "...",
                "links": links,
                "relevance": relevance_score,
                "extracted_info": extracted_info,
                "next_url": next_action.get("url", ""),
                "form_detected": next_action.get("form_detected", False),
                "research_state": research_state
            }
            
        except Exception as e:
            self.logger.error(f"Error navigating to {url}: {e}")
            return {
                "status": "error",
                "url": url,
                "action": ActionType.GO_BACK.value,
                "message": str(e)
            }
    
    def decide_next_action(self, content: str, research_state: Dict[str, Any]) -> Dict[str, Any]:
        """决定下一步动作：继续浏览、返回、填表单或结束"""
        decision_prompt = f"""
        基于当前的研究状态和已获取的信息，决定下一步最佳行动：
        
        当前页面：{self.current_page}
        已访问页面数：{len(self.visited_urls)}
        已收集笔记数：{len(self.notes)}
        研究重点：{self.research_focus}
        当前页面相关性：{self.content_quality_scores.get(self.current_page, 0)}
        
        研究状态概述：
        {json.dumps(research_state, indent=2)}
        
        可选操作：
        1. NAVIGATE - 访问新链接（指定链接ID）
        2. GO_BACK - 返回上一页
        3. FILL_FORM - 填写表单
        4. EXIT - 完成浏览，返回收集的信息
        
        可用链接：
        {json.dumps(self.navigable_links, indent=2)}
        
        决定最佳下一步行动，并提供理由。返回JSON格式：
        {{
            "action": "选择的操作（NAVIGATE/GO_BACK/FILL_FORM/EXIT）",
            "url": "如果是NAVIGATE，提供目标链接URL",
            "link_id": "如果是NAVIGATE，提供链接ID",
            "form_detected": true/false,
            "reason": "决策理由"
        }}
        """
        
        try:
            decision = self.llm_provider.generate_json(decision_prompt)
            
            # 确保决策格式正确
            action = decision.get("action", "EXIT")
            
            # 转换为标准动作类型
            if action.upper() == "NAVIGATE":
                action = ActionType.NAVIGATE.value
            elif action.upper() == "GO_BACK":
                action = ActionType.GO_BACK.value
            elif action.upper() == "FILL_FORM":
                action = ActionType.FILL_FORM.value
            elif action.upper() == "EXIT":
                action = ActionType.EXIT.value
            
            return {
                "action": action,
                "url": decision.get("url", ""),
                "link_id": decision.get("link_id", ""),
                "form_detected": decision.get("form_detected", False),
                "reason": decision.get("reason", "")
            }
                
        except Exception as e:
            self.logger.error(f"Error deciding next action: {e}")
            # 默认为退出操作
            return {
                "action": ActionType.EXIT.value,
                "reason": "Error in decision process"
            }
    
    def fill_form(self, form_data: Dict[str, str]) -> Dict[str, Any]:
        """填写网页表单"""
        form_inputs = self.browser.get_form_inputs()
        
        if not form_inputs:
            return {
                "status": "no_form_found",
                "action": ActionType.GO_BACK.value
            }
            
        # 匹配提供的表单数据与实际表单字段
        field_values = {}
        for field in form_inputs:
            if field["name"] in form_data:
                field_values[field["name"]] = form_data[field["name"]]
        
        # 填写表单并提交
        fill_result = self.browser.fill_form_inputs(field_values)
        
        if not fill_result:
            return {
                "status": "form_fill_failed",
                "action": ActionType.GO_BACK.value
            }
            
        submission_result = self.browser.submit_form()
        
        if not submission_result:
            return {
                "status": "form_submission_failed",
                "action": ActionType.GO_BACK.value
            }
        
        # 更新当前页面和状态
        new_url = self.browser.get_current_url()
        self.current_page = new_url
        self.visited_urls.add(new_url)
        
        # 获取并分析新页面内容
        content = self.browser.get_page_content()
        links = self.browser.get_links()
        self.navigable_links = links
        
        # 评估内容
        relevance_score = self._evaluate_content_relevance(content)
        self.content_quality_scores[new_url] = relevance_score
        
        # 提取关键信息
        extracted_info = self._extract_key_information(content)
        
        if extracted_info:
            self.notes.append({
                "url": new_url,
                "info": extracted_info,
                "relevance": relevance_score,
                "from_form_submission": True,
                "timestamp": time.time()
            })
        
        return {
            "status": "success",
            "action": ActionType.NAVIGATE.value,
            "url": new_url,
            "previous_url": self.current_page,
            "content_summary": content[:500] + "...",
            "links": links,
            "relevance": relevance_score,
            "extracted_info": extracted_info
        }
    
    def _go_back(self) -> Dict[str, Any]:
        """返回上一页或搜索结果"""
        # 在实际实现中，这里应实现返回上一页的逻辑
        # 这里简化处理，返回退出操作
        return {
            "status": "go_back",
            "action": ActionType.EXIT.value,
            "message": "Navigating back"
        }
    
    def _extract_key_information(self, content: str) -> str:
        """从页面内容中提取与研究重点相关的关键信息"""
        extract_prompt = f"""
        从以下网页内容中提取与研究主题"{self.research_focus}"高度相关的关键信息：
        
        1. 重要事实和数据
        2. 专家观点和论据
        3. 定义和概念解释
        4. 案例和实例
        5. 时间线和历史发展
        
        仅提取相关性高的信息，忽略无关内容。为每条信息标注可靠性评估。
        
        网页内容：
        {content[:5000]}  # 截取内容以避免超过上下文窗口
        """
        
        extraction = self.llm_provider.generate(extract_prompt)
        return extraction
    
    def _evaluate_content_relevance(self, content: str) -> float:
        """评估页面内容与当前研究重点的相关性"""
        if not self.research_focus:
            return 0.5  # 默认中等相关性
            
        relevance_prompt = f"""
        评估以下网页内容与研究主题"{self.research_focus}"的相关性。
        评分范围从0.0（完全不相关）到1.0（高度相关）。
        
        考虑以下因素：
        - 直接提及关键概念的频率
        - 提供的信息深度
        - 信息的专业性和权威性
        - 内容的新颖性（与已知信息相比）
        
        网页内容：
        {content[:3000]}  # 截取部分内容
        
        仅返回一个0.0到1.0之间的数字作为相关性评分。
        """
        
        try:
            relevance_text = self.llm_provider.generate(relevance_prompt).strip()
            # 提取数字
            relevance_match = re.search(r'(\d+\.\d+|\d+)', relevance_text)
            if relevance_match:
                relevance = float(relevance_match.group(1))
                return max(0.0, min(1.0, relevance))  # 确保在0-1范围内
            return 0.5  # 默认中等相关性
        except Exception as e:
            self.logger.error(f"Error evaluating content relevance: {e}")
            return 0.5  # 解析失败时返回默认值
    
    def get_collected_notes(self) -> List[Dict[str, Any]]:
        """获取所有收集的笔记"""
        return sorted(self.notes, key=lambda x: x.get("relevance", 0), reverse=True)
    
    def close(self):
        """关闭浏览器"""
        self.browser.close()

class ContentAnalysisAgent(BaseAgent):
    """内容分析代理，负责对获取的内容进行深入分析"""
    
    def __init__(self, llm_provider: LLMProvider, config: DeepResearcherConfig):
        super().__init__(llm_provider, config)
        self.parsers = {
            "html": HTMLParser(),
            "pdf": PDFParser(),
            "table": TableParser(),
            "code": CodeParser()
        }
        self.analyzed_contents = []
    
    def process(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """处理内容分析任务"""
        action = task.get("action")
        research_focus = task.get("research_focus", "")
        content = task.get("content", "")
        content_type = task.get("content_type", "html")
        url = task.get("url", "")
        
        if action == ActionType.ANALYZE.value:
            # 分析内容
            analysis = self.analyze_content(content, content_type, research_focus)
            
            # 提取引用（如果需要）
            citations = []
            if self.config.include_citations:
                citations = self.extract_citations(content, content_type)
            
            # 评估信息质量
            quality_assessment = self.assess_information_quality(analysis)
            
            return {
                "status": "success",
                "action": ActionType.INTEGRATE.value,
                "analysis": analysis,
                "citations": citations,
                "quality_assessment": quality_assessment,
                "url": url,
                "research_focus": research_focus
            }
        
        elif action == ActionType.EXTRACT.value:
            # 仅提取引用
            citations = self.extract_citations(content, content_type)
            
            return {
                "status": "success",
                "action": ActionType.INTEGRATE.value,
                "citations": citations,
                "url": url
            }
            
        return {
            "status": "error",
            "message": f"Unsupported action: {action}",
            "action": ActionType.EXIT.value
        }
    
    def analyze_content(self, content: str, content_type: str, research_focus: str) -> Dict[str, Any]:
        """分析内容，提取结构化信息"""
        # 使用合适的解析器预处理内容
        parsed_content = self._parse_content(content, content_type)
        
        analysis_prompt = f"""
        深入分析以下与"{research_focus}"相关的内容。
        
        提取以下类型的信息：
        1. 核心概念与定义
        2. 关键事实与数据（包含数字、统计、日期等）
        3. 主要论点与支持证据
        4. 作者观点与偏向
        5. 方法论与研究过程（如适用）
        6. 结论与发现
        
        对提取的每条信息评估：
        - 相关性：与研究主题的关联度
        - 重要性：内容在该领域的重要程度
        - 可靠性：基于来源、方法和内容一致性
        - 新颖性：是否提供新见解
        
        内容：
        {parsed_content[:8000]}  # 限制长度
        
        以JSON格式返回分析结果，使用以下结构：
        {
            "core_concepts": [{"concept": "概念", "definition": "定义", "relevance": 相关性评分},...],
            "key_facts": [{"fact": "事实", "importance": 重要性评分, "reliability": 可靠性评分},...],
            "main_arguments": [{"argument": "论点", "evidence": "证据", "strength": 强度评分},...],
            "author_perspective": {"perspective": "观点", "bias": "偏向程度", "credibility": 可信度评分},
            "methodology": {"approach": "方法", "rigor": 严谨性评分, "limitations": "局限性"},
            "conclusions": [{"conclusion": "结论", "importance": 重要性评分, "novelty": 新颖性评分},...]
        }
        """
        
        try:
            analysis_json = self.llm_provider.generate_json(analysis_prompt)
            
            # 记录分析结果
            self.analyzed_contents.append({
                "content_type": content_type,
                "research_focus": research_focus,
                "analysis": analysis_json,
                "timestamp": time.time()
            })
            
            return analysis_json
            
        except Exception as e:
            self.logger.error(f"Error analyzing content: {e}")
            # 返回简单分析作为后备
            return self._fallback_analysis(parsed_content, research_focus)
    
    def extract_citations(self, content: str, content_type: str) -> List[Dict[str, str]]:
        """提取内容中的引用和参考文献"""
        parsed_content = self._parse_content(content, content_type)
        
        citation_prompt = f"""
        从以下内容中提取所有引用和参考文献信息。
        
        对于每个引用，提取：
        1. 作者名
        2. 出版物/来源名
        3. 发表年份
        4. 标题
        5. DOI或URL（如有）
        
        内容：
        {parsed_content[:8000]}
        
        以JSON格式返回所有引用，每个引用包含上述字段（如可获取）。格式如下：
        [
            {{
                "authors": "作者名列表",
                "source": "出版物/来源名",
                "year": 发表年份,
                "title": "标题",
                "url": "DOI或URL"
            }},
            ...
        ]
        """
        
        try:
            citations = self.llm_provider.generate_json(citation_prompt)
            
            if isinstance(citations, list):
                return citations
            return []
                
        except Exception as e:
            self.logger.error(f"Error extracting citations: {e}")
            return []
    
    def assess_information_quality(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """评估提取信息的质量和可靠性"""
        quality_prompt = f"""
        评估以下研究信息的质量和可靠性：
        
        {json.dumps(analysis, indent=2)}
        
        对以下维度进行1-10评分：
        1. 数据完整性：信息是否完整，没有明显缺失
        2. 一致性：不同部分的信息是否存在矛盾
        3. 信源可靠性：信息来源的权威性和可信度
        4. 时效性：信息的时间相关性和更新状态
        5. 方法论健全性：研究方法和过程的科学性
        
        为每个维度提供评分和简短解释。返回JSON格式：
        {
            "completeness": {"score": 评分, "explanation": "解释"},
            "consistency": {"score": 评分, "explanation": "解释"},
            "source_reliability": {"score": 评分, "explanation": "解释"},
            "timeliness": {"score": 评分, "explanation": "解释"},
            "methodological_soundness": {"score": 评分, "explanation": "解释"},
            "overall_quality": 综合评分
        }
        """
        
        try:
            quality_assessment = self.llm_provider.generate_json(quality_prompt)
            
            # 计算综合评分
            if "overall_quality" not in quality_assessment:
                scores = [
                    quality_assessment.get("completeness", {}).get("score", 5),
                    quality_assessment.get("consistency", {}).get("score", 5),
                    quality_assessment.get("source_reliability", {}).get("score", 5),
                    quality_assessment.get("timeliness", {}).get("score", 5),
                    quality_assessment.get("methodological_soundness", {}).get("score", 5)
                ]
                quality_assessment["overall_quality"] = sum(scores) / len(scores)
            
            return quality_assessment
                
        except Exception as e:
            self.logger.error(f"Error assessing information quality: {e}")
            # 返回简单评估作为后备
            return {
                "completeness": {"score": 5, "explanation": "Default assessment"},
                "consistency": {"score": 5, "explanation": "Default assessment"},
                "source_reliability": {"score": 5, "explanation": "Default assessment"},
                "timeliness": {"score": 5, "explanation": "Default assessment"},
                "methodological_soundness": {"score": 5, "explanation": "Default assessment"},
                "overall_quality": 5
            }
    
    def _parse_content(self, content: str, content_type: str) -> str:
        """根据内容类型选择合适的解析器预处理内容"""
        parser = self.parsers.get(content_type.lower(), self.parsers["html"])
        return parser.parse(content)
    
    def _fallback_analysis(self, parsed_content: str, research_focus: str) -> Dict[str, Any]:
        """分析失败时的后备实现"""
        # 简单提取前几句作为核心概念
        sentences = re.split(r'[.!?]', parsed_content)
        core_concepts = [
            {"concept": s.strip(), "definition": "", "relevance": 5}
            for s in sentences[:3] if len(s.strip()) > 10
        ]
        
        # 返回简化结构
        return {
            "core_concepts": core_concepts,
            "key_facts": [],
            "main_arguments": [],
            "author_perspective": {"perspective": "Unknown", "bias": "Unknown", "credibility": 5},
            "methodology": {"approach": "Unknown", "rigor": 5, "limitations": "Unknown"},
            "conclusions": []
        }

class KnowledgeIntegrationAgent(BaseAgent):
    """知识整合代理，负责汇总和整合从多个来源获得的信息"""
    
    def __init__(self, llm_provider: LLMProvider, config: DeepResearcherConfig):
        super().__init__(llm_provider, config)
        self.citations = []
        self.findings = []
        self.conflicts = []
    
    def process(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """处理知识整合任务"""
        action = task.get("action")
        research_topic = task.get("research_topic", "")
        analyses = context.get("analyses", [])
        
        if action == ActionType.INTEGRATE.value:
            # 整合分析结果
            integrated_knowledge = self.integrate_information(analyses, research_topic)
            
            # 生成研究报告
            report = self.generate_research_report(
                integrated_knowledge, 
                research_topic, 
                include_citations=self.config.include_citations
            )
            
            return {
                "status": "success",
                "action": ActionType.EXIT.value,
                "integrated_knowledge": integrated_knowledge,
                "research_report": report,
                "conflicts": self.conflicts,
                "research_topic": research_topic
            }
            
        return {
            "status": "error",
            "message": f"Unsupported action: {action}",
            "action": ActionType.EXIT.value
        }
    
    def integrate_information(self, content_analyses: List[Dict[str, Any]], research_topic: str) -> Dict[str, Any]:
        """整合多个来源的内容分析结果"""
        # 提取所有关键发现
        all_findings = self._extract_all_findings(content_analyses)
        
        # 检测和解决冲突
        conflicts, resolved_findings = self._detect_and_resolve_conflicts(all_findings)
        self.conflicts = conflicts
        self.findings = resolved_findings
        
        integration_prompt = f"""
        整合以下关于"{research_topic}"的研究发现，创建一个连贯的知识合成：
        
        核心发现：
        {json.dumps(resolved_findings, indent=2)}
        
        已解决的冲突：
        {json.dumps(conflicts, indent=2)}
        
        合成应包括：
        1. 主要主题和子主题组织
        2. 关键发现的综合描述
        3. 不同来源的观点比较
        4. 对矛盾信息的解释
        5. 研究领域中的共识与分歧
        6. 关键知识缺口
        
        以连贯、客观的叙述形式组织内容，确保逻辑流畅。返回JSON格式：
        {
            "main_themes": [
                {
                    "theme": "主题名称",
                    "description": "主题描述",
                    "sub_themes": ["子主题1", "子主题2", ...],
                    "key_findings": ["关键发现1", "关键发现2", ...],
                    "consensus": "该主题的共识",
                    "disagreements": "该主题的分歧"
                },
                ...
            ],
            "integrated_findings": "整合的发现描述（长文本）",
            "knowledge_gaps": ["知识缺口1", "知识缺口2", ...],
            "overall_assessment": "整体评估"
        }
        """
        
        try:
            integrated_knowledge = self.llm_provider.generate_json(integration_prompt)
            return integrated_knowledge
                
        except Exception as e:
            self.logger.error(f"Error integrating information: {e}")
            # 返回简单整合作为后备
            return self._fallback_integration(resolved_findings, research_topic)
    
    def generate_research_report(self, integrated_knowledge: Dict[str, Any], research_topic: str, include_citations: bool = True) -> str:
        """生成最终研究报告"""
        report_prompt = f"""
        基于以下整合知识，为研究主题"{research_topic}"创建一份全面的研究报告：
        
        {json.dumps(integrated_knowledge, indent=2)}
        
        报告应包括：
        1. 执行摘要（200字以内）
        2. 研究背景和问题陈述
        3. 主要发现（按主题组织）
        4. 关键见解和结论
        5. 知识缺口和未来研究方向
        {"6. 参考文献和引用" if include_citations else ""}
        
        报告风格应专业、客观，提供深入分析而非简单概述。
        使用清晰的章节结构，适当使用小标题。
        {"在每个关键陈述后添加相关引用。" if include_citations else ""}
        
        以Markdown格式完成报告。
        """
        
        report = self.llm_provider.generate(report_prompt)
        
        if include_citations and self.citations:
            report = self._add_citations_to_report(report)
            
        return report
    
    def _extract_all_findings(self, content_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从所有内容分析中提取关键发现"""
        all_findings = []
        
        for analysis in content_analyses:
            analysis_content = analysis.get("analysis", {})
            
            if "core_concepts" in analysis_content:
                for concept in analysis_content.get("core_concepts", []):
                    all_findings.append({
                        "type": "concept",
                        "content": concept.get("concept", ""),
                        "definition": concept.get("definition", ""),
                        "relevance": concept.get("relevance", 5),
                        "source_url": analysis.get("url", "")
                    })
            
            if "key_facts" in analysis_content:
                for fact in analysis_content.get("key_facts", []):
                    all_findings.append({
                        "type": "fact",
                        "content": fact.get("fact", ""),
                        "importance": fact.get("importance", 5),
                        "reliability": fact.get("reliability", 5),
                        "source_url": analysis.get("url", "")
                    })
                
            if "conclusions" in analysis_content:
                for conclusion in analysis_content.get("conclusions", []):
                    all_findings.append({
                        "type": "conclusion",
                        "content": conclusion.get("conclusion", ""),
                        "importance": conclusion.get("importance", 5),
                        "novelty": conclusion.get("novelty", 5),
                        "source_url": analysis.get("url", "")
                    })
            
            # 收集引用
            if "citations" in analysis:
                self.citations.extend(analysis.get("citations", []))
        
        return all_findings
    
    def _detect_and_resolve_conflicts(self, findings: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """检测并解决发现中的冲突"""
        if len(findings) <= 1:
            return [], findings
            
        conflict_prompt = f"""
        分析以下研究发现，识别并解决任何矛盾或冲突：
        
        {json.dumps(findings, indent=2)}
        
        对于每个冲突：
        1. 描述矛盾的内容
        2. 涉及的发现ID（使用列表索引）
        3. 可能的解释
        4. 建议的解决方案
        
        以JSON格式返回：
        {{
            "conflicts": [
                {{
                    "description": "冲突描述",
                    "finding_ids": [涉及发现的索引],
                    "explanation": "可能的解释",
                    "resolution": "建议的解决方案"
                }},
                ...
            ],
            "resolved_findings": [修订后的发现列表]
        }}
        """
        
        try:
            resolution_data = self.llm_provider.generate_json(conflict_prompt)
            
            conflicts = resolution_data.get("conflicts", [])
            resolved_findings = resolution_data.get("resolved_findings", findings)
            
            return conflicts, resolved_findings
                
        except Exception as e:
            self.logger.error(f"Error detecting conflicts: {e}")
            # 解析失败时返回原始发现
            return [], findings
    
    def _add_citations_to_report(self, report: str) -> str:
        """向报告中添加引用"""
        if not self.citations:
            return report
            
        citation_prompt = f"""
        为以下研究报告添加适当的引用标记：
        
        报告：
        {report}
        
        可用引用：
        {json.dumps(self.citations, indent=2)}
        
        在每个关键陈述、数据点或特定观点后添加引用标记[n]，其中n是引用编号。
        然后在报告末尾添加完整的参考文献列表。
        
        保持原始的Markdown格式。
        """
        
        report_with_citations = self.llm_provider.generate(citation_prompt)
        return report_with_citations
    
    def _fallback_integration(self, findings: List[Dict[str, Any]], research_topic: str) -> Dict[str, Any]:
        """整合失败时的后备实现"""
        # 按类型分组
        concepts = [f for f in findings if f.get("type") == "concept"]
        facts = [f for f in findings if f.get("type") == "fact"]
        conclusions = [f for f in findings if f.get("type") == "conclusion"]
        
        # 创建主题（使用前3个概念作为主题）
        main_themes = []
        for i, concept in enumerate(concepts[:3]):
            main_themes.append({
                "theme": concept.get("content", f"Theme {i+1}"),
                "description": concept.get("definition", "No description available"),
                "sub_themes": [],
                "key_findings": [f.get("content", "") for f in facts[:3]],
                "consensus": "Information not available",
                "disagreements": "Information not available"
            })
        
        # 关键发现
        key_findings = "\n\n".join([
            f"Finding {i+1}: {f.get('content', '')}"
            for i, f in enumerate(facts[:5])
        ])
        
        # 知识缺口
        knowledge_gaps = [
            "Comprehensive data analysis",
            "Long-term trends",
            "Alternative perspectives"
        ]
        
        return {
            "main_themes": main_themes,
            "integrated_findings": key_findings,
            "knowledge_gaps": knowledge_gaps,
            "overall_assessment": f"Basic assessment of research topic: {research_topic}"
        }

class AgentRouter:
    """代理路由系统，负责任务分析和代理选择"""
    
    def __init__(self, config: DeepResearcherConfig):
        self.config = config
        self.llm_provider = LLMProvider(config)
        self.agents = {}
        self.task_history = []
        self.context = {"analyses": []}
        
        # 初始化各类代理
        self.agents["search"] = SearchAgent(self.llm_provider, config)
        self.agents["browser"] = BrowserAgent(self.llm_provider, config)
        self.agents["content"] = ContentAnalysisAgent(self.llm_provider, config)
        self.agents["knowledge"] = KnowledgeIntegrationAgent(self.llm_provider, config)
    
    def route_task(self, query: str) -> Dict[str, Any]:
        """根据查询分析，规划研究流程并分配任务"""
        analysis = self.analyze_query(query)
        research_plan = self._generate_research_plan(analysis, query)
        
        # 执行研究计划
        results = self._execute_research_plan(research_plan)
        return results
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """分析用户查询，确定研究策略"""
        # 使用LLM分析查询意图和复杂度
        analysis_prompt = f"""
        分析以下研究查询，确定：
        1. 主要研究领域
        2. 核心问题点
        3. 可能的信息来源
        4. 研究复杂度（1-5，其中5最复杂）
        
        查询: {query}
        
        以JSON格式返回分析结果：
        {{
            "research_field": "主要研究领域",
            "core_questions": ["核心问题1", "核心问题2", ...],
            "possible_sources": ["可能的信息来源1", "可能的信息来源2", ...],
            "complexity": 复杂度评分,
            "estimated_research_depth": 建议的研究深度,
            "key_search_terms": ["关键搜索词1", "关键搜索词2", ...]
        }}
        """
        
        try:
            analysis = self.llm_provider.generate_json(analysis_prompt)
            logger.info(f"Query analysis: {json.dumps(analysis, indent=2)}")
            return analysis
                
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            # 返回简单分析作为后备
            return {
                "research_field": "General",
                "core_questions": [query],
                "possible_sources": ["Web search", "Academic papers"],
                "complexity": 3,
                "estimated_research_depth": 2,
                "key_search_terms": query.split()
            }
    
    def _generate_research_plan(self, analysis: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
        """生成研究计划，包括各代理的任务安排"""
        # 根据查询复杂度确定研究深度
        complexity = analysis.get("complexity", 3)
        research_depth = min(complexity, self.config.max_research_depth)
        
        # 为研究任务生成一个唯一标识符
        research_id = f"research_{int(time.time())}"
        
        # 提取研究主题
        research_topic = query
        
        # 创建初始任务
        initial_search_task = {
            "agent": "search",
            "action": ActionType.SEARCH.value,
            "research_topic": research_topic,
            "research_id": research_id,
            "search_depth": research_depth,
            "key_search_terms": analysis.get("key_search_terms", [])
        }
        
        # 返回任务序列
        return [initial_search_task]
    
    def _execute_research_plan(self, research_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """执行研究计划，协调各代理的任务执行"""
        task_queue = research_plan.copy()
        final_results = {}
        research_topic = research_plan[0].get("research_topic", "")
        
        # 记录开始时间
        start_time = time.time()
        max_research_time = self.config.max_research_time
        
        # 初始化上下文
        self.context = {
            "research_topic": research_topic,
            "start_time": start_time,
            "analyses": [],
            "visited_urls": set(),
            "knowledge_gaps": [],
            "existing_knowledge": ""
        }
        
        while task_queue and (time.time() - start_time < max_research_time):
            # 取出下一个任务
            current_task = task_queue.pop(0)
            agent_name = current_task.get("agent")
            
            if agent_name not in self.agents:
                logger.error(f"Unknown agent: {agent_name}")
                continue
            
            # 执行任务
            logger.info(f"Executing task with agent {agent_name}: {current_task.get('action')}")
            agent = self.agents[agent_name]
            result = agent.process(current_task, self.context)
            
            # 记录任务历史
            self.task_history.append({
                "task": current_task,
                "result_status": result.get("status"),
                "timestamp": time.time()
            })
            
            # 处理结果
            if result.get("status") == "success":
                # 根据结果创建下一个任务
                next_action = result.get("action")
                
                if next_action == ActionType.NAVIGATE.value:
                    # 浏览器任务
                    if "url" in result:
                        url = result["url"]
                        if url and url not in self.context["visited_urls"]:
                            self.context["visited_urls"].add(url)
                            browser_task = {
                                "agent": "browser",
                                "action": ActionType.NAVIGATE.value,
                                "url": url,
                                "research_topic": research_topic
                            }
                            task_queue.append(browser_task)
                
                elif next_action == ActionType.ANALYZE.value:
                    # 内容分析任务
                    if "content" in result:
                        analysis_task = {
                            "agent": "content",
                            "action": ActionType.ANALYZE.value,
                            "content": result["content"],
                            "content_type": result.get("content_type", "html"),
                            "url": result.get("url", ""),
                            "research_topic": research_topic
                        }
                        task_queue.append(analysis_task)
                
                elif next_action == ActionType.INTEGRATE.value:
                    # 存储分析结果
                    if "analysis" in result:
                        self.context["analyses"].append({
                            "analysis": result["analysis"],
                            "url": result.get("url", ""),
                            "citations": result.get("citations", []),
                            "quality_assessment": result.get("quality_assessment", {})
                        })
                    
                    # 如果所有分析完成，添加知识整合任务
                    if not task_queue or len(self.context["analyses"]) >= 5:
                        integration_task = {
                            "agent": "knowledge",
                            "action": ActionType.INTEGRATE.value,
                            "research_topic": research_topic
                        }
                        task_queue.append(integration_task)
                
                elif next_action == ActionType.EXIT.value:
                    # 完成任务
                    if "research_report" in result:
                        final_results = result
                        break
                
                # 提取和更新知识缺口
                if "knowledge_gaps" in result:
                    self.context["knowledge_gaps"] = list(set(
                        self.context.get("knowledge_gaps", []) + 
                        result.get("knowledge_gaps", [])
                    ))
                
                # 更新现有知识
                if "extracted_info" in result:
                    existing_knowledge = self.context.get("existing_knowledge", "")
                    new_info = result.get("extracted_info", "")
                    if new_info:
                        self.context["existing_knowledge"] = f"{existing_knowledge}\n\n{new_info}"
            
            else:
                # 任务失败，记录错误
                logger.warning(f"Task failed: {result.get('message', 'Unknown error')}")
        
        # 检查是否因为超时而退出
        if time.time() - start_time >= max_research_time:
            logger.warning("Research terminated due to time limit")
            
            # 如果没有完成报告，强制生成一个
            if "research_report" not in final_results and self.context["analyses"]:
                integration_task = {
                    "agent": "knowledge",
                    "action": ActionType.INTEGRATE.value,
                    "research_topic": research_topic
                }
                agent = self.agents["knowledge"]
                final_results = agent.process(integration_task, self.context)
        
        # 如果没有结果，返回错误
        if not final_results:
            final_results = {
                "status": "error",
                "message": "Research failed to produce results",
                "research_topic": research_topic
            }
        
        # 清理资源
        self._cleanup()
        
        return final_results
    
    def _cleanup(self):
        """清理资源"""
        # 关闭浏览器
        if "browser" in self.agents:
            self.agents["browser"].close()

# ======================== 主程序 ========================

class DeepResearcher:
    """Deep Researcher主类，提供用户接口"""
    
    def __init__(self, config_file: str = None):
        # 加载配置
        if config_file and os.path.exists(config_file):
            self.config = DeepResearcherConfig.from_file(config_file)
        else:
            self.config = DeepResearcherConfig()
        
        # 创建工作目录
        os.makedirs(self.config.work_dir, exist_ok=True)
        os.makedirs(self.config.cache_dir, exist_ok=True)
        
        # 初始化代理路由系统
        self.router = AgentRouter(self.config)
        
        logger.info("Deep Researcher initialized")
    
    def research(self, query: str) -> Dict[str, Any]:
        """执行研究"""
        logger.info(f"Starting research on query: {query}")
        
        start_time = time.time()
        
        try:
            # 通过路由系统执行研究
            results = self.router.route_task(query)
            
            # 保存研究报告
            if "research_report" in results:
                self._save_report(results["research_report"], query)
            
            # 计算研究时间
            research_time = time.time() - start_time
            results["research_time"] = research_time
            
            logger.info(f"Research completed in {research_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Error during research: {e}")
            return {
                "status": "error",
                "message": f"Research failed: {str(e)}",
                "query": query
            }
    
    def _save_report(self, report: str, query: str) -> str:
        """保存研究报告"""
        # 创建文件名
        query_slug = re.sub(r'[^\w\s-]', '', query.lower())
        query_slug = re.sub(r'[-\s]+', '-', query_slug).strip('-_')
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{query_slug}-{timestamp}.md"
        
        # 保存路径
        save_path = os.path.join(self.config.work_dir, filename)
        
        # 写入文件
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        logger.info(f"Report saved to: {save_path}")
        return save_path

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deep Researcher - Advanced Research Assistant")
    parser.add_argument("--config", help="Path to configuration file", default="config.ini")
    parser.add_argument("--query", help="Research query")
    args = parser.parse_args()
    
    # 初始化Deep Researcher
    researcher = DeepResearcher(args.config)
    
    if args.query:
        # 从命令行参数获取查询
        query = args.query
    else:
        # 从用户输入获取查询
        query = input("Enter your research query: ")
    
    # 执行研究
    results = researcher.research(query)
    
    if results.get("status") == "success":
        print("\n=== Research Report ===\n")
        print(results.get("research_report", "No report generated"))
    else:
        print(f"Research failed: {results.get('message', 'Unknown error')}")

if __name__ == "__main__":
    main()
