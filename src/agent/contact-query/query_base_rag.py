from agno.agent import Agent
from agno.models.openai import OpenAILike
from agno.vectordb.lancedb import LanceDb
from agno.vectordb.search import SearchType
from agno.embedder.openai import OpenAIEmbedder
from agno.tools.reasoning import ReasoningTools
from agno.knowledge import AgentKnowledge

#------------------ settings ------------------
import os
from dotenv import load_dotenv
load_dotenv()
api_key=os.getenv("QWEN_API_KEY")
base_url=os.getenv("QWEN_API_BASE_URL")
local_base_url = 'http://192.168.0.166:8000/v1'
local_model_name = 'Qwen3-235B'
model_name = 'qwen-plus-latest'
embedding_model_id = 'text-embedding-v4'

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

settings = qwen_settings
#------------------ settings ------------------

vector_db = LanceDb(
    table_name="contact_table",
    uri="E:\\PythonProject\\LLM-Explore\\src\\agent\\contact-query\\tmp\\contact_vectors.lancedb",
    search_type=SearchType.hybrid,
    embedder=OpenAIEmbedder(id=embedding_model_id,api_key=api_key,base_url=base_url, dimensions=2048),
)


knowledge_base = AgentKnowledge(vector_db=vector_db)

# add_references=True 就是传统rag的做法，会将找到的相关文档都放在用户的context中，而现代的做法会通过search_knowledge_base进行工具查询然后得到相关结果，建议先不使用add_references=True，如果性能有问题再进行相关测试进行评估

agent = Agent(
    model=OpenAILike(**settings),
    name='Contact_Query_Agent',
    instructions=['查询合同详情的时候请列出所有数据，严禁遗漏任何条目','禁止虚构和假设任何数据','如果需要进行合同比对的时候，请按需**分别**查出所有项目后再进行比对','必须使用简体中文回复'],
    knowledge=knowledge_base,
    add_history_to_messages=True,
    num_history_responses=20,
    tools=[ReasoningTools(add_instructions=True)],
    markdown=True,
    # add_references=True,
    stream=True,
    stream_intermediate_steps=True,
    telemetry=False,
    debug_mode=True,
)

agent.print_response(message='包钢余热锅炉和宝钢余热锅炉价格相差多少？')