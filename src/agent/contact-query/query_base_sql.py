from agno.agent import Agent
from agno.models.openai import OpenAILike
from agno.vectordb.lancedb import LanceDb
from agno.vectordb.search import SearchType
from agno.embedder.openai import OpenAIEmbedder
from agno.document.base import Document
from agno.knowledge.document import DocumentKnowledgeBase
from agno.tools.reasoning import ReasoningTools


#------------------ settings ------------------
import os
from dotenv import load_dotenv
load_dotenv()
api_key=os.getenv("QWEN_API_KEY")
base_url=os.getenv("QWEN_API_BASE_URL")
local_base_url = 'http://192.168.0.166:8000/v1'
local_model_name = 'Qwen3-235B'
model_name = 'qwen-plus-latest'
embedding_model_id = 'text-embedding-v3'

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
    table_name="contact_info_vectors",
    uri="tmp/lancedb",
    search_type=SearchType.hybrid,
    embedder=OpenAIEmbedder(id=embedding_model_id,api_key=api_key,base_url=base_url, dimensions=1024),
)

contact_facts = """
湖南华菱湘潭钢铁有限公司2×360m2烧结余热发电项目
本溪北营钢铁（集团）股份有限公司400m2烧结机余热利用项目
"""

# Load documents from the data/docs directory
documents = [Document(content=contact_facts)]

knowledge_base = DocumentKnowledgeBase(
    documents=documents,
    vector_db=vector_db,
)

# Load the knowledge base
# knowledge_base.load(recreate=False)

from pymongo import MongoClient
import pandas as pd
mongo_uri = "mongodb://192.168.48.132:27017/"
def get_mongo_client(uri):
    try:
        return MongoClient(uri)
    except Exception as e:
        print(f"连接失败: {str(e)}")

# 执行MongoDB查询
def execute_mongo_query(project_name:str):
    try:
        client = get_mongo_client(mongo_uri)
        db = client['equipment_db']
        collection = db['equipment_collection_copy']
        
        # 执行查询
        try:
            results = list(collection.find({"project_name":project_name
            }).sort({
                'project_name': 1,
                'table_index': 1,
                'table_row_index': 1
            }))
            
            if not results:
                print("没有找到匹配的文档")
                return None, "没有找到匹配的记录"
                
        except Exception as e:
            print(f"查询失败: {str(e)}")
            return None, f"查询失败: {str(e)}"
        
        df = pd.json_normalize(results)
         # 定义需要显示的字段及其对应的中文表头
        display_columns = {
            'device_name': '设备',
            'unit_price': '单价',
            'quantity': '数量',
            'total_price': '总价',
            'price_unit': '价格单位',
            'unit': '单位',
            # 'project_name': '项目名称',
            # 'contact_name': '合同名称',
            # 'subitem_name': '子项名称',
            'contract_number': '合同号',
            # 'contact_type': '合同类型',
            'manufacturer': '制造商',
            'table_row_index': '表格行',
            'table_index': '表格索引'
        }
        
        # 筛选出指定的字段
        available_columns = [col for col in display_columns.keys() if col in df.columns]
        if not available_columns:
            print("查询结果中不包含指定的字段")
            return None, "查询结果中不包含指定的字段"
        
        df = df[available_columns]
        
        # 重命名列为中文表头
        df = df.rename(columns={col: display_columns[col] for col in available_columns})
        
        return df
    
    except Exception as e:
        return None, f"查询错误: {str(e)}"
    

from agno.tools import tool

@tool()
def get_contact_info_by_project_name(project_name:str)->str:
    '''
    Get the contact info by project name.
    
    Args:
        project_name (str): The name of the project.
        
    Returns:
        str: The contact info of the project.
    '''
    df = execute_mongo_query(project_name)
    return df.to_json(orient='records', force_ascii=False)


agent = Agent(
    model=OpenAILike(**settings),
    instructions=['首先根据知识库中存储的项目名称，选择最相关的项目返回','根据知识库中的查询到的标准项目名称，使用get_contact_info_by_project_name工具查询的相关的采购合同明细数据','查询合同详情的时候请列出所有数据，严禁遗漏任何条目','禁止虚构和假设任何数据','如果需要进行合同比对的时候，请按需查出所有项目后再进行比对'],
    knowledge=knowledge_base,
    tools=[ReasoningTools(add_instructions=True),get_contact_info_by_project_name],
    markdown=True,
    stream=True,
    stream_intermediate_steps=True,
    telemetry=False,
    debug_mode=True,
)

agent.print_response(message='请对比北营400烧结余热项目和华菱湘钢2*360中余热锅炉的价格')