import csv
from agno.agent import Agent
from agno.models.openai import OpenAILike
import mammoth
from textwrap import dedent
from pydantic import BaseModel, Field
from typing import List, Tuple
import logging
import pandas as pd
import json
from openai import OpenAI, max_retries

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s:%(filename)s:%(lineno)d - %(message)s',datefmt='%Y-%m-%d %H:%M:%S',level=logging.INFO) 

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

settings = local_settings
#------------------ settings ------------------

# 这里的year和total_price默认值应该给0，否则会在没有数值的时候由LLM赋值无导致无法正确解析pydantic模型
class ContractMeta(BaseModel):
    contact_no: str = Field(..., description="买方合同编号")
    # supplier_contact_no: str = Field(..., description="卖方合同编号")
    # contact_type: str = Field(..., description="合同类型")
    project_name: str = Field(..., description="项目名称")
    subitem_name: str = Field(..., description="子项名称")
    total_price_str: str = Field(..., description="合同金文本")
    total_price: float = Field(default=0, description="合同金额")
    date: str = Field(..., description="合同签订日期")
    year: int = Field(default=0, description="合同签订年份")
    # buyer: str = Field(..., description="买方名称")
    supplier: str = Field(..., description="卖方名称")
    main_equipments: List[str] = Field(..., description="主要设备")
    sub_equipments: List[str] = Field(..., description="分项设备")

def exact_docx_text(docx_path)->str:
    style_map = dedent("""\
      p =>
      b =>
      i =>""")
    contact_info = mammoth.convert_to_html(docx_path,style_map=style_map,include_default_style_map=False)
    logging.info(contact_info.value[:100]+'...'+contact_info.value[-100:])
    return contact_info.value

def run_extract_agent(content):
    instructions = ["严格按照expected_output中的键解析合同中的信息",
                "直接输出合同信息文本，严禁使用json格式或者```markdown```包裹",
                "严禁遗漏在expected_output中提到的任何信息",
                "合同类型：通常为:订货合同|供货合同，也可据实填写，无填None",
                "如果存在合同双方信息：分为买方和卖方两项单独分别列出",
                "如果存在最终供货设备表，禁止遗漏任何设备和价格信息",
                "如果存在分项报价表中，禁止遗漏任何设备和价格信息",
                "最终供货一览表和分项报价表按原始格式输出，(例如原始格式可能是html格式，需要保留对应的结构)"
                "最后按照expected_output中的键值对输出：买方合同编号:xxx，卖方合同编号：xxx...格式"
                ]
    expected_output = dedent('''买方合同编号
    合同类型
    项目名称
    子项名称
    买方名称
    卖方名称
    合同双方
      - 名称
      - 地址 
      - 联系人
      - 电话
      - 银行账号
      - 纳税人登记号
      - 单位名称
    合同金额
    不含税总价
    税额
    合同签订日期
    最终供货一览表
    分项报价表''')
    agent = Agent(model=OpenAILike(**settings,temperature=0,max_retries=3),description="你是一位合同解析专员，严格严谨遵守约定",instructions=instructions,expected_output=expected_output,telemetry=False)
    response = agent.run(message=content)
    return response.content

def extract_contact_info(content):
    max_retries = 3
    for i in range(max_retries):
        try:
            response = run_extract_agent(content)        
            if response:
                contact_exact_result = response.strip()
                logging.info(f"Contact extraction result: {contact_exact_result}")
                return contact_exact_result
        except Exception as e:
            logging.error(f"Error in contact extraction: {e}")
    return None

# 转换函数
def dict_to_str_with_mapping(d, mapping):
    list_chunk_size = 10  # 设备列表截断数，防止语义向量化之后被平均
    items = []
    for k, v in d.items():
        if k not in mapping:
            continue  # 跳过 mapping 中不存在的键
        new_key = mapping[k]
        if isinstance(v, list):
            v_str = ','.join(str(i) for i in v[:list_chunk_size])
        else:
            v_str = str(v)
        items.append(f"{new_key}: {v_str}")
    return ' '.join(items)


def transfer(meta_data:ContractMeta):
    # 键名映射表
    mapping = {
        'contact_no': '合同编号',
        'project_name': '项目名称',
        'subitem_name':'子项名称',
        'supplier': '供应商',
        'main_equipments':'主要设备',
        'sub_equipments':'分项设备'
    }
    logging.info(meta_data)
    contact_meta_dict = meta_data.model_dump()
    # meta数据转换为字符串
    result = dict_to_str_with_mapping(contact_meta_dict, mapping)
    logging.info(result)
    return result


def extract_contact_meta_data(content)->Tuple[str,ContractMeta]:
    meta_instructions = ["从合同数据中提取关键信息","没有值或者为None的字段，填:无","主要设备从最终供货一览表中提取","分项设备从分项报价表中提取","数据提取严格对应，不要遗漏，不要错对提取源"]
    meta_agent = Agent(model=OpenAILike(**deepseek_settings,temperature=0,max_retries=10),instructions=meta_instructions,response_model=ContractMeta,use_json_mode=True,telemetry=False)
    meta_agent_response = meta_agent.run(message=content)
    meta_data:ContractMeta = meta_agent_response.content
    meta_data_str = transfer(meta_data)
    return meta_data_str,meta_data
    

def extract_contact_meta_data_from_file(file_path)->Tuple[str,str,ContractMeta]:
    full_doc = exact_docx_text(file_path)
    llm_size_chunk = 32000 # llm最大处理的size
    full_doc = full_doc[:llm_size_chunk]
    extract_doc_content = extract_contact_info(full_doc)
    meta_data_str,meta_contract = extract_contact_meta_data(extract_doc_content)
    return extract_doc_content,meta_data_str,meta_contract

def append_to_csv(file_path, data_dict):
    file_exists = os.path.exists(file_path)
    if file_exists:
        df_existing = pd.read_csv(file_path, encoding='utf-8-sig')
        df_new = pd.DataFrame([data_dict])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(file_path, index=False, encoding='utf-8-sig')
    else:
        df_new = pd.DataFrame([data_dict])
        df_new.to_csv(file_path, index=False, encoding='utf-8-sig')

def read_processed_files(csv_path):
    '''读取 CSV，将每一行的文件路径存入集合，便于后续查重'''
    processed_files = set()
    try:
        with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if row:
                    processed_files.add(row[0])
    except FileNotFoundError:
        pass
    return processed_files

def list_files_in_folder(folder_path):
    '''该函数递归遍历文件夹，返回所有文件的完整路径。'''
    files = []
    for root, dirs, filenames in os.walk(folder_path):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

def write_processed_file(csv_path, file_path):
    '''在处理新文件后，将其路径追加写入 CSV。'''
    with open(csv_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([file_path])

def process_contact_file(file_path):
    '''处理文件，提取合同信息，转换为向量，存储到数据库'''
    db_csv_path = r"C:\Lee\work\contract\csv\contract_data.csv"
    
    extract_doc,meta_data_str,meta_contract = extract_contact_meta_data_from_file(file_path)
    embedding = get_embedding(meta_data_str)
    embedding_json = json.dumps(embedding)
    store_data = {
      "vector": embedding_json, 
      "meta_str": meta_data_str,
      "doc": extract_doc,
      "contact_no": meta_contract.contact_no,
      "project_name": meta_contract.project_name,
      "subitem_name": meta_contract.subitem_name,
      "total_price_str": meta_contract.total_price_str,
      "total_price": meta_contract.total_price,
      "date": meta_contract.date,
      "year": meta_contract.year,
      "supplier": meta_contract.supplier,
      "main_equipments": ','.join(meta_contract.main_equipments),
      "sub_equipments": ','.join(meta_contract.sub_equipments)
    }
    
    append_to_csv(db_csv_path, store_data)

def process_files(folder_path, csv_path):
    '''先读取已处理文件集合，然后遍历所有文件，未处理的才执行处理函数并记录。'''
    processed_files = read_processed_files(csv_path)
    files = list_files_in_folder(folder_path)
    for file in files:
        if file not in processed_files:
            process_contact_file(file)
            write_processed_file(csv_path, file)

def get_embedding(text,model='text-embedding-v4',dimensions=2048):
    client = OpenAI(
        api_key=api_key, 
        base_url=base_url
    )

    completion = client.embeddings.create(
        model=model,
        input=text,
        dimensions=dimensions, # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
        encoding_format="float"
    )
    
    return completion.data[0].embedding

if __name__ == '__main__':
    contact_path = r"C:\Lee\work\contract\demo"
    processed_csv_path = r'C:\Lee\work\contract\csv\processed_files.csv'
    process_files(contact_path, processed_csv_path)
    # process_contact_file(contact_path)
