import csv
from agno.agent import Agent
from agno.models.openai import OpenAILike
import mammoth
from textwrap import dedent
from pydantic import BaseModel, Field
from typing import Optional, Tuple,List
import logging
import pandas as pd
import json
import re

# 使用绝对导入，不知为何相对导入失败
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))
from src.utils import logger

# logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s:%(filename)s:%(lineno)d - %(message)s',datefmt='%Y-%m-%d %H:%M:%S',level=logging.INFO) 

#------------------ settings ------------------
import os
from dotenv import load_dotenv
load_dotenv()
api_key=os.getenv("QWEN_API_KEY")
base_url=os.getenv("QWEN_API_BASE_URL")
model_name = 'qwen-plus'

qwen_settings = {
  'api_key' : api_key,
  'base_url' : base_url,
  'id' : model_name,
  'temperature':0.01
}

settings = qwen_settings
#------------------ settings ------------------

class Party(BaseModel):
    name: str = Field(default="", description="公司名称")
    address: str = Field(default="", description="公司地址")
    contact_person: Optional[str] = Field(default=None, description="联系人")
    phone: Optional[str] = Field(default=None, description="联系电话")
    bank_account: Optional[str] = Field(default=None, description="银行账号")
    tax_id: Optional[str] = Field(default=None, description="纳税人登记号")

class Contract(BaseModel):
    buyer_contract_number: Optional[str] = Field(default=None, description="买方合同编号")
    seller_contract_number: Optional[str] = Field(default=None, description="卖方合同编号")
    contract_type: str = Field(default="", description="合同类型")
    project_name: Optional[str] = Field(default=None, description="项目名称")
    sub_project_name: Optional[str] = Field(default=None, description="子项名称")
    buyer: Party = Field(default_factory=Party, description="买方信息")
    seller: Party = Field(default_factory=Party, description="卖方信息")
    total_amount: Optional[float] = Field(default=None, description="合同金额（万元）")
    total_amount_str: Optional[str] = Field(default=None, description="合同金额（万元），字符串形式")
    tax_excluded_amount: Optional[float] = Field(default=None, description="不含税总价（万元）")
    tax_amount: Optional[float] = Field(default=None, description="税额（万元）")
    signing_date: Optional[str] = Field(default=None, description="合同签订日期（格式：YYYY-MM）")

class ContractTableJudge(BaseModel):
    reason: str = Field(default="", description="判断理由和依据")
    is_equipment_table: bool = Field(default=False, description="是否属于设备供货价格信息表或清单")

#------------------ models ------------------

# 转换时过滤图片
def remove_images(element):
    return []

def remove_img_tags(text):
    # 使用正则表达式匹配 <img /> 标签
    return re.sub(r'<img\s*/>', '', text)

def filter_table_text_from_docs(docx_path)->Tuple[List[str],str]:
    """正则化提取docx文档中的表格，返回表格内容和转换文本"""
    style_map = dedent("""
    p =>
    b =>
    i =>
    """)
    contact_info = mammoth.convert_to_html(docx_path,style_map=style_map,convert_image=mammoth.images.img_element(remove_images),include_default_style_map=False)
    logging.info("docx转换成功，数据："+contact_info.value[:100]+'...'+contact_info.value[-100:])
    
    exacted_string = contact_info.value
    
    # 使用正则表达式匹配<table>标签及其内容（包括嵌套标签）
    pattern = r'<table\b[^>]*>.*?</table>'
    # re.DOTALL 使 . 匹配包括换行符在内的所有字符
    tables = re.findall(pattern, exacted_string, re.DOTALL)
    
    removed_img_exacted_string = remove_img_tags(exacted_string)
    
    return tables,removed_img_exacted_string

def run_extract_agent(content)->Contract:
    """解析合同信息成结构化数据"""
    try:
        json_mode_agent = Agent(
            model=OpenAILike(**settings),
            description="你是一位合同解析AI助手，严格严谨遵守约定，严谨虚构任何用户未提供的合同信息，如果不存在或无法判断，保持空白。",
            response_model=Contract,
            use_json_mode=True,
            retries=3,
            telemetry=False,
        )
        response = json_mode_agent.run(message=content)
        logging.info(f"解析结构化数据成功：{response.content.model_dump()}")
    except Exception as e:
        logging.error(f"Error in contact extraction: {e}")
    return response.content

def judge_table_type(contract_info:Contract,table_html:str)->ContractTableJudge:
    """判断表格类型"""
    instructions = ["根据用户提供的合同结构数据和信息表格或清单，判断该表格或清单是否属于设备供货价格信息表或清单",
                "设备供货价格信息表或清单必定包含的条目有：设备名称或品名、价格，注意这是必备条目，如果没有出现价格特征，表明该表格或清单不是设备供货价格信息表或清单",
                "设备供货价格信息表或清单可能包含的条目有：规格信号、数量等",
                "核心依据是：表格中还必须包含有效上述必备条目（设备名称或品名、价格）的的设备数据，如果没有数据，那么是无效的设备供货价格信息表或清单"
                ]
    judge_equipment_table_agent = Agent(
        model=OpenAILike(**settings),
        description="你是一位合同解析AI助手，严格严谨遵守约定，严谨虚构任何用户未提供的合同信息，如果不存在或无法判断，保持空白。",
        instructions=instructions,
        response_model=ContractTableJudge,
        use_json_mode=True,
        retries=3,
        telemetry=False,
    ) 
    message=f"用户提供的合同信息：{contract_info.model_dump()}。用户提供的设备表格或清单: {table_html}"
    response = judge_equipment_table_agent.run(message=message)  
    return response.content


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
    '''处理文件,提取合同信息,存储到csv文件中.'''
    db_csv_path = r"C:\Lee\work\contract\csv\v3\contract_data.csv"
    
    tables, contact_full_doc_string = filter_table_text_from_docs(file_path)
    if len(tables) == 0:
        logging.info(f"No table found in {file_path}")

    # 解析合同信息,使用前32000字符进行解析
    partially_extracted_contract = contact_full_doc_string[:32000]
    contract_meta = run_extract_agent(partially_extracted_contract)
    store_data = {
      "contract_meta": contract_meta.model_dump(),
      "doc": contact_full_doc_string,
      "contact_no": contract_meta.buyer_contract_number,
      "project_name": contract_meta.project_name,
      "subitem_name": contract_meta.sub_project_name,
      "total_price_str": contract_meta.total_amount_str,
      "total_price": contract_meta.total_amount,
      "date": contract_meta.signing_date,
      "supplier": contract_meta.seller.name,
      "equipment_table":[]
    }
    for table_html in tables:
        judge_result = judge_table_type(contract_meta,table_html)
        if judge_result.is_equipment_table:
            store_data["equipment_table"].append(table_html)    
    
    logging.info(f"最终解析并存储信息：{store_data}")
    append_to_csv(db_csv_path, store_data)

def process_files(folder_path, csv_path):
    '''先读取已处理文件集合，然后遍历所有文件，未处理的才执行处理函数并记录。'''
    processed_files = read_processed_files(csv_path)
    files = list_files_in_folder(folder_path)
    for file in files:
        if file not in processed_files:
            process_contact_file(file)
            write_processed_file(csv_path, file)


if __name__ == '__main__':
    contact_path = r"C:\Lee\work\contract\demo"
    processed_csv_path = r'C:\Lee\work\contract\csv\v3\processed_files.csv'
    process_files(contact_path, processed_csv_path)
    # process_contact_file(contact_path)
