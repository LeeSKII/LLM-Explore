import lancedb
from openai import OpenAI
import os
from dotenv import load_dotenv
import pyarrow as pa

load_dotenv()  # 加载环境变量

api_key=os.getenv("QWEN_API_KEY")
base_url=os.getenv("QWEN_API_BASE_URL")
local_base_url = 'http://192.168.0.166:8000/v1'
local_model_name = 'Qwen3-235B'
model_name = 'qwen-plus-latest'
embedding_model_id = 'text-embedding-v4'
dashscope_api_key = os.getenv("QWEN_API_KEY")

temperature = 0.1
local_settings = {
  'api_key' : '123',
  'base_url' : local_base_url,
  'id' : local_model_name,
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

client = OpenAI(
    api_key=dashscope_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
)

data_payload = '衣服的质量杠杠的，很漂亮，不枉我等了这么久啊，喜欢，以后还来这里买'

completion = client.embeddings.create(
    model=embedding_model_id,
    input=data_payload,
    dimensions=2048,
    encoding_format="float"
)

# print(completion.model_dump_json())

vector = completion.data[0].embedding

print(vector)

db = lancedb.connect("C:/Lee/work/db/my_lancedb")  # 指定数据库目录，不存在会自动创建

schema = pa.schema([
    pa.field("vector", pa.list_(pa.float32())),
    pa.field("content", pa.utf8()),
])

table = db.create_table("tmp_table", schema=schema)

new_data = [
    {"vector": vector, "content": data_payload},
]
table.add(new_data)