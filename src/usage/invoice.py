import sys
sys.path.append(r'E:\PythonProject\LLM-Explore\src') 
 # 添加上级目录从而可以导入本地包
from utils.tools import encode_image
from utils.llms import chat_with_llm
from dotenv import load_dotenv
import os


load_dotenv()
api_key = "123"
base_url = os.getenv("SELF_HOST_URL")

def exact_info_from_invoice(invoice_path):
    base64_image = encode_image(invoice_path)
    message = [
        {     
            'role':'user','content':[
                {
                    "type": "text",
                    "text": '''这是一张发票，请识别发票信息，thinking it step by step，写出思考过程，输出成json格式，如果字段没有在发票中提供，请直接设置为''，任何数值均保留原始精度，禁止**推断任何未提供的信息**，参考的json格式如下：
                    ```json
                            {
                            "发票代码": "*",
                            "发票号码": "*",
                            "开票日期": "year-month-day",
                            "校验码": "*",
                            "购买方名称": "*",
                            "购买方纳税人识别号": "*",
                            "购买方地址、电话": "*",
                            "购买方开户行及账号": "*",
                            "货物": [
                                {
                                "名称": "*",
                                "规格型号": "*",
                                "单位": "*",
                                "数量": number,
                                "单价": number,
                                "金额": number,
                                "税率": "*",
                                "税额": number
                                }
                            ],
                            "合计金额": number,
                            "价税合计": number,
                            "销售方名称": "*",
                            "销售方纳税人识别号": "*",
                            "销售方地址、电话": "*",
                            "销售方开户行及账号": "*",
                            "备注": "*"
                            }
                        ```
                    。'''
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                }
            ]
        }
    ]
    result = chat_with_llm(message=message,api_key=api_key,base_url=base_url,model_name='/root/Qwen2.5-VL-32B-Instruct-AWQ',temperature=0)
    return result


if __name__ == '__main__':
    image_path = r"E:\Temp\zzsptfp\b191.jpg"
    result = exact_info_from_invoice(image_path)
    print(result)