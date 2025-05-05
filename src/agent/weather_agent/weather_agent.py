from dotenv import load_dotenv
import time
import jwt
import httpx
import re
import xml.etree.ElementTree as ET
from typing import Dict, Tuple, List
import json
from litellm import completion
import os
from enum import StrEnum

import sys
sys.path.append('../')  # 添加上级目录从而可以导入本地包

from weather_prompt import weather_system_prompt_cot

load_dotenv()

class Memory:
    def __init__(self):
        self.tag=None
        self.weight=None
        self.description=None

class BaseAgent:
    '''
    Retrieval: 实现了LLM的主动提问以及向用户回答
    Tools: 实现了LLM调用外部工具补充自身知识或者需要权威知识以及根据工具的响应进行分析
    Memory: 实现了LLM的记忆机制，在新消息到达的时候可以并行的write记忆，然后可以recall memory进行回忆，主要目的是focus LLM's attention。
    '''
    def __init__(self,messages,system_prompt,model_name, api_key, base_url,temperature=0.2, num_retries=3,is_debug=True):
        self.is_debug = is_debug
        # dict key:long memory,short memory
        self.memory:Dict[str,List[Memory]] = {}
        self.messages = messages
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.num_retries = num_retries
        self.temperature = temperature
        
    def get_assistant_response(self):
        try:
            # 使用 litellm 的 completion 方法
            stream = completion(
                model=self.model_name,
                messages=[{"role": "system", "content": self.system_prompt}] + [{"role": m["role"], "content": m["content"]} for m in self.messages],
                api_key=self.api_key,
                base_url=self.base_url,
                stream=True,
                temperature=self.temperature,
                num_retries=self.num_retries  # 配置自动重试次数
            )
            return stream
        except Exception as e:
            print(f"获取助手回复时出错: {e}")
            return None
     
    def strip_outer_tag(self,xml_str: str) -> str:
        """移除字符串XML的最外层标签"""
        start = xml_str.find('>') + 1
        end = xml_str.rfind('<')
        return xml_str[start:end].strip()

    def parse_input_text(self,input_text: str) -> Tuple[str, Dict]:
        """
        解析输入文本，提取thinking内容和action中的工具调用信息
        
        参数:
            input_text: 输入文本，包含<thinking>和<action>标签
            
        返回:
            Tuple[str, Dict]: 
                第一个元素是thinking内容，
                第二个元素是包含工具名和参数字典的字典
        """
        # 解析thinking内容
        thinking_start = input_text.find("<thinking>") + len("<thinking>")
        thinking_end = input_text.find("</thinking>")
        thinking_content = input_text[thinking_start:thinking_end].strip()
        
        # 解析action内容
        action_start = input_text.find("<action>") + len("<action>")
        action_end = input_text.find("</action>")
        action_content = input_text[action_start:action_end].strip()
        
        # 解析工具调用信息
        tool_info = {}
        
        try:
            # 包裹在根标签中确保XML格式正确
            root = ET.fromstring(f"<root>{action_content}</root>")
            if len(root) > 0:
                # 工具名是第一个子元素的标签名
                tool_element = root[0]
                tool_name = tool_element.tag
                tool_info["tool_name"] = tool_name
                
                # 解析参数 - 移除最外层标签
                params = {}
                for param in tool_element:
                    param_xml = ET.tostring(param, encoding='unicode').strip()
                    # 移除最外层标签
                    if param.text or len(param) > 0:  # 有内容或子元素
                        params[param.tag] = self.strip_outer_tag(param_xml)
                    else:  # 空标签
                        params[param.tag] = ""
                
                tool_info["parameters"] = params
                
        except ET.ParseError as e:
            print(f"解析XML时出错: {e}")
            return thinking_content, {"error": str(e)}
        
        return thinking_content, tool_info

    def hook_interactive(self,tool_name):
        if tool_name in ['ask_followup_question','attempt_completion']:
            return True
        else:
            return False

    def execute_action(self,action_data):
        """
        执行动作的工具方法
        
        :param action_data: 动作数据，格式如 {'tool_name': 'city_lookup', 'parameters': {'location': '北京'}}
        :return: 是否需要等待用户输入，动作执行结果
        """
        tool_name = action_data.get('tool_name')
        parameters = action_data.get('parameters', {})
        
        if not hasattr(self, tool_name):
            tool_result = [{
                "type": "text",
                "text": f"[{tool_name}] Result:"
            },
            {
                "type": "text",
                "text": '客户端没有名为 {tool_name} 的工具方法，请仔细检查可用工具，并选择正确的工具和参数。'
            }]
            return False,tool_result
        if self.hook_interactive(tool_name=tool_name):
            tool_result = [{
                "type": "interactive",
                "text": f"[{tool_name}] wait for user input"
            }]
            return True,tool_result
        try:
            method = getattr(self, tool_name)
            method_result = method(**parameters)
            tool_result = [{
                "type": "text",
                "text": f"[{tool_name}] Result:"
            },
            {
                "type": "text",
                "text": json.dumps(method_result, ensure_ascii=False)
            }]
            return False,tool_result
        except Exception as e:
            raise ValueError(f"执行 {tool_name} 工具方法时出错,传递参数: {parameters}, 错误信息:{e}")

    def tool_process(self,response):
        '''使用工具客户端处理llm返回的消息'''
        thinking,action = self.parse_input_text(response)
        is_interactive,tool_result = self.execute_action(action)
        if self.is_debug:
            print('=====:','tool_process 处理结果:')
            print('Thinking:',thinking)
            print('Action:',action)
            print('Tool Result:',tool_result)
            print('======','completed tool_process')
        return is_interactive,tool_result,action

    def build_tool_result_messages(self,is_interactive,tool_result,action,input_message):
        if is_interactive:
            # 进入到用户交互环节
            # 1.先将Action中的结果显示给用户,如果是`attempt_completion`,显示`result`,如果是`ask_followup_question`,显示`question`和`follow_up`
            # 2.搜集用户输入的消息拼接msg发送给LLM
            self.messages.append({'role':'user','content':[{
                    "type": "text",
                    "text": f"[{action.get('tool_name')}] Result:"
                },
                {
                    "type": "text",
                    "text": input_message   # user input message
                }]})
        else:
            self.messages.append({'role':'user','content':tool_result})
    
    def run(self):
        '''启动Agent，开始对话'''
        is_user_turn= False
        while True:
            role = "assistant"
            stream = self.get_assistant_response()
            if stream:
                response = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        response += chunk.choices[0].delta.content
                if self.is_debug:
                    print('======')
                    print(f"LLm: \n{response}")
                    print('======')
                if is_user_turn:
                    role = "user"
                else:
                    role = "assistant"
                self.messages.append({"role": role, "content": response})
                is_user_turn = not is_user_turn

            else:
                print("No response from the model")
                
            is_interactive,tool_result,action = self.tool_process(response = response)
            if is_interactive:
                user_input = input("User: ")
                if user_input == "exit":
                    break
                else:
                    self.build_tool_result_messages(is_interactive,tool_result,action,input_message = user_input) 
                    is_interactive = False
                is_user_turn = False
            else:
                self.build_tool_result_messages(is_interactive,tool_result,action,input_message = "")
                is_user_turn = not is_user_turn
        
    
    def ask_followup_question(self,question,follow_up):
        '''Retrieval(Query/Results):ask_followup_question'''
        return {'status': 'completed','question': question, 'follow_up': follow_up}
    
    def attempt_completion(self,result):
        '''Retrieval(Query/Results):attempt_completion'''
        return {'status': 'completed','result': result}
    
    # TODO: 实现一个回溯历史消息的上下文分析工具，根据已有消息进行回答，而不是请求特定的获取信息的工具   
    def context_analysis(self,messages):
        '''Retrieval(Query/Results):attempt_completion'''
        pass
    
    # TODO: 实现一个深度思考研究工具，这个工具可以自定义更加高级|更全面|更具体的system_prompt，实现从深层次角度理解用户意图，允许LLM从原环境的系统提示词的限定中解放出来，尝试从深度思考的角度进行回答，提升LLM的智能程度。
    def deeper_thinking(self,system_prompt):
        pass
    
    # TODO: 实现一个工具结果分析工具，分析工具的输出结果，提取出有用的信息，并将其转换为可读性更好或者更精简的格式，如将API返回的json数据转换为可读性更好的文字或适合LLM阅读的格式描述。
    def tool_result_analysis(self,tool_result):
        pass
    
    # TODO: 实现一个memory机制（这里的memory这是一个原型，memory的实现本质是实现一个历史消息的分析系统），当messages长度超过一定数量（或者其它约束条件）时，将历史消息按照语义分类为`long memory,short memory,memory的结构包含三部分:tag,weight,description,tag是记忆标签（限定标签数量，由系统定义）;weight是这类tag的message在历史对话中的权重,可由类似于token数进行计算占比;description是这个tag中message的摘要,并将long memory存储到memory库中，减低messages的复杂度，如果有关联后续做memory recall进行回忆。
    def memory(self,messages):
        pass
    
    # TODO: 实现一个记忆唤醒的工具，当用户输入的消息与历史消息相似度较高时，触发记忆唤醒，将之前的消息进行记忆唤醒，然后检索相关信息作为context，提升LLM的智能程度。
    def memory_recall(self,messages):
        pass
class WeatherAgent(BaseAgent):
    def __init__(self,messages,system_prompt,model_name, api_key, base_url,temperature=0.2, num_retries=3,is_debug=True):
        super().__init__(messages,system_prompt,model_name, api_key, base_url,temperature, num_retries,is_debug)
        self.private_key = """-----BEGIN PRIVATE KEY-----
MC4CAQAwBQYDK2VwBCIEIJIE87KurF9ZlyQQdyfMeiWbO+rNAoCxvJVTC//JnYMQ
-----END PRIVATE KEY-----"""
        self.project_id = "3AE3REGEEV"
        self.key_id = 'CMWDXN77PG'
        self.api_host = 'https://mr6r6t9rj9.re.qweatherapi.com'
        self.token = self.get_weather_jwt()

    def format_location(self,location):
        # 正则表达式校验：匹配 "数字,数字" 经纬度格式（可以是浮点数或整数）
        pattern = r'^-?\d+(\.\d+)?,-?\d+(\.\d+)?$'
        
        if not re.fullmatch(pattern, location):
            return location
        
        # 分割字符串
        lon, lat = location.split(',')
        
        # 转换为浮点数并格式化为小数点后两位
        formatted_lon = "{:.2f}".format(float(lon))
        formatted_lat = "{:.2f}".format(float(lat))
        
        # 重新组合成字符串
        formatted_location = f"{formatted_lon},{formatted_lat}"
        
        return formatted_location

    def get_weather_jwt(self):
        payload = {
            'iat': int(time.time()) - 100,
            'exp': int(time.time()) + 86300,
            'sub': self.project_id
        }
        headers = {
            'kid': self.key_id
        }

        # Generate JWT
        encoded_jwt = jwt.encode(payload, self.private_key, algorithm='EdDSA', headers = headers)
        if self.is_debug:
            print(encoded_jwt)
        return encoded_jwt

    def city_lookup(self,location):
        '''
        城市搜索API提供全球地理位位置、全球城市搜索服务，支持经纬度坐标反查、多语言、模糊搜索等功能。
        
        参数:
            location: (必选)需要查询地区的名称，支持文字、以英文逗号分隔的经度,纬度坐标（十进制，最多支持小数点后两位）、LocationID或Adcode（仅限中国城市）。例如 location=北京 或 location=116.41,39.92
            
        返回:
            json格式数据
        '''
        location = self.format_location(location)
        path = '/geo/v2/city/lookup'
        url = f'{self.api_host}{path}?location={location}'
        if self.is_debug:
            print("======:URL 请求相关信息")
            print(url)
            print(self.token)
        # 发送GET请求
        headers={"Authorization":f"Bearer {self.token}"}
        try:
            response = httpx.get(url,headers=headers)
            if self.is_debug:
                print('======: city_lookup 响应相关信息')
                print(response.status_code)  # 状态码
                print(response.text)         # 响应内容
            return response.json()
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {'status': 'error','message': '请求city_lookup失败'}
    
    def top_cities(self,number=None):
        '''查找热门城市'''
        if not number:
            number = 10
        else:
            number = int(number)
        path = '/geo/v2/city/top'
        url = f'{self.api_host}{path}?range=cn&number={number}'
        # 发送GET请求
        headers={"Authorization":f"Bearer {self.token}"}
        try:
            response = httpx.get(url,headers=headers)
            if self.is_debug:
                print(response.status_code)  # 状态码
                print(response.text)         # 响应内容
            return response.json()
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {'status': 'error','message': '请求top_cities失败'}
        
    def poi_lookup(self,location,city=None,type='scenic',number=None):
        '''
        地点搜索API提供全球地理位置、POI（兴趣点）搜索服务，支持经纬度坐标、城市名称、POI类型、POI名称模糊搜索等功能。
        '''
        location = self.format_location(location)
        path = '/geo/v2/city/lookup'
        if city:
            url = f'{self.api_host}{path}?location={location}&type={type}'
        else:
            url = f'{self.api_host}{path}?location={location}&type={type}&city={city}'

        headers={"Authorization":f"Bearer {self.token}"}
        try:
            response = httpx.get(url,headers=headers)
            if self.is_debug:
                print(response.text)         # 响应内容
            return response.json()
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {'status': 'error','message': '请求poi_lookup失败'}
        
    def poi_range_search(self,location,type='scenic',radius=None,number=None):
        '''
        范围搜索API提供全球范围搜索服务，支持经纬度坐标、POI类型、搜索半径、POI数量等功能。

        参数：
            location: 经纬度坐标，格式为“经度,纬度”
        '''
        location = self.format_location(location)
        path = '/geo/v2/poi/range'
        if radius:
            url = f'{self.api_host}{path}?location={location}&radius={radius}&type={type}'
        else:
            url = f'{self.api_host}{path}?location={location}&type={type}'

        headers={"Authorization":f"Bearer {self.token}"}
        try:
            response = httpx.get(url,headers=headers)
            if self.is_debug:
                print(response.text)         # 响应内容
            return response.json()
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {'status': 'error','message': '请求poi_range_search失败'}
        
    # 城市天气API组
    def city_weather_now(self,location):
        '''
        实况天气API提供全球城市实况天气查询服务，支持经纬度坐标、城市名称、多语言、数据更新时间等功能。

        参数：
            location: (必选)需要查询地区的LocationID或以英文逗号分隔的经度,纬度坐标（十进制，最多支持小数点后两位），LocationID可通过GeoAPI获取。例如 location=101010100 或 location=116.41,39.92
        '''
        location = self.format_location(location)
        path = '/v7/weather/now'
        url = f'{self.api_host}{path}?location={location}'
        headers={"Authorization":f"Bearer {self.token}"}
        try:
            response = httpx.get(url,headers=headers)
            if self.is_debug:
                print(response.text)         # 响应内容
            return response.json()
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {'status': 'error','message': '请求weather_now失败'}
        
    def city_weather_daily_forecast(self,location,forecast_days=None):
        '''
        每日天气预报API，提供全球城市未来3-30天天气预报，包括：日出日落、月升月落、最高最低温度、天气白天和夜间状况、风力、风速、风向、相对湿度、大气压强、降水量、露点温度、紫外线强度、能见度等。

        参数：
            location: (必选)需要查询地区的LocationID或以英文逗号分隔的经度,纬度坐标（十进制，最多支持小数点后两位），LocationID可通过GeoAPI获取。例如 location=101010100 或 location=116.41,39.92
            forecast_days: (必选)需要查未来多少天的天气预报，取值枚举：3,7,10,15,30
        '''
        location = self.format_location(location)
        if not forecast_days:
            forecast_days = 3
        else:
            forecast_days = int(forecast_days)
        path = '/v7/weather/'
        if forecast_days == 3:
            path += '3d'
        elif forecast_days == 7:
            path += '7d'
        elif forecast_days == 10:
            path += '10d'
        elif forecast_days == 15:
            path += '15d'
        elif forecast_days == 30:
            path += '30d'
        else:
            return {'status': 'error','message': '请求weather_daily_forecast失败,forecast_days参数错误,请选择枚举:3|7|10|15|30'}
        url = f'{self.api_host}{path}?location={location}'
        headers={"Authorization":f"Bearer {self.token}"}
        try:
            response = httpx.get(url,headers=headers)
            if self.is_debug:
                print(response.text)         # 响应内容
            return response.json()
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {'status': 'error','message': '请求weather_daily_forecast失败'}
        
    def city_weather_hourly_forecast(self,location,hours=None):
        '''
        逐小时预报API提供全球城市逐小时天气预报查询服务，支持经纬度坐标、城市名称、多语言、数据更新时间等功能。

        参数：
            location: (必选)需要查询地区的LocationID或以英文逗号分隔的经度,纬度坐标（十进制，最多支持小数点后两位），LocationID可通过GeoAPI获取。例如 location=101010100 或 location=116.41,39.92
            hours: (必选)需要查未来多少小时的天气预报，取值枚举：24,48,72,96,120
        '''
        location = self.format_location(location)
        if not hours:
            hours = 24
        else:
            hours = int(hours)
        path = '/v7/weather/'
        if hours == 24:
            path += '24h'
        elif hours == 72:
            path += '72h'
        elif hours == 168:
            path += '168h'
        else:
            return {'status': 'error','message': '请求weather_hourly_forecast失败,hours参数错误,请选择枚举:24|72|168'}
        url = f'{self.api_host}{path}?location={location}'
        headers={"Authorization":f"Bearer {self.token}"}
        try:
            response = httpx.get(url,headers=headers)
            if self.is_debug:
                print(response.text)         # 响应内容
            return response.json()
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {'status': 'error','message': '请求weather_hourly_forecast失败'}
        
    # 分钟预报API组
    def weather_rainy_forecast_minutes(self,location):
        '''
        分钟级降水（临近预报）支持中国1公里精度的未来2小时每5分钟降雨预报数据。

        参数:
            location(必选)需要查询地区的以英文逗号分隔的经度,纬度坐标（十进制，最多支持小数点后两位）。例如 location=116.41,39.92
        '''
        location = self.format_location(location)
        path = '/v7/minutely/5m'
        url = f'{self.api_host}{path}?location={location}'
        headers={"Authorization":f"Bearer {self.token}"}
        try:
            response = httpx.get(url,headers=headers)
            if self.is_debug:
                print(response.text)         # 响应内容
            return response.json()
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {'status': 'error','message': '请求weather_rainy_forecast_minutes失败'}
    
    # 格点天气API组
    # 以经纬度为基准的全球高精度、公里级、格点化天气预报产品，包括任意经纬度的实时天气和天气预报。
    def gird_weather_now(self,location):
        '''
        基于全球任意坐标的高精度实时天气，精确到3-5公里范围，包括：温度、湿度、大气压、天气状况、风力、风向等。

        参数：
            location(必选)需要查询地区的以英文逗号分隔的经度,纬度坐标（十进制，最多支持小数点后两位）。例如 location=116.41,39.92
        '''
        location = self.format_location(location)
        path = '/v7/grid-weather/now'
        url = f'{self.api_host}{path}?location={location}'
        headers={"Authorization":f"Bearer {self.token}"}
        try:
            response = httpx.get(url,headers=headers)
            if self.is_debug:
                print(response.text)         # 响应内容
            return response.json()
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {'status': 'error','message': '请求gird_weather_now失败'}
        
    def gird_weather_forecast(self,location,forecast_days=None):
        '''
        基于全球任意坐标的高精度每日天气预报，精确到3-5公里范围，包括温度、湿度、大气压、天气状况、风力、风向等。

        参数：
            location(必选)需要查询地区的以英文逗号分隔的经度,纬度坐标（十进制，最多支持小数点后两位）。例如 location=116.41,39.92
            forecast_days(必选)需要查未来多少天的天气预报，取值枚举：3,7,10,15,30
        '''
        location = self.format_location(location)
        if not forecast_days:
            forecast_days = 3
        else:
            forecast_days = int(forecast_days)
        path = '/v7/grid-weather/'
        if forecast_days == 3:
            path += '3d'
        elif forecast_days == 7:
            path += '7d'
        else:
            return {'status': 'error','message': '请求gird_weather_forecast失败,forecast_days参数错误,请选择枚举:3|7'}
        url = f'{self.api_host}{path}?location={location}'
        headers={"Authorization":f"Bearer {self.token}"}
        try:
            response = httpx.get(url,headers=headers)
            if self.is_debug:
                print(response.text)         # 响应内容
            return response.json()
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {'status': 'error','message': '请求gird_weather_forecast失败'}
        
    def gird_weather_hourly_forecast(self,location,hours=None):
        '''
        基于全球任意坐标的高精度逐小时天气预报，精确到3-5公里范围，包括温度、湿度、大气压、天气状况、风力、风向等。

        参数：
            location(必选)需要查询地区的以英文逗号分隔的经度,纬度坐标（十进制，最多支持小数点后两位）。例如 location=116.41,39.92
            hours(必选)需要查未来多少小时的天气预报，取值枚举：24,48,72,96,120
        '''
        location = self.format_location(location)
        if not hours:
            hours = 24
        else:
            hours = int(hours)
        path = '/v7/grid-weather/'
        if hours == 24:
            path += '24h'
        elif hours == 72:
            path += '72h'
        else:
            return {'status': 'error','message': '请求gird_weather_hourly_forecast失败,hours参数错误,请选择枚举:24|72'}
        url = f'{self.api_host}{path}?location={location}'
        headers={"Authorization":f"Bearer {self.token}"}
        try:
            response = httpx.get(url,headers=headers)
            if self.is_debug:
                print(response.text)         # 响应内容
            return response.json()
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {'status': 'error','message': '请求gird_weather_hourly_forecast失败'}
        
    # 天气指数预报组
    def weather_indices(self,location,forecast_days=None):
        '''
        获取中国和全球城市天气生活指数预报数据。
        中国天气生活指数：舒适度指数、洗车指数、穿衣指数、感冒指数、运动指数、旅游指数、紫外线指数、空气污染扩散条件指数、空调开启指数、过敏指数、太阳镜指数、化妆指数、晾晒指数、交通指数、钓鱼指数、防晒指数
        海外天气生活指数：运动指数、洗车指数、紫外线指数、钓鱼指数
        '''
        location = self.format_location(location)
        if not forecast_days:
            forecast_days = 1
        else:
            forecast_days = int(forecast_days)
        path = '/v7/indices/'
        if forecast_days == 1:
            path += '1d'
        elif forecast_days == 3:
            path += '3d'
        else:
            return {'status': 'error','message': '请求weather_indices失败,forecast_days参数错误,请选择枚举:1|3'}
        # TODO: 后续可增加type参数，可选1,2,3,4
        url = f'{self.api_host}{path}?location={location}&type=0'
        headers={"Authorization":f"Bearer {self.token}"}
        try:
            response = httpx.get(url,headers=headers)
            if self.is_debug:
                print(response.text)         # 响应内容
            return response.json()
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {'status': 'error','message': '请求weather_indices失败'}
        
    # 空气质量API组
    def air_quality(self,latitude,longitude):
        '''
        实时空气质量API提供指定地点的实时空气质量数据，精度为1x1公里。

        基于各个国家或地区当地标准的AQI、AQI等级、颜色和首要污染物
        和风天气通用AQI
        污染物浓度值、分指数
        健康建议
        相关联的监测站信息

        参数：
            latitude(必选)需要查询地区的纬度坐标（十进制，最多支持小数点后2位）
            longitude(必选)需要查询地区的经度坐标（十进制，最多支持小数点后2位）
        '''
        latitude = "{:.2f}".format(float(latitude))
        longitude = "{:.2f}".format(float(longitude))
        path = '/airquality/v1/current/'
        url = f'{self.api_host}{path}{latitude}/{longitude}'
        headers={"Authorization":f"Bearer {self.token}"}
        try:
            response = httpx.get(url,headers=headers)
            if self.is_debug:
                print(response.text)         # 响应内容
            return response.json()
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {'status': 'error','message': '请求air_quality失败'}
    
    def air_quality_hourly_forecast(self,latitude,longitude):
        '''
        空气质量小时预报API提供未来24小时空气质量的数据，包括AQI、污染物浓度、分指数以及健康建议。
        '''
        latitude = "{:.2f}".format(float(latitude))
        longitude = "{:.2f}".format(float(longitude))
        path = '/airquality/v1/hourly/'
        url = f'{self.api_host}{path}{latitude}/{longitude}'
        headers={"Authorization":f"Bearer {self.token}"}
        try:
            response = httpx.get(url,headers=headers)
            if self.is_debug:
                print(response.text)         # 响应内容
            return response.json()
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {'status': 'error','message': '请求air_quality_hourly_forecast失败'}
        
    def air_quality_daily_forecast(self,latitude,longitude):
        '''
        空气质量每日预报API提供未来3天的空气质量（AQI）预报、污染物浓度值和健康建议。
        '''
        latitude = "{:.2f}".format(float(latitude))
        longitude = "{:.2f}".format(float(longitude))
        path = '/airquality/v1/daily/'
        url = f'{self.api_host}{path}{latitude}/{longitude}'
        headers={"Authorization":f"Bearer {self.token}"}
        try:
            response = httpx.get(url,headers=headers)
            if self.is_debug:
                print(response.text)         # 响应内容
            return response.json()
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {'status': 'error','message': '请求air_quality_daily_forecast失败'}
        
    def air_quality_station_data(self,LocationID):
        '''
        监测站数据API提供各个国家或地区监测站的污染物浓度值。
        '''
        path = '/airquality/v1/station/'
        url = f'{self.api_host}{path}{LocationID}'
        headers={"Authorization":f"Bearer {self.token}"}
        try:
            response = httpx.get(url,headers=headers)
            if self.is_debug:
                print(response.text)         # 响应内容
            return response.json()
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {'status': 'error','message': '请求air_quality_station_data失败'}
        
    # 历史天气信息API组
    def weather_history(self,location,date):
        '''
        获取最近10天的天气历史再分析数据。

        参数：
            location(必选需要查询的地区，仅支持LocationID，LocationID可通过GeoAPI获取。例如 location=101010100
            date(必选)需要查询日期，格式为yyyyMMdd，例如 date=20200531
        '''
        path = '/v7/historical/weather'
        url = f'{self.api_host}{path}?location={location}&date={date}'
        headers={"Authorization":f"Bearer {self.token}"}
        try:
            response = httpx.get(url,headers=headers)
            if self.is_debug:
                print(response.text)         # 响应内容
            return response.json()
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {'status': 'error','message': '请求weather_history失败'}
        
    def air_quality_history(self,location,date):
        '''
        获取最近10天的中国空气质量历史再分析数据。

        参数：
            location(必选需要查询的地区，仅支持LocationID，LocationID可通过GeoAPI获取。例如 location=101010100
            date(必选)需要查询日期，格式为yyyyMMdd，例如 date=20200531
        '''
        path = '/v7/historical/air'
        url = f'{self.api_host}{path}?location={location}&date={date}'
        headers={"Authorization":f"Bearer {self.token}"}
        try:
            response = httpx.get(url,headers=headers)
            if self.is_debug:
                print(response.text)         # 响应内容
            return response.json()
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {'status': 'error','message': '请求air_quality_history失败'}


if __name__ == '__main__':
    # ==============Model Choice====================
    model_choices = {
        "deepseek-chat": {
            'model_name': 'deepseek/deepseek-chat',
            'api_key': os.getenv("DEEPSEEK_API_KEY"),
            'base_url': os.getenv("DEEPSEEK_API_BASE_URL")
        },
        'open-router-gemini-flash': {
            'model_name': 'openrouter/google/gemini-2.5-flash-preview',
            'api_key': os.getenv("OPENROUTER_API_KEY"),
            'base_url': os.getenv("OPENROUTER_BASE_URL")
        }
    }

    class ModelChoice(StrEnum):
        OPENER_ROUTER_GEMINI = 'open-router-gemini-flash'
        DEEPSEEK = "deepseek-chat"
        
    def initialize_client(model_choice: ModelChoice):
        if model_choice not in model_choices:
            print(f"Invalid model choice: {model_choice}")
        model_info = model_choices[model_choice]
        api_key = model_info['api_key']
        base_url = model_info['base_url']
        model_name = model_info['model_name']
        print(f"Initializing for {model_name} with API_KEY={api_key} and BASE_URL={base_url}")
        if not api_key:
           print("API_KEY is not set")
        if not base_url:
           print("BASE_URL is not set")
        return model_name, api_key, base_url
    # Initialize client
    MAX_MESSAGES = 20
    MAX_INPUT_LENGTH = 1000
    MODEL_NAME, API_KEY, BASE_URL = initialize_client(ModelChoice.DEEPSEEK)
    
    # ==========End of Model Choice=================
    
    # ==========Weather Agent=======================
    
    messages = [{'role':'user','content':input("User: ")}]
    
    weather_agent = WeatherAgent(messages=messages,system_prompt=weather_system_prompt_cot, model_name=MODEL_NAME, api_key=API_KEY, base_url=BASE_URL)
    weather_agent.run()
    
