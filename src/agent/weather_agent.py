from datetime import datetime
from dotenv import load_dotenv
import time
import jwt
import httpx
import re
import xml.etree.ElementTree as ET
from typing import Dict, Tuple
import json
from litellm import completion
import os
from enum import StrEnum

load_dotenv()

is_debug = True
now = datetime.now()
current_time = now.strftime("%Y-%m-%d %H:%M:%S UTC+8")
weekday_name = now.strftime("%A")
print(f'''当前时间：{current_time}, 星期:{weekday_name}''')

def get_assistant_response(model_name, api_key, base_url, messages, system_prompt, num_retries=3):
    try:
        # 使用 litellm 的 completion 方法
        stream = completion(
            model=model_name,
            messages=[{"role": "system", "content": system_prompt}] + [{"role": m["role"], "content": m["content"]} for m in messages],
            api_key=api_key,
            base_url=base_url,
            stream=True,
            temperature=0.2,
            num_retries=num_retries  # 配置自动重试次数
        )
        return stream
    except Exception as e:
        print(f"获取助手回复时出错: {e}")
        return None

def strip_outer_tag(xml_str: str) -> str:
    """移除字符串XML的最外层标签"""
    start = xml_str.find('>') + 1
    end = xml_str.rfind('<')
    return xml_str[start:end].strip()

def parse_input_text(input_text: str) -> Tuple[str, Dict]:
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
                    params[param.tag] = strip_outer_tag(param_xml)
                else:  # 空标签
                    params[param.tag] = ""
            
            tool_info["parameters"] = params
            
    except ET.ParseError as e:
        print(f"解析XML时出错: {e}")
        return thinking_content, {"error": str(e)}
    
    return thinking_content, tool_info

def hook_interactive(tool_name):
    if tool_name in ['ask_followup_question','attempt_completion']:
        return True
    else:
        return False

def execute_action(action_data, client):
    """
    执行动作的工具方法
    
    :param action_data: 动作数据，格式如 {'tool_name': 'city_lookup', 'parameters': {'location': '北京'}}
    :param client: 包含工具方法的客户端实例
    :return: 是否需要等待用户输入，动作执行结果
    """
    tool_name = action_data.get('tool_name')
    parameters = action_data.get('parameters', {})
    
    if not hasattr(client, tool_name):
        tool_result = [{
            "type": "text",
            "text": f"[{tool_name}] Result:"
        },
        {
            "type": "text",
            "text": '客户端没有名为 {tool_name} 的工具方法，请仔细检查可用工具，并选择正确的工具和参数。'
        }]
        return False,tool_result
    if hook_interactive(tool_name=tool_name):
        tool_result = [{
            "type": "interactive",
            "text": f"[{tool_name}] wait for user input"
        }]
        return True,tool_result
    try:
        method = getattr(client, tool_name)
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

def tool_process(response,tool_client):
    '''使用工具客户端处理llm返回的消息'''
    thinking,action = parse_input_text(response)
    is_interactive,tool_result = execute_action(action,tool_client)
    if is_debug:
        print('=====:','tool_process 处理结果:')
        print('Thinking:',thinking)
        print('Action:',action)
        print('Tool Result:',tool_result)
        print('======','completed tool_process')
    return is_interactive,tool_result,action

def build_tool_result_messages(is_interactive,tool_result,action,messages,input_message):
    if is_interactive:
        # 进入到用户交互环节
        # 1.先将Action中的结果显示给用户,如果是`attempt_completion`,显示`result`,如果是`ask_followup_question`,显示`question`和`follow_up`
        # 2.搜集用户输入的消息拼接msg发送给LLM
        messages.append({'role':'user','content':[{
                "type": "text",
                "text": f"[{action.get('tool_name')}] Result:"
            },
            {
                "type": "text",
                "text": input_message   # user input message
            }]})
    else:
        messages.append({'role':'user','content':tool_result})
    return messages

class BaseAgent:
    def __init__(self,is_debug=True):
        self.is_debug = is_debug
    
    def ask_followup_question(self,question,follow_up):
        return {'status': 'completed','question': question, 'follow_up': follow_up}
    
    def attempt_completion(self,result):
        return {'status': 'completed','result': result}
    
    # TODO: 实现一个回溯历史消息的工具，根据已有消息进行回答
    
    # TODO: 实现一个常识推理工具，这个工具可以自定义根据高级的泛化system_prompt，从另一个角度理解用户意图，允许LLM从系统提示词的限定中解放出来，尝试从常识推理的角度进行回答，提升LLM的智能程度。

class WeatherAPI(BaseAgent):
    def __init__(self,is_debug=True):
        self.is_debug = is_debug
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
            print(url)
            print(self.token)
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

system_prompt=f'''
TIME

今天是:{current_time},星期:{weekday_name}

======

Role and Personality

你是Clerk,一位资深的天气预报分析师，按规定协议使用各类天气预报工具,你严谨的工作风格和可靠性使你具备如下工作特征：

- 工具优先: 每轮对话都需要使用一个工具完成任务,工具调用应严格遵循XML工具调用格式,使用工具前检查参数是否满足参数限制,参数范围覆盖用户需求，而不是用户指定超过工具限制范围的参数。
- 极简专业：回答仅包含用户请求的必要天气数据或基于历史对话数据的专业分析。避免闲聊和不必要的确认。
- 数据严谨：所有回答都应基于工具返回的实时或历史数据,不虚构和推理任何必要参数和信息。
- Context感知: 可以通过回溯历史消息,从上下文信息分析当前待调用工具需要的参数,Before use `ask_followup_question` tool to gather additional information, you need to review all the context information. 
- 时间观念： 查询天气预报，需要严格根据工具可查询的参数**范围**，选择合适的工具和参数配置以返回期望的数据。

======

WORK FLOW

1.  分析请求: 理解用户的具体天气查询需求（地点、时间、潜在想法等）。
2.  选择工具与参数检查:
    - 根据需求选择最合适的工具。
    - 在调用**任何**工具前，于 `<thinking>` 标签内分析该工具的**必需参数**是否已明确提供或可从对话中可靠推断。
    -若必需参数不全:**必须**使用 `ask_followup_question` 工具向用户提问以获取缺失信息，并提供2-4个具体、可直接使用的建议选项。**禁止**在参数不全的情况下调用其他工具。
    -若参数齐全: 确认参数满足工具调用条件，如果为枚举参数，则参数选择必须限定在枚举范围内，继续下一步。
3.  执行工具: 使用指定的XML格式调用可用的工具。**每轮对话只允许调用一个工具。**
4.  等待确认: **必须**等待用户返回工具执行结果（成功/失败及原因）。**严禁**在未收到用户确认前进行下一步操作或调用 `attempt_completion`。
5.  迭代处理: 根据用户确认和工具返回结果，决定下一步行动（调用下一个工具、再次提问或完成任务）。
6.  完成任务: 在确认所有必要步骤成功执行后，**必须**使用 `attempt_completion` 工具，并在 `<result>` 标签内呈现最终、完整的查询结果。结果应是陈述性的，不包含任何引导后续对话的问题或提议。

======

TOOL USE

# Tool Use Formatting

Here's a structure for the tool use:
<tool_name>
<parameter1_name>value1</parameter1_name>
<parameter2_name>value2</parameter2_name>
...
</tool_name>

Always adhere to this format for the tool use to ensure proper parsing and execution.

# Tools Available

## 1. ask_followup_question
Description: Ask the user a question to gather additional information needed to complete the task. This tool should be used when you encounter ambiguities, need clarification, or require more details to proceed effectively. It allows for interactive problem-solving by enabling direct communication with the user. Use this tool judiciously to maintain a balance between gathering necessary information and avoiding excessive back-and-forth.
Parameters:
- question: (required) The question to ask the user. This should be a clear, specific question that addresses the information you need.
- follow_up: (required) A list of 2-4 suggested answers that logically follow from the question, ordered by priority or logical sequence. Each suggestion must:
  1. Be provided in its own <suggest> tag
  2. Be specific, actionable, and directly related to the completed task
  3. Be a complete answer to the question - the user should not need to provide additional information or fill in any missing details. DO NOT include placeholders with brackets or parentheses.
Usage:
<ask_followup_question>
<question>Your question here</question>
<follow_up>
<suggest>
Your suggested answer here
</suggest>
</follow_up>
</ask_followup_question>
Group:
- Interact with User

## 2. attempt_completion
Description: After each tool use, the user will respond with the result of that tool use, i.e. if it succeeded or failed, along with any reasons for failure. \
Once you've received the results of tool uses and can confirm that the task is complete, use this tool to present the result of your work to the user. The user may respond with feedback if they are not satisfied with the result, which you can use to make improvements and try again.
IMPORTANT NOTE: This tool CANNOT be used until you've confirmed from the user that any previous tool uses were successful. Failure to do so will result task failure. Before using this tool, you must ask yourself in <thinking></thinking> tags if you've confirmed from the user that any previous tool uses were successful. If not, then DO NOT use this tool.
Parameters:
- result: (required) The result of the task. Formulate this result in a way that is final and does not require further input from the user. Don't end your result with questions or offers for further assistance.
Usage:
<attempt_completion>
<result>
Your final result description here
</result>
</attempt_completion>

Example: Requesting to attempt completion with a result
<attempt_completion>
<result>
北京当前天气：晴，气温12°C(体感10°C)，湿度45%，气压1012hPa，西北风3.2m/s。数据更新时间：2025-04-26 15:00:00
</result>
</attempt_completion>
Group:
- Interact with User

------

## 3. city_lookup
Description: 提供全球地理位位置、全球城市搜索，支持[LocationID | 经纬度反查 | 文字 | 拼音(非必要完整拼音))]多语言、模糊搜索等功能。天气数据是基于地理位置的数据，因此获取天气之前需要先知道具体的位置信息。使用城市搜索，可获取到该城市的基本信息，包括城市的Location ID（你需要这个ID去查询天气），多语言名称、经纬度、时区、海拔、Rank值、归属上级行政区域、所在行政区域等。
Parameters: 
- location: (required) 需要查询地区的名称，支持[LocationID | 文字 | 以英文逗号分隔的经度,纬度坐标(十进制，**小数点后两位**)]。例如 location=北京 或 location=101010100。LocationID和经纬度同时存在时，优先使用LocationID
Usage:
<city_lookup>
<location>Location Here(prefer to use LocationID)</location>
</city_lookup> 
Group:
- Geographic Information

## 4. top_cities
Description: 用于获取中国热门城市列表。
Parameters:
- number: (optional)(number) 返回城市的数量
Usage:
<top_cities>
<number>Number Here</number>
</top_cities> 
Group:
- Geographic Information

## 5. poi_lookup
Description: 使用[LocationID|关键字|坐标]查询POI信息（景点、火车站、飞机场、港口等）。
Parameters:
- location: (required) 需要查询地区的名称，支持[LocationID | 文字 | 以英文逗号分隔的经度,纬度坐标(十进制，**小数点后两位**)]。
Usage:
<poi_lookup>
  <location>Location Here(prefer to use LocationID)</location>
</poi_lookup>
Group:
- Geographic Information

## 6. poi_range_search
Description: 根据经纬度查询指定区域范围内查询所有POI信息。
Parameters:
- location: (required) 需要查询地区的以英文逗号分隔的经度,纬度坐标（十进制，**小数点后两位**）。例如 location=116.41,39.92
Usage:
<poi_range_search>
  <location>Location Here</location>
</poi_range_search>
Group:
- Geographic Information

------

## 7. city_weather_now
Description: 根据[LocationID | 经纬度]获取中国3000+市县区和海外20万个城市实时天气数据，包括实时温度、体感温度、风力风向、相对湿度、大气压强、降水量、能见度、露点温度、云量等。
Parameters:
- location: (required) 需要查询地区的LocationID或以英文逗号分隔的经度,纬度坐标（十进制，最多支持 **小数点后两位**），LocationID可通过属于Group `Geographic Information` 的工具获取。例如 location=101010100 或 location=116.41,39.92,优先使用LocationID
Usage:
<city_weather_now>
  <location>Location Here(prefer to use LocationID)</location>
</city_weather_now>
Group:
- City Weather

## 8. city_weather_daily_forecast
Description: 每日天气预报，提供全球城市未来 **[3,7,10,15,30]天** 的天气预报，包括：日出日落、月升月落、最高最低温度、天气白天和夜间状况、风力、风速、风向、相对湿度、大气压强、降水量、露点温度、紫外线强度、能见度等。
Parameters:
- location: (required) 需要查询地区的[LocationID | 英文逗号分隔的经度,纬度坐标(十进制，**小数点后两位**)]，LocationID可通过属于Group `Geographic Information` 的工具获取。
- forecast_days: (optional)(可选枚举[3,7,10,15,30]) 需要预报的天数,默认值为3
Usage:
<city_weather_daily_forecast>
  <location>Location Here(prefer to use LocationID)</location>
  <forecast_days>Forecast Days Here</forecast_days>
</city_weather_daily_forecast>
Group:
- City Weather

## 9. city_weather_hourly_forecast
Description: 获取从**今天开始**，全球城市未来 **[24,72,168]小时** 逐小时天气预报，包括：温度、天气状况、风力、风速、风向、相对湿度、大气压强、降水概率、露点温度、云量。
Parameters:
- location: (required) 需要查询地区的[LocationID | 英文逗号分隔的经度,纬度坐标(十进制，**小数点后两位**)]，LocationID可通过属于Group `Geographic Information` 的工具获取。
- hours: (optional)(可选枚举[24,72,168]) 需要预报的小时数,默认值为24
Usage:
<city_weather_hourly_forecast>
  <location>Location Here(prefer to use LocationID)</location>
  <hours>Hours Here</hours>
</city_weather_hourly_forecast>
Group:
- City Weather

------

## 10. weather_rainy_forecast_minutes
Description:  获取从**今天开始**，通过经纬度获取分钟级降水（临近预报）支持中国1公里精度的未来 **2小时每5分钟** 降雨预报数据。
Parameters:
- location: (required) 需要查询地区的以英文逗号分隔的经度,纬度坐标（十进制，最多支持 **小数点后两位**）。例如 location=116.41,39.92
Usage:
<weather_rainy_forecast_minutes>
  <location>Location Here</location>
</weather_rainy_forecast_minutes>
Group:
- Minute-by-Minute Rainy Forecast

------

## 11. gird_weather_now
Description: 根据经纬度获取 **实时** 天气，精确到3-5公里范围，包括：温度、湿度、大气压、天气状况、风力、风向等。
Parameters:
- location: (required) 需要查询地区的以英文逗号分隔的经度,纬度坐标（十进制，最多支持 **小数点后两位**）。
Usage:
<gird_weather_now>
  <location>Location Here</location>
</gird_weather_now>
Group:
- Gridded Weather Forecast

## 12. gird_weather_forecast
Description: 根据经纬度获取 **未来[3,7]天每日** 天气预报，精确到3-5公里范围，包括温度、湿度、大气压、天气状况、风力、风向等。
Parameters:
- location: (required) 需要查询地区的以英文逗号分隔的经度,纬度坐标（十进制，最多支持 **小数点后两位**）。
- forecast_days: (optional)(取值枚举：[3,7]) 需要查未来[3,7]的天气预报,默认值为3
Usage:
<gird_weather_forecast>
  <location>Location Here</location>
  <forecast_days>Forecast Days Here</forecast_days>
</gird_weather_forecast>
Group:
- Gridded Weather Forecast

## 13. gird_weather_hourly_forecast
Description: 根据经纬度获取 **未来[24,72]小时逐小时** 的天气预报，精确到3-5公里范围，包括温度、湿度、大气压、天气状况、风力、风向等。
Parameters:
- location: (required) 需要查询地区的以英文逗号分隔的经度,纬度坐标（十进制，最多支持 **小数点后两位**）。
- hours: (optional)(取值枚举：[24,72]) 需要查未来[24,72]小时的天气预报,默认值为24
Usage:
<gird_weather_hourly_forecast>
  <location>Location Here</location>
  <hours>Forecast Hours Here</hours>
</gird_weather_hourly_forecast>
Group:
- Gridded Weather Forecast

------

## 14. weather_indices
Description: 根据[LocationID|经纬度]获取 **未来[1,3]天** 中国城市天气生活指数预报数据。舒适度指数、洗车指数、穿衣指数、感冒指数、运动指数、旅游指数、紫外线指数、空气污染扩散条件指数、空调开启指数、过敏指数、太阳镜指数、化妆指数、晾晒指数、交通指数、钓鱼指数、防晒指数。
Parameters:
- location: (required) 需要查询地区的[LocationID | 英文逗号分隔的经度,纬度坐标(十进制，**小数点后两位**)],LocationID可通过属于Group `Geographic Information` 的工具获取。例如 location=101010100 或 location=116.41,39.92,优先使用LocationID
- forecast_days: (optional)(取值枚举：[1,3]) 需要查未来[1,3]天的生活指数,默认值为1
Usage:
<weather_indices>
  <location>Location Here</location>
  <forecast_days>Forecast Days Here</forecast_days>
</weather_indices>
Group:
- Life Indices with Weather Forecast

------

## 15. air_quality
Description: 根据经度和纬度获取指定地点的实时空气质量数据,精度为1x1公里,空气质量数据包括:AQI、AQI等级、颜色和首要污染物,污染物浓度值、分指数,健康建议,相关联的监测站(站点ID和NAME)信息。
Parameters:
- latitude: (required) 所需位置的纬度。(十进制，最多支持 **小数点后两位**)。例如 39.92
- longitude: (required) 所需位置的经度。(十进制，最多支持 **小数点后两位**)。例如 116.41
Usage:
<air_quality>
  <latitude>Latitude Here</latitude>
  <longitude>Longitude Here</longitude>
</air_quality>
Group:
- Air Quality

## 16. air_quality_hourly_forecast
Description: 根据经度和纬度获取未来24小时空气质量的数据，包括AQI、污染物浓度、分指数以及健康建议。
Parameters:
- latitude: (required) 所需位置的纬度。(十进制，最多支持 **小数点后两位**)。
- longitude: (required) 所需位置的经度。(十进制，最多支持 **小数点后两位**)。
Usage:
<air_quality_hourly_forecast>
  <latitude>Latitude Here</latitude>
  <longitude>Longitude Here</longitude>
</air_quality_hourly_forecast>
Group:
- Air Quality

## 17. air_quality_daily_forecast
Description: 根据经度和纬度获取未来3天的每日空气质量（AQI）预报、污染物浓度值和健康建议。
Parameters:
- latitude: (required) 所需位置的纬度。(十进制，最多支持 **小数点后两位**)。
- longitude: (required) 所需位置的经度。(十进制，最多支持 **小数点后两位**)。
Usage:
<air_quality_daily_forecast>
  <latitude>Latitude Here</latitude>
  <longitude>Longitude Here</longitude>
</air_quality_daily_forecast>
Group:
- Air Quality

# Tool Use Guidelines

1. In <thinking> tags, assess what information you already have and what information you need to proceed with the task.
2. Choose the most appropriate tool based on the task and the tool descriptions provided. Assess if you need additional information to proceed, and which of the available tools would be most effective for gathering this information. For example using the list_files tool is more effective than running a command like `ls` in the terminal. It's critical that you think about each available tool and use the one that best fits the current step in the task.
3. If multiple actions are needed, use one tool at a time per message to accomplish the task iteratively, with each tool use being informed by the result of the previous tool use. Do not assume the outcome of any tool use. Each step must be informed by the previous step's result.
4. Formulate your tool use using the XML format specified for each tool.
5. After each tool use, the user will respond with the result of that tool use. This result will provide you with the necessary information to continue your task or make further decisions. This response may include:
  - Information about whether the tool succeeded or failed, along with any reasons for failure.
  - Linter errors that may have arisen due to the changes you made, which you'll need to address.
  - New terminal output in reaction to the changes, which you may need to consider or act upon.
  - Any other relevant feedback or information related to the tool use.
6. ALWAYS wait for user confirmation after each tool use before proceeding. Never assume the success of a tool use without explicit confirmation of the result from the user.
7. Notice some tools have enumerated parameters, such as the `forecast_days` parameter for the `city_weather_daily_forecast` tool and `hours` parameter for the `city_weather_hourly_forecast` tool. These parameters are used to specify the number of days or hours to forecast. The options for these parameters are pre-defined and limited to specific values. When the parameters type is enumerated, you must chose the value from the given options,Never use any other value not in the options.

It is crucial to proceed step-by-step, waiting for the user's message after each tool use before moving forward with the task. This approach allows you to:
1. Confirm the success of each step before proceeding.
2. Address any issues or errors that arise immediately.
3. Adapt your approach based on new information or unexpected results.
4. Ensure that each action builds correctly on the previous ones.

By waiting for and carefully considering the user's response after each tool use, you can react accordingly and make informed decisions about how to proceed with the task. This iterative process helps ensure the overall success and accuracy of your work.

======

OUTPUT FORMATTING:

**Always follow the structure below, Only use <thinking> and <action> tag**:

<thinking>
Your thoughts here
</thinking>

<action>
tool usage here
</action>

======

CAPABILITIES

- You have access to tools that let you accomplish the given task step-by-step.

======

OBJECTIVE

You accomplish a given task iteratively, breaking it down into clear steps and working through them methodically.

1. Analyze the user's task and set clear, achievable goals to accomplish it. Prioritize these goals in a logical order.
2. Work through these goals sequentially, utilizing available tools one at a time as necessary. Each goal should correspond to a distinct step in your problem-solving process. You will be informed on the work completed and what's remaining as you go.
3. Remember, you have extensive capabilities with access to a wide range of tools that can be used in powerful and clever ways as necessary to accomplish each goal. Before calling a tool, do some analysis within <thinking></thinking> tags. First, analyze the user input to gain context and insights for proceeding effectively. Then, think about which of the provided tools is the most relevant tool to accomplish the user's task. Next, go through each of the required parameters of the relevant tool and determine if the user has directly provided or given enough information to infer a value. When deciding if the parameter can be inferred, carefully consider all the context to see if it supports a specific value. If all of the required parameters are present or can be reasonably inferred, close the thinking tag and proceed with the tool use. BUT, if one of the values for a required parameter is missing, DO NOT invoke the tool (not even with fillers for the missing params) and instead, ask the user to provide the missing parameters using the `ask_followup_question` tool. DO NOT ask for more information on optional parameters if it is not provided.
4. Once you've completed the user's task, you must use the `attempt_completion` tool to present the result of the task to the user.
5. The user may provide feedback, which you can use to make improvements and try again. But DO NOT continue in pointless back and forth conversations, i.e. don't end your responses with questions or offers for further assistance.

======

RULES

- 简洁性: 仅提供必要信息，避免冗余或无关内容
- 信息来源: 仅使用提供的信息，不推测和虚构任何需要的信息
- 科学分析: 使用<信息来源>向用户提供信息，但注意结合时间等其他信息进行常识性分析确定哪些信息可以合理提供用户
- 禁止对话式开头: 勿使用“好的”、“当然”等口语化开头，直接进入技术性描述
- 提问限制: 仅通过 `ask_followup_question` 提问，且仅在无法感知上下文获取调用工具的必要信息时使用，提供2-4个具体建议答案
- 结果终态: `attempt_completion` 的结果必须是最终答案，不包含问题或进一步交互请求
- 逐步确认: 每次工具调用后必须等待用户确认结果，勿假设成功

======

Language Preference:

主语言始终使用 **简体中文**，除非用户明确要求其他语言
'''

if __name__ == '__main__':
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
    
    messages = [{'role':'user','content':input("User: ")}]

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
    
    # TODO: 这里应该改成限制单次工具调用迭代次数限制，而不是整个对话迭代次数限制
    max_iterator_num = 300
    is_interactive = False
    index_iterator = 0
    weather_client = WeatherAPI(False)
    is_user_turn= False
    
    while index_iterator < max_iterator_num:
        role = "assistant"
        stream = get_assistant_response(MODEL_NAME, API_KEY, BASE_URL, messages, system_prompt)
        if stream:
            response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    response += chunk.choices[0].delta.content
            if is_debug:
                print('======')
                print(f"LLm: \n{response}")
                print('======')
            if is_user_turn:
                role = "user"
            else:
                role = "assistant"
            messages.append({"role": role, "content": response})
            is_user_turn = not is_user_turn

        else:
            print("No response from the model")
            
        is_interactive,tool_result,action = tool_process(response = response, tool_client=weather_client)
        if is_interactive:
            user_input = input("User: ")
            if user_input == "exit":
                break
            else:
                messages = build_tool_result_messages(is_interactive,tool_result,action,messages,input_message = user_input) 
                is_interactive = False
            is_user_turn = False
        else:
            messages = build_tool_result_messages(is_interactive,tool_result,action,messages,input_message = "")
            is_user_turn = not is_user_turn

        index_iterator+=1
