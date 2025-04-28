import time
import jwt
import httpx
import re

class BaseAgent:
    def __init__(self,is_debug=True):
        self.is_debug = is_debug
    
    def ask_followup_question(self,question,follow_up):
        return {'status': 'completed','question': question, 'follow_up': follow_up}
    
    def attempt_completion(self,result):
        return {'status': 'completed','result': result}

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

if __name__ == '__main__':
    client = WeatherAPI()
    # client.city_lookup('北京')
    # client.top_cities(10)
    # client.poi_range_search('116.40528,39.90498')
    # client.city_weather_now('101010100')
    # client.city_weather_daily_forecast('101010100')
    # client.city_weather_hourly_forecast('101010100')
    # client.weather_rainy_forecast_minutes('116.41,39.92')
    # client.gird_weather_now('116.41,39.92')
    client.gird_weather_forecast('112.91,28.21','30')
    # client.gird_weather_hourly_forecast('116.41,39.92')
    # client.weather_indices('28.21304,112.91159','1')
    # client.air_quality('39.92','116.41')
    # client.air_quality_hourly_forecast('39.92','116.41')
    # client.air_quality_daily_forecast('39.92','116.41')
    # client.air_quality_station_data('P58911')
    # client.weather_history('101010100','20250331')
    # client.airquality_history('101010100','20250331')


