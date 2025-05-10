import streamlit as st
from dotenv import load_dotenv
import time
import jwt
import httpx
import re
from lxml import etree
from typing import Dict, Tuple, List, Any, Generator
import json
from litellm import completion
import os
from strenum import StrEnum
import sys

# Add parent directory to import local packages if your structure is app/main.py and weather_prompt.py is in app/
# If weather_prompt.py is in the same directory as this streamlit script, this sys.path.append might not be needed
# or should be adjusted. For simplicity, I'll assume weather_prompt.py is accessible.
# If you run this script directly from the directory containing weather_prompt.py:
try:
    from weather_prompt import weather_system_prompt_cot
except ImportError:
    # If running from a subdirectory (e.g. pages/) and weather_prompt is in parent
    sys.path.append('../')
    from weather_prompt import weather_system_prompt_cot

load_dotenv()

# --- Agent Code (Slightly Modified for Streamlit) ---

class BaseAgent:
    def __init__(self, messages: List[Dict[str, Any]], system_prompt: str, model_name: str, api_key: str, base_url: str, temperature: float = 0.2, num_retries: int = 3, is_debug: bool = True):
        self.is_debug = is_debug
        self.messages = messages # Initial messages
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.num_retries = num_retries
        self.temperature = temperature
        self.last_thinking_content: str = "" # To store thinking for display

    def get_assistant_response_stream(self) -> Generator[str, None, None]:
        try:
            stream = completion(
                model=self.model_name,
                messages=[{"role": "system", "content": self.system_prompt}] + [{"role": m["role"], "content": m["content"]} for m in self.messages],
                api_key=self.api_key,
                base_url=self.base_url,
                stream=True,
                temperature=self.temperature,
                num_retries=self.num_retries
            )
            full_response = ""
            reasoning_content = ""
            content = ""
            for chunk in stream:
                if hasattr(chunk.choices[0].delta, "reasoning_content"):
                    if chunk.choices[0].delta.reasoning_content is not None:                 
                        content_piece = chunk.choices[0].delta.reasoning_content
                        reasoning_content += content_piece
                        yield content_piece
                else:
                    if chunk.choices[0].delta.content is not None:
                        content_piece = chunk.choices[0].delta.content
                        content += content_piece
                        yield content_piece
            
            full_response = f'''<reasoning>{reasoning_content}</reasoning>\n<response>{content}</response>'''
            # self.messages.append({"role": "assistant", "content": full_response}) # Add response after full stream
            # Streamlit app will handle adding the full response to history for display.
            # Agent's internal messages will be updated more strategically.
            if self.is_debug:
                print('====== LLM Raw Response ======')
                print(full_response)
                print('====== End LLM Raw Response ======')

        except Exception as e:
            error_msg = f"Error getting assistant response: {e}"
            print(error_msg)
            yield f"<thinking>Error: Could not get response from LLM. {e}</thinking><action><attempt_completion><result>I encountered an error trying to process your request. Please try again.</result></attempt_completion></action>"


    def strip_outer_tag(self, xml_str: str) -> str:
        start = xml_str.find('>') + 1
        end = xml_str.rfind('<')
        if start > 0 and end > -1 and end > start:
             return xml_str[start:end].strip()
        return xml_str # Return original if not typical tag structure

    def parse_input_text(self, input_text: str) -> Tuple[str, Dict[str, Any]]:
        # 初始化返回变量
        thinking_content = ""
        tool_info: Dict[str, Any] = {}
        
        # 解析<thinking>标签
        try:
            thinking_match = re.search(r"<thinking>(.*?)</thinking>", input_text, re.DOTALL)
            if thinking_match:
                thinking_content = thinking_match.group(1).strip()
            self.last_thinking_content = thinking_content
        except Exception as e:
            print(f"Error parsing thinking tag: {e}")
            self.last_thinking_content = f"Error parsing thinking: {e}"

        # 解析<action>标签
        action_content_str = ""
        try:
            action_match = re.search(r"<action>(.*?)</action>", input_text, re.DOTALL)
            if action_match:
                action_content_str = action_match.group(1).strip()
        except Exception as e:
            print(f"Error parsing action tag: {e}")
            return self.last_thinking_content, {"error": f"Error parsing action tag: {e}", "details": "Agent will attempt completion."}

        # 如果没有action内容，尝试完成
        if not action_content_str:
            llm_direct_response = re.sub(r"<thinking>.*?</thinking>", "", input_text, flags=re.DOTALL).strip()
            if not llm_direct_response and not self.last_thinking_content:
                llm_direct_response = "I'm not sure how to respond to that. Can you try rephrasing?"
            elif not llm_direct_response and self.last_thinking_content:
                llm_direct_response = self.last_thinking_content
            tool_info["tool_name"] = "attempt_completion"
            tool_info["parameters"] = {"result": llm_direct_response}
            return self.last_thinking_content, tool_info

        try:
            # 使用lxml的容错解析器
            parser = etree.XMLParser(recover=True, remove_blank_text=True)
            
            # 直接解析action内容
            try:
                tool_element = etree.fromstring(action_content_str, parser=parser)
                tool_name = tool_element.tag
                tool_info["tool_name"] = tool_name
                
                params = {}
                for param_element in tool_element.iterchildren():
                    param_tag = param_element.tag
                    
                    # 收集所有内部内容
                    inner_content_parts = []
                    if param_element.text:
                        inner_content_parts.append(param_element.text)
                    
                    for child_node in param_element.iterchildren():
                        inner_content_parts.append(etree.tostring(child_node, encoding='unicode'))
                        if child_node.tail:
                            inner_content_parts.append(child_node.tail)
                    
                    param_value_str = "".join(inner_content_parts).strip()
                    params[param_tag] = param_value_str
                
                tool_info["parameters"] = params

                # 解析建议（如果是ask_followup_question工具）
                if tool_name == "ask_followup_question" and "follow_up" in params:
                    follow_up_content = params["follow_up"]
                    suggestions = []
                    try:
                        # 直接解析suggest标签，假设格式正确
                        suggest_parser = etree.XMLParser(recover=True)
                        suggest_elements = etree.fromstring(follow_up_content, parser=suggest_parser)
                        
                        # 处理单个或多个suggest标签
                        if suggest_elements.tag == "suggest":
                            # 单个suggest标签情况
                            if suggest_elements.text:
                                suggestions.append(suggest_elements.text.strip())
                        else:
                            # 多个suggest标签情况（需要遍历）
                            for suggest_element in suggest_elements.xpath("//suggest"):
                                if suggest_element.text:
                                    suggestions.append(suggest_element.text.strip())
                        
                        if suggestions:
                            tool_info["parameters"]["suggestions"] = suggestions
                    except etree.XMLSyntaxError as pe:
                        print(f"Could not parse <suggest> tags in follow_up: {pe}. Content: {follow_up_content}")
                        
            except etree.XMLSyntaxError as e:
                print(f"XML parsing error: {e}")
                raise ValueError("Invalid XML format in action content")
                
        except Exception as e:
            error_detail = f"Error parsing XML in action tag: {e}. Content: '{action_content_str}'"
            print(error_detail)
            tool_info["tool_name"] = "attempt_completion"
            tool_info["parameters"] = {"result": f"I had trouble processing my internal action steps. The details were: {action_content_str}"}
            tool_info["error"] = error_detail
            return self.last_thinking_content, tool_info
        
        return self.last_thinking_content, tool_info
    
    def hook_interactive(self, tool_name: str) -> bool:
        return tool_name in ['ask_followup_question', 'attempt_completion'] # Make attempt_completion potentially interactive for confirmation

    def execute_action(self, action_data: Dict[str, Any]) -> Tuple[bool, List[Dict[str, str]], Dict[str, Any]]:
        tool_name = action_data.get('tool_name', 'unknown_tool')
        parameters = action_data.get('parameters', {})
        
        tool_result_payload: List[Dict[str, str]] = []
        is_interactive = False

        try:
            if not hasattr(self, tool_name):
                tool_result_text = f"Error: Tool '{tool_name}' not found. Please check available tools."
                tool_result_payload = [{"type": "text", "text": f"[{tool_name}] Result: {tool_result_text}"}]
                return False, tool_result_payload, action_data

            if self.hook_interactive(tool_name=tool_name):
                is_interactive = True
                # For interactive tools, the "result" is a marker; actual interaction happens in UI
                # The payload here is mostly for logging/debugging the fact that an interactive tool was called.
                # The 'question' or 'result' from parameters will be shown to user by Streamlit app.
                display_text = ""
                if tool_name == "ask_followup_question":
                    display_text = parameters.get("question", "I have a follow-up question.")
                elif tool_name == "attempt_completion":
                    display_text = parameters.get("result", "I have an answer for you.")
                
                tool_result_payload = [{"type": "interactive_marker", "tool_name": tool_name, "text": display_text}]
                return True, tool_result_payload, action_data
        
            method = getattr(self, tool_name)
            method_result = method(**parameters)
            
            tool_result_payload = [
                {"type": "text", "text": f"[{tool_name}] Result: {json.dumps(method_result, ensure_ascii=False)}"}
            ]
            return False, tool_result_payload, action_data

        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}' with params {parameters}: {e}"
            print(error_msg)
            tool_result_payload = [{"type": "text", "text": f"[{tool_name}] Result: Error - {error_msg}"}]
            # We return False for is_interactive because the tool execution itself failed, not asking for input.
            return False, tool_result_payload, action_data

    def build_tool_result_message_for_llm(self, tool_result_payload: List[Dict[str, str]], action_details: Dict[str, Any], user_interactive_input: str = ""):
        """
        Prepares the message to be sent back to the LLM after a tool action.
        """
        tool_name_from_details = action_details.get('tool_name', 'unknown_tool_from_details')
        # Get parameters from action_details, as this dict contains the state from when the action was decided.
        action_parameters = action_details.get('parameters', {})
        llm_message_content_str = "" 

        if user_interactive_input:
            # This is when the user responds to an 'ask_followup_question'
            # We use 'action_parameters' which are the parameters of the 'ask_followup_question' tool call.
            question_that_was_asked = action_parameters.get('question', 'my previous question') # From the original <ask_followup_question> parameters
            llm_message_content_str = f"User's response to my question ('{question_that_was_asked}'): {user_interactive_input}"
            self.messages.append({'role': 'user', 'content': llm_message_content_str})

        elif tool_name_from_details == "attempt_completion":
            if self.is_debug: print(f"INFO: 'attempt_completion' executed. No 'tool result' message added to LLM history.")
            return 

        elif tool_name_from_details == "ask_followup_question":
            # This case is for when ask_followup_question is INITIATED by the LLM.
            # user_interactive_input will be empty here.
            if self.is_debug: print(f"INFO: 'ask_followup_question' initiated. No 'tool result' message added to LLM history yet.")
            return 

        else: # For actual, non-interactive tools that return data
            result_texts = []
            for item in tool_result_payload:
                if item.get("type") == "text" and "text" in item:
                    result_texts.append(item["text"])
                elif item.get("type") == "error": 
                    result_texts.append(f"Error from tool '{tool_name_from_details}': {item.get('text')}")

            if not result_texts:
                llm_message_content_str = f"Tool '{tool_name_from_details}' was called but returned no textual result to report."
            else:
                llm_message_content_str = "\n".join(result_texts)
            
            self.messages.append({'role': 'user', 'content': llm_message_content_str})


        if self.is_debug and llm_message_content_str: # Only print if a message was actually constructed
            print(f"===== Message Appended to Agent's Internal History (for LLM) =====")
            # Check the last message added, as self.messages could have been appended to differently if there was no llm_message_content_str
            if self.messages and self.messages[-1]['content'] == llm_message_content_str :
                print(f"Role: {self.messages[-1]['role']}")
                print(f"Content:\n{self.messages[-1]['content']}")
            else:
                print(f"DEBUG: A message might have been intended but llm_message_content_str was '{llm_message_content_str}' and last agent message is different or non-existent.")
            print(f"=================================================================")


    # --- Placeholder methods for BaseAgent ---
    def ask_followup_question(self, question:str, follow_up:str=""):
        '''Retrieval(Query/Results):ask_followup_question'''
        return {'status': 'completed', 'question': question, 'follow_up': follow_up}
    
    def attempt_completion(self, result:str):
        '''Retrieval(Query/Results):attempt_completion. This is the final answer.'''
        return {'status': 'completed', 'result': result}
    
    # TODOs from original code would go here if implemented

class WeatherAgent(BaseAgent):
    def __init__(self, messages: List[Dict[str, Any]], system_prompt: str, model_name: str, api_key: str, base_url: str, temperature: float = 0.2, num_retries: int = 3, is_debug: bool = True):
        super().__init__(messages, system_prompt, model_name, api_key, base_url, temperature, num_retries, is_debug)
        self.private_key = os.getenv("QWEATHER_PRIVATE_KEY", """-----BEGIN PRIVATE KEY-----
MC4CAQAwBQYDK2VwBCIEIJIE87KurF9ZlyQQdyfMeiWbO+rNAoCxvJVTC//JnYMQ
-----END PRIVATE KEY-----""") # Ensure this is set in .env or hardcoded
        self.project_id = os.getenv("QWEATHER_PROJECT_ID", "3AE3REGEEV")
        self.key_id = os.getenv("QWEATHER_KEY_ID", 'CMWDXN77PG')
        self.api_host = os.getenv("QWEATHER_API_HOST", 'https://mr6r6t9rj9.re.qweatherapi.com')
        self.token = None
        self.token_exp_time = 0
        self._ensure_valid_weather_jwt() # Initial token generation

    def _get_weather_jwt(self):
        iat = int(time.time()) - 100 # Issue time slightly in the past
        exp = iat + 86300            # Expiry time (24h - 100s from iat)
        payload = {
            'iat': iat,
            'exp': exp,
            'sub': self.project_id
        }
        headers = {'kid': self.key_id}
        try:
            encoded_jwt = jwt.encode(payload, self.private_key, algorithm='EdDSA', headers=headers)
            if self.is_debug: print(f"Generated QWeather JWT: {encoded_jwt[:20]}...")
            return encoded_jwt, exp # Return token and its expiry time
        except Exception as e:
            print(f"Error generating QWeather JWT: {e}")
            return None, 0
        
    def _ensure_valid_weather_jwt(self):
        # Refresh if no token, or if token expires in less than 5 minutes (300 seconds)
        if self.token is None or self.token_exp_time < (time.time() + 300):
            if self.is_debug and self.token is not None:
                print("QWeather JWT is expiring soon or invalid, refreshing...")
            self.token, self.token_exp_time = self._get_weather_jwt()
            if self.token is None:
                # Handle critical error: Weather tools will fail.
                # You might want to raise an exception or set a flag to disable weather tools.
                print("CRITICAL: Failed to generate/refresh QWeather JWT. Weather tools may not function.")


    def format_location(self, location:str):
        pattern = r'^-?\d+(\.\d+)?,-?\d+(\.\d+)?$'
        if not re.fullmatch(pattern, str(location)): # Ensure location is string
            return str(location)
        lon, lat = str(location).split(',')
        formatted_lon = "{:.2f}".format(float(lon))
        formatted_lat = "{:.2f}".format(float(lat))
        return f"{formatted_lon},{formatted_lat}"

    def _make_qweather_request(self, path:str, params:Dict[str,Any]):
        # Check if token is near expiry and refresh if needed (simplified: refresh every time for this example)
        # A better approach would be to check token's 'exp' claim.
        self._ensure_valid_weather_jwt() # Ensure token is valid before request
        if not self.token: # If token generation failed critically
             return {'status': 'error', 'code': '500', 'message': 'Failed to generate authentication token for QWeather.'}

        url = f'{self.api_host}{path}'
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            if self.is_debug: print(f"Requesting QWeather: {url} with params {params}")
            response = httpx.get(url, headers=headers, params=params)
            response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
            if self.is_debug:
                print(f"QWeather Response Status: {response.status_code}")
                print(f"QWeather Response Text: {response.text[:200]}...") # Log snippet
            return response.json()
        except httpx.HTTPStatusError as e:
            err_msg = f"QWeather API Error for {path}: {e.response.status_code} - {e.response.text}"
            print(err_msg)
            # Try to return a JSON-like error structure if possible
            try:
                return e.response.json()
            except json.JSONDecodeError:
                return {'status': 'error', 'code': str(e.response.status_code), 'message': err_msg}
        except Exception as e:
            err_msg = f"Unexpected error calling QWeather API for {path}: {e}"
            print(err_msg)
            return {'status': 'error', 'code': '500', 'message': err_msg}


    def city_lookup(self, location:str):
        location = self.format_location(location)
        return self._make_qweather_request('/geo/v2/city/lookup', {'location': location})
        
    def top_cities(self, number:str="10"): # LLM might pass number as string
        num_val = 10
        try:
            num_val = int(number)
        except ValueError:
            print(f"Warning: Invalid number '{number}' for top_cities, using default 10.")
        return self._make_qweather_request('/geo/v2/city/top', {'range': 'cn', 'number': num_val})

    def poi_lookup(self, location:str, type:str='scenic', city:str=None): # type is str
        location = self.format_location(location)
        params = {'location': location, 'type': type}
        if city: params['city'] = city
        return self._make_qweather_request('/geo/v2/poi/lookup', params) # Corrected path from original

    def poi_range_search(self, location:str, type:str='scenic', radius:str=None):
        location = self.format_location(location)
        params = {'location': location, 'type': type}
        if radius: params['radius'] = radius
        return self._make_qweather_request('/geo/v2/poi/range', params)

    def city_weather_now(self, locationID_or_latLon:str):
        location = self.format_location(locationID_or_latLon)
        return self._make_qweather_request('/v7/weather/now', {'location': location})

    def city_weather_daily_forecast(self, locationID_or_latLon:str, forecast_days:str="3"):
        location = self.format_location(locationID_or_latLon)
        valid_days_map = {"3": "3d", "7": "7d", "10": "10d", "15": "15d", "30": "30d"}
        if forecast_days not in valid_days_map:
            return {'status': 'error','message': f'Invalid forecast_days: {forecast_days}. Choose from {list(valid_days_map.keys())}'}
        path_segment = valid_days_map[forecast_days]
        return self._make_qweather_request(f'/v7/weather/{path_segment}', {'location': location})

    def city_weather_hourly_forecast(self, locationID_or_latLon:str, hours:str="24"):
        location = self.format_location(locationID_or_latLon)
        valid_hours_map = {"24": "24h", "72": "72h", "168": "168h"} # Common options
        if hours not in valid_hours_map:
             return {'status': 'error','message': f'Invalid hours: {hours}. Choose from {list(valid_hours_map.keys())}'}
        path_segment = valid_hours_map[hours]
        return self._make_qweather_request(f'/v7/weather/{path_segment}', {'location': location})

    def weather_rainy_forecast_minutes(self, latLon:str):
        location = self.format_location(latLon)
        return self._make_qweather_request('/v7/minutely/5m', {'location': location})

    def grid_weather_now(self, latLon:str): 
        location = self.format_location(latLon)
        return self._make_qweather_request('/v7/grid-weather/now', {'location': location})

    def grid_weather_forecast(self, latLon:str, forecast_days:str="3"):
        location = self.format_location(latLon)
        valid_days_map = {"3": "3d", "7": "7d"}
        if forecast_days not in valid_days_map:
            return {'status': 'error','message': f'Invalid forecast_days for grid: {forecast_days}. Choose from {list(valid_days_map.keys())}'}
        path_segment = valid_days_map[forecast_days]
        return self._make_qweather_request(f'/v7/grid-weather/{path_segment}', {'location': location})

    def grid_weather_hourly_forecast(self, latLon:str, hours:str="24"):
        location = self.format_location(latLon)
        valid_hours_map = {"24": "24h", "72": "72h"}
        if hours not in valid_hours_map:
            return {'status': 'error','message': f'Invalid hours for grid: {hours}. Choose from {list(valid_hours_map.keys())}'}
        path_segment = valid_hours_map[hours]
        return self._make_qweather_request(f'/v7/grid-weather/{path_segment}', {'location': location})
        
    def weather_indices(self, locationID_or_latLon:str, forecast_days:str="1"):
        location = self.format_location(locationID_or_latLon)
        valid_days_map = {"1": "1d", "3": "3d"}
        if forecast_days not in valid_days_map:
            return {'status': 'error','message': f'Invalid forecast_days for indices: {forecast_days}. Choose from {list(valid_days_map.keys())}'}
        path_segment = valid_days_map[forecast_days]
        return self._make_qweather_request(f'/v7/indices/{path_segment}', {'location': location, 'type': '0'}) # type=0 for all

    def air_quality(self, latitude:str, longitude:str):
        lat = "{:.2f}".format(float(latitude))
        lon = "{:.2f}".format(float(longitude))
        # Path for air quality is different, usually /air/now or similar, depends on QWeather exact API
        # The original code had path like /airquality/v1/current/latitude/longitude, which is unusual for QWeather
        # Assuming a more standard QWeather pattern: /v7/air/now?location=lon,lat
        location_param = f"{lon},{lat}" # QWeather usually takes lon,lat
        return self._make_qweather_request('/v7/air/now', {'location': location_param})
        # If the old path was correct, it would be:
        # return self._make_qweather_request(f'/airquality/v1/current/{lat}/{lon}', {})


    # ... Implement other weather methods similarly, ensuring paths and params match QWeather docs
    # For brevity, I'll skip repeating all of them but ensure they use _make_qweather_request
    # And handle parameter formatting (like lat,lon) correctly.
    
    def air_quality_hourly_forecast(self,latitude:str,longitude:str):
        lat = "{:.2f}".format(float(latitude))
        lon = "{:.2f}".format(float(longitude))
        location_param = f"{lon},{lat}"
        return self._make_qweather_request(f'/v7/air/5d', {'location': location_param}) # QWeather offers 5d for air, hourly is usually part of that or a premium feature not in free tier path
                                                                                       # Adapt if you have a specific path for hourly non-grid air quality.

    def air_quality_daily_forecast(self,latitude:str,longitude:str):
        # This is often the same as hourly for QWeather's /v7/air/5d
        return self.air_quality_hourly_forecast(latitude, longitude)

    def air_quality_station_data(self,LocationID:str):
        # This API might not exist in the standard QWeather free set or might be under a different path.
        # For now, providing a placeholder. Consult QWeather docs for the correct API.
        # return self._make_qweather_request(f'/airquality/v1/station/{LocationID}', {}) # Original path, likely incorrect for QWeather
        return {'status': 'error', 'message': f'Tool air_quality_station_data with LocationID {LocationID} is not implemented or path is unknown.'}

    def weather_history(self,location:str,date:str): # location is LocationID
        return self._make_qweather_request('/v7/historical/weather', {'location': location, 'date': date})
        
    def air_quality_history(self,location:str,date:str): # location is LocationID
        return self._make_qweather_request('/v7/historical/air', {'location': location, 'date': date})

# --- Streamlit App ---

MAX_MESSAGES_DISPLAY = 50
DEFAULT_SYSTEM_PROMPT = weather_system_prompt_cot if 'weather_system_prompt_cot' in globals() else "You are a helpful weather assistant."

class ModelChoice(StrEnum):
    QWEN_TURBO = 'openai/qwen-turbo-latest'
    DEEPSEEK = "deepseek-chat"
    DEEPSEEK_REASONER = "deepseek-reasoner"
    OPENER_ROUTER_GEMINI_1_5_FLASH = 'open-router-gemini-flash_1_5'
    OPENER_ROUTER_GEMINI_1_5_FLASH_8B = 'open-router-gemini-flash_1_5_8b'
    QWEN3_235B = 'openai/qwen3-235b-a22b'
    QWEN3_14B = 'openai/qwen3-14b'
    QWEN3_8B = 'openai/qwen3-8b'
    QWEN3_4B = 'openai/qwen3-4b'
    QWEN3_1_7B = 'openai/qwen3-1.7b'
    
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE_URL = os.getenv("DEEPSEEK_API_BASE_URL", "YOUR_DEEPSEEK_API_BASE_URL")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "YOUR_OPENROUTER_BASE_URL")

QWEN_API_KEY = os.getenv("QWEN_API_KEY", "YOUR_QWEN_API_KEY")
QWEN_API_BASE_URL = os.getenv("QWEN_API_BASE_URL", "YOUR_QWEN_API_BASE_URL")

MODEL_CONFIGS = {
    ModelChoice.DEEPSEEK: {
        'model_name': 'deepseek/deepseek-chat',
        'api_key': DEEPSEEK_API_KEY, 
        'base_url': DEEPSEEK_API_BASE_URL
    },
    ModelChoice.DEEPSEEK_REASONER: {
        'model_name': 'deepseek/deepseek-reasoner',
        'api_key': DEEPSEEK_API_KEY, 
        'base_url': DEEPSEEK_API_BASE_URL
    },
    ModelChoice.OPENER_ROUTER_GEMINI_1_5_FLASH: {
        'model_name': 'openrouter/google/gemini-flash-1.5',
        'api_key': OPENROUTER_API_KEY,
        'base_url': OPENROUTER_BASE_URL
    },
    ModelChoice.OPENER_ROUTER_GEMINI_1_5_FLASH_8B: {
        'model_name': 'openrouter/google/gemini-flash-1.5-8b',
        'api_key': OPENROUTER_API_KEY,
        'base_url': OPENROUTER_BASE_URL
    },
    ModelChoice.QWEN_TURBO: {
        'model_name': 'openai/qwen-turbo-latest',
        'api_key': QWEN_API_KEY,
        'base_url': QWEN_API_BASE_URL
    },
    ModelChoice.QWEN3_235B: {
        'model_name': 'openai/qwen3-235b-a22b',
        'api_key': QWEN_API_KEY,
        'base_url': QWEN_API_BASE_URL
    },
    ModelChoice.QWEN3_14B: {
        'model_name': 'openai/qwen3-14b',
        'api_key': QWEN_API_KEY,
        'base_url': QWEN_API_BASE_URL
    },
    ModelChoice.QWEN3_8B: {
        'model_name': 'openai/qwen3-8b',
        'api_key': QWEN_API_KEY,
        'base_url': QWEN_API_BASE_URL
    },
    ModelChoice.QWEN3_4B: {
        'model_name': 'openai/qwen3-4b',
        'api_key': QWEN_API_KEY,
        'base_url': QWEN_API_BASE_URL
    },
    ModelChoice.QWEN3_1_7B: {
        'model_name': 'openai/qwen3-1.7b',
        'api_key': QWEN_API_KEY,
        'base_url': QWEN_API_BASE_URL
    }
}

# ============sidebar settings=================

st.set_page_config(layout="wide", page_title="Weather Agent Chatbot")
st.sidebar.title("Agent Settings")
selected_model_key = st.sidebar.selectbox(
    "选择模型:", 
    options=list(ModelChoice), 
    format_func=lambda x: x.value,
    help="选择驱动 Agent 的大语言模型。"
)
MODEL_INFO = MODEL_CONFIGS[selected_model_key]

if 'system_prompt' not in st.session_state: st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
current_system_prompt = st.sidebar.text_area(
    "System Prompt:", 
    value=st.session_state.system_prompt, 
    height=300,
    help="定义 Agent 的核心行为和角色。修改后会开启新的对话。"
) 

if 'model_temperature' not in st.session_state:
    st.session_state.model_temperature = 0.1 # 默认温度值
st.session_state.model_temperature = st.sidebar.slider(
    "模型温度",
    min_value=0.0,
    max_value=1.0,
    value=st.session_state.model_temperature, # 从 session_state 读取当前值
    step=0.05, # 步长可以根据需要调整
    help="控制模型输出的随机性。较低的值使输出更具确定性和一致性，较高的值使其更具创造性和多样性。范围 0.0 - 1.0。"
)

if 'auto_expand_agent_process' not in st.session_state:
    st.session_state.auto_expand_agent_process = False # 默认折叠最新的思考过程 
st.session_state.auto_expand_agent_process = st.sidebar.toggle( # 或者 st.checkbox
    "思考过程",
    value=st.session_state.auto_expand_agent_process,
    help="开启后，最新的 Agent 处理步骤详情将默认展开。关闭则默认折叠。"
)

# 控制是否禁止用户输入
if 'disable_chat_input' not in st.session_state:
    st.session_state.disable_chat_input = False

if 'is_debug_mode' not in st.session_state: st.session_state.is_debug_mode = False

def initialize_agent(force_reinit=False):
    model_name = MODEL_INFO['model_name']
    api_key = MODEL_INFO['api_key']
    base_url = MODEL_INFO['base_url']
    if not api_key or "YOUR_" in api_key or not base_url or "YOUR_" in base_url : # Basic check for placeholder
        st.sidebar.error(f"API Key or Base URL for {selected_model_key.value} is not set correctly!")
        # st.stop() # Commented out for easier testing with mock agent
        print(f"Warning: API Key or Base URL for {selected_model_key.value} might not be set correctly. Using Mock Agent.")
    
    agent_needs_init = force_reinit or \
                       'weather_agent' not in st.session_state or \
                       st.session_state.weather_agent.model_name != model_name or \
                       st.session_state.weather_agent.system_prompt != st.session_state.system_prompt or \
                       st.session_state.weather_agent.temperature != st.session_state.model_temperature or \
                       st.session_state.weather_agent.api_key != api_key or \
                       st.session_state.weather_agent.base_url != base_url or \
                       st.session_state.weather_agent.is_debug != st.session_state.is_debug_mode
    if agent_needs_init:
        if st.session_state.is_debug_mode: print("Re-initializing WeatherAgent.")
        st.session_state.weather_agent = WeatherAgent( 
            messages=[], 
            system_prompt=st.session_state.system_prompt, model_name=model_name,temperature=st.session_state.model_temperature,
            api_key=api_key, base_url=base_url, is_debug=st.session_state.is_debug_mode
        )
        st.session_state.weather_agent.call_count = 0 # Reset for mock
    return st.session_state.weather_agent

if 'messages' not in st.session_state: st.session_state.messages = []
if current_system_prompt != st.session_state.system_prompt:
    st.session_state.system_prompt = current_system_prompt
    initialize_agent(force_reinit=True)
    st.session_state.messages = []
    st.rerun()   

initialize_agent()

if 'agent_is_waiting_for_input' not in st.session_state: st.session_state.agent_is_waiting_for_input = False
if 'interactive_tool_data' not in st.session_state: st.session_state.interactive_tool_data = None
if 'current_turn_intermediate_steps' not in st.session_state: st.session_state.current_turn_intermediate_steps = []
# --- NEW SESSION STATE VARIABLE ---
if 'new_user_message_to_process' not in st.session_state:
    st.session_state.new_user_message_to_process = None


if st.sidebar.button("🧐开始新对话", help="👋🏻清除当前对话历史并重置 Agent。"):
    st.session_state.messages = []
    initialize_agent(force_reinit=True) # Reinitialize agent with potentially new system prompt
    st.session_state.agent_is_waiting_for_input = False
    st.session_state.interactive_tool_data = None
    st.session_state.current_turn_intermediate_steps = []
    st.session_state.new_user_message_to_process = None # Reset this too
    st.rerun()

st.sidebar.markdown("---") # Add a separator

st.sidebar.markdown("**Author:** *Ski Lee*")


st.session_state.is_debug_mode = st.sidebar.checkbox(
    "Enable Agent Debug Mode", 
    value=st.session_state.is_debug_mode,
    help="开启后，控制台会输出详细的 Agent 运行日志，聊天界面会显示 Agent 的思考过程和工具调用详情。"
)

# ============main chat UI===================

st.title("Weather Agent 🤖🌦️")
st.badge(f"*当前模型: `{MODEL_INFO['model_name']}`*")

# --- Initial Conversation Starters ---
INITIAL_PROMPTS = [
    "长沙未来6个小时的天气怎么样？",
    "未来三天上海会下雨吗？",
    "本周末的天气适合在长沙进行哪些户外活动。",
    "查询广州当前空气质量指数。",
    "山东泰山明天和后天的天气预报是？"
]

if not st.session_state.messages or len(st.session_state.messages) == 1: # Only show if chat is empty and use length of messages equals 1 to fix streamlit bug when code not in this block but view still exist.
    st.markdown("❤️ **我们不会记录任何聊天记录。**")
    st.caption("模型速度和精度: QWen Turbo, 14b, 8b, Gemini Flash 1.5 ⚡️ | QWen-235b, DeepSeek 🕵️") 
    st.markdown("你好！我是天气助手智能体，我的运行逻辑完全由AI驱动。自主调用**和风天气Weather Tools**获取真实天气数据，并提供建议。你可以问我关于天气或者任何你感兴趣的问题，或者试试下面的常见问题：")
    
    # Create columns for a better layout, e.g., 2 or 3 buttons per row
    # Adjust the number of columns based on how many prompts you have
    num_cols = 2 
    cols = st.columns(num_cols)
    for i, prompt_text in enumerate(INITIAL_PROMPTS):
        button_key = f"initial_prompt_{i}"
        # Use use_container_width for buttons to fill column width
        if cols[i % num_cols].button(prompt_text, key=button_key, use_container_width=True,disabled=st.session_state.get('disable_chat_input', False)):
            # This will be picked up by the input handling logic below
            # We can reuse the 'clicked_suggestion' logic or a new state var
            # For simplicity, let's assume it sets current_run_user_input directly
            # and then the existing logic handles it.
            # To integrate with existing logic:
            st.session_state.clicked_suggestion = prompt_text 
            if st.session_state.is_debug_mode:
                print(f"Initial prompt button '{prompt_text}' clicked.")
            
            # Disable input until the next turn
            st.session_state.disable_chat_input = True
            # A rerun is needed for the input logic to pick up clicked_suggestion
            st.rerun() 
    st.markdown("---") # Add a separator

# ... (现有聊天记录显示代码) ...
# Display chat history from st.session_state.messages (UI display history)
for i, msg_data in enumerate(st.session_state.messages[-MAX_MESSAGES_DISPLAY:]): 
    avatar = "🧑‍💻" if msg_data["role"] == "user" else "🦖"
    with st.chat_message(msg_data["role"],avatar=avatar):
        if msg_data["role"] == "assistant" and "intermediate_steps" in msg_data and msg_data["intermediate_steps"]:
            is_last_message = (i == len(st.session_state.messages[-MAX_MESSAGES_DISPLAY:]) - 1)
            if st.session_state.auto_expand_agent_process:
                # 如果用户开启了自动展开，则根据是否是最后一条消息且Agent非等待状态来决定
                expanded_default = is_last_message and not st.session_state.agent_is_waiting_for_input
            else:
                # 如果用户关闭了自动展开，则始终默认折叠
                expanded_default = False

            with st.expander("查看智能体的所有执行步骤和工具返回信息 👀", expanded=expanded_default):
                for step in msg_data["intermediate_steps"]:
                    step_type = step.get("type", "unknown")
                    step_title = step.get("title", step_type.replace("_", " ").title())
                    step_content = step.get("content", "")
                    st.markdown(f"**{step_title}**")
                    if step_type == "llm_raw_response": st.code(step_content, language='xml')
                    elif step_type in ["thinking", "info", "error"]: st.markdown(f"```\n{step_content}\n```")
                    elif step_type in ["action_parsed", "action_executed"]: st.json(step_content)
                    elif step_type == "tool_result_payload":
                        if isinstance(step_content, list):
                            for item_idx, item_data in enumerate(step_content):
                                item_text = item_data.get('text', json.dumps(item_data))
                                # st.markdown(f"- Part {item_idx+1} ({item_data.get('type', 'unknown')}):")
                                try:
                                    parsed_json_candidate = item_text
                                    if "Result: " in item_text: 
                                        parsed_json_candidate = item_text.split("Result: ", 1)[-1]
                                    if isinstance(parsed_json_candidate, dict): st.json(parsed_json_candidate)
                                    else: st.json(json.loads(parsed_json_candidate))
                                except (json.JSONDecodeError, TypeError): st.markdown(f"  ```\n  {item_text}\n  ```")
                        else: st.markdown(f"```\n{json.dumps(step_content, indent=2)}\n```")
                    else: st.write(step_content) 
        main_content = msg_data.get("content_display", msg_data["content"])
        if isinstance(main_content, list): 
            for block in main_content:
                if isinstance(block, dict) and block.get("type") == "text": st.markdown(block["text"])
                else: st.markdown(f"```json\n{json.dumps(block, indent=2)}\n```") 
        else:
            st.markdown(str(main_content))


def run_full_agent_turn_and_manage_ui(initial_user_input: str = None):
    # === Disable input at the start of agent processing ===
    st.session_state.disable_chat_input = True
    
    agent: WeatherAgent = st.session_state.weather_agent
    st.session_state.current_turn_intermediate_steps = []

    if initial_user_input:
        agent.messages.append({"role": "user", "content": initial_user_input})
        if agent.is_debug: print(f"Appended user message to agent's internal history: '{initial_user_input}'")
        st.session_state.current_turn_intermediate_steps.append(
            {"type": "info", "title": "ℹ️ User Input Received by Agent", "content": f"Agent processing: {initial_user_input}"}
        )

    MAX_AGENT_STEPS = 50 # Reduced for mock safety
    final_status = "error"
    final_message_for_ui = "Agent processing encountered an issue."

    try: # Wrap the whole loop to ensure disable_chat_input is reset
        for step_count in range(MAX_AGENT_STEPS):
            if agent.is_debug: print(f"Agent processing step {step_count + 1} of this turn.")
            llm_full_response_this_step = ""
            
            ephemeral_message_placeholder = st.empty() 
            with ephemeral_message_placeholder.chat_message("assistant", avatar="⏳"):
                expander_title = f"🧠 LLM Thinking... (Step {step_count + 1} Streaming)"
                if step_count == 0 and initial_user_input: expander_title = "🧠 LLM Initial Response (Streaming)"
                elif step_count > 0: expander_title = f"🧠 LLM Processing Tool Result (Step {step_count + 1} - Streaming)"

                with st.expander(expander_title, expanded=True):
                    stream_display_placeholder = st.empty()
                    stream_display_placeholder.markdown("正在连接语言模型并获取回应... ▍")
                    try:
                        for chunk in agent.get_assistant_response_stream():
                            llm_full_response_this_step += chunk
                            stream_display_placeholder.markdown(llm_full_response_this_step + "▌")
                        stream_display_placeholder.markdown(llm_full_response_this_step)
                    except Exception as e:
                        stream_display_placeholder.error(f"LLM API Error during stream: {e}")
                        llm_full_response_this_step = f"<thinking>LLM API Error: {e}</thinking><action><tool_name>attempt_completion</tool_name><parameters><result>I encountered an issue with the Language Model connection.</result></parameters></action>"
            
            st.session_state.current_turn_intermediate_steps.append(
                {"type": "llm_raw_response", "title": f"🧠 LLM Raw Output (Step {step_count+1})", "content": llm_full_response_this_step}
            )
            agent.messages.append({"role": "assistant", "content": llm_full_response_this_step})
            if agent.is_debug: print(f"Appended assistant (LLM) raw response (Step {step_count+1}) to agent's internal history.")

            thinking_content, action_details_parsed = agent.parse_input_text(llm_full_response_this_step)
            if thinking_content:
                st.session_state.current_turn_intermediate_steps.append(
                    {"type": "thinking", "title": f"🤔 Agent Thinking (Step {step_count+1})", "content": thinking_content}
                )
            # st.session_state.current_turn_intermediate_steps.append(
            #     {"type": "action_parsed", "title": f"🛠️ Action Parsed (Step {step_count+1})", "content": action_details_parsed}
            # )

            tool_name_from_parse = action_details_parsed.get("tool_name")
            tool_params_from_parse = action_details_parsed.get("parameters", {})
            
            tool_calling_ui_placeholder = st.empty()
            if tool_name_from_parse and tool_name_from_parse not in ["attempt_completion", "ask_followup_question"]:
                tool_call_info_md = f"⚙️ Preparing to use tool: **`{tool_name_from_parse}`**"
                if tool_params_from_parse:
                    params_str = json.dumps(tool_params_from_parse)
                    if len(params_str) > 100: params_str = params_str[:100] + "..."
                    tool_call_info_md += f" with parameters: `{params_str}`"
                with tool_calling_ui_placeholder.chat_message("system", avatar="⚙️"): st.markdown(tool_call_info_md)
                st.session_state.current_turn_intermediate_steps.append(
                    {"type": "info", "title": f"🛠️ Tool Call Initiated (Step {step_count+1})",
                    "content": f"Tool: {tool_name_from_parse}, Parameters: {json.dumps(tool_params_from_parse, indent=2)}"}
                )
            elif "error" in action_details_parsed and tool_name_from_parse != "attempt_completion":
                with tool_calling_ui_placeholder.chat_message("system", avatar="⚠️"):
                    st.warning(f"⚠️ Error parsing action from LLM: {action_details_parsed.get('error', 'Unknown parsing error')}. Agent will attempt to recover or complete.")

            is_interactive, tool_result_payload, executed_action_details = agent.execute_action(action_details_parsed)
            
        

            st.session_state.current_turn_intermediate_steps.append(
                {"type": "action_executed", "title": f"⚙️ Action Executed (Step {step_count+1})", "content": executed_action_details}
            )
            if tool_result_payload:
                st.session_state.current_turn_intermediate_steps.append(
                    {"type": "tool_result_payload", "title": f"✨ Tool Result Payload (Step {step_count+1})", "content": tool_result_payload}
                )

            tool_name_executed = executed_action_details.get("tool_name")
            tool_params_executed = executed_action_details.get("parameters", {})

            if tool_name_executed == "attempt_completion":
                final_status = "completion"
                final_message_for_ui = tool_params_executed.get("result", "Completed.")
                break
            elif is_interactive and tool_name_executed == "ask_followup_question":
                final_status = "interactive"
                final_message_for_ui = tool_params_executed.get("question", "Need more info.")
                st.session_state.agent_is_waiting_for_input = True
                st.session_state.interactive_tool_data = {
                    "action_details": executed_action_details, 
                    "prompt_to_user": final_message_for_ui,
                    "suggestions": tool_params_executed.get("suggestions", [])
                }
                break
            elif not is_interactive:
                agent.build_tool_result_message_for_llm(tool_result_payload, executed_action_details)
                if agent.is_debug: print(f"Continuing loop after non-interactive tool '{tool_name_executed}' (Step {step_count+1})")
                st.session_state.current_turn_intermediate_steps.append(
                    {"type": "info", "title": f"ℹ️ Looping (End of Step {step_count+1})", "content": f"Feeding result of '{tool_name_executed}' back to LLM."}
                )
                if step_count == MAX_AGENT_STEPS - 1:
                    final_status = "error"
                    final_message_for_ui = "Max steps reached during tool processing loop. Cannot complete."
                    st.session_state.current_turn_intermediate_steps.append(
                        {"type": "error", "title": "❌ Max Steps Reached", "content": final_message_for_ui}
                    )
            else:
                final_status = "error"
                final_message_for_ui = f"Unhandled agent state: tool='{tool_name_executed}', interactive={is_interactive} (Step {step_count+1})"
                st.session_state.current_turn_intermediate_steps.append(
                    {"type": "error", "title": "❌ Agent Logic Error", "content": final_message_for_ui}
                )
                break       
    finally: # Ensure placeholders are cleared and input is re-enabled
        tool_calling_ui_placeholder.empty() 
        ephemeral_message_placeholder.empty()
        # === Re-enable input when agent turn is fully complete or waiting for specific input ===
        st.session_state.disable_chat_input = False # Default to enable
        
    tool_calling_ui_placeholder.empty() 
    ephemeral_message_placeholder.empty()
    
    assistant_response_for_ui_history = {
        "role": "assistant",
        "content": final_message_for_ui, 
        "content_display": final_message_for_ui, 
        "intermediate_steps": list(st.session_state.current_turn_intermediate_steps) 
    }
    st.session_state.messages.append(assistant_response_for_ui_history)

    if len(st.session_state.messages) > MAX_MESSAGES_DISPLAY:
        st.session_state.messages = st.session_state.messages[-MAX_MESSAGES_DISPLAY:]
    
    st.session_state.current_turn_intermediate_steps = [] 


# --- MODIFIED INPUT HANDLING AND PROCESSING LOGIC ---

# --- Input Acquisition and Staging for Processing ---
# This section captures input, adds it to UI messages, and schedules it for agent processing.
current_run_user_input = None

if 'clicked_suggestion' in st.session_state and st.session_state.clicked_suggestion:
    current_run_user_input = st.session_state.clicked_suggestion
    if st.session_state.is_debug_mode:
        print(f"Input from clicked suggestion: {current_run_user_input}")
    del st.session_state.clicked_suggestion  # Consume it
else:
    # Only display/enable chat_input if not processing a clicked suggestion immediately
    # AND if the agent is not generally busy (controlled by disable_chat_input)
    chat_input_value = st.chat_input(
        "关于任何天气信息",
        key="main_chat_input_widget",
        disabled=st.session_state.get('disable_chat_input', False) # Use the flag
    )
    if chat_input_value:
        current_run_user_input = chat_input_value
        if st.session_state.is_debug_mode:
            print(f"Input from chat_input: {current_run_user_input}")

if current_run_user_input:
    # A new input was submitted by the user in this script run.
    # Add it to the UI display messages immediately.
    st.session_state.messages.append({
        "role": "user",
        "content": current_run_user_input,
        "content_display": current_run_user_input
    })
    # Store it to be processed by the agent in the next script run (after this rerun).
    st.session_state.new_user_message_to_process = current_run_user_input
    
    # === Disable input as we are about to process this new message ===
    st.session_state.disable_chat_input = True 
    
    if st.session_state.is_debug_mode:
        print(f"User input '{current_run_user_input}' added to messages. Rerunning for agent processing.")
    st.rerun() # Rerun to display the user's message and then proceed to agent processing.

# --- Agent Processing and Interaction UI ---
# This section runs after any potential rerun caused by new input submission.

# First, display suggestions if the agent is waiting AND we are not about to process a new message.
if st.session_state.agent_is_waiting_for_input and not st.session_state.new_user_message_to_process:
    interactive_data = st.session_state.interactive_tool_data
    # Ensure interactive_data and action_details exist before trying to access them for key generation
    if interactive_data and 'action_details' in interactive_data:
        suggestions = interactive_data.get("suggestions", [])
        if suggestions:
            st.markdown("💡 **或者，您可以选择以下建议之一：**") # Make suggestions more prominent
            cols = st.columns(min(len(suggestions), 4)) # Max 4 suggestions per row
            action_tool_name = interactive_data['action_details'].get('tool_name', 'followup')
            for i, suggestion_text in enumerate(suggestions):
                button_key = f"suggest_btn_{action_tool_name}_{i}_{suggestion_text[:20]}" # More unique key
                if cols[i % len(cols)].button(suggestion_text, key=button_key, use_container_width=True):
                    st.session_state.clicked_suggestion = suggestion_text
                    if st.session_state.is_debug_mode:
                        print(f"Suggestion button '{suggestion_text}' clicked. Will be processed on next run.")
                    st.rerun()
        # else: # If no suggestions, the chat_input is the primary way to respond
        #     st.markdown("_Agent 正在等待您的回复..._") # Subtle reminder
    elif st.session_state.is_debug_mode:
        print("Warning: agent_is_waiting_for_input is True, but interactive_tool_data is not as expected.")


# Now, check if there's a user message that was staged for processing from the PREVIOUS run.
if st.session_state.new_user_message_to_process:
    user_input_for_agent = st.session_state.new_user_message_to_process
    st.session_state.new_user_message_to_process = None  # Consume the staged message

    agent: WeatherAgent = st.session_state.weather_agent # Ensure agent is available

    if st.session_state.agent_is_waiting_for_input:
        # Agent was waiting for input, and `user_input_for_agent` is the response.
        if st.session_state.is_debug_mode:
            print(f"Processing interactive response for agent: {user_input_for_agent}")
        
        interactive_data = st.session_state.interactive_tool_data # Should still be valid
        if not interactive_data or 'action_details' not in interactive_data :
            if st.session_state.is_debug_mode: print("Error: Inconsistent state for interactive input processing.")
            # Potentially reset or show error
            st.session_state.agent_is_waiting_for_input = False 
            st.error("An issue occurred with interactive input. Please try again.")
            st.rerun()
        else:
            agent.build_tool_result_message_for_llm(
                tool_result_payload=[], 
                action_details=interactive_data['action_details'], 
                user_interactive_input=user_input_for_agent
            )
            if agent.is_debug:
                print("Built tool result from user's interactive input and added to agent's internal history.")
            
            st.session_state.agent_is_waiting_for_input = False 
            st.session_state.interactive_tool_data = None     
            
            run_full_agent_turn_and_manage_ui(initial_user_input=None) 
            st.rerun()

    else:
        # Agent was NOT waiting for input, so `user_input_for_agent` is a new query.
        if st.session_state.is_debug_mode:
            print(f"Processing new query for agent: {user_input_for_agent}")
        
        run_full_agent_turn_and_manage_ui(initial_user_input=user_input_for_agent)
        st.rerun()