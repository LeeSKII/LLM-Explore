import streamlit as st
from dotenv import load_dotenv
import time
import jwt
import httpx
import re
import xml.etree.ElementTree as ET
from typing import Dict, Tuple, List, Any, Generator
import json
from litellm import completion
import os
from enum import StrEnum
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
class Memory:
    def __init__(self):
        self.tag: Any = None
        self.weight: Any = None
        self.description: Any = None

class BaseAgent:
    def __init__(self, messages: List[Dict[str, Any]], system_prompt: str, model_name: str, api_key: str, base_url: str, temperature: float = 0.2, num_retries: int = 3, is_debug: bool = True):
        self.is_debug = is_debug
        self.memory: Dict[str, List[Memory]] = {}
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
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content_piece = chunk.choices[0].delta.content
                    full_response += content_piece
                    yield content_piece
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
        try:
            thinking_match = re.search(r"<thinking>(.*?)</thinking>", input_text, re.DOTALL)
            if thinking_match:
                thinking_content = thinking_match.group(1).strip()
            self.last_thinking_content = thinking_content 
        except Exception as e:
            print(f"Error parsing thinking tag: {e}")
            self.last_thinking_content = f"Error parsing thinking: {e}"

        action_content_str = "" # Renamed to avoid conflict
        try:
            action_match = re.search(r"<action>(.*?)</action>", input_text, re.DOTALL)
            if action_match:
                action_content_str = action_match.group(1).strip()
        except Exception as e:
            print(f"Error parsing action tag: {e}")
            return self.last_thinking_content, {"error": f"Error parsing action tag: {e}", "details": "Agent will attempt completion."}

        tool_info: Dict[str, Any] = {}
        if not action_content_str:
            # ... (fallback to attempt_completion as before) ...
            llm_direct_response = re.sub(r"<thinking>.*?</thinking>", "", input_text, flags=re.DOTALL).strip()
            if not llm_direct_response and not self.last_thinking_content:
                 llm_direct_response = "I'm not sure how to respond to that. Can you try rephrasing?"
            elif not llm_direct_response and self.last_thinking_content:
                 llm_direct_response = self.last_thinking_content 
            tool_info["tool_name"] = "attempt_completion"
            tool_info["parameters"] = {"result": llm_direct_response}
            return self.last_thinking_content, tool_info

        try:
            # ... (existing XML parsing logic up to extracting tool_name and params) ...
            tool_element_root = ET.fromstring(action_content_str) 
            tool_name = tool_element_root.tag
            tool_info["tool_name"] = tool_name
            
            params = {}
            for param_element in tool_element_root:
                param_tag = param_element.tag
                
                # Collect all inner text and XML, preserving structure for tags like <suggest>
                # This involves iterating through child nodes and their text/tail content.
                # A simpler way if params are expected to be mostly text or simple nested tags:
                inner_content_parts = []
                if param_element.text: # Text before the first child
                    inner_content_parts.append(param_element.text) # Keep leading/trailing spaces for now, strip later
                
                for child_node in param_element:
                    # Add the string representation of the child tag itself
                    inner_content_parts.append(ET.tostring(child_node, encoding='unicode'))
                    # Add text that comes after the child tag (tail)
                    # if child_node.tail: # Tail text is often stripped or part of next text node.
                    #    inner_content_parts.append(child_node.tail)

                param_value_str = "".join(inner_content_parts).strip()
                params[param_tag] = param_value_str
            
            tool_info["parameters"] = params

            # --- NEW: Parse suggestions if tool is ask_followup_question ---
            if tool_name == "ask_followup_question" and "follow_up" in params:
                follow_up_content = params["follow_up"] # This is a string like "<suggest>A</suggest><suggest>B</suggest>"
                suggestions = []
                try:
                    # Wrap in a root to parse potentially multiple suggest tags
                    suggest_root = ET.fromstring(f"<root_suggest>{follow_up_content}</root_suggest>")
                    for suggest_element in suggest_root.findall("suggest"):
                        if suggest_element.text:
                            suggestions.append(suggest_element.text.strip())
                    if suggestions:
                        tool_info["parameters"]["suggestions"] = suggestions
                        # Optionally remove raw follow_up if it's now redundant
                        # del tool_info["parameters"]["follow_up"] 
                except ET.ParseError as pe:
                    print(f"Could not parse <suggest> tags in follow_up: {pe}. Content: {follow_up_content}")
            # --- END NEW ---
            
        except ET.ParseError as e:
            # ... (error handling as before) ...
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
        self.token = self._get_weather_jwt()

    def _get_weather_jwt(self): # Renamed to avoid conflict if a tool is named get_weather_jwt
        payload = {
            'iat': int(time.time()) - 100,
            'exp': int(time.time()) + 86300, # 24 hours minus 100 seconds
            'sub': self.project_id
        }
        headers = {'kid': self.key_id}
        try:
            encoded_jwt = jwt.encode(payload, self.private_key, algorithm='EdDSA', headers=headers)
            if self.is_debug: print(f"Generated QWeather JWT: {encoded_jwt[:20]}...")
            return encoded_jwt
        except Exception as e:
            print(f"Error generating QWeather JWT: {e}")
            # Fallback or raise error, for now, returning a dummy to avoid crashing on init
            # In a real app, this should be handled more robustly (e.g., disable weather tools)
            return "dummy_jwt_error"


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
        if True: # Simplified: refresh token before each call or implement proper expiry check
            self.token = self._get_weather_jwt()
            if self.token == "dummy_jwt_error":
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

    def city_weather_now(self, location:str):
        location = self.format_location(location)
        return self._make_qweather_request('/v7/weather/now', {'location': location})

    def city_weather_daily_forecast(self, location:str, forecast_days:str="3"):
        location = self.format_location(location)
        days_val = 3
        valid_days_map = {"3": "3d", "7": "7d", "10": "10d", "15": "15d", "30": "30d"}
        if forecast_days not in valid_days_map:
            return {'status': 'error','message': f'Invalid forecast_days: {forecast_days}. Choose from {list(valid_days_map.keys())}'}
        path_segment = valid_days_map[forecast_days]
        return self._make_qweather_request(f'/v7/weather/{path_segment}', {'location': location})

    def city_weather_hourly_forecast(self, location:str, hours:str="24"):
        location = self.format_location(location)
        valid_hours_map = {"24": "24h", "72": "72h", "168": "168h"} # Common options
        if hours not in valid_hours_map:
             return {'status': 'error','message': f'Invalid hours: {hours}. Choose from {list(valid_hours_map.keys())}'}
        path_segment = valid_hours_map[hours]
        return self._make_qweather_request(f'/v7/weather/{path_segment}', {'location': location})

    def weather_rainy_forecast_minutes(self, location:str):
        location = self.format_location(location)
        return self._make_qweather_request('/v7/minutely/5m', {'location': location})

    def gird_weather_now(self, location:str): # Corrected typo: grid_weather_now
        location = self.format_location(location)
        return self._make_qweather_request('/v7/grid-weather/now', {'location': location})

    def gird_weather_forecast(self, location:str, forecast_days:str="3"): # Corrected typo
        location = self.format_location(location)
        valid_days_map = {"3": "3d", "7": "7d"}
        if forecast_days not in valid_days_map:
            return {'status': 'error','message': f'Invalid forecast_days for grid: {forecast_days}. Choose from {list(valid_days_map.keys())}'}
        path_segment = valid_days_map[forecast_days]
        return self._make_qweather_request(f'/v7/grid-weather/{path_segment}', {'location': location})

    def gird_weather_hourly_forecast(self, location:str, hours:str="24"): # Corrected typo
        location = self.format_location(location)
        valid_hours_map = {"24": "24h", "72": "72h"}
        if hours not in valid_hours_map:
            return {'status': 'error','message': f'Invalid hours for grid: {hours}. Choose from {list(valid_hours_map.keys())}'}
        path_segment = valid_hours_map[hours]
        return self._make_qweather_request(f'/v7/grid-weather/{path_segment}', {'location': location})
        
    def weather_indices(self, location:str, forecast_days:str="1"):
        location = self.format_location(location)
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
DEFAULT_SYSTEM_PROMPT = weather_system_prompt_cot

class ModelChoice(StrEnum):
    DEEPSEEK = "deepseek-chat"
    OPENER_ROUTER_GEMINI = 'open-router-gemini-flash'
MODEL_CONFIGS = {
    ModelChoice.DEEPSEEK: {
        'model_name': 'deepseek/deepseek-chat',
        'api_key': os.getenv("DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_API_KEY"), # Provide default for mock
        'base_url': os.getenv("DEEPSEEK_API_BASE_URL", "YOUR_DEEPSEEK_BASE_URL")
    },
    ModelChoice.OPENER_ROUTER_GEMINI: {
        'model_name': 'openrouter/google/gemini-flash-1.5',
        'api_key': os.getenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY"),
        'base_url': os.getenv("OPENROUTER_BASE_URL", "YOUR_OPENROUTER_BASE_URL")
    }
}

st.set_page_config(layout="wide", page_title="Weather Agent Chatbot")
st.sidebar.title("Weather Agent Configuration")
selected_model_key = st.sidebar.selectbox("Choose a Model:", options=list(ModelChoice), format_func=lambda x: x.value)
MODEL_INFO = MODEL_CONFIGS[selected_model_key]

if 'system_prompt' not in st.session_state: st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
current_system_prompt = st.sidebar.text_area("System Prompt:", value=st.session_state.system_prompt, height=300) # Reduced height for mock

if 'is_debug_mode' not in st.session_state: st.session_state.is_debug_mode = True
st.session_state.is_debug_mode = st.sidebar.checkbox("Enable Agent Debug Mode", value=st.session_state.is_debug_mode)

def initialize_agent(force_reinit=False):
    model_name = MODEL_INFO['model_name']
    api_key = MODEL_INFO['api_key']
    base_url = MODEL_INFO['base_url']
    if not api_key or not base_url:
        st.sidebar.error(f"API Key or Base URL for {selected_model_key.value} is not set!")
        st.stop()
    agent_needs_init = force_reinit or \
                       'weather_agent' not in st.session_state or \
                       st.session_state.weather_agent.model_name != model_name or \
                       st.session_state.weather_agent.system_prompt != st.session_state.system_prompt or \
                       st.session_state.weather_agent.api_key != api_key or \
                       st.session_state.weather_agent.base_url != base_url or \
                       st.session_state.weather_agent.is_debug != st.session_state.is_debug_mode
    if agent_needs_init:
        if st.session_state.is_debug_mode: print("Re-initializing WeatherAgent.")
        st.session_state.weather_agent = WeatherAgent( # Using Mock Agent
            messages=[{"role": "system", "content": st.session_state.system_prompt}], # Agent needs system prompt in its history
            system_prompt=st.session_state.system_prompt, model_name=model_name,
            api_key=api_key, base_url=base_url, is_debug=st.session_state.is_debug_mode
        )
        # Reset call count for mock agent on re-init
        if hasattr(st.session_state.weather_agent, 'call_count'):
            st.session_state.weather_agent.call_count = 0
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


if st.sidebar.button("New Conversation"):
    st.session_state.messages = []
    initialize_agent(force_reinit=True)
    st.session_state.agent_is_waiting_for_input = False
    st.session_state.interactive_tool_data = None
    st.session_state.current_turn_intermediate_steps = []
    st.rerun()

st.title("Weather Agent Chatbot 🤖🌦️")
st.markdown(f"Using Model: `{MODEL_INFO['model_name']}` (Mocked Behavior)")

# Display chat history from st.session_state.messages (UI display history)
for i, msg_data in enumerate(st.session_state.messages[-MAX_MESSAGES_DISPLAY:]): # Display only last N messages
    with st.chat_message(msg_data["role"]):
        if msg_data["role"] == "assistant" and "intermediate_steps" in msg_data and msg_data["intermediate_steps"]:
            # Expand the last assistant message's process if it's not waiting for input OR if it just completed.
            # If it's waiting for input, we usually want to see its question, not necessarily its full thought process expanded by default.
            # Let's refine this: expand if it's the very last message AND agent is NOT waiting.
            # Or if it IS waiting, but it's the one that asked the question.
            is_last_message = (i == len(st.session_state.messages[-MAX_MESSAGES_DISPLAY:]) - 1)
            
            expanded_default = is_last_message and not st.session_state.agent_is_waiting_for_input

            with st.expander("View Agent's Process", expanded=expanded_default):
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
                                st.markdown(f"- Part {item_idx+1} ({item_data.get('type', 'unknown')}):")
                                try:
                                    # Attempt to parse if it looks like JSON within a string
                                    parsed_json_candidate = item_text
                                    if "Result: " in item_text: # Specific for some mock payloads
                                        parsed_json_candidate = item_text.split("Result: ", 1)[-1]
                                    
                                    # Check if it's already a dict (from direct JSON step_content)
                                    if isinstance(parsed_json_candidate, dict):
                                        st.json(parsed_json_candidate)
                                    else: # Try to load string as JSON
                                        parsed_json = json.loads(parsed_json_candidate)
                                        st.json(parsed_json)
                                except (json.JSONDecodeError, TypeError):
                                    st.markdown(f"  ```\n  {item_text}\n  ```")
                        else: # If step_content is not a list but some other structure
                            st.markdown(f"```\n{json.dumps(step_content, indent=2)}\n```")
                    else: st.write(step_content) # Fallback for unknown step types
        main_content = msg_data.get("content_display", msg_data["content"])
        if isinstance(main_content, list): # Handle Claude-style list of content blocks
            for block in main_content:
                if isinstance(block, dict) and block.get("type") == "text": st.markdown(block["text"])
                else: st.markdown(f"```json\n{json.dumps(block, indent=2)}\n```") # Display other blocks as JSON
        else:
            st.markdown(str(main_content))


def run_full_agent_turn_and_manage_ui(initial_user_input: str = None):
    agent: WeatherAgent = st.session_state.weather_agent
    st.session_state.current_turn_intermediate_steps = []

    if initial_user_input:
        # Add to agent's internal history (system prompt is already there)
        agent.messages.append({"role": "user", "content": initial_user_input})
        if agent.is_debug: print(f"Appended user message to agent's internal history: '{initial_user_input}'")
        st.session_state.current_turn_intermediate_steps.append(
            {"type": "info", "title": "ℹ️ User Input Received", "content": f"User asked: {initial_user_input}"}
        )

    MAX_AGENT_STEPS = 7
    final_status = "error"
    final_message_for_ui = "Agent processing encountered an issue."

    for step_count in range(MAX_AGENT_STEPS):
        if agent.is_debug: print(f"Agent processing step {step_count + 1} of this turn.")
        llm_full_response_this_step = ""
        
        ephemeral_message_placeholder = st.empty() # Create a placeholder for the entire ephemeral message
        with ephemeral_message_placeholder.chat_message("assistant", avatar="⏳"):
            expander_title = f"🧠 LLM Thinking (Step {step_count + 1} - Streaming)"
            if step_count == 0 and initial_user_input: expander_title = "🧠 LLM Initial Response (Streaming)"
            elif step_count > 0: expander_title = f"🧠 LLM Processing Tool Result (Step {step_count + 1} - Streaming)"

            with st.expander(expander_title, expanded=True):
                stream_display_placeholder = st.empty()
                stream_display_placeholder.markdown("▍")
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
        # Add LLM's raw response to agent's *internal* message history
        agent.messages.append({"role": "assistant", "content": llm_full_response_this_step})
        if agent.is_debug: print(f"Appended assistant (LLM) raw response (Step {step_count+1}) to agent's internal history.")

        thinking_content, action_details_parsed = agent.parse_input_text(llm_full_response_this_step)
        if thinking_content:
            st.session_state.current_turn_intermediate_steps.append(
                {"type": "thinking", "title": f"🤔 Agent Thinking (Step {step_count+1})", "content": thinking_content}
            )
        st.session_state.current_turn_intermediate_steps.append(
            {"type": "action_parsed", "title": f"🛠️ Action Parsed (Step {step_count+1})", "content": action_details_parsed}
        )

        tool_name_from_parse = action_details_parsed.get("tool_name")
        tool_params_from_parse = action_details_parsed.get("parameters", {})
        
        tool_calling_ui_placeholder = st.empty()
        if tool_name_from_parse and tool_name_from_parse not in ["attempt_completion", "ask_followup_question"]:
            tool_call_info_md = f"⚙️ Preparing to use tool: **`{tool_name_from_parse}`**"
            if tool_params_from_parse:
                params_str = json.dumps(tool_params_from_parse)
                if len(params_str) > 100: params_str = params_str[:100] + "..."
                tool_call_info_md += f" with parameters: `{params_str}`"
            with tool_calling_ui_placeholder.chat_message("system", avatar="⚙️"):
                st.markdown(tool_call_info_md)
            st.session_state.current_turn_intermediate_steps.append(
                {"type": "info", "title": f"🛠️ Tool Call Initiated (Step {step_count+1})",
                 "content": f"Tool: {tool_name_from_parse}, Parameters: {json.dumps(tool_params_from_parse, indent=2)}"}
            )
        elif "error" in action_details_parsed and tool_name_from_parse != "attempt_completion":
             with tool_calling_ui_placeholder.chat_message("system", avatar="⚠️"):
                 st.warning(f"⚠️ Error parsing action from LLM: {action_details_parsed.get('error', 'Unknown parsing error')}. Agent will attempt to recover or complete.")

        is_interactive, tool_result_payload, executed_action_details = agent.execute_action(action_details_parsed)
        
        tool_calling_ui_placeholder.empty() # Clear "Preparing to use tool"
        # Clear the ephemeral LLM streaming message *after* action is executed and parsed.
        # The final assistant message will consolidate everything.
        ephemeral_message_placeholder.empty()


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
            # We don't add tool_result to LLM for attempt_completion as it's the final step for LLM.
            # The agent.messages already contains the LLM's action with attempt_completion.
            break

        elif is_interactive and tool_name_executed == "ask_followup_question":
            final_status = "interactive"
            final_message_for_ui = tool_params_executed.get("question", "Need more info.")
            st.session_state.agent_is_waiting_for_input = True
            st.session_state.interactive_tool_data = {
                "action_details": executed_action_details, # Contains tool_name, original params (question, suggestions)
                "prompt_to_user": final_message_for_ui,
                "suggestions": tool_params_executed.get("suggestions", [])
            }
            # For ask_followup_question, the LLM has *asked* the question.
            # We don't immediately send a tool_result back to LLM.
            # We wait for user input. That input will then be formatted as a tool_result.
            # The agent.messages already has the LLM's <action>ask_followup_question</action>.
            break
        
        elif not is_interactive:
            # Non-interactive tool executed, feed its result back to agent's internal history for next LLM call
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
                # No explicit message to agent here, as the loop ends. The final_message_for_ui will be displayed.
        else:
            final_status = "error"
            final_message_for_ui = f"Unhandled agent state: tool='{tool_name_executed}', interactive={is_interactive} (Step {step_count+1})"
            st.session_state.current_turn_intermediate_steps.append(
                {"type": "error", "title": "❌ Agent Logic Error", "content": final_message_for_ui}
            )
            break
    # --- End of Agent Processing Loop ---

    # The final assistant message for the UI chat history
    assistant_response_for_ui_history = {
        "role": "assistant",
        "content": final_message_for_ui, # This is the actual content to be stored for the LLM if it were a real turn
        "content_display": final_message_for_ui, # This is what user sees as the main response
        "intermediate_steps": list(st.session_state.current_turn_intermediate_steps) # Copy
    }
    st.session_state.messages.append(assistant_response_for_ui_history)

    if len(st.session_state.messages) > MAX_MESSAGES_DISPLAY:
        st.session_state.messages = st.session_state.messages[-MAX_MESSAGES_DISPLAY:]
    
    st.session_state.current_turn_intermediate_steps = [] # Clear for next turn


# --- Input Handling ---
user_prompt_input = None

# 1. Check for clicked suggestion first
if 'clicked_suggestion' in st.session_state and st.session_state.clicked_suggestion:
    user_prompt_input = st.session_state.clicked_suggestion
    if st.session_state.is_debug_mode: print(f"Input from clicked suggestion: {user_prompt_input}")
    del st.session_state.clicked_suggestion # Consume it
    # A rerun will happen from the button click itself if this path is taken.
    # The subsequent `if user_prompt_input:` block will handle it.

# 2. If no suggestion was clicked, get input from the main chat_input
# This will be None if nothing was submitted in the current script run.
# It will have a value if the user typed and pressed Enter OR if a suggestion button caused a rerun.
if user_prompt_input is None: # Only check chat_input if a suggestion wasn't already processed
    # The placeholder text for chat_input will be generic.
    # The agent's question (if any) is already visible as the last message in the chat history.
    user_prompt_input = st.chat_input("What would you like to know about the weather?")
    if user_prompt_input and st.session_state.is_debug_mode:
        print(f"Input from chat_input: {user_prompt_input}")


# --- Process Input ---
if st.session_state.agent_is_waiting_for_input:
    interactive_data = st.session_state.interactive_tool_data
    
    # Display suggestion buttons if available for the interactive follow-up
    suggestions = interactive_data.get("suggestions", [])
    if suggestions:
        # Create columns for buttons. Use a simple layout for now.
        # Text before buttons can be useful.
        st.markdown("Or choose a suggestion:")
        cols = st.columns(min(len(suggestions), 5)) # Max 5 suggestions per row
        for i, suggestion_text in enumerate(suggestions):
            # Use a unique key for each button to avoid conflicts
            if cols[i % len(cols)].button(suggestion_text, key=f"suggest_btn_{interactive_data['action_details'].get('tool_name', 'followup')}_{i}"):
                st.session_state.clicked_suggestion = suggestion_text
                if st.session_state.is_debug_mode: print(f"Suggestion button '{suggestion_text}' clicked.")
                # When a button is clicked, Streamlit reruns the script.
                # The `user_prompt_input` logic at the top will pick up `st.session_state.clicked_suggestion`.
                st.rerun()
    
    # If user_prompt_input has a value (either from a button click that caused a rerun, or from typing into the main chat_input)
    if user_prompt_input:
        if st.session_state.is_debug_mode: print(f"Processing interactive response: {user_prompt_input}")
        user_interactive_response = user_prompt_input

        # Add user's interactive response to UI chat history
        st.session_state.messages.append({"role": "user", "content": user_interactive_response, "content_display": user_interactive_response})
        
        agent: WeatherAgent = st.session_state.weather_agent
        # This response is the "result" of the "ask_followup_question" tool
        agent.build_tool_result_message_for_llm(
            tool_result_payload=[], # No payload from user text, it IS the payload
            action_details=interactive_data['action_details'], # The original action that asked the question
            user_interactive_input=user_interactive_response
        )
        if agent.is_debug: print("Built tool result from user's interactive input and added to agent's internal history.")
        
        st.session_state.agent_is_waiting_for_input = False # Reset waiting state
        st.session_state.interactive_tool_data = None     # Clear interactive data
        
        # Agent continues its turn with the user's interactive input
        run_full_agent_turn_and_manage_ui(initial_user_input=None) # `None` because user input already added to agent.messages
        st.rerun()

elif user_prompt_input: # This means it's a new query (agent was not waiting for input)
    if st.session_state.is_debug_mode: print(f"Processing new query: {user_prompt_input}")
    # Add user's new query to UI chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt_input, "content_display": user_prompt_input})
    
    # Agent starts a new turn with this initial user input
    run_full_agent_turn_and_manage_ui(initial_user_input=user_prompt_input)
    st.rerun()

# Final check: If still waiting for input (e.g., user hasn't responded yet),
# ensure the suggestions are visible if the script reruns for other reasons.
# This block is mostly for ensuring suggestions stay on screen if a rerun happens
# *before* the user interacts with a suggestion button or the chat input.
# The actual processing of the input is handled above.
if st.session_state.agent_is_waiting_for_input and not user_prompt_input:
    # This case can happen if the script reruns for some reason while waiting,
    # and no new input was provided in *that specific rerun*.
    # The suggestions would have been drawn by the block above this `elif`.
    # So, technically, this block might not be strictly necessary if suggestion drawing
    # is always inside the `if st.session_state.agent_is_waiting_for_input:` block.
    # Let's ensure the suggestion display logic is robust within that main `if` block.
    # The current structure seems to handle it: suggestions are displayed if agent_is_waiting_for_input is true.
    pass