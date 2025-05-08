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
        thinking_content = "Could not parse thinking."
        try:
            thinking_match = re.search(r"<thinking>(.*?)</thinking>", input_text, re.DOTALL)
            if thinking_match:
                thinking_content = thinking_match.group(1).strip()
            self.last_thinking_content = thinking_content # Store for display
        except Exception as e:
            print(f"Error parsing thinking tag: {e}")
            self.last_thinking_content = f"Error parsing thinking: {e}"


        action_content = ""
        try:
            action_match = re.search(r"<action>(.*?)</action>", input_text, re.DOTALL)
            if action_match:
                action_content = action_match.group(1).strip()
        except Exception as e:
            print(f"Error parsing action tag: {e}")
            return self.last_thinking_content, {"error": f"Error parsing action tag: {e}"}


        tool_info: Dict[str, Any] = {}
        if not action_content:
            # If no action tag, LLM might be trying to converse or failed format
            # Treat as an attempt_completion with the raw input_text (minus thinking, if any)
            # This is a fallback. Ideally, LLM always uses <action>.
            llm_direct_response = re.sub(r"<thinking>.*?</thinking>", "", input_text, flags=re.DOTALL).strip()
            if not llm_direct_response and not self.last_thinking_content: # Empty response
                 llm_direct_response = "I'm not sure how to respond to that. Can you try rephrasing?"
            elif not llm_direct_response and self.last_thinking_content: # Only thinking
                 llm_direct_response = self.last_thinking_content # Or a canned response

            tool_info["tool_name"] = "attempt_completion"
            tool_info["parameters"] = {"result": llm_direct_response}
            print(f"Warning: No <action> tag found. Fallback to attempt_completion with content: {llm_direct_response}")
            return self.last_thinking_content, tool_info

        try:
            # Wrap in a root tag only if action_content itself isn't a single well-formed XML tool call
            # A bit heuristic: if it starts with < and ends with > and contains another <, it might be okay
            if not (action_content.startswith("<") and action_content.endswith(">") and action_content.count("<") > 1) :
                # This case is unlikely if LLM follows prompt for single tool in action
                # However, if it puts text directly in <action>text</action>
                root = ET.fromstring(f"<root><action_wrapper>{action_content}</action_wrapper></root>")
                tool_element = root.find('action_wrapper')
                if tool_element is not None and tool_element.text and not list(tool_element): # if it's just text
                    tool_info["tool_name"] = "attempt_completion" # Treat as direct response
                    tool_info["parameters"] = {"result": tool_element.text.strip()}
                    return self.last_thinking_content, tool_info

            # Try parsing action_content directly
            # It should be <tool><param>...</param></tool>
            tool_element_root = ET.fromstring(action_content) # This assumes action_content is like <tool_name>...</tool_name>
            tool_name = tool_element_root.tag
            tool_info["tool_name"] = tool_name
            
            params = {}
            for param_element in tool_element_root:
                # For parameters that might contain complex XML/HTML, get the inner content
                # ET.tostring includes the tag itself. We need to strip it.
                inner_xml = "".join(ET.tostring(e, encoding='unicode') for e in param_element)
                if param_element.text and not inner_xml: # Simple text node
                    params[param_element.tag] = param_element.text.strip()
                elif inner_xml: # Has child elements or mixed content
                    # Strip the outer tag of the param_element itself
                    # e.g. if param_xml is <city>New <b>York</b></city>, we want "New <b>York</b>"
                    # This logic might need refinement based on actual LLM output for complex params
                    param_content_str = ET.tostring(param_element, encoding='unicode')
                    params[param_element.tag] = self.strip_outer_tag(param_content_str)

                else: # Empty tag
                    params[param_element.tag] = ""
            
            tool_info["parameters"] = params
            
        except ET.ParseError as e:
            print(f"Error parsing XML in action tag: {e}. Content: '{action_content}'")
            # Fallback: treat the action_content as part of a direct response
            tool_info["tool_name"] = "attempt_completion"
            tool_info["parameters"] = {"result": f"Error in processing my action: {action_content}"}
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
        This message should be in the 'user' role to simulate the tool's output.
        """
        content_blocks = []
        if user_interactive_input: # This is for when user responds to an interactive tool
            # For ask_followup_question, user_interactive_input is the answer.
            # For attempt_completion (if made interactive for confirmation), user_interactive_input could be "yes", "no", or a correction.
            content_blocks.append({
                "type": "text",
                "text": f"[{action_details.get('tool_name')}] User's response: {user_interactive_input}"
            })
        else: # This is for non-interactive tools or the initial marker of an interactive tool
            content_blocks.extend(tool_result_payload)
        
        self.messages.append({'role': 'user', 'content': content_blocks })
        if self.is_debug:
            print("===== Tool Result Message for LLM =====")
            print(json.dumps(self.messages[-1], indent=2, ensure_ascii=False))
            print("======================================")


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

# Constants for Streamlit
MAX_MESSAGES_DISPLAY = 50 # Max messages to keep in st.session_state.messages for display
DEFAULT_SYSTEM_PROMPT = weather_system_prompt_cot # Use the imported one

# Model Choice (Simplified for Streamlit)
class ModelChoice(StrEnum):
    DEEPSEEK = "deepseek-chat"
    OPENER_ROUTER_GEMINI = 'open-router-gemini-flash'
    # Add other models from your original list if needed

MODEL_CONFIGS = {
    ModelChoice.DEEPSEEK: {
        'model_name': 'deepseek/deepseek-chat',
        'api_key': os.getenv("DEEPSEEK_API_KEY"),
        'base_url': os.getenv("DEEPSEEK_API_BASE_URL")
    },
    ModelChoice.OPENER_ROUTER_GEMINI: {
        'model_name': 'openrouter/google/gemini-flash-1.5', # Updated name if applicable
        'api_key': os.getenv("OPENROUTER_API_KEY"),
        'base_url': os.getenv("OPENROUTER_BASE_URL")
    }
}

st.set_page_config(layout="wide", page_title="Weather Agent Chatbot")

# --- Initialization & Sidebar ---
st.sidebar.title("Weather Agent Configuration")

# Model Selection
selected_model_key = st.sidebar.selectbox(
    "Choose a Model:",
    options=list(ModelChoice),
    format_func=lambda x: x.value
)
MODEL_INFO = MODEL_CONFIGS[selected_model_key]
MODEL_NAME = MODEL_INFO['model_name']
API_KEY = MODEL_INFO['api_key']
BASE_URL = MODEL_INFO['base_url']

if not API_KEY or not BASE_URL:
    st.sidebar.error(f"API Key or Base URL for {selected_model_key.value} is not set in .env file!")
    st.stop()

# System Prompt
if 'system_prompt' not in st.session_state:
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
current_system_prompt = st.sidebar.text_area(
    "System Prompt (Edit carefully):",
    value=st.session_state.system_prompt,
    height=300
)
if current_system_prompt != st.session_state.system_prompt:
    st.session_state.system_prompt = current_system_prompt
    # Force re-initialization of agent if prompt changes
    if 'weather_agent' in st.session_state:
        del st.session_state['weather_agent']
    st.rerun()

# Debug Mode Toggle
if 'is_debug_mode' not in st.session_state:
    st.session_state.is_debug_mode = True # Default to True for more verbosity
st.session_state.is_debug_mode = st.sidebar.checkbox("Enable Agent Debug Mode (Console Logs)", value=st.session_state.is_debug_mode)


# Initialize or retrieve session state variables
if 'messages' not in st.session_state: # For chat display
    st.session_state.messages = []
if 'weather_agent' not in st.session_state: # Agent instance
    # Agent's internal messages start empty for a new session/agent
    st.session_state.weather_agent = WeatherAgent(
        messages=[], # Agent keeps its own message history internally
        system_prompt=st.session_state.system_prompt,
        model_name=MODEL_NAME,
        api_key=API_KEY,
        base_url=BASE_URL,
        is_debug=st.session_state.is_debug_mode
    )
if 'agent_is_waiting_for_input' not in st.session_state:
    st.session_state.agent_is_waiting_for_input = False
if 'interactive_tool_data' not in st.session_state: # Holds {'action_details': ..., 'prompt_to_user': ...}
    st.session_state.interactive_tool_data = None

# Button to start a new conversation
if st.sidebar.button("New Conversation"):
    st.session_state.messages = []
    # Re-initialize agent with empty internal messages
    st.session_state.weather_agent = WeatherAgent(
        messages=[],
        system_prompt=st.session_state.system_prompt,
        model_name=MODEL_NAME,
        api_key=API_KEY,
        base_url=BASE_URL,
        is_debug=st.session_state.is_debug_mode
    )
    st.session_state.agent_is_waiting_for_input = False
    st.session_state.interactive_tool_data = None
    st.rerun()

# --- Main Chat UI ---
st.title("Weather Agent Chatbot 🤖🌦️")
st.markdown(f"Using Model: `{MODEL_NAME}`")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], list): # Handle tool result blocks
            for item in msg["content"]:
                if item["type"] == "text":
                    st.markdown(item["text"])
                # Could add more complex rendering for other types if needed
        else:
            st.markdown(msg["content"]) # Standard text messages

# --- Agent Interaction Loop ---
def run_agent_turn(user_input_for_llm: str = None):
    """
    Handles a full turn of the agent:
    1. (Optional) Adds user input to agent's internal messages.
    2. Gets LLM response stream.
    3. Parses LLM response for thinking/action.
    4. Executes action.
    5. Builds tool result message for LLM (for its next turn).
    Returns the final assistant message to display, or sets up for interactive input.
    """
    agent: WeatherAgent = st.session_state.weather_agent

    if user_input_for_llm:
        # Add user's direct message to agent's internal history
        agent.messages.append({"role": "user", "content": user_input_for_llm})

    # 1. Get LLM response
    llm_full_response = ""
    with st.chat_message("assistant"): # Placeholder for streaming
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...") # "Thinking..." in Thai
        for chunk in agent.get_assistant_response_stream():
            llm_full_response += chunk
            message_placeholder.markdown(llm_full_response + "▌")
        message_placeholder.markdown(llm_full_response)

    # Add raw LLM response (thinking+action) to agent's internal messages
    # This is what the agent "said" before tool processing
    agent.messages.append({"role": "assistant", "content": llm_full_response})
    
    # Display LLM's raw output (Thinking & Action)
    # This is good for debugging what the LLM actually produced
    with st.expander("Agent's Raw Output (LLM Thought & Action Plan)", expanded=False):
        st.markdown(f"```xml\n{llm_full_response}\n```")

    # 2. Parse LLM response
    thinking_content, action_details = agent.parse_input_text(llm_full_response)

    # Display Thinking
    if thinking_content:
        with st.chat_message("assistant"): # Could use a different icon/role for "system"
            with st.expander("Agent Thinking 🤔", expanded=True):
                st.markdown(thinking_content)
    
    # Display Action
    if action_details and "error" not in action_details:
        with st.chat_message("assistant"):
            with st.expander("Agent Action 🛠️", expanded=True):
                st.json(action_details)
    elif "error" in action_details:
         with st.chat_message("assistant"):
            st.error(f"Error in parsing action: {action_details['error']}")
            # Agent needs to handle this, this display is for user awareness
            # The agent's fallback in parse_input_text might already make it an attempt_completion


    # 3. Execute Action
    is_interactive, tool_result_payload, executed_action_details = agent.execute_action(action_details)
    
    # Display Tool Result (raw output from tool)
    if tool_result_payload:
        # Filter out interactive markers for direct display, they are handled by is_interactive logic
        displayable_tool_result = [item for item in tool_result_payload if item.get("type") != "interactive_marker"]
        if displayable_tool_result:
            with st.chat_message("system", avatar="⚙️"): # System avatar for tool results
                 with st.expander("Tool Result ✨", expanded=True):
                    for item in displayable_tool_result:
                        st.markdown(f"**{item.get('text', '')}**")


    # 4. Handle based on action type (interactive, completion, or needs more processing)
    
    tool_name = executed_action_details.get("tool_name")
    tool_params = executed_action_details.get("parameters", {})

    if is_interactive:
        if tool_name == "ask_followup_question":
            prompt_to_user = tool_params.get("question", "I need more information. Can you clarify?")
            st.session_state.messages.append({"role": "assistant", "content": prompt_to_user}) # Display agent's question
            st.session_state.agent_is_waiting_for_input = True
            st.session_state.interactive_tool_data = {
                "action_details": executed_action_details,
                "prompt_to_user": prompt_to_user
            }
            # Agent's `build_tool_result_message_for_llm` will be called *after* user provides interactive input.
            st.rerun() # Rerun to show input field for interactive response
            return # Stop processing here, wait for user's interactive input

        elif tool_name == "attempt_completion":
            final_answer = tool_params.get("result", "I've completed the request.")
            # This is a final answer from the agent.
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            # No further LLM call needed for this turn if it's a completion.
            # Add the marker to LLM history so it knows it tried to complete.
            agent.build_tool_result_message_for_llm(tool_result_payload, executed_action_details)
            st.rerun()
            return
    else:
        # Non-interactive tool was called, or an error occurred during tool call.
        # The result of this tool needs to be fed back to the LLM for further processing.
        agent.build_tool_result_message_for_llm(tool_result_payload, executed_action_details)
        # Now, recursively call run_agent_turn to let LLM process the tool result.
        # Pass no user_input_for_llm, as the agent is reacting to its own tool use.
        run_agent_turn()


# --- Input Handling ---
if st.session_state.agent_is_waiting_for_input:
    # Agent is waiting for user's response to an interactive tool (e.g., ask_followup_question)
    agent: WeatherAgent = st.session_state.weather_agent
    interactive_data = st.session_state.interactive_tool_data
    
    st.info(f"Agent asks: {interactive_data['prompt_to_user']}") # Show the agent's question
    
    user_interactive_response = st.chat_input(f"Your response to the agent:", key="interactive_response_input")

    if user_interactive_response:
        # Add user's interactive response to display history
        st.session_state.messages.append({"role": "user", "content": user_interactive_response})
        
        # Build the message for the LLM using the user's interactive response
        agent.build_tool_result_message_for_llm(
            tool_result_payload=[], # No direct tool payload, user_interactive_input is key
            action_details=interactive_data['action_details'],
            user_interactive_input=user_interactive_response
        )
        
        st.session_state.agent_is_waiting_for_input = False
        st.session_state.interactive_tool_data = None
        
        # Agent continues processing with the new user input
        run_agent_turn() # No direct user input here, agent processes the interactive response
        st.rerun()

else:
    # Standard chat input
    if prompt := st.chat_input("What would you like to know about the weather?"):
        # Add user message to display history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): # Display it immediately
            st.markdown(prompt)

        # Limit display message history length
        if len(st.session_state.messages) > MAX_MESSAGES_DISPLAY:
            st.session_state.messages = st.session_state.messages[-MAX_MESSAGES_DISPLAY:]
        
        # Start agent processing turn with the new user prompt
        run_agent_turn(user_input_for_llm=prompt)
        st.rerun()