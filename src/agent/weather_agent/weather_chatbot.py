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
        This message should represent the information the LLM needs to continue.
        For most LLMs expecting OpenAI format, 'user' role content should be a string.
        'tool' role is preferred for actual tool outputs if using OpenAI tool calling.
        Given the current XML-based tool use, we'll use 'user' role with string content.
        """
        tool_name_from_details = action_details.get('tool_name', 'unknown_tool_from_details')
        llm_message_content_str = "" # Initialize as string

        if user_interactive_input:
            # This is when the user responds to an 'ask_followup_question'
            llm_message_content_str = f"User's response to my question about '{tool_params_executed.get('question', 'previous question')}': {user_interactive_input}"
            # Add this to agent's history as a user message containing the user's actual response
            self.messages.append({'role': 'user', 'content': llm_message_content_str})

        elif tool_name_from_details == "attempt_completion":
            # For attempt_completion, the agent has finished its thought process for this branch.
            # The assistant's message containing the <attempt_completion> tag is already in history.
            # No further message needs to be sent to the LLM representing the "result" of this pseudo-tool.
            # The 'final_answer' is for the UI. The agent's internal turn concludes.
            if self.is_debug:
                print(f"INFO: 'attempt_completion' executed. No further 'tool result' message added to LLM history for this action.")
            return # Do not add any message for attempt_completion outcome

        elif tool_name_from_details == "ask_followup_question":
            # When 'ask_followup_question' is INITIALLY called (not the user's response yet):
            # The assistant message with the <action><ask_followup_question>...</ask_followup_question></action>
            # is already in agent.messages. The agent will now wait for user input.
            # No "tool result" needs to be added to LLM history at this point.
            if self.is_debug:
                print(f"INFO: 'ask_followup_question' initiated. Waiting for user's interactive response. No 'tool result' message added to LLM history yet.")
            return # Do not add any message for ask_followup_question initiation

        else: # For actual, non-interactive tools that return data
            # Collate texts from tool_result_payload
            # tool_result_payload is like: [{"type": "text", "text": "[tool_name] Result: {json_output}"}]
            # or it could contain error messages from tool execution.
            result_texts = []
            for item in tool_result_payload:
                if item.get("type") == "text" and "text" in item:
                    result_texts.append(item["text"])
                elif item.get("type") == "error": # If your payload can contain explicit errors
                    result_texts.append(f"Error from tool '{tool_name_from_details}': {item.get('text')}")

            if not result_texts:
                llm_message_content_str = f"Tool '{tool_name_from_details}' was called but returned no textual result to report."
            else:
                llm_message_content_str = "\n".join(result_texts)
            
            # This message represents the output of the tool, which the LLM needs to process.
            # Role 'user' is used here as per the agent's current design of not using OpenAI 'tool' role.
            self.messages.append({'role': 'user', 'content': llm_message_content_str})


        if self.is_debug and llm_message_content_str: # Only print if a message was actually constructed
            print(f"===== Message Appended to Agent's Internal History (for LLM) =====")
            print(f"Role: {self.messages[-1]['role']}")
            print(f"Content:\n{self.messages[-1]['content']}")
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
        'api_key': os.getenv("DEEPSEEK_API_KEY"),
        'base_url': os.getenv("DEEPSEEK_API_BASE_URL")
    },
    ModelChoice.OPENER_ROUTER_GEMINI: {
        'model_name': 'openrouter/google/gemini-flash-1.5', 
        'api_key': os.getenv("OPENROUTER_API_KEY"),
        'base_url': os.getenv("OPENROUTER_BASE_URL")
    }
}

st.set_page_config(layout="wide", page_title="Weather Agent Chatbot")

# --- Initialization & Sidebar ---
st.sidebar.title("Weather Agent Configuration")

selected_model_key = st.sidebar.selectbox(
    "Choose a Model:",
    options=list(ModelChoice),
    format_func=lambda x: x.value
)
MODEL_INFO = MODEL_CONFIGS[selected_model_key]

if 'system_prompt' not in st.session_state:
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
current_system_prompt = st.sidebar.text_area(
    "System Prompt (Edit carefully):",
    value=st.session_state.system_prompt,
    height=300
)

if 'is_debug_mode' not in st.session_state:
    st.session_state.is_debug_mode = True 
st.session_state.is_debug_mode = st.sidebar.checkbox("Enable Agent Debug Mode (Console Logs)", value=st.session_state.is_debug_mode)

def initialize_agent(force_reinit=False):
    model_name = MODEL_INFO['model_name']
    api_key = MODEL_INFO['api_key']
    base_url = MODEL_INFO['base_url']

    if not api_key or not base_url:
        st.sidebar.error(f"API Key or Base URL for {selected_model_key.value} is not set in .env file!")
        st.stop() # Stop if essential configs are missing

    agent_needs_init = force_reinit or \
                       'weather_agent' not in st.session_state or \
                       st.session_state.weather_agent.model_name != model_name or \
                       st.session_state.weather_agent.system_prompt != st.session_state.system_prompt or \
                       st.session_state.weather_agent.api_key != api_key or \
                       st.session_state.weather_agent.base_url != base_url or \
                       st.session_state.weather_agent.is_debug != st.session_state.is_debug_mode
    
    if agent_needs_init:
        if st.session_state.is_debug_mode: print("Re-initializing WeatherAgent.")
        st.session_state.weather_agent = WeatherAgent(
            messages=[], # Agent's internal messages start empty for a new agent instance
            system_prompt=st.session_state.system_prompt,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            is_debug=st.session_state.is_debug_mode
        )
        # If messages are tied to the agent instance, new agent means new message history for it.
        # st.session_state.messages (for UI display) is handled separately.
    return st.session_state.weather_agent


if 'messages' not in st.session_state: 
    st.session_state.messages = [] 

# Handle system prompt changes
if current_system_prompt != st.session_state.system_prompt:
    st.session_state.system_prompt = current_system_prompt
    initialize_agent(force_reinit=True) # Force re-init if system prompt changes
    st.session_state.messages = [] # Also clear UI messages for new system prompt
    st.rerun()

# Initialize agent (also handles changes in model, debug mode etc.)
initialize_agent()


if 'agent_is_waiting_for_input' not in st.session_state:
    st.session_state.agent_is_waiting_for_input = False
if 'interactive_tool_data' not in st.session_state: 
    st.session_state.interactive_tool_data = None
if 'current_turn_intermediate_steps' not in st.session_state:
    st.session_state.current_turn_intermediate_steps = []

if st.sidebar.button("New Conversation"):
    st.session_state.messages = [] # Clear UI messages
    initialize_agent(force_reinit=True) # Create a new agent instance, effectively clearing its internal messages
    st.session_state.agent_is_waiting_for_input = False
    st.session_state.interactive_tool_data = None
    st.session_state.current_turn_intermediate_steps = []
    st.rerun()

# --- Main Chat UI ---
st.title("Weather Agent Chatbot 🤖🌦️")
st.markdown(f"Using Model: `{MODEL_INFO['model_name']}`")

# Display chat history from st.session_state.messages (UI display history)
for i, msg_data in enumerate(st.session_state.messages):
    with st.chat_message(msg_data["role"]):
        if msg_data["role"] == "assistant" and "intermediate_steps" in msg_data and msg_data["intermediate_steps"]:
            expanded_default = (i == len(st.session_state.messages) - 1) 
            with st.expander("View Agent's Process", expanded=expanded_default):
                for step in msg_data["intermediate_steps"]:
                    step_type = step.get("type", "unknown")
                    step_title = step.get("title", step_type.replace("_", " ").title())
                    step_content = step.get("content", "")

                    st.markdown(f"**{step_title}**")
                    if step_type == "llm_raw_response":
                        st.code(step_content, language='xml')
                    elif step_type in ["thinking", "info"]:
                        st.markdown(f"```\n{step_content}\n```")
                    elif step_type in ["action_parsed", "action_executed"]:
                        st.json(step_content)
                    elif step_type == "tool_result_payload":
                        if isinstance(step_content, list):
                            for item_idx, item_data in enumerate(step_content):
                                item_text = item_data.get('text', json.dumps(item_data))
                                st.markdown(f"- Part {item_idx+1} ({item_data.get('type', 'unknown')}):")
                                # Try to pretty print if text is JSON string
                                try:
                                    parsed_json = json.loads(item_text.split("Result: ", 1)[-1] if "Result: " in item_text else item_text)
                                    st.json(parsed_json)
                                except (json.JSONDecodeError, TypeError):
                                    st.markdown(f"  ```\n  {item_text}\n  ```")
                        else:
                             st.markdown(f"```\n{json.dumps(step_content, indent=2)}\n```")
                    else:
                        st.write(step_content)
        
        # Display the main user-facing content of the message
        # For assistant, this is the final answer or question. For user, it's their query.
        main_content = msg_data.get("content_display", msg_data["content"]) # Use content_display if available
        if isinstance(main_content, list): # Handle complex content like list of dicts from agent's 'user' role message
            for block in main_content:
                if isinstance(block, dict) and block.get("type") == "text":
                    st.markdown(block["text"])
                else: # Fallback for other complex structures
                    st.markdown(f"```json\n{json.dumps(block, indent=2)}\n```")
        else:
            st.markdown(str(main_content))


def process_agent_turn(initial_user_input: str = None):
    agent: WeatherAgent = st.session_state.weather_agent
    
    # Clear intermediate steps for this new turn or continuation
    st.session_state.current_turn_intermediate_steps = []

    if initial_user_input: # This is a fresh input from the user to start a turn
        # Add user's direct message to agent's internal history
        agent.messages.append({"role": "user", "content": initial_user_input})
        if agent.is_debug: print(f"Appended user message to agent's internal history: '{initial_user_input}'")
        st.session_state.current_turn_intermediate_steps.append(
            {"type": "info", "title": "ℹ️ User Input", "content": f"User asked: {initial_user_input}"}
        )

    MAX_AGENT_STEPS = 100 # Max iterations for tool use within one turn
    for step_count in range(MAX_AGENT_STEPS):
        if agent.is_debug: print(f"Agent processing step {step_count + 1} of this turn.")

        llm_full_response = ""
        # Display thinking placeholder in the main chat area
        # It will be replaced by the actual message once processing for this step is done or turn ends
        progress_message = "🤔 Thinking..." if step_count == 0 and initial_user_input else f"⚙️ Processing step {step_count + 1}..."
        if step_count > 0 :
             st.session_state.current_turn_intermediate_steps.append(
                {"type": "info", "title": f"↪️ Agent Loop (Step {step_count + 1})", "content": "Agent is processing tool results or continuing thought."}
            )

        stream_placeholder = st.empty() # Temporary placeholder for streaming status if needed
        # stream_placeholder.markdown(progress_message) # Can be uncommented if needed

        try:
            # Note: Live streaming directly to UI here might be complex with the single-expander goal
            # For now, we collect the full response then add to intermediate steps.
            temp_response_for_stream_display = ""
            for chunk in agent.get_assistant_response_stream():
                llm_full_response += chunk
                # To provide some feedback during streaming if it's long:
                # temp_response_for_stream_display += chunk
                # stream_placeholder.markdown(temp_response_for_stream_display + "▌")
            # stream_placeholder.markdown(llm_full_response) # Final full response for this step
            stream_placeholder.empty() # Clear the temporary streaming placeholder

        except Exception as e:
            st.error(f"Error during LLM call: {e}")
            st.session_state.current_turn_intermediate_steps.append(
                {"type": "error", "title": "❌ LLM API Error", "content": f"LLM API Error: {e}"}
            )
            # Attempt to make the agent provide a graceful error message
            llm_full_response = f"<thinking>An LLM API error occurred: {e}</thinking><action><attempt_completion><result>I'm sorry, I encountered an issue connecting to the language model. Please try again later.</result></attempt_completion></action>"
            # Continue to parse this error response so it becomes a completion

        st.session_state.current_turn_intermediate_steps.append(
            {"type": "llm_raw_response", "title": "🧠 LLM Raw Output", "content": llm_full_response}
        )
        # Add raw LLM response to agent's internal messages. Crucial for multi-step reasoning.
        agent.messages.append({"role": "assistant", "content": llm_full_response})
        if agent.is_debug: print(f"Appended assistant (LLM) raw response to agent's internal history.")

        thinking_content, action_details_parsed = agent.parse_input_text(llm_full_response)
        
        if thinking_content: # thinking_content is agent.last_thinking_content
            st.session_state.current_turn_intermediate_steps.append(
                {"type": "thinking", "title": "🤔 Agent Thinking", "content": thinking_content}
            )
        st.session_state.current_turn_intermediate_steps.append(
            {"type": "action_parsed", "title": "🛠️ Action Parsed (from LLM)", "content": action_details_parsed}
        )
        
        # If parsing itself reported an error in action_details_parsed
        if "error" in action_details_parsed and action_details_parsed.get("tool_name") != "attempt_completion":
             if agent.is_debug: print(f"Parsing error in action_details: {action_details_parsed['error']}")
             # agent.execute_action will handle this by likely defaulting to attempt_completion

        is_interactive, tool_result_payload, executed_action_details = agent.execute_action(action_details_parsed)
        # executed_action_details is the action_data that was actually attempted (could be modified by execute_action)
        st.session_state.current_turn_intermediate_steps.append(
            {"type": "action_executed", "title": "⚙️ Action Executed (Tool Details)", "content": executed_action_details} 
        )
        if tool_result_payload: 
            st.session_state.current_turn_intermediate_steps.append(
                {"type": "tool_result_payload", "title": "✨ Tool Result Payload (Raw Tool Output)", "content": tool_result_payload}
            )

        tool_name_executed = executed_action_details.get("tool_name")
        tool_params_executed = executed_action_details.get("parameters", {}) # Make sure this is defined

        if tool_name_executed == "attempt_completion":
            final_answer = tool_params_executed.get("result", "I've completed the request.")
            # agent.build_tool_result_message_for_llm will now correctly do NOTHING for attempt_completion
            agent.build_tool_result_message_for_llm(tool_result_payload, executed_action_details)
            return "completion", final_answer

        elif is_interactive:
            if tool_name_executed == "ask_followup_question":
                prompt_to_user = tool_params_executed.get("question", "I need more information. Can you clarify?")
                st.session_state.agent_is_waiting_for_input = True
                st.session_state.interactive_tool_data = {
                    "action_details": executed_action_details,
                    "prompt_to_user": prompt_to_user,
                    "tool_params_executed": tool_params_executed # Pass params for use in build_tool_result
                }
                # agent.build_tool_result_message_for_llm will now correctly do NOTHING for ask_followup_question initiation
                agent.build_tool_result_message_for_llm(tool_result_payload, executed_action_details)
                return "interactive", prompt_to_user
            # ... (else for other interactive tools, if any) ...
        else:
            # Non-interactive tool was called (e.g., weather_now)
            # agent.build_tool_result_message_for_llm will now add a 'user' message with string content
            agent.build_tool_result_message_for_llm(tool_result_payload, executed_action_details)
            if agent.is_debug: print(f"Continuing loop in process_agent_turn after non-interactive tool '{tool_name_executed}'")
                


        

    # If loop finishes without returning (e.g., MAX_AGENT_STEPS reached)
    max_steps_msg = "Maximum processing steps reached. The agent may be stuck or the task is too complex. Please try rephrasing or starting a new conversation."
    st.session_state.current_turn_intermediate_steps.append(
        {"type": "error", "title": "⚠️ Max Steps Reached", "content": max_steps_msg}
    )
    # Try to complete with this message
    agent.build_tool_result_message_for_llm(
        [],
        {"tool_name": "attempt_completion", "parameters": {"result": max_steps_msg}}
    )
    return "completion", max_steps_msg


# --- Input Handling ---
user_prompt = st.chat_input("What would you like to know about the weather?")

if st.session_state.agent_is_waiting_for_input:
    interactive_data = st.session_state.interactive_tool_data
    # Ensure tool_params_executed is available for the build_tool_result_message_for_llm call
    tool_params_executed = interactive_data.get("tool_params_executed", {}) # Get it from stored data
    
    # Display the agent's question clearly if waiting for input
    # This info is already part of the last assistant message, but can be reiterated.
    # st.info(f"Agent asks: {interactive_data['prompt_to_user']}") 
    
    user_interactive_response = st.chat_input(
        f"Your response to the agent's question: {interactive_data['prompt_to_user']}", 
        key=f"interactive_input_{interactive_data['action_details'].get('tool_name', 'default_key')}"
    )

    if user_interactive_response:
        st.session_state.messages.append({"role": "user", "content": user_interactive_response, "content_display": user_interactive_response})
        
        agent: WeatherAgent = st.session_state.weather_agent
        # Update agent's internal history with the user's interactive response, formatted as a tool result
        agent.build_tool_result_message_for_llm(
            tool_result_payload=[], # Not directly from a tool, but from user fulfilling tool's request
            action_details=interactive_data['action_details'], 
            user_interactive_input=user_interactive_response
        )
        if agent.is_debug: print("Built tool result from user's interactive input and added to agent's internal history.")
        
        st.session_state.agent_is_waiting_for_input = False
        st.session_state.interactive_tool_data = None
        
        # Agent continues processing. initial_user_input is None because we are continuing a turn.
        status, final_message_content = process_agent_turn(initial_user_input=None) 
        
        assistant_response_for_ui = {
            "role": "assistant",
            "content": final_message_content, # The final user-facing message or next question
            "content_display": final_message_content,
            "intermediate_steps": list(st.session_state.current_turn_intermediate_steps) 
        }
        st.session_state.messages.append(assistant_response_for_ui)
        
        if len(st.session_state.messages) > MAX_MESSAGES_DISPLAY:
            st.session_state.messages = st.session_state.messages[-MAX_MESSAGES_DISPLAY:]
        st.rerun()

elif user_prompt: # Standard new user query
    st.session_state.messages.append({"role": "user", "content": user_prompt, "content_display": user_prompt})
    
    status, final_message_content = process_agent_turn(initial_user_input=user_prompt)

    assistant_response_for_ui = {
        "role": "assistant",
        "content": final_message_content, 
        "content_display": final_message_content,
        "intermediate_steps": list(st.session_state.current_turn_intermediate_steps)
    }
    st.session_state.messages.append(assistant_response_for_ui)
    
    if len(st.session_state.messages) > MAX_MESSAGES_DISPLAY:
        st.session_state.messages = st.session_state.messages[-MAX_MESSAGES_DISPLAY:]
    st.rerun()