import streamlit as st
import os
from dotenv import load_dotenv
from litellm import completion

# Constants
MAX_MESSAGES = 20
MAX_INPUT_LENGTH = 1000
DEFAULT_SYSTEM_PROMPT = 'You are a chatbot.'

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

# Load environment variables
load_dotenv()

from enum import StrEnum

class ModelChoice(StrEnum):
    OPENER_ROUTER_GEMINI = 'open-router-gemini-flash'
    DEEPSEEK = "deepseek-chat"

@st.cache_resource
def initialize_client(model_choice: ModelChoice):
    if model_choice not in model_choices:
        st.error(f"Invalid model choice: {model_choice}. Please choose from {list(model_choices.keys())}.")
        st.stop()
    model_info = model_choices[model_choice]
    api_key = model_info['api_key']
    base_url = model_info['base_url']
    model_name = model_info['model_name']
    print(f"Initializing for {model_name} with API_KEY={api_key} and BASE_URL={base_url}")
    if not api_key:
        st.error("API_KEY is not set. Please configure it in the .env file.")
        st.stop()
    return model_name, api_key, base_url

def get_assistant_response(model_name, api_key, base_url, messages, system_prompt, num_retries=3):
    try:
        # 使用 litellm 的 completion 方法
        stream = completion(
            model=model_name,
            messages=[{"role": "system", "content": system_prompt}] + [{"role": m["role"], "content": m["content"]} for m in messages],
            api_key=api_key,
            base_url=base_url,
            stream=True,
            num_retries=num_retries  # 配置自动重试次数
        )
        return stream
    except Exception as e:
        st.error(f"Failed to get response: {str(e)}")
        return None

# Initialize client
MODEL_NAME, API_KEY, BASE_URL = initialize_client(ModelChoice.DEEPSEEK)

# Set up sidebar for system prompt configuration and new conversation
st.sidebar.header("System Prompt Configuration")
if 'system_prompt' not in st.session_state:
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT

system_prompt = st.sidebar.text_area(
    "System Prompt",
    value=st.session_state.system_prompt,
    height=500,
    help="Enter the system prompt to define the AI's behavior."
)
if system_prompt != st.session_state.system_prompt:
    st.session_state.system_prompt = system_prompt

# Add button to start a new conversation
if st.sidebar.button("New Conversation"):
    st.session_state.messages = []
    st.rerun()  # Force rerun to refresh the UI

# Set up main UI
st.title('Chat with AI Assistant')

# Initialize session state for messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("What is up?"):
    if prompt.strip():
        if len(prompt) > MAX_INPUT_LENGTH:
            st.error(f"Input too long. Maximum length is {MAX_INPUT_LENGTH} characters.")
        else:
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Limit message history
            if len(st.session_state.messages) > MAX_MESSAGES:
                st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("正在请求AI助理...")
                
                # Get and process stream
                stream = get_assistant_response(MODEL_NAME, API_KEY, BASE_URL, st.session_state.messages, st.session_state.system_prompt)
                if stream:
                    response = ""
                    for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            response += chunk.choices[0].delta.content
                            message_placeholder.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    message_placeholder.markdown("无法获取响应，请稍后再试。")
    else:
        st.warning("Please enter a message.")