import os, time
from typing import List
from functools import wraps
from enum import Enum
from dataclasses import dataclass
from typing import List

# Define the core wrapper function that contains the common logic
def timeit_wrapper(func, class_name=None):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        func_name = func.__name__
        
        if class_name:
            base_key = f"{class_name}.{func_name}"
        else:
            if len(args) == 0:
                base_key = func_name
            else:
                actual_class_name = args[0].__class__.__name__
                base_key = f"{actual_class_name}.{func_name}"
                
        print(f'Function {func_name} Took {total_time:.2f} seconds')
        
        return result
    return wrapper


# Create the decorator factory with a default parameter
def timeit(class_name=None):
    if callable(class_name):
        # If class_name is callable, it means decorator is used without parameters
        func = class_name
        return timeit_wrapper(func)
    else:
        # Create the actual decorator if class_name is not callable
        def decorator(func):
            return timeit_wrapper(func, class_name)
        return decorator



class ChatMainLLM(Enum):
    GPTLLM = "OPENAI"
    BEDROCKLLM = "BEDROCK"
    VERTEXLLM = "VERTEX"
    AZUREGPTLLM = "AZUREOPENAI"
    AZUREGPTCURLLLM = "AZUREOPENAICURL"

def populate_pdfs(pdf_links: List[str], model_arch: ChatMainLLM) -> List[dict]:
    if model_arch == ChatMainLLM.VERTEXLLM:
        return [
            {
                "fileData": {
                    "mimeType": "application/pdf",
                    "fileUri": pdf_link
                }
            } for pdf_link in pdf_links
        ]
    elif model_arch == ChatMainLLM.AZUREGPTCURLLLM:
        #no pdf support for azure
        raise NotImplementedError()



class VertexCredentials:
    def __init__(self) -> None:
        self.is_initialized = False

    def initialize(self, vertex_keys_path=None):
        import google.auth
        import google.auth.transport.requests
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") == None:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = vertex_keys_path if vertex_keys_path else os.path.join(os.getcwd(), "vertex_keys.json")
        if not os.path.exists(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]):
            raise FileNotFoundError(f"Vertex credentials not found at {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")
        creds, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
        auth_req = google.auth.transport.requests.Request()
        creds.refresh(auth_req)
        self.token = creds.token
        self.project_id="vertexai-keys"
        self.is_initialized = True


class ChatInputContentType(Enum):
    TEXT = "TEXT"
    IMAGE_PNG = "IMAGE_PNG"
    IMAGE_JPEG = "IMAGE_JPEG"
    PDF = "PDF"
    VOICE_TEXT = "VOICE_TEXT"


class LLMConfig:
    def __init__(self, config: dict):
        self._from_dict(config)

    def _from_dict(self, config: dict):
        self.model_arch = config.get("model_arch")
        self.model_name = config.get("model_name")
        self.temperature = config.get("temperature")
        self.safety_settings = config.get("safety_settings")
        self.max_output_tokens = config.get("max_output_tokens")
        self.top_k = config.get("top_k")
        self.top_p = config.get("top_p")
        self.llm_region = config.get("llm_region", "europe-west3") # INFO: Useful especially for the Vertex
        self.response_mimetype = config.get("response_mimetype")
        self.second_speaker = config.get("second_speaker", "assistant")
        self.token_per_stream_msg = config.get("token_per_stream_msg", 5)

    def to_dict(self) -> dict:
        return {
            "model_arch": self.model_arch,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "safety_settings": self.safety_settings,
            "max_output_tokens": self.max_output_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "llm_region": self.llm_region,
            "response_mimetype": self.response_mimetype,
            "second_speaker": self.second_speaker,
            "token_per_stream_msg": self.token_per_stream_msg
        }

    def __dict__(self):
        return self.to_dict()
    
    def __str__(self):
        return str(self.to_dict())



class ChatMainLLMName(Enum):
    AZURE_GPT_3_5 = "madlen-gpt-35"
    AZURE_GPT_4o = "madlen-gpt4o"
    AZURE_GPT_4o_MINI = "madlen-gpt-4o-mini"
    AZURE_GPT_4o_AUGUST = "madlen-gpt4o-august"
    LLAMA_3_8B = "meta.llama3-8b-instruct-v1:0"
    LLAMA_3_70B = "meta.llama3-70b-instruct-v1:0"
    CLAUDE_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
    CLAUDE_SONNET_3_5 = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    CLAUDE_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
    CLAUDE_OPUS = "anthropic.claude-3-opus-20240229-v1:0"
    VERTEX_GEMINI_15_PRO_1 = "gemini-1.5-pro-001"
    VERTEX_GEMINI_15_PRO_2 = "gemini-1.5-pro-002"
    VERTEX_GEMINI_PRO = "gemini-1.0-pro-002"
    VERTEX_GEMINI_FLASH = "gemini-1.5-flash-001"
    VERTEX_GEMINI_FLASH_2 = "gemini-1.5-flash-002"
    VERTEX_GEMINI_FLASH_2_0 = "gemini-2.0-flash-exp"
    VERTEX_SONNET_35 = "claude-3-5-sonnet@20240620"
    VERTEX_GEMINI_25_FLASH_PREVIEW = "gemini-2.5-flash-preview-04-17"


@dataclass
class Message():
    role: str | None
    message: str | None
    message_type: str | None
    message_uri: str | None
    message_content_type: ChatInputContentType


@dataclass
class LlmResponseItem():
    content: str
    is_harmful: bool
    input_token_count: int
    output_token_count: int
    model_name: str
    
