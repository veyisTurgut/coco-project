from llm.mylib import *
import base64, requests, json, copy, traceback
from requests.exceptions import Timeout, ReadTimeout, ConnectTimeout


class CurlVertex():
    def __init__(self, llm_config: LLMConfig, logger=None):
        self.logger = logger
        self.model_name = llm_config.model_name
        
        # ! WARNING : This is not safe, we will be enumerating this in the near future
        self.llm_region = llm_config.llm_region if llm_config.llm_region else "europe-west3"
        
        # fields to send in CURL
        self.generation_config = {
            "temperature": llm_config.temperature,
            "topP": llm_config.top_p,
            "topK": llm_config.top_k,
            "maxOutputTokens": llm_config.max_output_tokens,
        }
        if llm_config.response_mimetype:
            self.generation_config["responseMimeType"] = llm_config.response_mimetype
        if llm_config.safety_settings is None:
            self.safety_settings = [
                {
                    "category": i,
                    "threshold": 3,  # 4 (NONE) is not working: You can get access either (a) through an allowlist via your Google account team, or (b) by switching your account type to monthly invoiced billing via this instruction: https://cloud.google.com/billing/docs/how-to/invoiced-billing.
                    "method": 0
                } for i in range(1, 5)
            ]
        else:
            self.safety_settings = llm_config.safety_settings

        self.logger.info("Initialize curl vertex")

    def populate_content(self, chat_history: List[Message]) -> List[dict]:
        chat_history = copy.deepcopy(chat_history)
        content = []
        for item in chat_history:
            single_content = {"role": item.role}

            if item.message_content_type == ChatInputContentType.TEXT or item.message_content_type == ChatInputContentType.VOICE_TEXT:
                parts = [{"text": item.message}]

            elif item.message_content_type in [ChatInputContentType.IMAGE_JPEG, ChatInputContentType.IMAGE_PNG, ChatInputContentType.PDF]:
                if item.message_content_type == ChatInputContentType.PDF:
                    mime_type = "application/pdf"
                elif item.message_content_type == ChatInputContentType.IMAGE_JPEG:
                    mime_type = "image/jpeg"
                elif item.message_content_type == ChatInputContentType.IMAGE_PNG:
                    mime_type = "image/png"
                encoded_image = base64.b64encode(requests.get(item.message_uri).content).decode('utf-8')
                #parts = [{"inline_data": {"mimeType": mime_type, "data": f"data:{mime_type};base64,{encoded_image}"}}, {"text": item.message}]
                parts = [{"inline_data": {"mimeType": mime_type, "data": f"{encoded_image}"}}, {"text": item.message}]
            else:
                raise ValueError(f"Unsupported content type in ChatHistoryModule: {item.message_content_type}")

            single_content["parts"] = parts
            content.append(single_content)
                  
        return content
    

    def generate(self,
                 instruction_prompt: str = None,
                 chat_history: List[Message] = None,
                 content_dict: List[dict] | None = None,
                 is_streaming: bool = True,
                 pdf_files: List[str] = None,
                 return_tokens: bool = False,
                 timeout: int = 60,
                 **kwargs):
        assert instruction_prompt or chat_history or content_dict, "At least one of instruction_prompt, chat_history or content_dict must be given."
        # get json_data with config
        config = copy.deepcopy(self.generation_config)
        config.update(kwargs)
        config = {k: v for k, v in config.items() if v is not None}
        if "max_tokens" in config:
            config["maxOutputTokens"] = config["max_tokens"]
            del config["max_tokens"]
        if self.model_name == ChatMainLLMName.VERTEX_GEMINI_25_FLASH_PREVIEW:
            config["thinkingConfig"] = { "thinkingBudget": 0 }

        json_data = {
            "contents": None,
            "systemInstruction": None,
            "safetySettings": self.safety_settings,
            "generationConfig": config
        }

        # create the content
        if content_dict is None:
            content_dict = {self.model_name: self.populate_content(chat_history) if chat_history else []}
        elif chat_history and content_dict:
            self.logger.warning("Both chat history and content are given. Using content.")
        if len(content_dict[self.model_name]) == 0:
            # add system instruction as a content from user
            content_dict = {self.model_name: [
                {
                    "role": "user",
                    "parts": [
                        {"text": instruction_prompt}
                    ]
                }
            ]}
            if pdf_files:
                content_dict[self.model_name][0]["parts"].extend(populate_pdfs(pdf_files, ChatMainLLM.VERTEXLLM))
        else:
            if instruction_prompt:
                # add system instruction
                json_data["systemInstruction"] = {'parts': [{'text': instruction_prompt}]}

        json_data["contents"] = content_dict[self.model_name]

        PROJECT_ID = vertex_credentials.project_id
        stream = requests.Session()
        
        # INFO: REST URL Format
        # POST https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/publishers/google/models/gemini-1.5-pro:streamGenerateContent
        
        url_generate_content_endpoint = ":streamGenerateContent" if is_streaming else ":generateContent"
        url = 'https://' + self.llm_region + '-aiplatform.googleapis.com/v1/projects/' + PROJECT_ID + '/locations/' + self.llm_region + '/publishers/google/models/' + self.model_name.value + url_generate_content_endpoint

        def handle_json(myjson):
            try:
                data: dict = json.loads(myjson)
                input_tokens = 0
                output_tokens = 0
                if "usageMetadata" in data:
                    if "promptTokenCount" in data["usageMetadata"]:
                        input_tokens = data["usageMetadata"]["promptTokenCount"]
                    if "candidatesTokenCount" in data["usageMetadata"]:
                        output_tokens = data["usageMetadata"]["candidatesTokenCount"]
                candidates = data.get('candidates', [])
                if len(candidates) > 0:
                    parts = data.get('candidates', [])[0].get('content', {}).get('parts', [])
                    if len(parts) < 0:
                        if return_tokens:
                            return LlmResponseItem("", False, input_tokens, output_tokens, self.model_name.value)
                        return LlmResponseItem("", False, None, None, self.model_name.value)
                    content = parts[0].get('text', '')
                    finish_reason = candidates[0].get('finishReason', "")
                    contains_harmful_content = finish_reason == "SAFETY"
                    if return_tokens:
                        return LlmResponseItem(content, contains_harmful_content, input_tokens, output_tokens, self.model_name.value)
                    return LlmResponseItem(content, contains_harmful_content, None, None, self.model_name.value)
                else:
                    if return_tokens:
                        return LlmResponseItem("", False, input_tokens, output_tokens, self.model_name.value)
                    return LlmResponseItem("", False, None, None, self.model_name.value)
            except Exception as e:
                raise Exception(f"Error in parsing response: {myjson}\nPreviously sent this json: {json_data}\nError: {traceback.format_exc()}")

        headers = {
            'Authorization': 'Bearer ' + vertex_credentials.token,
            'Content-Type': 'application/json',
        }
        
        buffered_json = ""
        try:
            with stream.post(url=url, headers=headers, stream=True, json=json_data, timeout=timeout) as resp:
                if resp.status_code // 100 > 3:
                    self.logger.error(f"Error in response: {resp.text} \n Previously sent this json: {json_data} to this url: {url} with headers: {headers}")
                    raise Exception(f"{resp.status_code} Error in response: {resp.text}")
                for line in resp.iter_lines():
                    if line:
                        line_str = line.decode()
                        if line_str == '{' or line_str == "[{":
                            buffered_json = "{"
                        elif line_str == '}' or line_str == "}]":
                            buffered_json += "}"
                            yield handle_json(buffered_json)
                        else:
                            buffered_json += line_str
        except (Timeout, ReadTimeout, ConnectTimeout) as e:
            raise Timeout(f"Timeout in {url}")
        
vertex_credentials = VertexCredentials()
vertex_credentials.initialize()