import logging, os
from curl_vertex import CurlVertex
from mylib import *
from google.cloud import storage

@timeit
def upload_to_gcs(image_path):
    service_account_key_path = "./vertex_keys.json"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_key_path
    storage_client = storage.Client()

    bucket_name = "veyis_bucket"
    destination_blob_name = image_path 

    # Create a blob object
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Upload the file to GCS
    with open(image_path, "rb") as f:
        blob.upload_from_file(f)

    print(f"File uploaded to GCS: {blob.public_url}")
    return blob.public_url

@timeit
def agent_1_extract_subjects(message_uri, subjects):
    # Set the model name
    model_arch = ChatMainLLM.VERTEXLLM.value
    model_name = ChatMainLLMName.VERTEX_GEMINI_15_PRO_2
    # Set the instruction prompt
    instruction_prompt = """
       TODO
        """
        
    instruction_prompt = instruction_prompt.format(subjects=subjects)

    # Set the chat history
    chat_history = [
        Message(role="user", 
                message=".", 
                message_content_type=ChatInputContentType.IMAGE_PNG, message_uri=message_uri, message_type=None),
    ]

    # Set the timeout
    timeout = 60

    # Create the CurlVertex object
    curl_vertex = CurlVertex(logger = logger, llm_config=LLMConfig({"model_arch": model_arch, "model_name": model_name, "temperature": 0.5, "top_k": 10, "top_p": 0.95}))

    # Generate the response
    buffer = ""
    for resp in curl_vertex.generate(instruction_prompt=instruction_prompt, 
                                                 chat_history=chat_history, 
                                                 is_streaming=False, 
                                                 return_tokens=True, 
                                                 timeout=timeout):
        resp: LlmResponseItem
        print("input_token_count", resp.input_token_count)
        print("output_token_count", resp.output_token_count)
        buffer += resp.content
    return buffer

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    #image = sys.argv[1]
    context = ""

    #image_url = upload_to_gcs(image) #      
    image_url = "https://storage.googleapis.com/veyis_bucket/questions/Screenshot%20from%202024-10-15%2011-24-01.png"
    agent_1_out = agent_1_extract_subjects(image_url, context) 
    print(agent_1_out)
    print()