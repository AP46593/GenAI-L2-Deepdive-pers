import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)


## Need to check MODEL - Model not supported 

# model = meta-llama/Meta-Llama-3-8B-Instruct
# model = meta-llama/Meta-Llama-3-70B-Instruct
# model = mistralai/Mixtral-8x7B-Instruct-v0.1
# model = google/gemma-7b-it
# model = Qwen/Qwen1.5-72B-Chat
# model = microsoft/Phi-3-mini-4k-instructgoogle/gemma-7b-it"  
#

resp = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",  # not working
    messages=[
        {"role": "user", "content": "Hello, can you tell me a joke about AI?"}
    ],
)

print(resp.choices[0].message.content)