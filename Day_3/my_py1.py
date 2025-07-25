import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

response = client.chat.completions.create(
  model="gpt-4.1-nano-2025-04-14",
  messages=[
    {"role": "user", "content": "write a small para about Sachin Tendulkar"}
  ],
  max_tokens=50,
)
print(response.choices[0].message.content)
