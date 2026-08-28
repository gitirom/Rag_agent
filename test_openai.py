import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY is not set")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

response = client.chat.completions.create(
    model="minimax/minimax-m3:free",
    messages=[
        {
            "role": "user",
            "content": "what a RAG system is?"
        }
    ]
)

print("✅ OpenRouter API is working!")
print()
print(response.choices[0].message.content)