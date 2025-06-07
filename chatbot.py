from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable not set. Please set it before running the app.")

client = Groq(api_key=api_key)

def ask_bot(user_input):
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful assistant answering medical questions about PCOS in simple terms."},
            {"role": "user", "content": user_input}
        ],
        temperature=0.7,
        max_tokens=500,
        top_p=1
    )
    return completion.choices[0].message.content
