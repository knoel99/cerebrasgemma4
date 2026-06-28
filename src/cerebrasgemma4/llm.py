import os

from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras

load_dotenv()

MODEL = "gemma-4-31b"
_client = Cerebras(api_key=os.environ["CEREBRAS_API_KEY"])


def stream(messages, **kwargs):
    for chunk in _client.chat.completions.create(
        messages=messages,
        model=MODEL,
        stream=True,
        max_completion_tokens=32_768,
        temperature=1.0,
        top_p=0.95,
        **kwargs,
    ):
        yield chunk.choices[0].delta.content or ""